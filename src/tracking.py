"""Registro del equity real de la cuenta en cada corrida, guardia de drawdown
persistente entre corridas, y comparación del desempeño en vivo contra la
banda esperada (percentiles del Monte Carlo del backtest) para detectar que
la estrategia "se rompió" lo antes posible, no meses después.

Guardado en SQLite (no CSV/JSON sueltos): con varios perfiles y varios brokers
corriendo por cron -- potencialmente al mismo tiempo -- un `to_csv()` que
trunca y reescribe todo el archivo puede corromperse si dos corridas escriben
a la vez. SQLite maneja eso solo: cada escritura es una transacción atómica, y
con `PRAGMA busy_timeout` un escritor concurrente ESPERA en vez de fallar o
pisar al otro.

Todas las funciones aceptan `reports_dir` opcional -- cada PERFIL (conservador
vs. agresivo vs. cripto) Y cada BROKER dentro de un perfil (ej. Alpaca real vs.
el broker virtual, o Binance vs. Bitso vs. virtual en cripto) deben pasar un
`reports_dir` distinto -- ver `scripts/run_live_once.py` y
`scripts/run_crypto_live_once.py`, que arman `reports_dir/<broker>/` para que
dos brokers del mismo perfil nunca compartan guardia de drawdown ni historial
de equity (antes de este cambio sí lo compartían -- era un bug real).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = ROOT / "reports"

# Compatibilidad hacia atrás para código/tests que importan estos nombres directamente.
REPORTS_DIR = DEFAULT_REPORTS_DIR
DB_PATH = REPORTS_DIR / "tracking.sqlite3"
EXPECTED_BAND_FILE = REPORTS_DIR / "monte_carlo.json"


def _db_path(reports_dir: Path | None = None) -> Path:
    d = reports_dir or DEFAULT_REPORTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / "tracking.sqlite3"


@contextmanager
def _connect(reports_dir: Path | None = None):
    conn = sqlite3.connect(_db_path(reports_dir), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")  # espera hasta 10s si otra corrida tiene el lock, en vez de fallar
    conn.execute("CREATE TABLE IF NOT EXISTS equity_log (date TEXT PRIMARY KEY, equity REAL NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def append_equity(date: pd.Timestamp, equity: float, reports_dir: Path | None = None) -> pd.DataFrame:
    date_str = pd.Timestamp(date).normalize().strftime("%Y-%m-%d")
    with _connect(reports_dir) as conn:
        conn.execute(
            "INSERT INTO equity_log (date, equity) VALUES (?, ?) "
            "ON CONFLICT(date) DO UPDATE SET equity = excluded.equity",
            (date_str, float(equity)),
        )
    return load_equity_log(reports_dir)


def load_equity_log(reports_dir: Path | None = None) -> pd.DataFrame:
    with _connect(reports_dir) as conn:
        df = pd.read_sql_query("SELECT date, equity FROM equity_log ORDER BY date", conn)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
    else:
        df["date"] = pd.to_datetime(df.get("date", pd.Series(dtype="object")))
    return df


def load_state(reports_dir: Path | None = None) -> dict:
    with _connect(reports_dir) as conn:
        rows = conn.execute("SELECT key, value FROM state").fetchall()
    state = {key: json.loads(value) for key, value in rows}
    state.setdefault("peak_equity", None)
    state.setdefault("guard_active", False)
    state.setdefault("exposure_scale", 1.0)
    return state


def save_state(state: dict, reports_dir: Path | None = None) -> None:
    with _connect(reports_dir) as conn:
        for key, value in state.items():
            conn.execute(
                "INSERT INTO state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )


def _historical_peak_from_log(fallback: float, reports_dir: Path | None = None) -> float:
    """Semilla del pico cuando todavía no hay estado guardado. Usar `fallback`
    (el equity de hoy) sería un bug: si el bot empieza a correr justo en medio
    de una caída ya en curso, tomaría ese valor bajo como "el pico" y nunca
    activaría la guardia para esa caída. En vez de eso, se busca el máximo ya
    registrado en el log de equity (que puede incluir el valor de hoy si
    `append_equity` ya corrió antes de esta llamada)."""
    df = load_equity_log(reports_dir)
    if len(df):
        return float(df["equity"].max())
    return fallback


def update_drawdown_guard(equity: float, threshold: float = -0.15, recover: float = -0.07,
                           dd_guard_scale: float = 0.5, guard_ramp_days: int = 5,
                           reports_dir: Path | None = None) -> tuple[float, float, bool, bool]:
    """Misma lógica de histéresis que `src/backtest.py`, pero persistida en disco
    entre corridas diarias (el backtest la lleva en memoria dentro de un solo run).

    Reactivación GRADUAL (mismo mecanismo que `run_backtest` en `src/backtest.py`):
    una vez que el drawdown se recupera, la exposición sube en pasos iguales
    CORRIDA A CORRIDA (una corrida = normalmente un día, vía cron) hasta volver a
    1.0, en vez de saltar de golpe de `dd_guard_scale` a exposición completa el
    mismo día que se cumple la condición de recuperación -- eso evita quedar
    totalmente expuesto de nuevo justo cuando la recuperación todavía podría ser
    un rebote falso. `guard_ramp_days=0` recupera el salto instantáneo (el
    comportamiento de antes de este cambio).

    Devuelve (drawdown_actual, exposure_scale_actual [1.0 = exposición completa,
    `dd_guard_scale` = mínimo con la guardia recién activada], guardia_activa_ahora
    [True mientras exposure_scale < 1.0, ya sea recién activada o a mitad de la
    rampa de vuelta], cambio_de_estado_en_esta_corrida [solo al activarse por
    primera vez o al completar la rampa de vuelta a 1.0])."""
    state = load_state(reports_dir)
    peak = state.get("peak_equity")
    if peak is None:
        peak = _historical_peak_from_log(fallback=equity, reports_dir=reports_dir)
    peak = max(peak, equity)
    dd = equity / peak - 1.0

    was_active = state.get("guard_active", False)
    prev_scale = state.get("exposure_scale", 1.0)

    if dd <= threshold:
        scale = dd_guard_scale
    elif dd >= recover:
        if guard_ramp_days > 0:
            step = (1.0 - dd_guard_scale) / guard_ramp_days
            scale = min(1.0, prev_scale + step)
        else:
            scale = 1.0
    else:
        scale = prev_scale  # histéresis: ni se activa ni se libera, mantiene lo que había

    guard_active = scale < 1.0 - 1e-9
    changed = guard_active != was_active
    save_state({"peak_equity": peak, "guard_active": guard_active, "exposure_scale": scale}, reports_dir)
    return dd, scale, guard_active, changed


def load_expected_band(reports_dir: Path | None = None, band_dir: Path | None = None) -> dict | None:
    """`band_dir`: carpeta donde vive `monte_carlo.json` (lo genera el backtest,
    ej. `reports_crypto/`), si es distinta de `reports_dir` (donde vive el
    tracking de ESTE broker en particular, ej. `reports_crypto/binance/`).
    Si no se pasa, se asume igual a `reports_dir` (el caso de acciones, donde
    hoy no hay sub-carpeta por broker separada del backtest)."""
    d = band_dir if band_dir is not None else (reports_dir or DEFAULT_REPORTS_DIR)
    path = d / "monte_carlo.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def check_drift(min_days: int = 20, reports_dir: Path | None = None, band_dir: Path | None = None) -> str | None:
    """None si todo está dentro de lo esperado (o si aún no hay suficiente
    historial en vivo para juzgar). Un string con el mensaje de alerta si el
    retorno mensual promedio realizado cae por debajo del percentil 5 que el
    Monte Carlo del backtest consideraba plausible. Ver `load_expected_band`
    para qué es `band_dir`."""
    df = load_equity_log(reports_dir)
    if len(df) < min_days:
        return None

    band = load_expected_band(reports_dir, band_dir)
    if not band:
        return None

    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    months = max(days / 30.4, 1e-6)
    realized_total_return = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1
    realized_avg_monthly = (1 + realized_total_return) ** (1 / months) - 1

    p5 = band["avg_monthly_return"]["p5"]
    if realized_avg_monthly < p5:
        return (
            f"Retorno mensual promedio realizado en vivo ({realized_avg_monthly * 100:.2f}%) "
            f"está por debajo del percentil 5 que el backtest consideraba plausible "
            f"({p5 * 100:.2f}%). Posible cambio de régimen o decay de la estrategia -- "
            f"revisa manualmente antes de seguir operando."
        )
    return None
