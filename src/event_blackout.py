"""Blackouts (opt-in) para evitar operar justo alrededor de eventos binarios
conocidos que pueden mover el precio de golpe (gap), sin que sea una señal de
tendencia real -- rebalancear justo ESE día puede ejecutar a precios de ruido
del evento, no de la tendencia que la estrategia cree estar siguiendo.

Dos mecanismos, con alcances y limitaciones DISTINTOS -- léelos con atención:

  1. Blackout de FOMC (`freeze_weights_on_blackout_days`): en las fechas de
     reunión de la Reserva Federal (`config/macro_calendar.yaml`), el
     portafolio COMPLETO no rebalancea -- mantiene los pesos del día hábil
     anterior. Funciona tanto en el BACKTEST (las fechas son históricas y
     públicas) como en vivo. LIMITACIÓN HONESTA: `config/macro_calendar.yaml`
     viene con una lista de partida, NO un calendario mantenido
     automáticamente -- revísala y actualízala contra el calendario oficial
     (https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) antes
     de confiar en el backtest para medir su efecto histórico completo. Una
     fecha faltante simplemente no aplica el blackout ese día -- no hay ningún
     efecto "silencioso" incorrecto, solo cobertura incompleta si la lista
     está desactualizada.

  2. Blackout de earnings (`get_earnings_blackout_tickers`): para acciones
     individuales, cerca de su reporte de resultados. SOLO PARA USO EN VIVO --
     yfinance solo expone la PRÓXIMA fecha de reporte de cada ticker (un
     calendario hacia ADELANTE), no un historial point-in-time confiable para
     reconstruir en el backtest, así que esta guardia nunca aparece en ningún
     backtest de este proyecto (una discrepancia real y conocida entre
     backtest y ejecución en vivo, documentada acá en vez de escondida).
     "Best effort": si la consulta a yfinance falla, tarda, o no devuelve
     nada, ese ticker simplemente NO se bloquea -- nunca se asume "en
     blackout" por falta de datos (falla del lado seguro, no del lado
     paranoico).
"""
from __future__ import annotations

import datetime as dt
import pathlib

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent

DEFAULT_EARNINGS_BLACKOUT_DAYS_BEFORE = 1
DEFAULT_EARNINGS_BLACKOUT_DAYS_AFTER = 1


def load_macro_calendar(path: pathlib.Path | None = None) -> set[pd.Timestamp]:
    """Lee `config/macro_calendar.yaml` -> `fomc_decision_dates`. Si el archivo
    no existe o está vacío, devuelve un set vacío (sin blackout, no un error)."""
    path = path or (ROOT / "config" / "macro_calendar.yaml")
    if not path.exists():
        return set()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    dates = raw.get("fomc_decision_dates", []) or []
    return {pd.Timestamp(d).normalize() for d in dates}


def freeze_weights_on_blackout_days(weights: pd.DataFrame, blackout_dates: set[pd.Timestamp]) -> pd.DataFrame:
    """En cada fecha de `blackout_dates` presente en el índice de `weights`,
    reemplaza los pesos de ese día por los del último día hábil anterior que NO
    sea blackout -- "congela" la cartera en vez de rebalancear un día de
    volatilidad anormal por un evento binario conocido. Blackouts consecutivos
    encadenan correctamente (cada uno hereda del último día no-blackout). Si el
    primer día del histórico es blackout (no hay nada previo de qué heredar),
    cae a 0 en vez de dejar NaN -- conservador, nunca inventa un valor."""
    if not blackout_dates:
        return weights
    out = weights.copy()
    is_blackout = out.index.normalize().isin(blackout_dates)
    if not is_blackout.any():
        return out
    out.loc[is_blackout] = np.nan
    return out.ffill().fillna(0.0)


def get_earnings_blackout_tickers(
    tickers: list[str], as_of: dt.date,
    days_before: int = DEFAULT_EARNINGS_BLACKOUT_DAYS_BEFORE,
    days_after: int = DEFAULT_EARNINGS_BLACKOUT_DAYS_AFTER,
) -> set[str]:
    """SOLO para uso en vivo -- ver el docstring del módulo. Devuelve el
    subconjunto de `tickers` cuyo próximo (o más reciente) reporte de
    resultados conocido cae dentro de `[as_of - days_before, as_of + days_after]`."""
    import yfinance as yf  # import perezoso -- el backtest nunca llama a esta función

    blocked = set()
    for t in tickers:
        try:
            cal = yf.Ticker(t).get_earnings_dates(limit=4)
            if cal is None or cal.empty:
                continue
            for ts in cal.index:
                event_date = pd.Timestamp(ts).tz_localize(None).normalize().date()
                window_start = event_date - dt.timedelta(days=days_before)
                window_end = event_date + dt.timedelta(days=days_after)
                if window_start <= as_of <= window_end:
                    blocked.add(t)
                    break
        except Exception:
            continue  # best-effort: un ticker que falla no bloquea a los demás ni hace fallar la corrida
    return blocked


def apply_earnings_blackout(target_weights: dict[str, float], blackout_tickers: set[str]) -> dict[str, float]:
    """Pone en 0 el peso objetivo de cualquier ticker en `blackout_tickers` --
    el resto de los pesos NO se renormaliza (mismo criterio que el resto de los
    overlays de este proyecto, ver `src/portfolio_overlays.py`)."""
    return {t: (0.0 if t in blackout_tickers else w) for t, w in target_weights.items()}
