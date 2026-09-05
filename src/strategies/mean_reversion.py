"""Estrategia de reversión a la media: RSI(2) corto plazo, solo en tendencia alcista de fondo.

Regla por activo (Connors RSI-2 simplificado):
  - Filtro de tendencia: precio > SMA de largo plazo (ej. 200d) -> solo se buscan compras.
  - Entrada: RSI(2) cae por debajo de `entry_rsi` (sobreventa extrema de corto plazo).
  - Salida: RSI(2) sube por encima de `exit_rsi`, o pasan `max_hold_days`, o el precio
    cae por debajo del stop (lo que ocurra primero). El stop es TRAILING por defecto
    (`use_trailing_stop=True`): se mide desde el máximo alcanzado DESDE LA ENTRADA, no
    desde el precio de entrada -- si la posición sube y después retrocede, protege la
    ganancia ya generada en vez de esperar a que retroceda todo el camino de vuelta al
    precio de entrada. `use_trailing_stop=False` recupera el stop fijo (`stop_loss_pct`,
    medido siempre desde la entrada).

Tamaño de posición por PARIDAD DE RIESGO: en vez de un peso fijo por posición,
el tamaño se escala inversamente a la volatilidad del activo en el momento de
la entrada (un activo el doble de volátil recibe la mitad de tamaño), capado
por `max_weight_by_ticker` (acciones individuales con tope más bajo que ETFs,
por riesgo de supervivencia).

Es una estrategia "state machine" (necesita saber si hay posición abierta) --
el loop día a día no se puede vectorizar con operaciones de pandas, pero SÍ se
puede compilar: si `numba` está instalado, el loop corre compilado a código
máquina (10-50x más rápido que Python puro) en vez de interpretado. Si no está
instalado, cae de vuelta al mismo loop en Python normal -- mismo resultado,
solo más lento. `numba` es opcional a propósito (no todos los entornos lo
instalan limpio) -- ver requirements.txt.
"""
import numpy as np
import pandas as pd

from src.indicators import sma, rsi, realized_vol

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def _decorator(func):
            return func
        return _decorator if not (args and callable(args[0])) else args[0]

DEFAULT_PARAMS = dict(
    trend_sma=200,
    rsi_window=2,
    entry_rsi=10.0,
    exit_rsi=70.0,
    max_hold_days=10,
    stop_loss_pct=0.06,
    max_concurrent_positions=5,
    weight_per_position=0.10,
    vol_lookback=20,
    reference_vol=0.15,  # volatilidad anualizada "típica" contra la que se escala el tamaño
    periods_per_year=252,  # 252 (días hábiles, acciones) o 365 (cripto, cotiza todos los días)
    # Stop TRAILING (desde el máximo alcanzado desde la entrada) en vez de fijo (desde el
    # precio de entrada): protege ganancias ya generadas si la posición sube y después
    # retrocede, en vez de dar de vuelta toda la ganancia antes de salir. Con
    # use_trailing_stop=True, `trailing_stop_pct` reemplaza a `stop_loss_pct` como
    # distancia de salida (el mismo valor por defecto, pero medida desde el máximo, no
    # desde la entrada -- antes de que el precio suba, máximo == entrada, así que el
    # comportamiento en la entrada es idéntico al stop fijo; solo diverge después de que
    # la posición ya está en ganancia). use_trailing_stop=False recupera el stop fijo.
    use_trailing_stop=True,
    trailing_stop_pct=0.06,
)


@njit(cache=True)
def _state_machine_loop(close: np.ndarray, trend_ok: np.ndarray, rsi_vals: np.ndarray, vol_vals: np.ndarray,
                         entry_rsi: float, exit_rsi: float, max_hold_days: int, stop_loss_pct: float,
                         reference_vol: float, use_trailing_stop: bool,
                         trailing_stop_pct: float) -> tuple[np.ndarray, np.ndarray]:
    n = close.shape[0]
    signal = np.zeros(n)
    size = np.zeros(n)

    in_position = False
    entry_price = 0.0
    peak_price = 0.0
    entry_idx = -1
    size_mult = 1.0

    for i in range(n):
        if not in_position:
            if trend_ok[i] and rsi_vals[i] < entry_rsi:
                in_position = True
                entry_price = close[i]
                peak_price = close[i]
                entry_idx = i
                v = vol_vals[i]
                if v > 0:
                    sm = reference_vol / v
                    if sm < 0.5:
                        sm = 0.5
                    elif sm > 2.0:
                        sm = 2.0
                    size_mult = sm
                else:
                    size_mult = 1.0
            signal[i] = 1.0 if in_position else 0.0
            size[i] = size_mult if in_position else 0.0
        else:
            days_held = i - entry_idx
            price = close[i]
            if price > peak_price:
                peak_price = price
            if use_trailing_stop:
                hit_stop = price <= peak_price * (1 - trailing_stop_pct)
            else:
                hit_stop = price <= entry_price * (1 - stop_loss_pct)
            hit_exit_rsi = rsi_vals[i] > exit_rsi
            hit_max_hold = days_held >= max_hold_days
            signal[i] = 1.0
            size[i] = size_mult
            if hit_stop or hit_exit_rsi or hit_max_hold:
                in_position = False
                entry_price = 0.0
                peak_price = 0.0
                entry_idx = -1

    return signal, size


def _signals_for_ticker(close: pd.Series, params: dict) -> tuple[pd.Series, pd.Series]:
    """Devuelve (señal 0/1 de tener posición, tamaño relativo por paridad de riesgo
    fijado el día de la entrada y sostenido mientras dure la posición)."""
    trend_ok = close > sma(close, params["trend_sma"])
    r = rsi(close, params["rsi_window"])
    vol = realized_vol(close.pct_change(), window=params["vol_lookback"], periods_per_year=params["periods_per_year"])

    # NaN en cualquiera de estos (calentamiento de los indicadores) se comporta
    # igual que en pandas: cualquier comparación con NaN da False -- por eso es
    # seguro pasar los arrays "crudos" (con NaN) al loop compilado sin limpiarlos antes.
    signal_arr, size_arr = _state_machine_loop(
        close.to_numpy(dtype=np.float64),
        trend_ok.to_numpy(dtype=bool),
        r.to_numpy(dtype=np.float64),
        vol.to_numpy(dtype=np.float64),
        float(params["entry_rsi"]), float(params["exit_rsi"]),
        int(params["max_hold_days"]), float(params["stop_loss_pct"]),
        float(params["reference_vol"]), bool(params.get("use_trailing_stop", True)),
        float(params.get("trailing_stop_pct", params["stop_loss_pct"])),
    )
    return pd.Series(signal_arr, index=close.index), pd.Series(size_arr, index=close.index)


def generate_weights(close: pd.DataFrame, tickers: list[str], params: dict | None = None,
                      max_weight_by_ticker: dict[str, float] | None = None) -> pd.DataFrame:
    p = {**DEFAULT_PARAMS, **(params or {})}
    sub = close[tickers]

    signals, sizes = {}, {}
    for t in tickers:
        sig, sz = _signals_for_ticker(sub[t].dropna(), p)
        signals[t] = sig
        sizes[t] = sz

    raw_signal = pd.DataFrame(signals).reindex(sub.index).fillna(0.0)
    size_mult = pd.DataFrame(sizes).reindex(sub.index).fillna(0.0)

    n_positions = raw_signal.sum(axis=1)
    scale_down = (p["max_concurrent_positions"] / n_positions.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)

    base_weight = raw_signal.mul(p["weight_per_position"], axis=0) * size_mult
    weights = base_weight.mul(scale_down, axis=0)

    if max_weight_by_ticker:
        cap = pd.Series({t: max_weight_by_ticker.get(t, p["weight_per_position"] * 2) for t in tickers})
        weights = weights.clip(upper=cap, axis=1)
    return weights
