"""Indicadores técnicos calculados solo con pandas/numpy (sin dependencias externas)."""
import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out[avg_loss == 0] = 100.0
    return out


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return lower, mid, upper


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1)
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def realized_vol(returns: pd.Series, window: int = 20, annualize: bool = True,
                  periods_per_year: int = 252) -> pd.Series:
    vol = returns.rolling(window, min_periods=window).std()
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def momentum_12_1(close: pd.Series) -> pd.Series:
    """Momentum de 12 meses excluyendo el último mes (factor clásico 12-1), en
    días hábiles de bolsa (252/año). Para activos que cotizan todos los días
    del año (cripto), usa `momentum_n_m` con `long_period=365, short_period=30`."""
    return momentum_n_m(close, long_period=252, short_period=21)


def momentum_n_m(close: pd.Series, long_period: int, short_period: int) -> pd.Series:
    """Momentum "N-M": retorno entre hace `long_period` y hace `short_period`
    barras (excluye el tramo más reciente, que tiende a revertir en el muy
    corto plazo). Generaliza `momentum_12_1` a cualquier calendario de cotización."""
    return close.shift(short_period) / close.shift(long_period) - 1.0
