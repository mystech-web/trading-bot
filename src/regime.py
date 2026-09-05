"""Filtro de régimen macro: baja la exposición ANTES de que el drawdown ya
haya ocurrido, en vez de solo reaccionar después (eso ya lo hace la guardia de
drawdown en `backtest.py`/`tracking.py`). Es el complemento proactivo.

Hasta TRES señales independientes, combinadas tomando la MÁS conservadora (el
mínimo de todas) día a día:

  1. Tendencia de precio: distancia del benchmark (SPY) a su media móvil de
     200 días. Por encima -> mercado "sano", exposición completa. Por debajo
     -> exposición se reduce suavemente (no de golpe) hasta un piso mínimo.
  2. Volatilidad REALIZADA relativa: razón entre la volatilidad realizada
     reciente (`vol_window` días) y la volatilidad "normal" de ese mismo
     benchmark (mediana de los últimos `vol_baseline_window` días). Un salto de
     volatilidad suele preceder a las caídas grandes -- esta señal reacciona a
     ESO, no al precio, así que puede activarse en mercados que técnicamente
     siguen sobre su SMA200 pero ya empezaron a moverse de forma errática
     (mismo principio que la guardia proactiva de `backtest.py`, aplicado acá
     al régimen macro en vez de al drawdown del portafolio).
  3. Volatilidad IMPLÍCITA (VIX, opcional -- solo si se pasa `vix_close`): a
     diferencia de la señal 2 (que mide qué tan volátil ESTUVO el mercado en
     los últimos días), el VIX es la volatilidad que el mercado de OPCIONES
     espera hacia adelante -- a veces se adelanta a la volatilidad realizada
     (el mercado "sabe" antes de que el precio se mueva de verdad). Umbrales
     ABSOLUTOS (no una razón como la señal 2): VIX < `vix_full_exposure_below`
     (default 20) es un mercado tranquilo; VIX > `vix_floor_above` (default 35)
     ya es estrés serio (niveles típicos de un bear market o un crash agudo).
     Solo aplica a acciones de EE.UU. -- no hay un equivalente líquido y
     gratuito para cripto, así que esta señal simplemente no se usa ahí.

Ninguna de las tres llega a cero por completo -- eso se lo dejamos a la guardia
de drawdown reactiva, que sí puede justificarlo con una caída ya confirmada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import sma

DEFAULT_PARAMS = dict(
    sma_window=200,
    full_exposure_above_pct=0.0,   # a partir de qué distancia (sobre la SMA) ya es exposición 100%
    floor_below_pct=-0.15,          # a qué distancia (bajo la SMA) se toca el piso mínimo
    min_scale=0.40,
    vol_enabled=True,
    vol_window=20,                   # ventana corta: volatilidad realizada "actual"
    vol_baseline_window=252,          # ventana larga: qué es "normal" para este benchmark
    vol_full_exposure_ratio=1.3,       # razón actual/normal por debajo de la cual es exposición 100%
                                         # (>1.0 a propósito: el ruido normal de una ventana corta de
                                         # 20 días ya fluctúa por encima/debajo de la mediana de 252 días
                                         # la mitad del tiempo -- un margen evita "falsos positivos" por
                                         # puro ruido de muestreo en un mercado genuinamente calmado)
    vol_floor_ratio=2.5,                # razón por encima de la cual se toca el piso de volatilidad
    vol_min_scale=0.40,
    vix_enabled=True,
    vix_full_exposure_below=20.0,       # VIX absoluto por debajo del cual exposición completa
    vix_floor_above=35.0,               # VIX absoluto por encima del cual se toca el piso
    vix_min_scale=0.40,
)


def _price_trend_scale(benchmark_close: pd.Series, p: dict) -> pd.Series:
    ma = sma(benchmark_close, p["sma_window"])
    dist = benchmark_close / ma - 1.0
    lo, hi = p["floor_below_pct"], p["full_exposure_above_pct"]
    scale = (dist - lo) / (hi - lo)
    scale = p["min_scale"] + scale.clip(lower=0.0, upper=1.0) * (1.0 - p["min_scale"])
    return scale.fillna(1.0)


def _volatility_scale(benchmark_close: pd.Series, p: dict) -> pd.Series:
    rets = benchmark_close.pct_change()
    short_vol = rets.rolling(p["vol_window"], min_periods=p["vol_window"]).std()
    baseline_vol = rets.rolling(p["vol_baseline_window"], min_periods=p["vol_window"]).std()
    ratio = (short_vol / baseline_vol.replace(0, np.nan)).fillna(1.0)

    lo, hi = p["vol_full_exposure_ratio"], p["vol_floor_ratio"]
    # razón BAJA (<=lo, vol tranquila) -> exposición completa; razón ALTA (>=hi) -> piso.
    scale = (hi - ratio) / (hi - lo)
    scale = p["vol_min_scale"] + scale.clip(lower=0.0, upper=1.0) * (1.0 - p["vol_min_scale"])
    return scale.fillna(1.0)


def _implied_vol_scale(vix_close: pd.Series, p: dict) -> pd.Series:
    lo, hi = p["vix_full_exposure_below"], p["vix_floor_above"]
    # VIX BAJO (<=lo, mercado tranquilo) -> exposición completa; VIX ALTO (>=hi) -> piso.
    scale = (hi - vix_close) / (hi - lo)
    scale = p["vix_min_scale"] + scale.clip(lower=0.0, upper=1.0) * (1.0 - p["vix_min_scale"])
    return scale.fillna(1.0)


def compute_regime_scale(benchmark_close: pd.Series, params: dict | None = None,
                          vix_close: pd.Series | None = None) -> pd.Series:
    """`vix_close` es opcional (default `None`, sin señal de VIX) para que el
    resto del código (crypto, tests existentes) siga funcionando sin tener que
    pasar VIX -- ver `scripts/run_backtest.py`/`run_live_once.py` para de dónde
    sale (descarga separada de `^VIX`, no forma parte del universo invertible)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    scales = [_price_trend_scale(benchmark_close, p)]
    if p.get("vol_enabled", True):
        scales.append(_volatility_scale(benchmark_close, p))
    if p.get("vix_enabled", True) and vix_close is not None:
        vix_aligned = vix_close.reindex(benchmark_close.index).ffill()
        scales.append(_implied_vol_scale(vix_aligned, p))
    if len(scales) == 1:
        return scales[0]
    return pd.concat(scales, axis=1).min(axis=1)
