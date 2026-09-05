"""Modelo de costos de transacción sensible a liquidez.

El modelo anterior usaba `cost_bps` fijo para cualquier operación, en cualquier
activo. En la realidad, comprar/vender un ETF gigante como SPY cuesta mucho
menos (spread angosto, casi sin impacto de mercado) que mover el mismo % del
portafolio en una acción menos líquida -- y el costo también crece si la orden
es grande respecto al volumen diario típico del activo (impacto de mercado).

Esta versión modela: costo = spread_base + impacto_de_mercado(participación),
donde participación = tamaño_de_la_orden_en_dólares / volumen_promedio_en_dólares.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def liquidity_adjusted_cost(
    delta_weights: pd.DataFrame,
    avg_dollar_volume: pd.DataFrame,
    base_capital: float = 100_000.0,
    base_spread_bps: float = 3.0,
    impact_coeff: float = 15.0,
    max_cost_bps: float = 50.0,
) -> pd.Series:
    """Costo total diario como fracción del equity.

    `base_capital` es un tamaño de cuenta de referencia usado SOLO para estimar
    qué tan grande es cada orden en dólares (no para llevar la cuenta de equity
    real -- eso lo hace `backtest.py`). Es una aproximación: no compone con el
    crecimiento real del equity, pero es suficiente para capturar la idea de
    que "mover 10% del portafolio en un activo poco líquido cuesta más caro".
    """
    tickers = [c for c in delta_weights.columns if c in avg_dollar_volume.columns]
    dw = delta_weights[tickers]
    adv = avg_dollar_volume[tickers].reindex(dw.index)

    order_notional = dw.abs() * base_capital
    participation = (order_notional / adv.replace(0, np.nan)).fillna(0.0)

    cost_bps_per_asset = base_spread_bps + impact_coeff * participation
    cost_bps_per_asset = cost_bps_per_asset.clip(upper=max_cost_bps)

    cost_frac_per_asset = dw.abs() * (cost_bps_per_asset / 10_000.0)
    return cost_frac_per_asset.sum(axis=1)


def average_dollar_volume(close: pd.DataFrame, volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    tickers = [c for c in close.columns if c in volume.columns]
    dollar_vol = close[tickers] * volume[tickers]
    return dollar_vol.rolling(window, min_periods=max(5, window // 4)).mean()
