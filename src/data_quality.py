"""Detecta y limpia probables ERRORES DE DATOS (bad ticks del proveedor) en una
matriz de precios de cierre -- NO detecta ni suaviza crashes reales de mercado,
al contrario: el criterio está diseñado específicamente para distinguir entre
ambos y dejar los crashes intactos (son la señal que las estrategias necesitan
ver, sobre todo para el stress test).

Dos categorías de error que se limpian:
  1. Precio inválido (<= 0) -- nunca es un dato de mercado real.
  2. "Bad tick" -- un salto de precio extremo en un solo día que se REVIERTE
     casi por completo en los días siguientes. Un crash real (ej. COVID
     marzo 2020) también tiene caídas grandes en un día, pero no revierte así
     de rápido -- por eso el criterio combina "movimiento extremo" CON
     "reversión rápida", no solo uno de los dos. Sin la segunda condición, un
     filtro ingenuo de "|retorno| > X%" borraría precisamente los días de
     crash que el stress test necesita conservar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def flag_and_clean_outliers(close: pd.DataFrame, max_daily_return: float = 0.50,
                             reversion_window: int = 3, min_reversion_fraction: float = 0.6,
                             verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    `max_daily_return`: retorno diario absoluto a partir del cual un día se
    considera "extremo" y candidato a revisión (0.50 = 50% -- ya es un
    movimiento diario enorme incluso para una acción individual volátil).

    `reversion_window`: cuántos días hacia adelante se revisan buscando una
    reversión hacia el precio previo al salto.

    `min_reversion_fraction`: qué fracción del salto original debe revertirse
    dentro de esa ventana para tratarlo como bad tick (1.0 = reversión total,
    0.6 = default, revierte al menos el 60% del movimiento).

    Los días marcados se ponen en NaN y se interpolan linealmente entre los
    precios válidos vecinos (nunca se eliminan filas -- rompería la
    alineación de fechas entre tickers en `close`).

    Devuelve (close_limpio, reporte). `reporte` tiene una fila por outlier
    detectado: ticker, fecha, tipo, retorno original, fracción revertida.
    """
    cleaned = close.copy()
    flagged_rows: list[dict] = []
    tickers_to_interpolate: set[str] = set()

    for ticker in close.columns:
        s = close[ticker].dropna()
        if len(s) < reversion_window + 2:
            continue

        invalid_days = set(s.index[s <= 0])
        for day in invalid_days:
            cleaned.loc[day, ticker] = np.nan
            tickers_to_interpolate.add(ticker)
            flagged_rows.append(dict(ticker=ticker, date=day, type="precio_invalido",
                                      original_return=np.nan, reverted_fraction=np.nan))

        rets = s.pct_change()
        suspect_days = rets[rets.abs() > max_daily_return].index
        for day in suspect_days:
            if day in invalid_days:
                continue  # ya cubierto arriba como precio inválido
            loc = s.index.get_loc(day)
            if loc == 0:
                continue
            price_before = s.iloc[loc - 1]
            price_at = s.iloc[loc]
            move = price_at - price_before
            if move == 0 or price_before <= 0:
                continue

            end_loc = min(loc + reversion_window, len(s) - 1)
            future_prices = s.iloc[loc + 1:end_loc + 1]
            if future_prices.empty:
                continue
            closest_to_before = future_prices.sub(price_before).abs().min()
            reverted_fraction = 1.0 - (closest_to_before / abs(move))

            if reverted_fraction >= min_reversion_fraction:
                cleaned.loc[day, ticker] = np.nan
                tickers_to_interpolate.add(ticker)
                flagged_rows.append(dict(ticker=ticker, date=day, type="bad_tick_revertido",
                                          original_return=round(rets.loc[day], 4),
                                          reverted_fraction=round(reverted_fraction, 3)))

    for ticker in tickers_to_interpolate:
        cleaned[ticker] = cleaned[ticker].interpolate(method="linear", limit_direction="both")

    if flagged_rows and verbose:
        print(f"[data_quality] {len(flagged_rows)} outlier(s) de precio detectados y limpiados "
              f"(errores de datos probables, NO crashes reales -- ver reporte para el detalle).")

    report = pd.DataFrame(flagged_rows)
    if not report.empty:
        report = report.sort_values(["date", "ticker"]).reset_index(drop=True)
    return cleaned, report
