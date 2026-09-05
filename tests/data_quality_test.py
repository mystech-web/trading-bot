"""Valida `src.data_quality.flag_and_clean_outliers` con casos sintéticos donde
se conoce de antemano qué DEBERÍA pasar: un crash real (caída grande, sin
reversión) tiene que sobrevivir intacto -- es la señal que el stress test
necesita ver -- mientras que un bad tick (salto que se revierte) y un precio
inválido (<=0) sí deben limpiarse.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data_quality import flag_and_clean_outliers


def _base_prices(n=60, start=100.0):
    # random walk suave, sin outliers, como base común para todos los tickers.
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.008, n)
    return start * np.exp(np.cumsum(rets))


def test_real_crash_is_not_touched():
    dates = pd.bdate_range("2020-01-01", periods=60)
    prices = _base_prices(60)
    # Crash real: cae 40% en un día y NO se recupera en los siguientes 10 días
    # (se queda ahí, como un crash de mercado real -- no un glitch de datos).
    prices = prices.copy()
    prices[30] *= 0.60
    for i in range(31, 60):
        prices[i] = prices[30] * (1 + np.random.default_rng(i).normal(0, 0.005))

    close = pd.DataFrame({"CRASH_TICKER": prices}, index=dates)
    cleaned, report = flag_and_clean_outliers(close, verbose=False)

    assert report.empty or "CRASH_TICKER" not in set(report.get("ticker", [])), \
        "un crash real (sin reversión) no debería marcarse como outlier"
    pd.testing.assert_series_equal(cleaned["CRASH_TICKER"], close["CRASH_TICKER"], check_names=False)


def test_bad_tick_is_detected_and_cleaned():
    dates = pd.bdate_range("2020-01-01", periods=60)
    prices = _base_prices(60)
    prices = prices.copy()
    original_day30 = prices[30]
    original_day31 = prices[31]
    # Bad tick: el precio se dispara 3x en un día y AL DÍA SIGUIENTE vuelve
    # exactamente a donde hubiera estado sin el salto -- la firma clásica de
    # un error de proveedor de datos que se autocorrige.
    prices[30] = prices[30] * 3.0
    prices[31] = original_day31  # vuelve a la trayectoria "normal"

    close = pd.DataFrame({"GLITCH_TICKER": prices}, index=dates)
    cleaned, report = flag_and_clean_outliers(close, verbose=False)

    assert not report.empty, "el bad tick debería haberse detectado"
    tickers_flagged = set(report["ticker"])
    assert "GLITCH_TICKER" in tickers_flagged

    day30 = dates[30]
    # el precio limpio en el día del glitch debería estar mucho más cerca del
    # valor original (interpolado) que del salto 3x.
    assert abs(cleaned.loc[day30, "GLITCH_TICKER"] - original_day30) < abs(prices[30] - original_day30)
    # el resto de la serie (fuera del día del glitch) no debería tocarse.
    untouched_mask = close.index != day30
    pd.testing.assert_series_equal(
        cleaned.loc[untouched_mask, "GLITCH_TICKER"], close.loc[untouched_mask, "GLITCH_TICKER"], check_names=False)


def test_invalid_nonpositive_price_always_cleaned():
    dates = pd.bdate_range("2020-01-01", periods=30)
    prices = _base_prices(30)
    prices = prices.copy()
    prices[15] = -5.0  # precio inválido, nunca un dato real de mercado

    close = pd.DataFrame({"BAD_PRICE": prices}, index=dates)
    cleaned, report = flag_and_clean_outliers(close, verbose=False)

    assert not report.empty
    assert "precio_invalido" in set(report["type"])
    assert (cleaned["BAD_PRICE"] > 0).all(), "no debería quedar ningún precio <= 0 después de limpiar"


def test_multi_ticker_matrix_shape_preserved():
    """La matriz limpia debe conservar exactamente las mismas fechas y
    columnas que la original -- nunca se eliminan filas ni columnas."""
    dates = pd.bdate_range("2020-01-01", periods=40)
    close = pd.DataFrame({
        "A": _base_prices(40, start=50),
        "B": _base_prices(40, start=200),
        "C": _base_prices(40, start=10),
    }, index=dates)
    close.loc[dates[20], "B"] = close.loc[dates[19], "B"] * 4  # bad tick en B
    close.loc[dates[21], "B"] = close.loc[dates[19], "B"] * 1.01  # revierte

    cleaned, report = flag_and_clean_outliers(close, verbose=False)
    assert list(cleaned.columns) == list(close.columns)
    assert cleaned.index.equals(close.index)
    assert cleaned.notna().all().all(), "no debería quedar ningún NaN tras interpolar"


def main():
    print("[1/4] Probando que un crash real (sin reversión) NO se toca...")
    test_real_crash_is_not_touched()
    print("[2/4] Probando que un bad tick (salto que revierte) se detecta y limpia...")
    test_bad_tick_is_detected_and_cleaned()
    print("[3/4] Probando que un precio <= 0 siempre se limpia...")
    test_invalid_nonpositive_price_always_cleaned()
    print("[4/4] Probando que la forma de la matriz (fechas/columnas) se preserva...")
    test_multi_ticker_matrix_shape_preserved()
    print("\nDATA QUALITY TEST OK: el filtro de outliers distingue crashes reales de errores de datos.")


if __name__ == "__main__":
    main()
