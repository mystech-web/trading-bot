"""Valida la dimensión de volatilidad del filtro de régimen macro
(`src.regime.compute_regime_scale`) con datos sintéticos: un tramo calmado
(vol normal), seguido de un salto de volatilidad SIN que el precio caiga bajo
su SMA200 -- el escenario exacto donde la señal de precio sola no reacciona,
pero un mercado real que empieza a moverse erráticamente sí suele ser una
señal de alerta temprana."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.regime import compute_regime_scale, DEFAULT_PARAMS


def _make_benchmark(n=400, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    drift = 0.0008
    calm_vol = 0.006

    # Tramo calmado largo (350 días) + un tramo corto de "whipsaw" al final (vol
    # ~6x, oscilando alrededor de la misma tendencia, sin romper la SMA200 hacia
    # abajo). El tramo calmado tiene que ser bien más largo que la ventana base
    # de volatilidad (`vol_baseline_window`) -- si no, la propia ventana larga
    # termina "absorbiendo" el whipsaw y dejando de detectarlo como anómalo.
    calm = rng.normal(drift, calm_vol, 350)
    whipsaw = np.array([drift + (0.04 if i % 2 == 0 else -0.038) for i in range(n - 350)])
    returns = np.concatenate([calm, whipsaw])
    price = 100 * np.cumprod(1 + returns)
    return pd.Series(price, index=dates)


def test_volatility_dampens_exposure_even_above_sma():
    benchmark = _make_benchmark()
    ma200 = benchmark.rolling(200).mean()

    price_only = compute_regime_scale(benchmark, dict(vol_enabled=False))
    vol_aware = compute_regime_scale(benchmark, dict(vol_enabled=True, vol_window=10, vol_baseline_window=252))

    # Solo los primeros días del whipsaw (no todo el tramo): la ventana base de
    # 252 días recién empieza a "contaminarse" con datos del whipsaw a medida que
    # este avanza -- el efecto es más claro justo al arrancar.
    tail = benchmark.index[350:365]
    still_above_sma = (benchmark.loc[tail] > ma200.loc[tail]).all()
    assert still_above_sma, "el escenario debía mantenerse sobre la SMA200 durante el whipsaw (así está diseñado)"

    print(f"  escala solo-precio al inicio del whipsaw (sobre SMA200): "
          f"{price_only.loc[tail].round(3).unique()}")
    print(f"  escala con volatilidad en el mismo tramo: {vol_aware.loc[tail].min():.3f} (mínimo)")

    assert (price_only.loc[tail] == 1.0).all(), \
        "la señal de solo-precio no debería reaccionar (el precio nunca cruza bajo la SMA200)"
    assert (vol_aware.loc[tail] < 1.0).any(), \
        "la señal con volatilidad SÍ debería reducir exposición durante el whipsaw, aunque el precio siga arriba"
    assert (vol_aware <= price_only + 1e-9).all(), \
        "la escala combinada (mínimo de las dos señales) nunca debería ser más alta que la de solo-precio"


def test_vol_disabled_matches_price_only_exactly():
    benchmark = _make_benchmark()
    price_only = compute_regime_scale(benchmark, dict(vol_enabled=False))
    default_would_be_price_only = compute_regime_scale(
        benchmark, {**DEFAULT_PARAMS, "vol_enabled": False})
    pd.testing.assert_series_equal(price_only, default_would_be_price_only)
    assert (price_only >= DEFAULT_PARAMS["min_scale"] - 1e-9).all()
    assert (price_only <= 1.0).all()


def test_calm_market_stays_at_full_exposure():
    """En el tramo calmado, la volatilidad reciente es igual a la baseline (misma
    distribución) -> razón ~1.0 -> no debería activar el piso de volatilidad."""
    benchmark = _make_benchmark()
    vol_aware = compute_regime_scale(benchmark, dict(vol_enabled=True, vol_window=10, vol_baseline_window=100))
    calm_middle = benchmark.index[150:250]  # bien adentro del tramo calmado, con historia suficiente
    assert (vol_aware.loc[calm_middle] > 0.9).mean() > 0.8, \
        "en un tramo calmado y estable, la mayoría de los días deberían quedar cerca de exposición completa"


def main():
    print("[1/3] Probando que la señal de volatilidad reduce exposición aunque el precio siga sobre la SMA200...")
    test_volatility_dampens_exposure_even_above_sma()
    print("\n[2/3] Probando que vol_enabled=False reproduce exactamente la señal de solo-precio...")
    test_vol_disabled_matches_price_only_exactly()
    print("\n[3/3] Probando que un mercado calmado se queda cerca de exposición completa...")
    test_calm_market_stays_at_full_exposure()
    print("\nREGIME VOLATILITY TEST OK: la dimensión de volatilidad del filtro de régimen funciona correctamente.")


if __name__ == "__main__":
    main()
