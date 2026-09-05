"""Valida la señal de volatilidad IMPLÍCITA (VIX) del filtro de régimen macro
(`src.regime.compute_regime_scale`): un pico de VIX debería reducir la
exposición aunque el precio del benchmark siga sano (sobre su SMA200) y su
volatilidad REALIZADA siga tranquila -- exactamente el caso donde las otras
dos señales no reaccionarían, pero el mercado de opciones ya está señalando
estrés hacia adelante.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.regime import compute_regime_scale, DEFAULT_PARAMS, _price_trend_scale


def _make_benchmark(n=400, seed=9):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    rets = rng.normal(0.0006, 0.004, n)  # tendencia alcista sostenida + vol realizada calma
    price = 100 * np.cumprod(1 + rets)
    return pd.Series(price, index=dates)


def _make_vix(benchmark: pd.Series, spike_start=300, spike_end=320, calm=15.0, spike=45.0) -> pd.Series:
    vix = pd.Series(calm, index=benchmark.index)
    vix.iloc[spike_start:spike_end] = spike
    return vix


def test_vix_spike_dampens_exposure_even_when_price_and_realized_vol_are_calm():
    benchmark = _make_benchmark()
    vix = _make_vix(benchmark)

    price_scale = _price_trend_scale(benchmark, DEFAULT_PARAMS)
    only_vix = compute_regime_scale(benchmark, dict(vol_enabled=False), vix_close=vix)

    peak_idx = benchmark.index[305]
    assert price_scale.loc[peak_idx] == 1.0, \
        "el escenario está diseñado para que el precio se mantenga sano (sin reaccionar) durante el pico de VIX"
    assert only_vix.loc[peak_idx] < 1.0, \
        "un pico de VIX por sí solo debería reducir la exposición, aunque el precio no reaccione"
    assert only_vix.loc[peak_idx] == DEFAULT_PARAMS["vix_min_scale"], \
        f"con VIX=45 (por encima de vix_floor_above=35), debería tocar el piso vix_min_scale: {only_vix.loc[peak_idx]}"
    print(f"  OK: VIX=45 reduce la exposición a {only_vix.loc[peak_idx]:.2f} pese a que precio y vol realizada "
          f"siguen tranquilos")


def test_vix_none_is_backward_compatible_no_op():
    """Sin pasar `vix_close` (el default), el comportamiento debe ser IDÉNTICO
    al de antes de este cambio -- código/tests existentes que llaman
    `compute_regime_scale(benchmark, params)` sin el nuevo argumento no deberían
    verse afectados."""
    benchmark = _make_benchmark()
    no_vix_arg = compute_regime_scale(benchmark, dict(vol_enabled=False))
    explicit_none = compute_regime_scale(benchmark, dict(vol_enabled=False), vix_close=None)
    pd.testing.assert_series_equal(no_vix_arg, explicit_none)
    assert (no_vix_arg == 1.0).all(), "sin VIX, con vol_enabled=False, la escala debería ser exposición completa"


def test_vix_disabled_flag_ignores_vix_even_if_provided():
    benchmark = _make_benchmark()
    vix = _make_vix(benchmark)
    scale = compute_regime_scale(benchmark, dict(vol_enabled=False, vix_enabled=False), vix_close=vix)
    assert (scale == 1.0).all(), "con vix_enabled=False, un vix_close provisto no debería tener ningún efecto"


def test_combined_scale_is_minimum_across_all_three_signals():
    """Cuando la señal de volatilidad realizada Y la de VIX están activas a la
    vez, la escala combinada debería ser el mínimo de las tres (nunca más alta
    que ninguna de las señales individuales)."""
    benchmark = _make_benchmark()
    vix = _make_vix(benchmark)

    price_scale = _price_trend_scale(benchmark, DEFAULT_PARAMS)
    combined = compute_regime_scale(benchmark, dict(vol_enabled=True, vol_window=10, vol_baseline_window=252),
                                     vix_close=vix)
    only_vix = compute_regime_scale(benchmark, dict(vol_enabled=False), vix_close=vix)

    assert (combined <= price_scale + 1e-9).all(), "la combinada nunca debería superar la señal de precio"
    assert (combined <= only_vix + 1e-9).all(), "la combinada nunca debería superar la señal de VIX sola"


def test_vix_series_with_different_index_is_aligned_via_reindex_and_ffill():
    """`vix_close` puede venir con un índice de fechas levemente distinto al del
    benchmark (proveedores de datos distintos) -- debería alinearse con
    `reindex().ffill()`, no explotar ni desalinear la señal."""
    benchmark = _make_benchmark()
    vix = _make_vix(benchmark)
    # Simula un VIX con menos historia al principio (como si el proveedor solo
    # tuviera datos desde más tarde) -- los días previos deberían caer en NaN
    # tras el reindex y, por lo tanto, NO dampear (fillna(1.0), fail-safe).
    vix_short = vix.iloc[50:]

    scale = compute_regime_scale(benchmark, dict(vol_enabled=False), vix_close=vix_short)
    early_days = benchmark.index[:50]
    assert (scale.loc[early_days] == 1.0).all(), \
        "sin dato de VIX todavía (antes de que empiece su historia), no debería dampear -- default a 1.0"
    peak_idx = benchmark.index[305]
    assert scale.loc[peak_idx] == DEFAULT_PARAMS["vix_min_scale"], \
        "una vez que hay historia de VIX, el pico debería seguir detectándose igual"
    print("  OK: un VIX con menos historia se alinea correctamente (sin dampear antes de tener datos)")


def main():
    print("[1/5] Probando que un pico de VIX reduce exposición aunque precio y vol realizada estén tranquilos...")
    test_vix_spike_dampens_exposure_even_when_price_and_realized_vol_are_calm()
    print("\n[2/5] Probando que vix_close=None es un no-op retrocompatible...")
    test_vix_none_is_backward_compatible_no_op()
    print("\n[3/5] Probando que vix_enabled=False ignora el VIX aunque se provea...")
    test_vix_disabled_flag_ignores_vix_even_if_provided()
    print("\n[4/5] Probando que la escala combinada es el mínimo de las tres señales...")
    test_combined_scale_is_minimum_across_all_three_signals()
    print("\n[5/5] Probando que un VIX con índice distinto se alinea correctamente (reindex + ffill)...")
    test_vix_series_with_different_index_is_aligned_via_reindex_and_ffill()
    print("\nREGIME VIX TEST OK: la señal de volatilidad implícita funciona correctamente.")


if __name__ == "__main__":
    main()
