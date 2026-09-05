"""Valida `src.ensemble.apply_portfolio_vol_target` con datos sintéticos
deterministas: un tramo calmado (volatilidad realizada ~= el objetivo, no
debería escalar casi nada) seguido de un tramo de alta volatilidad (debería
reducir la exposición efectiva), y confirma que la decisión es CAUSAL -- el
primer día del tramo de alta volatilidad todavía no está escalado (la
volatilidad de ese día recién se conoce DESPUÉS de vivirlo).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.ensemble import apply_portfolio_vol_target


def _make_returns():
    """250 días calmados (vol diaria ~0.63%, target anualizado 10% => vol diaria
    objetivo = 0.10/sqrt(252) ~= 0.0063, así que el tramo calmado empieza
    prácticamente EN el objetivo) + 60 días de alta volatilidad (~3x)."""
    rng = np.random.default_rng(21)
    target_daily_vol = 0.10 / np.sqrt(252)
    calm = rng.normal(0.0002, target_daily_vol, 250)
    high_vol = rng.normal(0.0002, target_daily_vol * 3, 60)
    returns = np.concatenate([calm, high_vol])
    dates = pd.bdate_range("2021-01-04", periods=len(returns))
    return pd.Series(returns, index=dates), 250


def test_scales_down_during_high_volatility_period():
    returns, calm_len = _make_returns()
    scaled = apply_portfolio_vol_target(returns, vol_target=0.10, vol_lookback=20, max_gross_exposure=1.0)

    # Bien adentro del tramo de alta vol (después de que la ventana de 20 días
    # ya está mayormente compuesta de retornos de alta vol) -- ahí la escala
    # debería estar reduciendo exposición de forma clara.
    deep_high_vol = returns.index[calm_len + 30: calm_len + 55]
    ratio = (scaled.loc[deep_high_vol] / returns.loc[deep_high_vol]).abs()
    # (el ratio por día individual puede variar de signo si el retorno crudo es
    # chico, así que comparamos volatilidad realizada del tramo, más robusto)
    raw_vol = returns.loc[deep_high_vol].std()
    scaled_vol = scaled.loc[deep_high_vol].std()
    print(f"  volatilidad del tramo de alta vol -- sin escalar: {raw_vol:.4f}, escalada: {scaled_vol:.4f}")
    assert scaled_vol < raw_vol * 0.9, \
        "la volatilidad del ensamble escalado debería ser claramente menor durante el tramo de alta vol"


def test_causal_no_lookahead_on_first_high_vol_day():
    """El primer día del salto de volatilidad todavía no debería estar
    escalado -- la ventana rodante de volatilidad usada para decidir la escala
    de ESE día todavía no incluye ningún retorno de alta vol (se decide con lo
    conocido HASTA el día anterior)."""
    returns, calm_len = _make_returns()
    scaled = apply_portfolio_vol_target(returns, vol_target=0.10, vol_lookback=20, max_gross_exposure=1.0)

    first_high_vol_day = returns.index[calm_len]
    assert abs(scaled.loc[first_high_vol_day] - returns.loc[first_high_vol_day]) < 1e-9, \
        "el primer día de alta volatilidad no debería estar escalado todavía (decisión causal, sin fuga)"


def test_max_gross_exposure_caps_scale_up():
    """En un tramo de volatilidad MUY baja (mucho menor al objetivo), la escala
    cruda (vol_target/realizada) sería mayor a 1 -- debe quedar acotada en
    `max_gross_exposure`, nunca "apalancar" más allá de eso."""
    n = 100
    very_calm = np.full(n, 0.0001)  # retorno constante -> volatilidad realizada ~0
    dates = pd.bdate_range("2021-01-04", periods=n)
    returns = pd.Series(very_calm, index=dates)

    scaled = apply_portfolio_vol_target(returns, vol_target=0.10, vol_lookback=20, max_gross_exposure=1.0)
    tail = scaled.iloc[30:]
    raw_tail = returns.iloc[30:]
    # con exposición <= 1.0 siempre, el retorno escalado nunca debería superar (en magnitud) al crudo.
    assert (tail.abs() <= raw_tail.abs() + 1e-12).all(), \
        "con max_gross_exposure=1.0 el retorno escalado nunca debería exceder al retorno crudo"


def test_calm_market_near_target_stays_close_to_unscaled():
    returns, calm_len = _make_returns()
    scaled = apply_portfolio_vol_target(returns, vol_target=0.10, vol_lookback=20, max_gross_exposure=1.0)
    calm_tail = returns.index[100:200]  # bien adentro del tramo calmado, con historia suficiente
    diff = (scaled.loc[calm_tail] - returns.loc[calm_tail]).abs().mean()
    print(f"  diferencia promedio (calmado, vol ~= objetivo): {diff:.5f}")
    assert diff < 0.001, "si la vol realizada ya está cerca del objetivo, la escala debería rondar 1.0"


def main():
    print("[1/4] Probando que reduce exposición durante un tramo de alta volatilidad...")
    test_scales_down_during_high_volatility_period()
    print("\n[2/4] Probando que la decisión es causal (sin fuga de información)...")
    test_causal_no_lookahead_on_first_high_vol_day()
    print("\n[3/4] Probando que max_gross_exposure acota el escalado hacia arriba...")
    test_max_gross_exposure_caps_scale_up()
    print("\n[4/4] Probando que un mercado calmado cerca del objetivo casi no se toca...")
    test_calm_market_near_target_stays_close_to_unscaled()
    print("\nPORTFOLIO VOL TARGET TEST OK: el overlay de volatilidad a nivel ensamble funciona correctamente.")


if __name__ == "__main__":
    main()
