"""Valida la confirmación multi-horizonte de `src.strategies.momentum` con un
escenario sintético DETERMINISTA: un activo que cruza su SMA rápida por
encima de la lenta (dispara la señal clásica) pero cuyo momentum de varios
meses sigue siendo mayormente NEGATIVO (viene de una caída larga, el cruce es
un rebote de corto plazo, no una tendencia real todavía) -- la confirmación
multi-horizonte debería bloquear la entrada ahí, mientras que sin ella (o con
`use_multi_horizon=False`) el cruce solo sí generaría la señal.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.strategies import momentum


def _make_whipsaw_scenario():
    """400 días: una caída larga y sostenida (~300 días) seguida de un rebote
    corto pero fuerte (~30 días) que alcanza a cruzar la SMA rápida por encima
    de la lenta -- sin que el momentum de 6-12 meses (todavía dominado por la
    caída larga) se haya puesto positivo."""
    n_decline, n_rebound, n_tail = 300, 15, 85
    decline = np.full(n_decline, -0.004)           # caída larga y constante
    rebound = np.full(n_rebound, 0.012)             # rebote fuerte pero corto
    tail = np.full(n_tail, 0.0005)                    # continúa suave tras el rebote
    returns = np.concatenate([decline, rebound, tail])
    dates = pd.bdate_range("2020-01-02", periods=len(returns))
    price = 100 * np.cumprod(1 + returns)
    close = pd.DataFrame({"X": price}, index=dates)
    return close, n_decline, n_rebound


def test_multi_horizon_blocks_whipsaw_that_sma_alone_would_allow():
    close, n_decline, n_rebound = _make_whipsaw_scenario()

    # fast=10, slow=50 para que el cruce reaccione rápido al rebote corto.
    params_sma_only = dict(fast=10, slow=50, use_multi_horizon=False,
                            corr_dampening=0.0, max_weight_per_asset=1.0, vol_target=1.0, max_gross_exposure=10.0)
    params_multi = dict(fast=10, slow=50, use_multi_horizon=True,
                         multi_horizon_windows=(21, 63, 126, 252), multi_horizon_agreement=0.75,
                         corr_dampening=0.0, max_weight_per_asset=1.0, vol_target=1.0, max_gross_exposure=10.0)

    w_sma_only = momentum.generate_weights(close, ["X"], params=params_sma_only)
    w_multi = momentum.generate_weights(close, ["X"], params=params_multi)

    # Ventana justo después del rebote, antes de que el rebote alcance a mover el
    # momentum de 6-12 meses -- el cruce de medias ya debería estar activo ahí.
    check_window = close.index[n_decline + n_rebound - 5: n_decline + n_rebound + 5]

    sma_only_active_days = (w_sma_only.loc[check_window, "X"] > 0).sum()
    multi_active_days = (w_multi.loc[check_window, "X"] > 0).sum()

    print(f"  días con posición (solo cruce de medias): {sma_only_active_days}/{len(check_window)}")
    print(f"  días con posición (con confirmación multi-horizonte): {multi_active_days}/{len(check_window)}")

    assert sma_only_active_days > 0, \
        "el cruce de medias solo SÍ debería activarse con el rebote (así está diseñado el escenario)"
    assert multi_active_days == 0, \
        "la confirmación multi-horizonte debería BLOQUEAR la entrada -- el momentum largo sigue negativo"


def test_multi_horizon_allows_genuine_sustained_uptrend():
    """Control: en una tendencia alcista sostenida y genuina (no un rebote
    corto), la confirmación multi-horizonte NO debería bloquear la entrada --
    si no, sería un filtro demasiado estricto que nunca deja operar nada."""
    n = 400
    returns = np.full(n, 0.0015)  # tendencia alcista sostenida y constante
    dates = pd.bdate_range("2020-01-02", periods=n)
    price = 100 * np.cumprod(1 + returns)
    close = pd.DataFrame({"X": price}, index=dates)

    params_multi = dict(fast=10, slow=50, use_multi_horizon=True,
                         multi_horizon_windows=(21, 63, 126, 252), multi_horizon_agreement=0.75,
                         corr_dampening=0.0, max_weight_per_asset=1.0, vol_target=1.0, max_gross_exposure=10.0)
    weights = momentum.generate_weights(close, ["X"], params=params_multi)

    tail = weights.iloc[-30:]["X"]
    assert (tail > 0).all(), \
        "en una tendencia alcista sostenida y genuina, la confirmación multi-horizonte no debería bloquear la entrada"


def main():
    print("[1/2] Probando que la confirmación multi-horizonte bloquea un rebote corto (whipsaw)...")
    test_multi_horizon_blocks_whipsaw_that_sma_alone_would_allow()
    print("\n[2/2] Probando que SÍ deja operar una tendencia alcista genuina y sostenida...")
    test_multi_horizon_allows_genuine_sustained_uptrend()
    print("\nMOMENTUM MULTI-HORIZON TEST OK: la confirmación multi-horizonte funciona correctamente.")


if __name__ == "__main__":
    main()
