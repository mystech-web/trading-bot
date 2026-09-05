"""Valida el ajuste por correlación de `src.strategies.sector_rotation` con
datos sintéticos donde se conoce la estructura de correlación de antemano: 2
sectores que se mueven casi idénticos entre sí (redundantes) y 1 sector con su
propio factor de riesgo (diversificador) -- todos con tendencia alcista fuerte
para que los 3 califiquen como "picks" (les gana al cash) y `top_n=3` los
sostenga a todos, aislando así el efecto de la correlación del efecto de
selección por momentum.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.strategies import sector_rotation


def _make_close(n=500, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    drift = 0.0007
    vol = 0.011

    common = rng.normal(0, vol, n)  # factor de riesgo compartido (fluctuación, sin drift)
    idio_a = rng.normal(0, vol * 0.12, n)
    idio_b = rng.normal(0, vol * 0.12, n)
    independent_noise = rng.normal(0, vol, n)

    sector_a = 100 * np.exp(np.cumsum(drift + common + idio_a))       # redundante con B
    sector_b = 100 * np.exp(np.cumsum(drift + common + idio_b))       # redundante con A
    sector_c = 100 * np.exp(np.cumsum(drift + independent_noise))     # diversificador
    cash = 100 * np.exp(np.cumsum(np.full(n, 0.0001)))                # cash casi plano, todos le ganan

    close = pd.DataFrame({"SECTOR_A": sector_a, "SECTOR_B": sector_b, "SECTOR_C": sector_c,
                           "CASH": cash}, index=dates)
    return close


def test_redundant_sectors_lose_weight_to_diversifier():
    close = _make_close()
    tickers = ["SECTOR_A", "SECTOR_B", "SECTOR_C"]

    pure = sector_rotation.generate_weights(close, tickers, "CASH",
                                             params=dict(top_n=3, corr_dampening=0.0, max_weight_per_asset=1.0))
    corr_aware = sector_rotation.generate_weights(close, tickers, "CASH",
                                                   params=dict(top_n=3, corr_dampening=0.8, max_weight_per_asset=1.0))

    active = (pure.sum(axis=1) > 0) & (corr_aware.sum(axis=1) > 0)
    assert active.sum() > 10, "se esperaban varios rebalanceos con los 3 sectores activos"

    avg_pure_redundant = pure.loc[active, ["SECTOR_A", "SECTOR_B"]].sum(axis=1).mean()
    avg_corr_redundant = corr_aware.loc[active, ["SECTOR_A", "SECTOR_B"]].sum(axis=1).mean()
    avg_pure_diversifier = pure.loc[active, "SECTOR_C"].mean()
    avg_corr_diversifier = corr_aware.loc[active, "SECTOR_C"].mean()

    print(f"  peso combinado A+B (redundantes) -- igual: {avg_pure_redundant:.3f}, "
          f"por correlación: {avg_corr_redundant:.3f}")
    print(f"  peso de C (diversificador) -- igual: {avg_pure_diversifier:.3f}, "
          f"por correlación: {avg_corr_diversifier:.3f}")

    assert avg_corr_redundant < avg_pure_redundant, \
        "el par redundante (A, B) debería perder peso combinado con el ajuste por correlación"
    assert avg_corr_diversifier > avg_pure_diversifier, \
        "el sector diversificador (C) debería ganar peso con el ajuste por correlación"

    # Peso igual (corr_dampening=0) debe seguir siendo exactamente 1/3 en los días donde
    # los 3 sectores califican simultáneamente (algunos rebalanceos solo tienen 1 o 2
    # picks activos -- por eso el promedio general de arriba no da un 1/3 limpio).
    all_three_active = (pure[tickers] > 0).all(axis=1)
    assert all_three_active.sum() > 5, "se esperaban varios rebalanceos con los 3 sectores calificando a la vez"
    thirds = pure.loc[all_three_active, tickers]
    assert (thirds.round(4) == round(1 / 3, 4)).all().all(), \
        "con corr_dampening=0 el peso debería seguir siendo exactamente igual (comportamiento original)"


def test_weights_respect_cap_and_no_nan():
    close = _make_close()
    tickers = ["SECTOR_A", "SECTOR_B", "SECTOR_C"]
    weights = sector_rotation.generate_weights(close, tickers, "CASH",
                                                params=dict(top_n=3, corr_dampening=0.8, max_weight_per_asset=0.35))
    assert weights.notna().all().all(), "no debería haber NaN en los pesos"
    assert (weights <= 0.35 + 1e-9).all().all(), "ningún peso individual debería exceder max_weight_per_asset"
    assert (weights >= 0).all().all(), "no debería haber pesos negativos"


def main():
    print("[1/2] Probando que sectores redundantes pierden peso frente a un diversificador...")
    test_redundant_sectors_lose_weight_to_diversifier()
    print("\n[2/2] Probando que el cap por posición y la ausencia de NaN se respetan...")
    test_weights_respect_cap_and_no_nan()
    print("\nSECTOR ROTATION CORRELATION TEST OK: el ajuste por correlación funciona correctamente.")


if __name__ == "__main__":
    main()
