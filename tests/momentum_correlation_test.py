"""Valida el ajuste por correlación de `src.strategies.momentum` con datos
sintéticos donde se conoce de antemano la estructura de correlación: 3 activos
que se mueven casi idénticos entre sí (mismo factor de riesgo, redundantes) y
1 activo con ruido independiente (diversificador) -- todos con la MISMA
volatilidad y la MISMA tendencia alcista, para aislar el efecto de la
correlación del efecto de la volatilidad o del filtro de tendencia.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.strategies import momentum


def _make_close(n=400, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    drift = 0.0006
    vol = 0.012

    common_factor = rng.normal(drift, vol, n)
    idio = {f"CLUSTER_{i}": rng.normal(0, vol * 0.15, n) for i in range(3)}  # ruido chico -> muy correlacionados
    independent = rng.normal(drift, vol, n)  # su PROPIO factor, no comparte common_factor

    prices = {}
    for name, noise in idio.items():
        ret = common_factor + noise
        prices[name] = 100 * np.exp(np.cumsum(ret))
    prices["INDEPENDENT"] = 100 * np.exp(np.cumsum(independent))

    close = pd.DataFrame(prices, index=dates)
    return close


def test_correlated_assets_get_dampened_vs_pure_risk_parity():
    close = _make_close()
    tickers = list(close.columns)

    # max_weight_per_asset y vol_target/max_gross_exposure relajados a propósito:
    # con los defaults reales (cap 0.20 por activo), 4 activos de vol similar ya
    # saturan el cap por sí solos y esconden por completo el efecto del ajuste
    # por correlación (verificado manualmente antes de fijar el test) -- lo que
    # se quiere aislar aquí es la lógica de _rolling_corr_to_basket + penalty,
    # no la interacción con el cap de posición (que es responsabilidad de otro
    # parámetro y ya se prueba en otros tests/smoke tests).
    # use_multi_horizon=False: aísla el ajuste por correlación del filtro de
    # confirmación multi-horizonte (otra mejora, con su propio test dedicado en
    # tests/momentum_multi_horizon_test.py) -- combinar ambos gates reduce cuántos
    # días quedan "en tendencia" para comparar, sin aportar nada a ESTA prueba.
    relax = dict(max_weight_per_asset=1.0, vol_target=1.0, max_gross_exposure=10.0, use_multi_horizon=False)
    pure_rp = momentum.generate_weights(close, tickers, params=dict(corr_dampening=0.0, fast=10, slow=30, **relax))
    corr_aware = momentum.generate_weights(close, tickers, params=dict(corr_dampening=0.8, fast=10, slow=30, **relax))

    # Comparar promedio de peso relativo (peso del activo / suma de pesos ese día)
    # en días donde AMBAS versiones tienen exposición, para aislar el efecto del
    # ajuste por correlación del efecto del escalado global por vol_target.
    common_active = (pure_rp.sum(axis=1) > 0) & (corr_aware.sum(axis=1) > 0)
    pure_share = pure_rp.loc[common_active].div(pure_rp.loc[common_active].sum(axis=1), axis=0)
    corr_share = corr_aware.loc[common_active].div(corr_aware.loc[common_active].sum(axis=1), axis=0)

    cluster_cols = ["CLUSTER_0", "CLUSTER_1", "CLUSTER_2"]
    avg_pure_cluster_share = pure_share[cluster_cols].sum(axis=1).mean()
    avg_corr_cluster_share = corr_share[cluster_cols].sum(axis=1).mean()
    avg_pure_indep_share = pure_share["INDEPENDENT"].mean()
    avg_corr_indep_share = corr_share["INDEPENDENT"].mean()

    print(f"  participación combinada del cluster correlacionado -- paridad pura: {avg_pure_cluster_share:.3f}, "
          f"consciente de correlación: {avg_corr_cluster_share:.3f}")
    print(f"  participación del activo independiente -- paridad pura: {avg_pure_indep_share:.3f}, "
          f"consciente de correlación: {avg_corr_indep_share:.3f}")

    assert avg_corr_cluster_share < avg_pure_cluster_share, \
        "el cluster correlacionado debería perder participación combinada al activar el ajuste por correlación"
    assert avg_corr_indep_share > avg_pure_indep_share, \
        "el activo independiente (diversificador) debería GANAR participación al activar el ajuste por correlación"

    # No debería haber NaN ni pesos negativos (long-only), pase lo que pase con el ajuste por correlación.
    for df in (pure_rp, corr_aware):
        assert df.notna().all().all(), "no debería haber NaN en los pesos"
        assert (df >= 0).all().all(), "no debería haber pesos negativos (long-only)"

    # Con los parámetros DEFAULT (cap y vol_target reales, sin relajar) el pipeline
    # completo sigue produciendo pesos válidos y acotados -- valida que el ajuste
    # por correlación no rompe nada al combinarse con el cap de posición normal.
    default_weights = momentum.generate_weights(close, tickers, params=dict(corr_dampening=0.8, fast=10, slow=30))
    assert default_weights.notna().all().all()
    assert (default_weights.sum(axis=1) <= momentum.DEFAULT_PARAMS["max_gross_exposure"] + 1e-6).all()
    assert (default_weights >= 0).all().all()


def test_negative_correlation_boosts_weight():
    """Caso más directo: un activo con correlación casi -1 contra la canasta
    (se mueve en contra del resto) debería terminar con MÁS peso relativo que
    uno con la misma volatilidad pero correlación positiva alta."""
    n = 300
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2020-01-01", periods=n)
    vol = 0.01
    drift = 0.0005

    # Separar drift (compartido, para que AMBOS activos tiendan al alza y entren
    # en tendencia) de la fluctuación (compartida en signo opuesto -> correlación
    # negativa). Si el drift se mezclara dentro de "common" con signo invertido en
    # NEG, el drift neto de NEG terminaría en ~0 y casi nunca cruzaría la SMA.
    common_fluctuation = rng.normal(0, vol, n)
    noise_a = rng.normal(0, vol * 0.1, n)
    pos_corr = 100 * np.exp(np.cumsum(drift + common_fluctuation + noise_a))   # sigue a la canasta
    neg_corr = 100 * np.exp(np.cumsum(drift - common_fluctuation + noise_a))   # mismo drift, fluctuación opuesta

    close = pd.DataFrame({"POS": pos_corr, "NEG": neg_corr}, index=dates)
    # use_multi_horizon=False: aísla el ajuste por correlación del filtro de
    # confirmación multi-horizonte (otra mejora, con su propio test dedicado en
    # tests/momentum_multi_horizon_test.py) -- combinar ambos gates reduce cuántos
    # días quedan "en tendencia" para comparar, sin aportar nada a ESTA prueba.
    relax = dict(max_weight_per_asset=1.0, vol_target=1.0, max_gross_exposure=10.0, use_multi_horizon=False)
    weights = momentum.generate_weights(close, ["POS", "NEG"],
                                         params=dict(corr_dampening=0.8, fast=10, slow=30, **relax))

    both_active = weights[(weights["POS"] > 0) & (weights["NEG"] > 0)]
    assert len(both_active) > 20, "se esperaban suficientes días con ambos activos en tendencia para comparar"
    assert both_active["NEG"].mean() > both_active["POS"].mean(), \
        "el activo negativamente correlacionado con la canasta debería recibir más peso en promedio"


def main():
    print("[1/2] Probando que activos correlacionados (redundantes) pierden peso frente a un diversificador...")
    test_correlated_assets_get_dampened_vs_pure_risk_parity()
    print("[2/2] Probando que un activo con correlación negativa gana peso frente a uno positivamente correlacionado...")
    test_negative_correlation_boosts_weight()
    print("\nMOMENTUM CORRELATION TEST OK: el ajuste por correlación de paridad de riesgo funciona correctamente.")


if __name__ == "__main__":
    main()
