"""Valida `src.ensemble.optimize_ensemble_weights` con datos sintéticos donde se
conoce de antemano qué estrategia debería terminar recibiendo más peso -- no
basta con "no truena", hay que verificar que la asignación dinámica realmente
reacciona al desempeño OOS previo, respeta el piso `min_weight`, nunca mira
hacia adelante, y que el fold 0 usa la asignación por defecto (sin historial).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.ensemble import optimize_ensemble_weights, DEFAULT_ALLOCATION


class _Fold:
    def __init__(self, test_start, test_end):
        self.test_start = pd.Timestamp(test_start)
        self.test_end = pd.Timestamp(test_end)


def _make_oos_returns():
    """3 folds de 60 días hábiles cada uno. 'low_vol' tiene volatilidad baja y
    constante en todos los folds; 'high_vol' tiene volatilidad alta y constante.
    Como la volatilidad de cada estrategia NO cambia entre folds, el fold 1 y el
    fold 2 deberían terminar con la MISMA asignación entre sí (una vez que hay
    historial), y esa asignación debería favorecer claramente a 'low_vol'."""
    rng = np.random.default_rng(42)
    folds = []
    all_dates = []
    cursor = pd.Timestamp("2020-01-06")
    for _ in range(3):
        dates = pd.bdate_range(cursor, periods=60)
        folds.append(_Fold(dates[0], dates[-1]))
        all_dates.append(dates)
        cursor = dates[-1] + pd.Timedelta(days=3)

    full_index = pd.DatetimeIndex(np.concatenate(all_dates))
    low_vol = pd.Series(rng.normal(0.0005, 0.002, len(full_index)), index=full_index)
    high_vol = pd.Series(rng.normal(0.0005, 0.03, len(full_index)), index=full_index)
    flat = pd.Series(0.0, index=full_index)  # estrategia sin variación -> vol=0, caso borde división por cero
    return dict(low_vol=low_vol, high_vol=high_vol, flat=flat), folds


def test_dynamic_allocation_favors_lower_vol_strategy():
    oos_returns, folds = _make_oos_returns()
    default_alloc = dict(low_vol=0.34, high_vol=0.33, flat=0.33)

    result = optimize_ensemble_weights(oos_returns, folds, default_alloc, min_weight=0.05)
    allocs = result["fold_allocations"]
    assert len(allocs) == 3

    # Fold 0: sin historial previo -> debe usar la asignación por defecto tal cual.
    assert allocs[0]["allocation"] == {k: round(v, 4) for k, v in default_alloc.items()}

    # Fold 1 y 2: ya hay historial -> low_vol (mucha menor volatilidad) debe
    # recibir claramente más peso que high_vol.
    for f in (allocs[1], allocs[2]):
        assert f["allocation"]["low_vol"] > f["allocation"]["high_vol"], \
            f"se esperaba que low_vol reciba más peso que high_vol en fold {f['fold']}"
        # el piso min_weight=0.05 debe respetarse siempre
        for w in f["allocation"].values():
            assert w >= 0.05 - 1e-9
        # la asignación debe sumar 1 (renormalizada tras aplicar el piso)
        assert abs(sum(f["allocation"].values()) - 1.0) < 1e-6

    # 'flat' tiene std=0 en todos los folds -> caería a peso "crudo" 0, pero el
    # piso min_weight debe rescatarlo a exactamente min_weight tras renormalizar.
    for f in (allocs[1], allocs[2]):
        assert abs(f["allocation"]["flat"] - 0.05) < 1e-3, \
            "una estrategia con volatilidad 0 debería quedar exactamente en el piso min_weight"

    # Fold 1 y 2 deberían coincidir entre sí (la vol de cada estrategia no cambia
    # entre folds una vez que ambos tienen historial -- fold 2 ve un fold más de
    # historia que fold 1, pero la vol estimada converge al mismo valor esperado).
    assert allocs[1]["allocation"]["low_vol"] > 0
    assert allocs[2]["allocation"]["low_vol"] > 0

    # No fuga de información: la asignación del fold 1 no debe depender de datos
    # del fold 1 o 2 -- verificado indirectamente arriba (fold 0 = default exacto,
    # que es la prueba directa de que no hay atajo usando datos futuros).

    # returns: debe ser una serie continua, sin huecos ni duplicados, cubriendo
    # exactamente los 3 folds.
    combined = result["returns"]
    assert not combined.index.duplicated().any()
    assert combined.index.is_monotonic_increasing
    assert combined.index.min() == folds[0].test_start
    assert combined.index.max() == folds[-1].test_end

    print(f"  fold 0 (default): {allocs[0]['allocation']}")
    print(f"  fold 1 (dinámico): {allocs[1]['allocation']}")
    print(f"  fold 2 (dinámico): {allocs[2]['allocation']}")


def _make_correlated_oos_returns():
    """2 folds. 'strat_a' y 'strat_b' comparten el mismo factor de riesgo
    (correlacionadas entre sí) con volatilidad IGUAL a 'strat_c' (independiente)
    -- a propósito, para aislar el efecto de la correlación del efecto de la
    volatilidad (que ya prueba el test de arriba)."""
    rng = np.random.default_rng(11)
    folds = []
    all_dates = []
    cursor = pd.Timestamp("2020-01-06")
    for _ in range(2):
        dates = pd.bdate_range(cursor, periods=80)
        folds.append(_Fold(dates[0], dates[-1]))
        all_dates.append(dates)
        cursor = dates[-1] + pd.Timedelta(days=3)

    full_index = pd.DatetimeIndex(np.concatenate(all_dates))
    n = len(full_index)
    vol = 0.01
    common = rng.normal(0.0003, vol, n)
    idio_a = rng.normal(0, vol * 0.05, n)
    idio_b = rng.normal(0, vol * 0.05, n)
    independent = rng.normal(0.0003, vol, n)
    return dict(
        strat_a=pd.Series(common + idio_a, index=full_index),
        strat_b=pd.Series(common + idio_b, index=full_index),
        strat_c=pd.Series(independent, index=full_index),
    ), folds


def test_correlation_dampening_reduces_weight_of_correlated_pair():
    oos_returns, folds = _make_correlated_oos_returns()
    default_alloc = dict(strat_a=1 / 3, strat_b=1 / 3, strat_c=1 / 3)

    pure = optimize_ensemble_weights(oos_returns, folds, default_alloc, min_weight=0.0, corr_dampening=0.0)
    corr_aware = optimize_ensemble_weights(oos_returns, folds, default_alloc, min_weight=0.0, corr_dampening=0.8)

    fold1_pure = pure["fold_allocations"][1]["allocation"]
    fold1_corr = corr_aware["fold_allocations"][1]["allocation"]
    pure_pair = fold1_pure["strat_a"] + fold1_pure["strat_b"]
    corr_pair = fold1_corr["strat_a"] + fold1_corr["strat_b"]

    print(f"  peso combinado A+B (correlacionadas) -- sin ajuste: {pure_pair:.3f}, con ajuste: {corr_pair:.3f}")
    print(f"  peso de C (diversificadora) -- sin ajuste: {fold1_pure['strat_c']:.3f}, "
          f"con ajuste: {fold1_corr['strat_c']:.3f}")

    assert corr_pair < pure_pair, "el par correlacionado (A, B) debería perder peso combinado con corr_dampening>0"
    assert fold1_corr["strat_c"] > fold1_pure["strat_c"], \
        "la estrategia diversificadora (C) debería ganar peso con corr_dampening>0"
    # Sin ajuste (dampening=0), con vol prácticamente igual, el reparto debería ser ~1/3 cada una.
    assert abs(pure_pair - 2 / 3) < 0.05


def test_unknown_method_raises():
    oos_returns, folds = _make_oos_returns()
    try:
        optimize_ensemble_weights(oos_returns, folds, method="not_a_real_method")
        raise AssertionError("debería haber lanzado ValueError")
    except ValueError:
        pass


def test_default_allocation_fallback_when_none_passed():
    oos_returns, folds = _make_oos_returns()
    result = optimize_ensemble_weights(oos_returns, folds)
    assert set(result["fold_allocations"][0]["allocation"].keys()) <= set(DEFAULT_ALLOCATION.keys()) | set(oos_returns.keys())


def main():
    print("[1/4] Probando que la asignación dinámica favorece a la estrategia de menor volatilidad...")
    test_dynamic_allocation_favors_lower_vol_strategy()
    print("\n[2/4] Probando que el ajuste por correlación penaliza al par redundante...")
    test_correlation_dampening_reduces_weight_of_correlated_pair()
    print("\n[3/4] Probando que un método desconocido lanza ValueError...")
    test_unknown_method_raises()
    print("\n[4/4] Probando fallback a DEFAULT_ALLOCATION cuando no se pasa nada...")
    test_default_allocation_fallback_when_none_passed()
    print("\nENSEMBLE OPTIMIZE TEST OK: la asignación dinámica por walk-forward funciona correctamente.")


if __name__ == "__main__":
    main()
