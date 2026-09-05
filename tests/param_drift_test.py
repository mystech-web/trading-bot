"""Valida `src.param_drift.check_param_drift` con casos sintéticos: un
parámetro que coincide (no debería marcarse), uno que difiere poco (dentro de
la tolerancia, no debería marcarse), uno que difiere mucho (SÍ debería
marcarse), y un parámetro que el walk-forward nunca varía (no debería
compararse -- no está en el grid, así que no puede "driftear").
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

from src.param_drift import check_param_drift, format_drift_report


def _fold_report(best_params, test_start="2024-01-01", test_end="2024-12-31"):
    return dict(best_params=best_params, test_start=pd.Timestamp(test_start), test_end=pd.Timestamp(test_end))


def test_matching_params_produce_no_findings():
    fold_reports = {"momentum": [_fold_report(dict(fast=50, slow=200))]}
    live_params = {"momentum": dict(fast=50, slow=200, vol_target=0.10)}
    findings = check_param_drift(fold_reports, live_params)
    assert findings == []
    print("  parámetros idénticos OK: sin hallazgos")


def test_small_difference_within_tolerance_is_ignored():
    # slow: 200 (en vivo) vs 210 (walk-forward) -> 5% de diferencia, tolerancia default 20%.
    fold_reports = {"momentum": [_fold_report(dict(fast=50, slow=210))]}
    live_params = {"momentum": dict(fast=50, slow=200, vol_target=0.10)}
    findings = check_param_drift(fold_reports, live_params, tolerance_pct=0.20)
    assert findings == [], f"una diferencia de 5% no debería marcarse con tolerancia de 20%: {findings}"
    print("  diferencia chica (5%) dentro de tolerancia OK: sin hallazgos")


def test_large_difference_is_flagged():
    # fast: 50 (en vivo) vs 10 (walk-forward) -> 80% de diferencia, cruza la tolerancia.
    fold_reports = {"momentum": [_fold_report(dict(fast=10, slow=200))]}
    live_params = {"momentum": dict(fast=50, slow=200, vol_target=0.10)}
    findings = check_param_drift(fold_reports, live_params, tolerance_pct=0.20)
    assert len(findings) == 1
    f = findings[0]
    assert f["strategy"] == "momentum" and f["param"] == "fast"
    assert f["live_value"] == 50 and f["latest_wf_value"] == 10
    print(f"  diferencia grande (80%) OK: se marcó -- {f}")

    report = format_drift_report(findings)
    assert "momentum" in report and "fast" in report
    print(f"  formato de reporte legible OK")


def test_params_not_in_grid_are_never_compared():
    # 'vol_target' no está en best_params del walk-forward (nunca se barre en el
    # grid de momentum) -- aunque el valor en vivo sea distinto de cualquier cosa,
    # no debería aparecer como hallazgo porque no hay con qué compararlo.
    fold_reports = {"momentum": [_fold_report(dict(fast=50, slow=200))]}  # sin 'vol_target'
    live_params = {"momentum": dict(fast=50, slow=200, vol_target=0.99)}  # valor "raro" a propósito
    findings = check_param_drift(fold_reports, live_params)
    assert findings == [], "un parámetro que el walk-forward no varía nunca debería compararse"
    print("  parámetro fuera del grid (vol_target) OK: nunca se compara")


def test_multiple_strategies_and_empty_fold_reports():
    fold_reports = {
        "momentum": [_fold_report(dict(fast=10, slow=200))],   # diverge en 'fast'
        "mean_reversion": [_fold_report(dict(entry_rsi=10.0, exit_rsi=70.0))],  # coincide
        "sector_rotation": [],  # sin folds (walk-forward no corrió) -> se ignora sin romper
    }
    live_params = {
        "momentum": dict(fast=50, slow=200),
        "mean_reversion": dict(entry_rsi=10.0, exit_rsi=70.0),
        "sector_rotation": dict(top_n=3),
    }
    findings = check_param_drift(fold_reports, live_params)
    assert len(findings) == 1 and findings[0]["strategy"] == "momentum"
    print("  múltiples estrategias + fold_reports vacío OK: solo marca momentum, no rompe con lista vacía")


def main():
    print("[1/5] Probando que parámetros idénticos no generan hallazgos...")
    test_matching_params_produce_no_findings()
    print("\n[2/5] Probando que una diferencia chica (dentro de tolerancia) se ignora...")
    test_small_difference_within_tolerance_is_ignored()
    print("\n[3/5] Probando que una diferencia grande sí se marca...")
    test_large_difference_is_flagged()
    print("\n[4/5] Probando que un parámetro fuera del grid nunca se compara...")
    test_params_not_in_grid_are_never_compared()
    print("\n[5/5] Probando múltiples estrategias y fold_reports vacío...")
    test_multiple_strategies_and_empty_fold_reports()
    print("\nPARAM DRIFT TEST OK: la detección de drift de parámetros funciona correctamente.")


if __name__ == "__main__":
    main()
