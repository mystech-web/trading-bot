"""Compara los parámetros que el walk-forward MÁS RECIENTE eligió (el fold más
cercano a hoy) contra los que están configurados en `live_params.yaml` --
`run_live_once.py`/`run_crypto_live_once.py` NO usan los parámetros del
walk-forward, usan los fijos del YAML (a propósito, ver el comentario del
propio archivo -- re-optimización continua sería overfitting). Con el tiempo
eso puede quedarse "viejo": el mercado cambia y el óptimo se mueve, pero nadie
actualiza el YAML.

Esto NO auto-aplica ningún cambio -- solo avisa. Sigue siendo decisión tuya
revisar `reports/param_stability_*.csv` y decidir si vale la pena actualizar
`config/live_params.yaml` manualmente.
"""
from __future__ import annotations


def check_param_drift(fold_reports_by_strategy: dict[str, list[dict]], live_params_by_strategy: dict[str, dict],
                       tolerance_pct: float = 0.20) -> list[dict]:
    """Solo compara los parámetros que el propio grid de walk-forward barre por
    estrategia (ej. `fast`/`slow` para momentum -- `vol_target` no se compara,
    porque el walk-forward nunca lo varía, así que no puede "driftear").

    `tolerance_pct`: qué tanto puede diferir un valor numérico antes de
    marcarse (0.20 = 20% de diferencia relativa). Valores no numéricos
    (ej. strings) se comparan por igualdad exacta.

    Devuelve una lista de hallazgos (vacía si no hay drift relevante), cada uno
    con: strategy, param, live_value, latest_wf_value, fold_test_start, fold_test_end.
    """
    findings = []
    for strategy, fold_reports in fold_reports_by_strategy.items():
        if not fold_reports:
            continue
        latest = fold_reports[-1]
        latest_params = latest["best_params"]
        live_strategy_params = live_params_by_strategy.get(strategy, {})

        for param_name, wf_value in latest_params.items():
            if param_name not in live_strategy_params:
                continue
            live_value = live_strategy_params[param_name]

            if isinstance(wf_value, (int, float)) and isinstance(live_value, (int, float)):
                if live_value == 0:
                    diverged = wf_value != 0
                else:
                    diverged = abs(wf_value - live_value) / abs(live_value) > tolerance_pct
            else:
                diverged = wf_value != live_value

            if diverged:
                findings.append(dict(
                    strategy=strategy, param=param_name, live_value=live_value, latest_wf_value=wf_value,
                    fold_test_start=str(latest["test_start"].date()), fold_test_end=str(latest["test_end"].date()),
                ))
    return findings


def format_drift_report(findings: list[dict]) -> str:
    if not findings:
        return "Sin drift relevante: los parámetros en vivo coinciden con lo que eligió el fold más reciente."
    lines = [f"{len(findings)} parámetro(s) en config/live_params.yaml difieren del óptimo del fold más reciente:"]
    for f in findings:
        lines.append(f"  [{f['strategy']}] {f['param']}: en vivo={f['live_value']!r} vs. "
                      f"walk-forward más reciente={f['latest_wf_value']!r} "
                      f"(fold {f['fold_test_start']} -> {f['fold_test_end']})")
    lines.append("  Esto NO se aplica solo -- revisa reports/param_stability_*.csv antes de decidir si actualizar "
                  "el YAML a mano (un solo fold puede ser ruido, no una señal real de que el óptimo cambió).")
    return "\n".join(lines)
