"""Smoke test con datos sintéticos: valida que todo el pipeline (indicadores,
estrategias, motor de backtest, walk-forward, ensamble, métricas) corre de punta
a punta sin errores. NO valida que la estrategia sea rentable (los datos son
ruido aleatorio) -- eso requiere datos reales, que este sandbox no puede
descargar por política de red. Correr esto localmente sirve para confirmar que
el código no tiene bugs antes de usar datos reales en tu Mac/PC.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.metrics import summary_table
from src.walk_forward import build_folds, walk_forward_backtest, param_stability_table, param_stability_score
from src.ensemble import combine_returns
from src.strategies import momentum, mean_reversion, sector_rotation
from src.monte_carlo import monte_carlo_summary, summary_for_json
from src.stress_test import CRISIS_PERIODS, periods_covered, run_stress_test
from src.notify import get_logger, send_alert
from src.tracking import append_equity, update_drawdown_guard, check_drift, load_equity_log


def make_synthetic_close(n_days=8 * 252, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n_days)
    tickers = {
        "T_SPY": (0.0003, 0.010),
        "T_QQQ": (0.0004, 0.013),
        "SEC_A": (0.0003, 0.014),
        "SEC_B": (0.0002, 0.012),
        "SEC_C": (0.0001, 0.011),
        "SEC_D": (0.0005, 0.015),
        "STK_A": (0.0004, 0.017),
        "STK_B": (0.0002, 0.016),
        "CASH": (0.00008, 0.0005),
    }
    data = {}
    for t, (mu, sigma) in tickers.items():
        rets = rng.normal(mu, sigma, n_days)
        price = 100 * np.exp(np.cumsum(rets))
        data[t] = price
    return pd.DataFrame(data, index=dates)


def main():
    close = make_synthetic_close()
    print(f"Datos sintéticos: {close.shape[0]} días, {close.shape[1]} tickers "
          f"({close.index.min().date()} -> {close.index.max().date()})")

    momentum_mr_universe = ["T_SPY", "T_QQQ", "STK_A", "STK_B"]
    sector_universe = ["SEC_A", "SEC_B", "SEC_C", "SEC_D"]
    cash_ticker = "CASH"
    bt_kw = dict(cost_bps=5.0, dd_guard_threshold=-0.15, dd_guard_recover=-0.07, dd_guard_scale=0.5)

    print("\n[1/8] Probando generación de pesos + backtest simple por estrategia...")
    w_mom = momentum.generate_weights(close, momentum_mr_universe)
    bt_mom = run_backtest(close, w_mom, **bt_kw)
    assert bt_mom["equity"].notna().all(), "equity de momentum tiene NaN"
    print(f"  momentum: equity final = {bt_mom['equity'].iloc[-1]:.3f}, "
          f"turnover medio = {bt_mom['turnover'].mean():.4f}")

    w_mr = mean_reversion.generate_weights(close, momentum_mr_universe)
    bt_mr = run_backtest(close, w_mr, **bt_kw)
    assert bt_mr["equity"].notna().all(), "equity de mean_reversion tiene NaN"
    print(f"  mean_reversion: equity final = {bt_mr['equity'].iloc[-1]:.3f}, "
          f"posiciones abiertas prom = {(w_mr > 0).sum(axis=1).mean():.2f}")

    w_rot = sector_rotation.generate_weights(close, sector_universe, cash_ticker)
    bt_rot = run_backtest(close, w_rot, **bt_kw)
    assert bt_rot["equity"].notna().all(), "equity de sector_rotation tiene NaN"
    assert (w_rot.sum(axis=1) <= 1.0001).all(), "sector_rotation excede 100% de exposición"
    print(f"  sector_rotation: equity final = {bt_rot['equity'].iloc[-1]:.3f}")

    print("\n[2/8] Probando walk-forward (grids chicos, folds cortos, con embargo)...")
    analysis_start = close.index.min() + pd.DateOffset(years=1)
    folds = build_folds(close.index, analysis_start, train_years=2, test_years=1, step_years=1, embargo_days=5)
    assert len(folds) >= 3, f"muy pocos folds generados: {len(folds)}"
    for f in folds:
        assert f.test_start > f.train_end, "el embargo no está dejando hueco entre train y test"
    print(f"  {len(folds)} folds generados, embargo de 5 días verificado entre train_end y test_start")

    wf_mom = walk_forward_backtest(
        close, lambda p: momentum.generate_weights(close, momentum_mr_universe, p),
        [dict(fast=20, slow=100), dict(fast=50, slow=200)], folds, bt_kw,
    )
    wf_mr = walk_forward_backtest(
        close, lambda p: mean_reversion.generate_weights(close, momentum_mr_universe, p),
        [dict(entry_rsi=10.0, exit_rsi=70.0), dict(entry_rsi=5.0, exit_rsi=60.0)], folds, bt_kw,
    )
    wf_rot = walk_forward_backtest(
        close, lambda p: sector_rotation.generate_weights(close, sector_universe, cash_ticker, p),
        [dict(top_n=2), dict(top_n=3)], folds, bt_kw,
    )
    for name, wf in [("momentum", wf_mom), ("mean_reversion", wf_mr), ("sector_rotation", wf_rot)]:
        assert len(wf["returns"]) > 0, f"{name}: walk-forward sin retornos OOS"
        assert wf["returns"].notna().all(), f"{name}: walk-forward con NaN"
        assert len(wf["fold_reports"]) == len(folds), f"{name}: número de folds inconsistente"
    print("  walk-forward OK para las 3 estrategias, sin NaN, folds consistentes")

    print("\n[3/8] Probando reporte de estabilidad de parámetros...")
    for name, wf in [("momentum", wf_mom), ("mean_reversion", wf_mr), ("sector_rotation", wf_rot)]:
        table_params = param_stability_table(wf["fold_reports"])
        scores = param_stability_score(wf["fold_reports"])
        assert len(table_params) == len(wf["fold_reports"]), f"{name}: tabla de estabilidad con filas de más/menos"
        assert scores, f"{name}: no se calculó ningún score de estabilidad"
        assert all(0.0 <= s <= 1.0 for s in scores.values()), f"{name}: score de estabilidad fuera de [0,1]"
    print(f"  estabilidad calculada OK para las 3 estrategias (ej. momentum: {scores})")

    print("\n[4/8] Probando ensamble...")
    oos = {"momentum": wf_mom["returns"], "mean_reversion": wf_mr["returns"], "sector_rotation": wf_rot["returns"]}
    ens = combine_returns(oos)
    assert ens.notna().all()
    print(f"  ensemble: {len(ens)} días OOS, equity final = {(1+ens).cumprod().iloc[-1]:.3f}")

    print("\n[5/8] Probando tabla resumen de métricas...")
    table = summary_table({"momentum": oos["momentum"], "mean_reversion": oos["mean_reversion"],
                            "sector_rotation": oos["sector_rotation"], "ensemble": ens})
    print(table.to_string())
    assert not table.isna().all(axis=None), "tabla de métricas totalmente vacía"

    print("\n[6/8] Probando Monte Carlo (block bootstrap)...")
    mc = monte_carlo_summary(ens, n_sims=200, block_size=21, seed=1)
    assert mc["avg_monthly_return"]["p5"] <= mc["avg_monthly_return"]["p50"] <= mc["avg_monthly_return"]["p95"], \
        "los percentiles del Monte Carlo no están ordenados correctamente"
    assert 0.0 <= mc["prob_avg_monthly_in_target_0.5_2pct"] <= 1.0
    mc_json = summary_for_json(mc)
    assert "_raw_avg_monthly" not in mc_json, "summary_for_json no debería incluir el array crudo"
    print(f"  Monte Carlo OK: p5={mc['avg_monthly_return']['p5']*100:.2f}% "
          f"mediana={mc['avg_monthly_return']['p50']*100:.2f}% p95={mc['avg_monthly_return']['p95']*100:.2f}%")

    print("\n[7/8] Probando stress test contra crashes conocidos...")
    covered = periods_covered(ens.index, CRISIS_PERIODS)
    if covered:
        stress = run_stress_test({"ensemble": ens, "momentum": oos["momentum"]}, covered)
        assert len(stress) == len(covered), "el stress test no cubrió todos los períodos esperados"
        print(f"  stress test OK, {len(covered)} período(s) cubiertos por el rango sintético: {list(covered)}")
    else:
        print("  (ningún crash conocido cae dentro del rango sintético 2016-2023 -- normal, son fechas ficticias)")

    print("\n[8/8] Probando logging/alertas y tracking de equity en vivo (SQLite, sin red real)...")
    logger = get_logger("smoke-test")
    send_alert("smoke-test: sin canal configurado", "esto no debería lanzar excepción", logger)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tracking_dir = pathlib.Path(tmp)
        append_equity(pd.Timestamp("2024-01-02"), 10_000.0, reports_dir=tracking_dir)
        append_equity(pd.Timestamp("2024-01-03"), 9_000.0, reports_dir=tracking_dir)  # -10%, no activa la guardia
        assert (tracking_dir / "tracking.sqlite3").exists(), "debería haberse creado la base SQLite"
        log_df = load_equity_log(tracking_dir)
        assert len(log_df) == 2, f"esperaba 2 filas en el log, hay {len(log_df)}"

        dd, scale, guard_active, changed = update_drawdown_guard(9_000.0, threshold=-0.15, recover=-0.07,
                                                                   reports_dir=tracking_dir)
        assert abs(dd - (-0.10)) < 1e-9, f"drawdown calculado mal: {dd}"
        assert guard_active is False, "la guardia no debería activarse todavía con -10%"
        assert scale == 1.0, "sin guardia activa, la escala de exposición debería ser 100%"

        dd2, scale2, guard_active2, changed2 = update_drawdown_guard(8_000.0, threshold=-0.15, recover=-0.07,
                                                                       reports_dir=tracking_dir)
        assert guard_active2 is True, "la guardia debería activarse con -20% de drawdown"
        assert changed2 is True, "el cambio de estado de la guardia debería reportarse"
        assert scale2 == 0.5, "la guardia recién activada debería escalar la exposición a dd_guard_scale (50%)"

        drift = check_drift(min_days=1, reports_dir=tracking_dir)  # sin monte_carlo.json -> None, no explota
        assert drift is None or isinstance(drift, str)
        print("  tracking de equity (SQLite) + guardia de drawdown persistente + check_drift OK")

    print("\nSMOKE TEST OK: el pipeline completo (incluyendo Monte Carlo, stress test, "
          "estabilidad de parámetros y tracking en vivo) corre sin errores.")
    print("(Recuerda: estos números son de RUIDO ALEATORIO, no significan nada. "
          "Corre run_backtest.py con datos reales para resultados válidos.)")


if __name__ == "__main__":
    main()
