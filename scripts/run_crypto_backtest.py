"""Backtest completo del módulo cripto (Binance): descarga velas diarias,
corre las 3 estrategias (in-sample + walk-forward), arma el ensamble, y genera
el mismo tipo de reporte honesto que `run_backtest.py` (equities) -- Monte
Carlo, stress test contra crashes cripto conocidos, y estabilidad de
parámetros. Comparte casi todo el motor (`src/backtest.py`, `src/metrics.py`,
`src/monte_carlo.py`, `src/walk_forward.py`, `src/ensemble.py`) con el bot de
acciones -- lo único que cambia es la fuente de datos, el universo, y que acá
`periods_per_year=365` (cripto cotiza todos los días, no solo días hábiles).

Uso:
    python scripts/run_crypto_backtest.py
"""
import argparse
import functools
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.plot_style import apply_neon_style
apply_neon_style()

from src.crypto_data import (
    load_crypto_universe, load_crypto_live_params, all_crypto_symbols, download_prices,
    build_close_matrix, build_quote_volume_matrix, build_crypto_position_caps,
)
from src.data_quality import flag_and_clean_outliers
from src.backtest import run_backtest
from src.metrics import summary_table, monthly_returns
from src.walk_forward import build_folds, walk_forward_backtest, param_stability_table, param_stability_score
from src.param_drift import check_param_drift, format_drift_report
from src.ensemble import combine_returns, optimize_ensemble_weights, apply_portfolio_vol_target, DEFAULT_ALLOCATION
from src.portfolio_overlays import (
    sweep_idle_cash, compute_aggregate_correlation, correlation_based_cap_scale,
    tighten_caps_by_correlation, ramp_in_new_positions, apply_weight_overlays,
)
from src.event_blackout import load_macro_calendar, freeze_weights_on_blackout_days
from src.strategies import momentum, mean_reversion, sector_rotation
from src.monte_carlo import monte_carlo_summary, summary_for_json
from src.stress_test import CRYPTO_CRISIS_PERIODS, periods_covered, run_stress_test
from src.regime import compute_regime_scale
from src.tax import estimate_after_tax_cagr, estimate_after_tax_monthly

REPORTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "reports_crypto"
REPORTS_DIR.mkdir(exist_ok=True)

MOMENTUM_GRID = [dict(fast=f, slow=s) for f in (10, 20, 30) for s in (50, 100, 150) if f < s]
MEAN_REV_GRID = [dict(entry_rsi=e, exit_rsi=x) for e in (10.0, 15.0, 20.0) for x in (60.0, 70.0, 80.0)]
ROTATION_GRID = [dict(top_n=n) for n in (2, 3, 4)]

# Ver README, sección "Sesgo de asimetría cripto": el token base con el que se creó
# la señal es el que gana en el backtest -- el ranking de altcoins de este universo
# es el de HOY. No hay dataset gratuito de "membresía histórica de las top-N cripto
# por market cap" -- si un token estuvo entre los grandes en 2019 y hoy está muerto,
# no aparece acá.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                         help="Núcleos para paralelizar el walk-forward (default: todos menos uno).")
    parser.add_argument("--vol-target-max-exposure", type=float, default=None,
                         help="Diagnóstico/A-B: sobreescribe portfolio_vol_target.max_gross_exposure "
                              "(config/crypto_live_params.yaml, default: 1.0) SOLO para esta corrida. "
                              "Con 1.0 el overlay de vol-targeting del ensamble SOLO puede reducir "
                              "exposición, nunca aumentarla -- si la volatilidad realizada del ensamble "
                              "ya corre por debajo del objetivo (revisa 'ann_vol' de "
                              "ENSEMBLE_OOS_dynamic_alloc en summary.csv contra "
                              "portfolio_vol_target.vol_target), el overlay queda inerte y "
                              "ENSEMBLE_OOS_dynamic_alloc_vol_target sale idéntico a "
                              "ENSEMBLE_OOS_dynamic_alloc. Un valor >1.0 simula apalancamiento adicional "
                              "-- en cripto esto es MUCHO más riesgoso que en acciones (revisa el "
                              "max_drawdown resultante contra el de BTC buy-and-hold en el mismo reporte "
                              "antes de considerar usarlo en vivo). No cambia config/crypto_live_params.yaml "
                              "ni el bot en vivo.")
    args = parser.parse_args()

    universe = load_crypto_universe()
    live_params = load_crypto_live_params()
    symbols = all_crypto_symbols(universe)
    periods_per_year = live_params["periods_per_year"]

    print(f"Descargando {len(symbols)} símbolos de Binance (10y de análisis + buffer)...")
    raw_data = download_prices(symbols, years=11)
    close = build_close_matrix(raw_data, universe["quote_currency"])
    close = close.dropna(how="all")
    # max_daily_return más alto que en equities (0.60 vs. 0.50 default): un altcoin
    # chico SÍ puede moverse >50% en un día por noticias reales (listing, pump
    # genuino) sin ser un error de datos -- el filtro solo actúa si además el
    # precio REVIERTE rápido, así que subir el umbral reduce falsos positivos
    # sin dejar de atrapar los bad ticks reales de la API de Binance.
    close, outlier_report = flag_and_clean_outliers(close, max_daily_return=0.60)
    if not outlier_report.empty:
        outlier_report.to_csv(REPORTS_DIR / "data_quality_outliers.csv", index=False)
    volume = build_quote_volume_matrix(raw_data)
    print(f"Datos: {close.index.min().date()} -> {close.index.max().date()}, {close.shape[1]} símbolos "
          f"(algunos con menos historia -- altcoins listados después que BTC/ETH)")

    analysis_start = close.index.min() + pd.DateOffset(years=1)
    momentum_universe = all_crypto_symbols(universe)
    sector_universe = list(universe["altcoins"])
    cash_ticker = universe["quote_currency"]
    benchmark = universe["benchmark"]

    position_caps = build_crypto_position_caps(universe, live_params["position_caps"])
    print(f"Topes de peso por tipo de moneda: {live_params['position_caps']}")

    # ---------- Overlays de portafolio (ver src/portfolio_overlays.py) ----------
    overlay_cfg = live_params.get("portfolio_overlays", {})
    sweep_cash_ticker = cash_ticker if overlay_cfg.get("cash_sweep_enabled", True) else None
    cap_scale = None
    if overlay_cfg.get("dynamic_caps_enabled", True):
        corr_universe = [t for t in momentum_universe if t in close.columns]
        agg_corr = compute_aggregate_correlation(close[corr_universe].pct_change(),
                                                   window=overlay_cfg.get("corr_window", 60))
        cap_scale = correlation_based_cap_scale(
            agg_corr, full_cap_below=overlay_cfg.get("full_cap_below", 0.3),
            floor_above=overlay_cfg.get("floor_above", 0.7), min_scale=overlay_cfg.get("min_scale", 0.6),
        )
        pct_tightened = (cap_scale < 1.0).mean() * 100
        print(f"Topes de posición dinámicos por correlación activos: se ajustan en {pct_tightened:.1f}% "
              f"de los días del histórico.")

    ramp_max_daily_increase = overlay_cfg.get("ramp_max_daily_increase", 0.02) \
        if overlay_cfg.get("ramp_in_enabled", True) else None
    if ramp_max_daily_increase:
        print(f"Entrada escalonada de posiciones nuevas activa: máximo {ramp_max_daily_increase * 100:.1f} "
              f"puntos porcentuales de aumento por día.")

    # ---------- Blackout de eventos macro (FOMC, ver src/event_blackout.py) ----------
    blackout_cfg = live_params.get("event_blackout", {})
    blackout_dates = load_macro_calendar() if blackout_cfg.get("fomc_blackout_enabled", True) else set()
    if blackout_dates:
        dates_in_range = sum(1 for d in blackout_dates if close.index.min() <= d <= close.index.max())
        print(f"Blackout de eventos macro (FOMC) activo: {dates_in_range} fecha(s) conocidas dentro del "
              f"rango de datos.")

    def _finish_weights(w):
        if cap_scale is not None:
            w = tighten_caps_by_correlation(w, position_caps, cap_scale)
        if ramp_max_daily_increase:
            w = ramp_in_new_positions(w, max_daily_increase=ramp_max_daily_increase, cash_ticker=sweep_cash_ticker)
        w = sweep_idle_cash(w, sweep_cash_ticker)
        return freeze_weights_on_blackout_days(w, blackout_dates)

    regime_cfg = live_params["regime_filter"]
    regime_scale = compute_regime_scale(close[benchmark], regime_cfg) if regime_cfg.get("enabled") else None
    if regime_scale is not None:
        pct_derisked = (regime_scale < 1.0).mean() * 100
        print(f"Filtro de régimen macro activo (sobre {benchmark}): exposición reducida en "
              f"{pct_derisked:.1f}% de los días del histórico.")

    cost_cfg = live_params["costs"]
    backtest_kw = dict(
        volume=volume,
        use_liquidity_costs=cost_cfg["use_liquidity_costs"],
        liquidity_cost_kwargs=dict(
            base_capital=cost_cfg["base_capital"], base_spread_bps=cost_cfg["base_spread_bps"],
            impact_coeff=cost_cfg["impact_coeff"], max_cost_bps=cost_cfg["max_cost_bps"],
        ),
        regime_scale=regime_scale,
        dd_guard_threshold=-0.15, dd_guard_recover=-0.07, dd_guard_scale=0.5,
    )

    folds = build_folds(close.index, analysis_start, train_years=2, test_years=1, step_years=1, embargo_days=5)
    print(f"Walk-forward: {len(folds)} folds de {folds[0].train_start.date()} a {folds[-1].test_end.date()} "
          f"(embargo de 5 días entre train y test en cada fold)")

    # ---------- In-sample (referencia, optimista / sesgado a overfitting) ----------
    insample_returns = {}
    w_mom_is = _finish_weights(momentum.generate_weights(close, momentum_universe, None, position_caps))
    bt_mom = run_backtest(close, w_mom_is, **backtest_kw)
    insample_returns["momentum"] = bt_mom["returns"].loc[analysis_start:]
    w_mr_is = _finish_weights(mean_reversion.generate_weights(close, momentum_universe, None, position_caps))
    bt_mr = run_backtest(close, w_mr_is, **backtest_kw)
    insample_returns["mean_reversion"] = bt_mr["returns"].loc[analysis_start:]
    w_rot_is = _finish_weights(sector_rotation.generate_weights(close, sector_universe, cash_ticker))
    bt_rot = run_backtest(close, w_rot_is, **backtest_kw)
    insample_returns["sector_rotation"] = bt_rot["returns"].loc[analysis_start:]

    # ---------- Walk-forward (out-of-sample, estimación honesta) ----------
    # functools.partial (no lambda): tiene que ser "picklable" para --jobs > 1.
    print(f"Corriendo walk-forward de momentum ({args.jobs} núcleos)...")
    wf_mom = walk_forward_backtest(
        close, functools.partial(
            apply_weight_overlays,
            functools.partial(momentum.generate_weights, close, momentum_universe, max_weight_by_ticker=position_caps),
            sweep_cash_ticker, position_caps, cap_scale, blackout_dates, ramp_max_daily_increase,
        ),
        MOMENTUM_GRID, folds, backtest_kw, periods_per_year=periods_per_year, n_jobs=args.jobs,
    )
    print(f"Corriendo walk-forward de mean reversion ({args.jobs} núcleos, usa loop por moneda)...")
    wf_mr = walk_forward_backtest(
        close, functools.partial(
            apply_weight_overlays,
            functools.partial(mean_reversion.generate_weights, close, momentum_universe,
                               max_weight_by_ticker=position_caps),
            sweep_cash_ticker, position_caps, cap_scale, blackout_dates, ramp_max_daily_increase,
        ),
        MEAN_REV_GRID, folds, backtest_kw, periods_per_year=periods_per_year, n_jobs=args.jobs,
    )
    print(f"Corriendo walk-forward de rotación entre altcoins ({args.jobs} núcleos)...")
    wf_rot = walk_forward_backtest(
        close, functools.partial(
            apply_weight_overlays,
            functools.partial(sector_rotation.generate_weights, close, sector_universe, cash_ticker),
            sweep_cash_ticker, position_caps, cap_scale, blackout_dates, ramp_max_daily_increase,
        ),
        ROTATION_GRID, folds, backtest_kw, periods_per_year=periods_per_year, n_jobs=args.jobs,
    )

    oos_returns = {
        "momentum": wf_mom["returns"],
        "mean_reversion": wf_mr["returns"],
        "sector_rotation": wf_rot["returns"],
    }
    ensemble_oos = combine_returns(oos_returns, live_params["allocation"])

    # ---------- Asignación dinámica entre estrategias (alternativa a la mezcla fija) ----------
    dynamic = optimize_ensemble_weights(oos_returns, folds, live_params["allocation"], min_weight=0.05)
    ensemble_oos_dynamic = dynamic["returns"]
    with open(REPORTS_DIR / "ensemble_dynamic_allocations.json", "w") as f:
        json.dump(dynamic["fold_allocations"], f, indent=2)

    # ---------- Vol-targeting a nivel de portafolio (overlay sobre el ensamble dinámico) ----------
    pvt_cfg = live_params.get("portfolio_vol_target", {})
    max_gross_exposure = pvt_cfg.get("max_gross_exposure", 1.0)
    if args.vol_target_max_exposure is not None:
        max_gross_exposure = args.vol_target_max_exposure
        print(f"--vol-target-max-exposure: max_gross_exposure sobreescrito a {max_gross_exposure} "
              f"(config real: {pvt_cfg.get('max_gross_exposure', 1.0)}) -- solo para esta corrida de diagnóstico.")
    if pvt_cfg.get("enabled", True):
        ensemble_oos_vol_target = apply_portfolio_vol_target(
            ensemble_oos_dynamic, vol_target=pvt_cfg.get("vol_target", 0.20),
            vol_lookback=pvt_cfg.get("vol_lookback", 20),
            max_gross_exposure=max_gross_exposure,
            periods_per_year=periods_per_year,
        )
    else:
        ensemble_oos_vol_target = ensemble_oos_dynamic

    # ---------- Estabilidad de parámetros entre folds ----------
    print("\n" + "=" * 100)
    print("ESTABILIDAD DE PARÁMETROS ENTRE FOLDS")
    print("=" * 100)
    all_stability_scores = {}
    for name, wf in [("momentum", wf_mom), ("mean_reversion", wf_mr), ("sector_rotation", wf_rot)]:
        table_params = param_stability_table(wf["fold_reports"])
        scores = param_stability_score(wf["fold_reports"])
        all_stability_scores[name] = scores
        print(f"\n{name}:")
        print(table_params.to_string())
        print(f"  Estabilidad por parámetro: {scores}")
        table_params.to_csv(REPORTS_DIR / f"param_stability_{name}.csv")
    with open(REPORTS_DIR / "param_stability_scores.json", "w") as f:
        json.dump(all_stability_scores, f, indent=2)

    # ---------- Alerta de drift: ¿el parámetro en vivo sigue vigente? ----------
    drift_findings = check_param_drift(
        {"momentum": wf_mom["fold_reports"], "mean_reversion": wf_mr["fold_reports"],
         "sector_rotation": wf_rot["fold_reports"]},
        {"momentum": live_params["momentum"], "mean_reversion": live_params["mean_reversion"],
         "sector_rotation": live_params["sector_rotation"]},
    )
    print("\n" + "=" * 100)
    print("DRIFT DE PARÁMETROS (¿los de config/crypto_live_params.yaml siguen coincidiendo con el fold más reciente?)")
    print("=" * 100)
    print(format_drift_report(drift_findings))
    with open(REPORTS_DIR / "param_drift.json", "w") as f:
        json.dump(drift_findings, f, indent=2)

    # ---------- Benchmark ----------
    bench_returns = close[benchmark].pct_change().dropna()
    bench_returns = bench_returns.loc[oos_returns["momentum"].index.min():oos_returns["momentum"].index.max()]

    # ---------- Reporte ----------
    all_results = {
        f"benchmark_{benchmark}_buy_hold": bench_returns,
        "momentum_in_sample (sesgado)": insample_returns["momentum"],
        "mean_reversion_in_sample (sesgado)": insample_returns["mean_reversion"],
        "sector_rotation_in_sample (sesgado)": insample_returns["sector_rotation"],
        "momentum_OOS_walkforward": oos_returns["momentum"],
        "mean_reversion_OOS_walkforward": oos_returns["mean_reversion"],
        "sector_rotation_OOS_walkforward": oos_returns["sector_rotation"],
        "ENSEMBLE_OOS_walkforward": ensemble_oos,
        "ENSEMBLE_OOS_dynamic_alloc": ensemble_oos_dynamic,
        "ENSEMBLE_OOS_dynamic_alloc_vol_target": ensemble_oos_vol_target,
    }
    table = summary_table(all_results, periods_per_year=periods_per_year)
    print("\n" + "=" * 100)
    print("REPORTE COMPARATIVO -- MÓDULO CRIPTO (retornos en %, ann_vol/max_drawdown en %, monthly en %)")
    print("=" * 100)
    print(table.to_string())
    table.to_csv(REPORTS_DIR / "summary.csv")

    ens_avg_monthly = table.loc["ENSEMBLE_OOS_walkforward", "avg_monthly_return"]
    ens_cagr = table.loc["ENSEMBLE_OOS_walkforward", "cagr"]
    dyn_avg_monthly = table.loc["ENSEMBLE_OOS_dynamic_alloc", "avg_monthly_return"]
    vt_avg_monthly = table.loc["ENSEMBLE_OOS_dynamic_alloc_vol_target", "avg_monthly_return"]
    print(f"\nEnsamble cripto OOS mezcla FIJA (fuera de muestra, honesto): {ens_avg_monthly:.2f}% mensual promedio.")
    print(f"Ensamble cripto OOS asignación DINÁMICA: {dyn_avg_monthly:.2f}% mensual promedio.")
    print(f"Ensamble cripto OOS dinámica + vol-targeting: {vt_avg_monthly:.2f}% mensual promedio.")

    if live_params["tax"]["estimate_enabled"]:
        rate = live_params["tax"]["short_term_rate"]
        after_tax_cagr = estimate_after_tax_cagr(ens_cagr / 100, rate) * 100
        after_tax_monthly = estimate_after_tax_monthly(ens_avg_monthly / 100, rate) * 100
        print(f"[Estimación fiscal aproximada, tasa {rate*100:.0f}%] CAGR: {ens_cagr:.2f}% -> {after_tax_cagr:.2f}%, "
              f"mensual: {ens_avg_monthly:.2f}% -> {after_tax_monthly:.2f}%")
    else:
        print("[Sin estimación fiscal -- el tratamiento de cripto varía demasiado por país. "
              "Consulta a un contador de tu país, sobre todo en México donde no está bien definido.]")

    monthly_returns(ensemble_oos).to_csv(REPORTS_DIR / "ensemble_monthly_returns.csv")

    oos_returns_df = pd.DataFrame({
        "momentum": oos_returns["momentum"], "mean_reversion": oos_returns["mean_reversion"],
        "sector_rotation": oos_returns["sector_rotation"], "ensemble": ensemble_oos,
        "ensemble_dynamic_alloc": ensemble_oos_dynamic,
        "ensemble_dynamic_alloc_vol_target": ensemble_oos_vol_target,
    })
    oos_returns_df.index.name = "date"
    oos_returns_df.to_csv(REPORTS_DIR / "oos_returns.csv")

    # ---------- Monte Carlo ----------
    print("\n" + "=" * 100)
    print("MONTE CARLO (block bootstrap, bloques de 30 días de calendario)")
    print("=" * 100)
    mc = monte_carlo_summary(ensemble_oos, n_sims=1000, block_size=30, seed=7, periods_per_year=periods_per_year)
    am = mc["avg_monthly_return"]
    print(f"  p5={am['p5']*100:.2f}%  p25={am['p25']*100:.2f}%  mediana={am['p50']*100:.2f}%  "
          f"p75={am['p75']*100:.2f}%  p95={am['p95']*100:.2f}%")
    print(f"  Probabilidad de que el promedio mensual caiga en 0.5%-2%: "
          f"{mc['prob_avg_monthly_in_target_0.5_2pct']*100:.1f}%")
    print(f"  Probabilidad de que sea NEGATIVO: {mc['prob_avg_monthly_negative']*100:.1f}%")
    dd_p = mc["max_drawdown"]
    print(f"  Máximo drawdown -- p5={dd_p['p5']*100:.1f}%  mediana={dd_p['p50']*100:.1f}%  "
          f"p95(peor caso)={dd_p['p95']*100:.1f}%")

    with open(REPORTS_DIR / "monte_carlo.json", "w") as f:
        json.dump(summary_for_json(mc), f, indent=2)

    plt.figure()
    plt.hist(mc["_raw_avg_monthly"] * 100, bins=40)
    plt.axvspan(0.5, 2.0, alpha=0.18, color="#00ffc8", label="objetivo 0.5%-2%")
    plt.xlabel("Retorno mensual promedio simulado (%)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución Monte Carlo (ensamble cripto OOS)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "monte_carlo_hist.png", dpi=130)
    plt.close()

    # ---------- Stress test ----------
    print("\n" + "=" * 100)
    print("STRESS TEST: comportamiento en crashes cripto conocidos")
    print("=" * 100)
    insample_ensemble = combine_returns(insample_returns, live_params["allocation"])
    full_bench_returns = close[benchmark].pct_change().dropna()

    insample_stress_universe = {
        "momentum": insample_returns["momentum"], "mean_reversion": insample_returns["mean_reversion"],
        "sector_rotation": insample_returns["sector_rotation"], "ensemble": insample_ensemble,
        f"benchmark_{benchmark}": full_bench_returns,
    }
    covered_all = periods_covered(close.index, CRYPTO_CRISIS_PERIODS)
    if covered_all:
        print("\n[IN-SAMPLE, parámetros por defecto]")
        stress_insample = run_stress_test(insample_stress_universe, covered_all)
        print(stress_insample.to_string(index=False))
        stress_insample.to_csv(REPORTS_DIR / "stress_test_insample.csv", index=False)
    else:
        print("Ningún crash conocido cae dentro del rango de datos descargado.")

    oos_index = ensemble_oos.index
    covered_oos = periods_covered(oos_index, CRYPTO_CRISIS_PERIODS)
    if covered_oos:
        oos_stress_universe = {**oos_returns, "ensemble": ensemble_oos, f"benchmark_{benchmark}": bench_returns}
        print("\n[OUT-OF-SAMPLE walk-forward]")
        stress_oos = run_stress_test(oos_stress_universe, covered_oos)
        print(stress_oos.to_string(index=False))
        stress_oos.to_csv(REPORTS_DIR / "stress_test_oos.csv", index=False)
    else:
        print("\n[OUT-OF-SAMPLE] Ningún crash cripto conocido cae dentro del rango cubierto por el walk-forward.")

    for name, series in [("momentum", oos_returns["momentum"]), ("mean_reversion", oos_returns["mean_reversion"]),
                          ("sector_rotation", oos_returns["sector_rotation"]), ("ensemble", ensemble_oos),
                          ("ensemble_dynamic_alloc", ensemble_oos_dynamic),
                          ("ensemble_dynamic_alloc_vol_target", ensemble_oos_vol_target),
                          (f"benchmark_{benchmark}", bench_returns)]:
        equity = (1 + series).cumprod()
        plt.plot(equity.index, equity.values, label=name)
    plt.legend()
    plt.title("Curvas de equity OUT-OF-SAMPLE -- módulo cripto")
    plt.ylabel("Crecimiento de $1")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "equity_oos.png", dpi=130)

    print(f"\nArchivos guardados en {REPORTS_DIR}/ (incluye ensemble_dynamic_allocations.json). "
          f"Corre 'streamlit run dashboard.py' -> perfil 'Cripto' para verlo.")


if __name__ == "__main__":
    main()
