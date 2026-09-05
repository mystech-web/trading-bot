"""Punto de entrada principal: descarga datos, corre las 3 estrategias
(in-sample para referencia + walk-forward para la estimación honesta),
arma el ensamble, y genera un reporte comparativo.

Uso:
    python scripts/run_backtest.py                     # perfil conservador (default)
    python scripts/run_backtest.py --profile aggressive # perfil agresivo (apalancado, más riesgo -- ver README)
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
import numpy as np
import pandas as pd

from src.plot_style import apply_neon_style
apply_neon_style()

from src.data import (
    load_universe, load_live_params, resolve_profile, all_tickers, download_prices,
    build_close_matrix, build_field_matrix, build_position_caps,
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
from src.stress_test import CRISIS_PERIODS, periods_covered, run_stress_test
from src.regime import compute_regime_scale
from src.tax import estimate_after_tax_cagr, estimate_after_tax_monthly

# Grids más densos que la versión inicial -- más caro de correr, pero permite ver si
# el parámetro "óptimo" es consistente entre folds (ver param_stability_score) en vez
# de un único par que podría ser puro ruido de un rango chico de combinaciones.
MOMENTUM_GRID = [
    dict(fast=f, slow=s) for f in (10, 20, 30, 50) for s in (100, 150, 200, 252) if f < s
]
MEAN_REV_GRID = [
    dict(entry_rsi=e, exit_rsi=x) for e in (5.0, 10.0, 15.0) for x in (60.0, 70.0, 80.0)
]
ROTATION_GRID = [dict(top_n=n) for n in (2, 3, 4, 5)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["conservative", "aggressive"], default="conservative",
                         help="'conservative' (default) o 'aggressive' (apalancado -- ver README antes de usarlo).")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                         help="Núcleos para paralelizar el walk-forward (default: todos menos uno). "
                              "1 = secuencial, sin paralelismo.")
    parser.add_argument("--exclude-satellite", action="store_true",
                         help="Fuerza excluir international_etfs (EFA/EEM) y diversifier_etfs (VNQ/DBC/IEF) "
                              "de este backtest, sin importar 'include_satellite_etfs' en config/live_params*.yaml. "
                              "Es el default real desde que un backtest mostró que diluían avg_monthly_return "
                              "por debajo del objetivo -- este flag ya no debería cambiar nada salvo que hayas "
                              "activado 'include_satellite_etfs: true' en tu config.")
    parser.add_argument("--include-satellite", action="store_true",
                         help="Fuerza incluir EFA/EEM/VNQ/DBC/IEF en momentum/mean_reversion para este backtest, "
                              "sin importar config/live_params*.yaml -- para comparar A/B contra el default "
                              "(excluidos) y decidir si la diversificación extra vale el costo en retorno.")
    parser.add_argument("--vol-target-max-exposure", type=float, default=None,
                         help="Sobreescribe portfolio_vol_target.max_gross_exposure "
                              "(config/live_params*.yaml) SOLO para esta corrida. "
                              "Con 1.0 el overlay de vol-targeting del ensamble SOLO puede reducir "
                              "exposición, nunca aumentarla -- si la volatilidad realizada del ensamble "
                              "ya corre por debajo del objetivo (revisa 'ann_vol' de ENSEMBLE_OOS_dynamic_alloc "
                              "en summary.csv contra portfolio_vol_target.vol_target), el overlay queda "
                              "inerte y ENSEMBLE_OOS_dynamic_alloc_vol_target sale idéntico a "
                              "ENSEMBLE_OOS_dynamic_alloc. Un valor >1.0 simula apalancamiento adicional para "
                              "comparar -- el conservador ya usa 1.3 por default (ver config/live_params.yaml).")
    args = parser.parse_args()
    if args.exclude_satellite and args.include_satellite:
        parser.error("--exclude-satellite y --include-satellite son mutuamente excluyentes.")

    live_params_path, reports_dir = resolve_profile(args.profile)
    reports_dir.mkdir(exist_ok=True)
    global REPORTS_DIR
    REPORTS_DIR = reports_dir
    if args.profile == "aggressive":
        print("=" * 100)
        print("PERFIL AGRESIVO: usa apalancamiento (ETFs 3x) y mayor riesgo. Lee config/live_params_aggressive.yaml")
        print("y la sección correspondiente del README antes de operar esto con dinero real, ni siquiera en paper.")
        print("=" * 100)

    universe = load_universe()
    live_params = load_live_params(live_params_path)
    # include_satellite_etfs (config/live_params*.yaml, default False): un backtest real
    # (2015-2026) mostró que EFA/EEM/VNQ/DBC/IEF diluían avg_monthly_return del ensamble
    # por debajo del objetivo -- ver README, sección de diversificación. Quedan fuera de
    # momentum/mean_reversion por default; --include-satellite/--exclude-satellite fuerzan
    # lo contrario para este backtest sin tocar el config.
    include_satellite = live_params.get("include_satellite_etfs", False)
    if args.include_satellite:
        include_satellite = True
    if args.exclude_satellite:
        include_satellite = False
    if not include_satellite:
        universe = dict(universe, international_etfs=[], diversifier_etfs=[])
    print(f"Diversificación satélite (EFA/EEM/VNQ/DBC/IEF) en momentum/mean_reversion: "
          f"{'activa' if include_satellite else 'desactivada (default)'}.")
    tickers = all_tickers(universe)
    print(f"Descargando {len(tickers)} tickers (10y de análisis + 1y de calentamiento)...")
    raw_data = download_prices(tickers, years=11)
    close = build_close_matrix(raw_data)
    close = close.dropna(how="all")
    close, outlier_report = flag_and_clean_outliers(close)
    if not outlier_report.empty:
        outlier_report.to_csv(REPORTS_DIR / "data_quality_outliers.csv", index=False)
    volume = build_field_matrix(raw_data, "Volume")
    print(f"Datos: {close.index.min().date()} -> {close.index.max().date()}, {close.shape[1]} tickers")

    analysis_start = close.index.min() + pd.DateOffset(years=1)
    base_universe = sorted(set(universe["broad_etfs"]) | set(universe["liquid_stocks"])
                            | set(universe.get("international_etfs", []))
                            | set(universe.get("diversifier_etfs", [])))
    base_universe = [t for t in base_universe if t in close.columns]
    mean_reversion_universe = base_universe
    momentum_universe = base_universe
    if live_params.get("include_leveraged_etfs"):
        leveraged = [t for t in universe.get("leveraged_etfs", []) if t in close.columns]
        momentum_universe = sorted(set(base_universe) | set(leveraged))
        print(f"Perfil agresivo: se agregan {leveraged} al universo de MOMENTUM (no a mean_reversion, "
              f"por el decay que sufren los apalancados en estrategias de idas y vueltas cortas).")
    sector_universe = [t for t in universe["sector_etfs"] if t in close.columns]
    cash_ticker = universe["cash_proxy"]
    benchmark = universe["benchmark"]

    position_caps = build_position_caps(universe, live_params["position_caps"])
    print(f"Topes de peso por tipo de activo (mitiga sesgo de supervivencia en acciones "
          f"individuales): {live_params['position_caps']}")

    # ---------- Overlays de portafolio (ver src/portfolio_overlays.py) ----------
    # Se aplican a los pesos de CADA estrategia, antes de correr el backtest: barrido
    # de cash ocioso hacia el proxy de cash, y (opcional) topes de posición más
    # estrictos en días de correlación agregada alta entre TODO el universo.
    overlay_cfg = live_params.get("portfolio_overlays", {})
    sweep_cash_ticker = cash_ticker if overlay_cfg.get("cash_sweep_enabled", True) else None
    cap_scale = None
    if overlay_cfg.get("dynamic_caps_enabled", True):
        corr_universe = sorted(set(universe["broad_etfs"]) | set(universe["sector_etfs"])
                                | set(universe["liquid_stocks"]) | set(universe.get("international_etfs", []))
                                | set(universe.get("diversifier_etfs", [])))
        corr_universe = [t for t in corr_universe if t in close.columns]
        agg_corr = compute_aggregate_correlation(close[corr_universe].pct_change(),
                                                   window=overlay_cfg.get("corr_window", 60))
        cap_scale = correlation_based_cap_scale(
            agg_corr, full_cap_below=overlay_cfg.get("full_cap_below", 0.3),
            floor_above=overlay_cfg.get("floor_above", 0.7), min_scale=overlay_cfg.get("min_scale", 0.6),
        )
        pct_tightened = (cap_scale < 1.0).mean() * 100
        print(f"Topes de posición dinámicos por correlación activos: se ajustan en {pct_tightened:.1f}% "
              f"de los días del histórico (correlación agregada del universo por encima de "
              f"{overlay_cfg.get('full_cap_below', 0.3)}).")

    ramp_max_daily_increase = overlay_cfg.get("ramp_max_daily_increase", 0.02) \
        if overlay_cfg.get("ramp_in_enabled", True) else None
    if ramp_max_daily_increase:
        print(f"Entrada escalonada de posiciones nuevas activa: máximo {ramp_max_daily_increase * 100:.1f} "
              f"puntos porcentuales de aumento por día (las bajadas nunca se limitan).")

    # ---------- Blackout de eventos macro (FOMC, ver src/event_blackout.py) ----------
    blackout_cfg = live_params.get("event_blackout", {})
    blackout_dates = load_macro_calendar() if blackout_cfg.get("fomc_blackout_enabled", True) else set()
    if blackout_dates:
        dates_in_range = sum(1 for d in blackout_dates if close.index.min() <= d <= close.index.max())
        print(f"Blackout de eventos macro (FOMC) activo: {dates_in_range} fecha(s) conocidas dentro del "
              f"rango de datos -- esos días el portafolio no rebalancea (mantiene los pesos del día anterior).")

    def _finish_weights(w):
        if cap_scale is not None:
            w = tighten_caps_by_correlation(w, position_caps, cap_scale)
        if ramp_max_daily_increase:
            w = ramp_in_new_positions(w, max_daily_increase=ramp_max_daily_increase, cash_ticker=sweep_cash_ticker)
        w = sweep_idle_cash(w, sweep_cash_ticker)
        return freeze_weights_on_blackout_days(w, blackout_dates)

    regime_cfg = live_params["regime_filter"]
    vix_close = None
    if regime_cfg.get("enabled") and regime_cfg.get("vix_enabled", True):
        # ^VIX no es parte del universo invertible (no se puede comprar directamente) --
        # se descarga por separado, solo como señal para el filtro de régimen.
        vix_raw = download_prices(["^VIX"], years=11)
        vix_matrix = build_close_matrix(vix_raw)
        if "^VIX" in vix_matrix.columns and not vix_matrix["^VIX"].dropna().empty:
            vix_close = vix_matrix["^VIX"]
            print("Señal de régimen: VIX descargado -- se usa como tercera señal (volatilidad implícita).")
        else:
            print("[WARN] No se pudo descargar VIX -- el filtro de régimen sigue funcionando solo con "
                  "tendencia de precio + volatilidad realizada.")

    regime_scale = compute_regime_scale(close[benchmark], regime_cfg, vix_close=vix_close) \
        if regime_cfg.get("enabled") else None
    if regime_scale is not None:
        pct_derisked = (regime_scale < 1.0).mean() * 100
        print(f"Filtro de régimen macro activo: exposición reducida en {pct_derisked:.1f}% de los días "
              f"del histórico (SPY bajo su SMA{regime_cfg['sma_window']}).")

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

    folds = build_folds(close.index, analysis_start, train_years=3, test_years=1, step_years=1, embargo_days=5)
    print(f"Walk-forward: {len(folds)} folds de {folds[0].train_start.date()} a {folds[-1].test_end.date()} "
          f"(embargo de 5 días entre train y test en cada fold)")

    # ---------- In-sample (referencia, optimista / sesgado a overfitting) ----------
    insample_returns = {}
    w_mom_is = _finish_weights(momentum.generate_weights(close, momentum_universe, None, position_caps))
    bt_mom = run_backtest(close, w_mom_is, **backtest_kw)
    insample_returns["momentum"] = bt_mom["returns"].loc[analysis_start:]
    w_mr_is = _finish_weights(mean_reversion.generate_weights(close, mean_reversion_universe, None, position_caps))
    bt_mr = run_backtest(close, w_mr_is, **backtest_kw)
    insample_returns["mean_reversion"] = bt_mr["returns"].loc[analysis_start:]
    w_rot_is = _finish_weights(sector_rotation.generate_weights(close, sector_universe, cash_ticker))
    bt_rot = run_backtest(close, w_rot_is, **backtest_kw)
    insample_returns["sector_rotation"] = bt_rot["returns"].loc[analysis_start:]

    # ---------- Walk-forward (out-of-sample, estimación honesta) ----------
    # functools.partial (no lambda): tiene que ser "picklable" para mandarse a otros
    # procesos cuando --jobs > 1 -- un lambda o closure no se puede enviar entre
    # procesos con el contexto "spawn" (el default en macOS/Windows).
    print(f"Corriendo walk-forward de momentum ({args.jobs} núcleos)...")
    wf_mom = walk_forward_backtest(
        close, functools.partial(
            apply_weight_overlays,
            functools.partial(momentum.generate_weights, close, momentum_universe, max_weight_by_ticker=position_caps),
            sweep_cash_ticker, position_caps, cap_scale, blackout_dates, ramp_max_daily_increase,
        ),
        MOMENTUM_GRID, folds, backtest_kw, n_jobs=args.jobs,
    )
    print(f"Corriendo walk-forward de mean reversion ({args.jobs} núcleos, usa loop por ticker)...")
    wf_mr = walk_forward_backtest(
        close, functools.partial(
            apply_weight_overlays,
            functools.partial(mean_reversion.generate_weights, close, mean_reversion_universe,
                               max_weight_by_ticker=position_caps),
            sweep_cash_ticker, position_caps, cap_scale, blackout_dates, ramp_max_daily_increase,
        ),
        MEAN_REV_GRID, folds, backtest_kw, n_jobs=args.jobs,
    )
    print(f"Corriendo walk-forward de rotación sectorial ({args.jobs} núcleos)...")
    wf_rot = walk_forward_backtest(
        close, functools.partial(
            apply_weight_overlays,
            functools.partial(sector_rotation.generate_weights, close, sector_universe, cash_ticker),
            sweep_cash_ticker, position_caps, cap_scale, blackout_dates, ramp_max_daily_increase,
        ),
        ROTATION_GRID, folds, backtest_kw, n_jobs=args.jobs,
    )

    oos_returns = {
        "momentum": wf_mom["returns"],
        "mean_reversion": wf_mr["returns"],
        "sector_rotation": wf_rot["returns"],
    }
    ensemble_oos = combine_returns(oos_returns, DEFAULT_ALLOCATION)

    # ---------- Asignación dinámica entre estrategias (alternativa a la mezcla fija) ----------
    # Mismos `folds` que el walk-forward de arriba: para cada fold, pesa cada
    # estrategia según su volatilidad OOS en los folds ANTERIORES (nunca mira el
    # fold actual ni futuros -- ver docstring de optimize_ensemble_weights). El
    # fold 0 no tiene historial, así que arranca igual que el ensamble fijo.
    dynamic = optimize_ensemble_weights(oos_returns, folds, DEFAULT_ALLOCATION, min_weight=0.05)
    ensemble_oos_dynamic = dynamic["returns"]
    with open(REPORTS_DIR / "ensemble_dynamic_allocations.json", "w") as f:
        json.dump(dynamic["fold_allocations"], f, indent=2)

    # ---------- Vol-targeting a nivel de portafolio (overlay sobre el ensamble dinámico) ----------
    # Cada estrategia individual ya apunta a su propia volatilidad objetivo (ver
    # src/strategies/momentum.py), pero el ENSAMBLE combinado no tenía ningún
    # control de volatilidad propio -- si las 3 coinciden en modo agresivo al
    # mismo tiempo, el ensamble puede terminar más volátil de lo que cualquiera
    # apunta individualmente. Se aplica sobre el ensamble dinámico (la mejor
    # combinación disponible hasta acá) para tener una tercera comparación.
    pvt_cfg = live_params.get("portfolio_vol_target", {})
    max_gross_exposure = pvt_cfg.get("max_gross_exposure", 1.0)
    if args.vol_target_max_exposure is not None:
        max_gross_exposure = args.vol_target_max_exposure
        print(f"--vol-target-max-exposure: max_gross_exposure sobreescrito a {max_gross_exposure} "
              f"(config real: {pvt_cfg.get('max_gross_exposure', 1.0)}) -- solo para esta corrida de diagnóstico.")
    if pvt_cfg.get("enabled", True):
        ensemble_oos_vol_target = apply_portfolio_vol_target(
            ensemble_oos_dynamic, vol_target=pvt_cfg.get("vol_target", 0.10),
            vol_lookback=pvt_cfg.get("vol_lookback", 20),
            max_gross_exposure=max_gross_exposure,
        )
    else:
        ensemble_oos_vol_target = ensemble_oos_dynamic

    # ---------- Estabilidad de parámetros entre folds ----------
    print("\n" + "=" * 100)
    print("ESTABILIDAD DE PARÁMETROS ENTRE FOLDS (¿el 'óptimo' es real o es ruido?)")
    print("=" * 100)
    all_stability_scores = {}
    for name, wf in [("momentum", wf_mom), ("mean_reversion", wf_mr), ("sector_rotation", wf_rot)]:
        table_params = param_stability_table(wf["fold_reports"])
        scores = param_stability_score(wf["fold_reports"])
        all_stability_scores[name] = scores
        print(f"\n{name}:")
        print(table_params.to_string())
        print(f"  Estabilidad por parámetro (1.0 = siempre el mismo valor en todos los folds): {scores}")
        low_stability = [p for p, s in scores.items() if s < 0.5]
        if low_stability:
            print(f"  [AVISO] Parámetro(s) inestables ({', '.join(low_stability)}): el valor 'óptimo' "
                  f"cambia de fold a fold. Trátalo como ruido, no como una ventaja real de {name}.")
        table_params.to_csv(REPORTS_DIR / f"param_stability_{name}.csv")

    with open(REPORTS_DIR / "param_stability_scores.json", "w") as f:
        json.dump(all_stability_scores, f, indent=2)

    # ---------- Alerta de drift: ¿el parámetro en vivo sigue vigente? ----------
    # Compara SOLO lo que el walk-forward eligió en su fold MÁS RECIENTE contra lo
    # que está fijo en config/live_params.yaml (lo que run_live_once.py usa de
    # verdad) -- nunca auto-aplica nada, solo avisa. Ver src/param_drift.py.
    drift_findings = check_param_drift(
        {"momentum": wf_mom["fold_reports"], "mean_reversion": wf_mr["fold_reports"],
         "sector_rotation": wf_rot["fold_reports"]},
        {"momentum": live_params["momentum"], "mean_reversion": live_params["mean_reversion"],
         "sector_rotation": live_params["sector_rotation"]},
    )
    print("\n" + "=" * 100)
    print("DRIFT DE PARÁMETROS (¿los de config/live_params.yaml siguen coincidiendo con el fold más reciente?)")
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
    table = summary_table(all_results)
    print("\n" + "=" * 100)
    print("REPORTE COMPARATIVO (retornos en %, ann_vol/max_drawdown en %, monthly en %)")
    print("=" * 100)
    print(table.to_string())
    table.to_csv(REPORTS_DIR / "summary.csv")

    ens_avg_monthly = table.loc["ENSEMBLE_OOS_walkforward", "avg_monthly_return"]
    ens_cagr = table.loc["ENSEMBLE_OOS_walkforward", "cagr"]
    dyn_avg_monthly = table.loc["ENSEMBLE_OOS_dynamic_alloc", "avg_monthly_return"]
    vt_avg_monthly = table.loc["ENSEMBLE_OOS_dynamic_alloc_vol_target", "avg_monthly_return"]
    print(f"\nObjetivo del usuario: 0.5% - 2.0% de retorno mensual promedio.")
    print(f"Ensamble OOS mezcla FIJA (fuera de muestra, honesto) logró: {ens_avg_monthly:.2f}% mensual promedio.")
    print(f"Ensamble OOS asignación DINÁMICA (pesa más a la estrategia de menor vol reciente) logró: "
          f"{dyn_avg_monthly:.2f}% mensual promedio.")
    print(f"Ensamble OOS asignación dinámica + vol-targeting a nivel portafolio logró: "
          f"{vt_avg_monthly:.2f}% mensual promedio.")
    print("(Ninguna de las tres es automáticamente 'mejor' -- compara también ann_vol y max_drawdown "
          "en la tabla de arriba, y mira ensemble_dynamic_allocations.json para ver cómo cambió el reparto "
          "entre folds. Menos volatilidad no siempre implica mejor retorno.)")
    if 0.5 <= ens_avg_monthly <= 2.0:
        print("-> Mezcla fija: dentro del rango objetivo EN EL BACKTEST HISTÓRICO. Esto no garantiza el futuro.")
    else:
        print("-> Mezcla fija: fuera del rango objetivo en este backtest. Revisa parámetros/universo o ajusta expectativas.")

    # ---------- Estimación (aproximada) de drag fiscal ----------
    if live_params["tax"]["estimate_enabled"]:
        rate = live_params["tax"]["short_term_rate"]
        after_tax_cagr = estimate_after_tax_cagr(ens_cagr / 100, rate) * 100
        after_tax_monthly = estimate_after_tax_monthly(ens_avg_monthly / 100, rate) * 100
        print(f"\n[Estimación fiscal aproximada, cuenta gravable, tasa de corto plazo {rate*100:.0f}%]")
        print(f"  CAGR antes de impuestos: {ens_cagr:.2f}% -> después de impuestos (aprox.): {after_tax_cagr:.2f}%")
        print(f"  Mensual antes de impuestos: {ens_avg_monthly:.2f}% -> después (aprox.): {after_tax_monthly:.2f}%")
        print("  (Aproximación simple asumiendo ganancia de corto plazo cada año -- no es asesoría fiscal. "
              "No aplica si operas en una cuenta con ventajas fiscales.)")

    monthly_returns(ensemble_oos).to_csv(REPORTS_DIR / "ensemble_monthly_returns.csv")

    # ---------- Datos crudos para el dashboard ----------
    oos_returns_df = pd.DataFrame({
        "momentum": oos_returns["momentum"],
        "mean_reversion": oos_returns["mean_reversion"],
        "sector_rotation": oos_returns["sector_rotation"],
        "ensemble": ensemble_oos,
        "ensemble_dynamic_alloc": ensemble_oos_dynamic,
        "ensemble_dynamic_alloc_vol_target": ensemble_oos_vol_target,
    })
    oos_returns_df.index.name = "date"
    oos_returns_df.to_csv(REPORTS_DIR / "oos_returns.csv")

    # ---------- Monte Carlo (block bootstrap) sobre el ensamble OOS ----------
    print("\n" + "=" * 100)
    print("MONTE CARLO (block bootstrap, 1000 simulaciones sobre el ensamble OUT-OF-SAMPLE)")
    print("=" * 100)
    mc = monte_carlo_summary(ensemble_oos, n_sims=1000, block_size=21, seed=7)
    am = mc["avg_monthly_return"]
    print(f"Retorno mensual promedio -- distribución de escenarios plausibles (no un solo número):")
    print(f"  p5={am['p5']*100:.2f}%  p25={am['p25']*100:.2f}%  mediana={am['p50']*100:.2f}%  "
          f"p75={am['p75']*100:.2f}%  p95={am['p95']*100:.2f}%")
    print(f"  Probabilidad de que el promedio mensual caiga en el rango objetivo 0.5%-2%: "
          f"{mc['prob_avg_monthly_in_target_0.5_2pct']*100:.1f}%")
    print(f"  Probabilidad de que el promedio mensual sea NEGATIVO: "
          f"{mc['prob_avg_monthly_negative']*100:.1f}%")
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
    plt.title("Distribución Monte Carlo del retorno mensual promedio (ensamble OOS)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "monte_carlo_hist.png", dpi=130)
    plt.close()

    # ---------- Stress test contra crashes conocidos ----------
    print("\n" + "=" * 100)
    print("STRESS TEST: comportamiento en crashes conocidos")
    print("=" * 100)
    insample_ensemble = combine_returns(insample_returns, DEFAULT_ALLOCATION)
    full_bench_returns = close[benchmark].pct_change().dropna()

    insample_stress_universe = {
        "momentum": insample_returns["momentum"],
        "mean_reversion": insample_returns["mean_reversion"],
        "sector_rotation": insample_returns["sector_rotation"],
        "ensemble": insample_ensemble,
        f"benchmark_{benchmark}": full_bench_returns,
    }
    covered_all = periods_covered(close.index, CRISIS_PERIODS)
    if covered_all:
        print("\n[IN-SAMPLE, parámetros por defecto -- cubre todo el rango de datos]")
        stress_insample = run_stress_test(insample_stress_universe, covered_all)
        print(stress_insample.to_string(index=False))
        stress_insample.to_csv(REPORTS_DIR / "stress_test_insample.csv", index=False)
    else:
        print("Ningún crash conocido cae dentro del rango de datos descargado.")

    oos_index = ensemble_oos.index
    covered_oos = periods_covered(oos_index, CRISIS_PERIODS)
    if covered_oos:
        oos_stress_universe = {**oos_returns, "ensemble": ensemble_oos,
                                f"benchmark_{benchmark}": bench_returns}
        print("\n[OUT-OF-SAMPLE walk-forward -- solo períodos cubiertos por el rango de test de los folds]")
        stress_oos = run_stress_test(oos_stress_universe, covered_oos)
        print(stress_oos.to_string(index=False))
        stress_oos.to_csv(REPORTS_DIR / "stress_test_oos.csv", index=False)
    else:
        print("\n[OUT-OF-SAMPLE] Ningún crash conocido cae dentro del rango cubierto por el walk-forward "
              "(normal si el walk-forward empieza después de 2018-2020: usa el resultado in-sample de "
              "arriba como referencia de comportamiento en crashes, con la salvedad de que no es OOS).")

    for name, series in [("momentum", oos_returns["momentum"]),
                          ("mean_reversion", oos_returns["mean_reversion"]),
                          ("sector_rotation", oos_returns["sector_rotation"]),
                          ("ensemble", ensemble_oos),
                          ("ensemble_dynamic_alloc", ensemble_oos_dynamic),
                          ("ensemble_dynamic_alloc_vol_target", ensemble_oos_vol_target),
                          (f"benchmark_{benchmark}", bench_returns)]:
        equity = (1 + series).cumprod()
        plt.plot(equity.index, equity.values, label=name)
    plt.legend()
    plt.title("Curvas de equity OUT-OF-SAMPLE (walk-forward)")
    plt.ylabel("Crecimiento de $1")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "equity_oos.png", dpi=130)
    print(f"\nArchivos guardados en {REPORTS_DIR}/:")
    print("  summary.csv, equity_oos.png, ensemble_monthly_returns.csv, oos_returns.csv")
    print("  ensemble_dynamic_allocations.json (reparto de capital entre estrategias, fold a fold)")
    print("  param_stability_<estrategia>.csv (una por estrategia)")
    print("  monte_carlo.json, monte_carlo_hist.png")
    print("  stress_test_insample.csv" + (", stress_test_oos.csv" if covered_oos else ""))
    print("\nCorre 'streamlit run dashboard.py' para ver todo esto en un dashboard interactivo.")


if __name__ == "__main__":
    main()
