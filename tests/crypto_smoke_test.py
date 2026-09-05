"""Smoke test del módulo cripto con datos sintéticos (todos los días del año,
no solo hábiles) -- valida indicadores, estrategias, backtest, walk-forward,
Monte Carlo y stress test con `periods_per_year=365` de punta a punta, sin
tocar la red real (Binance está bloqueado desde este sandbox de todos modos).
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
from src.monte_carlo import monte_carlo_summary
from src.stress_test import CRYPTO_CRISIS_PERIODS, periods_covered, run_stress_test
from src.crypto_data import build_crypto_position_caps


def make_synthetic_crypto_close(n_days=6 * 365, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="D")  # TODOS los días, no bdate_range
    coins = {
        "BTCUSDT": (0.0006, 0.035),
        "ETHUSDT": (0.0007, 0.045),
        "BNBUSDT": (0.0005, 0.05),
        "SOLUSDT": (0.0008, 0.06),
        "XRPUSDT": (0.0002, 0.05),
        "ADAUSDT": (0.0002, 0.05),
        "LINKUSDT": (0.0004, 0.055),
        "AVAXUSDT": (0.0005, 0.06),
    }
    data = {}
    volume = {}
    for t, (mu, sigma) in coins.items():
        rets = rng.normal(mu, sigma, n_days)
        data[t] = 10 * np.exp(np.cumsum(rets))
        volume[t] = rng.uniform(1e6, 5e7, n_days)
    close = pd.DataFrame(data, index=dates)
    close["USDT"] = 1.0
    vol = pd.DataFrame(volume, index=dates)
    return close, vol


def main():
    close, quote_volume = make_synthetic_crypto_close()
    print(f"Datos sintéticos cripto: {close.shape[0]} días (calendario completo), {close.shape[1]} símbolos "
          f"({close.index.min().date()} -> {close.index.max().date()})")

    universe = dict(majors=["BTCUSDT", "ETHUSDT"],
                     altcoins=["BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT"],
                     quote_currency="USDT", benchmark="BTCUSDT")
    position_caps = build_crypto_position_caps(universe, dict(major_coin=0.30, altcoin=0.15))
    momentum_universe = universe["majors"] + universe["altcoins"]
    sector_universe = universe["altcoins"]
    cash_ticker = universe["quote_currency"]

    bt_kw = dict(cost_bps=10.0, dd_guard_threshold=-0.15, dd_guard_recover=-0.07, dd_guard_scale=0.5)
    crypto_momentum_params = dict(fast=20, slow=100, vol_target=0.20, vol_lookback=20, periods_per_year=365)
    crypto_mr_params = dict(trend_sma=100, entry_rsi=15.0, exit_rsi=70.0, max_hold_days=10, stop_loss_pct=0.10,
                             max_concurrent_positions=4, weight_per_position=0.08, vol_lookback=20,
                             reference_vol=0.50, periods_per_year=365)
    crypto_rot_params = dict(top_n=3, max_weight_per_asset=0.35, momentum_long=365, momentum_short=30,
                              rebalance_rule="ME")

    print("\n[1/6] Probando generación de pesos + backtest simple por estrategia (periods_per_year=365)...")
    w_mom = momentum.generate_weights(close, momentum_universe, crypto_momentum_params, position_caps)
    bt_mom = run_backtest(close, w_mom, **bt_kw)
    assert bt_mom["equity"].notna().all()
    print(f"  momentum: equity final = {bt_mom['equity'].iloc[-1]:.3f}")

    w_mr = mean_reversion.generate_weights(close, momentum_universe, crypto_mr_params, position_caps)
    bt_mr = run_backtest(close, w_mr, **bt_kw)
    assert bt_mr["equity"].notna().all()
    print(f"  mean_reversion: equity final = {bt_mr['equity'].iloc[-1]:.3f}")

    w_rot = sector_rotation.generate_weights(close, sector_universe, cash_ticker, crypto_rot_params)
    bt_rot = run_backtest(close, w_rot, **bt_kw)
    assert bt_rot["equity"].notna().all()
    assert (w_rot.sum(axis=1) <= 1.0001).all()
    print(f"  sector_rotation (rebalance_rule=ME, momentum 365/30): equity final = {bt_rot['equity'].iloc[-1]:.3f}")

    print("\n[2/6] Probando topes de posición por tipo de moneda (major_coin vs altcoin)...")
    for t in universe["majors"]:
        assert position_caps[t] == 0.30
    for t in universe["altcoins"]:
        assert position_caps[t] == 0.15
    assert (w_mom[universe["altcoins"]].max() <= 0.15 + 1e-6).all(), "una altcoin excede su tope"
    print("  topes OK: majors=0.30, altcoins=0.15, respetados en los pesos generados")

    print("\n[3/6] Probando walk-forward con periods_per_year=365...")
    analysis_start = close.index.min() + pd.DateOffset(years=1)
    folds = build_folds(close.index, analysis_start, train_years=2, test_years=1, step_years=1, embargo_days=5)
    assert len(folds) >= 2
    wf_mom = walk_forward_backtest(
        close, lambda p: momentum.generate_weights(close, momentum_universe, p, position_caps),
        [dict(fast=10, slow=50), dict(fast=20, slow=100)], folds, bt_kw, periods_per_year=365,
    )
    wf_mr = walk_forward_backtest(
        close, lambda p: mean_reversion.generate_weights(close, momentum_universe, p, position_caps),
        [dict(entry_rsi=15.0, exit_rsi=70.0), dict(entry_rsi=10.0, exit_rsi=60.0)], folds, bt_kw, periods_per_year=365,
    )
    wf_rot = walk_forward_backtest(
        close, lambda p: sector_rotation.generate_weights(close, sector_universe, cash_ticker, p),
        [dict(top_n=2), dict(top_n=3)], folds, bt_kw, periods_per_year=365,
    )
    for name, wf in [("momentum", wf_mom), ("mean_reversion", wf_mr), ("sector_rotation", wf_rot)]:
        assert len(wf["returns"]) > 0 and wf["returns"].notna().all(), f"{name}: walk-forward con problemas"
        table_params = param_stability_table(wf["fold_reports"])
        scores = param_stability_score(wf["fold_reports"])
        assert len(table_params) == len(folds) and scores
    print(f"  walk-forward OK para las 3 estrategias ({len(folds)} folds), estabilidad de parámetros calculada")

    print("\n[4/6] Probando ensamble + tabla de métricas (anualización 365 días)...")
    oos = {"momentum": wf_mom["returns"], "mean_reversion": wf_mr["returns"], "sector_rotation": wf_rot["returns"]}
    ens = combine_returns(oos, dict(momentum=0.45, mean_reversion=0.20, sector_rotation=0.35))
    table = summary_table({**oos, "ensemble": ens}, periods_per_year=365)
    assert not table.isna().all(axis=None)
    print(table.to_string())

    print("\n[5/6] Probando Monte Carlo (block_size=30, periods_per_year=365)...")
    mc = monte_carlo_summary(ens, n_sims=200, block_size=30, seed=3, periods_per_year=365)
    assert mc["avg_monthly_return"]["p5"] <= mc["avg_monthly_return"]["p50"] <= mc["avg_monthly_return"]["p95"]
    print(f"  Monte Carlo OK: p5={mc['avg_monthly_return']['p5']*100:.2f}% "
          f"p95={mc['avg_monthly_return']['p95']*100:.2f}%")

    print("\n[6/6] Probando stress test contra crashes cripto conocidos...")
    covered = periods_covered(ens.index, CRYPTO_CRISIS_PERIODS)
    if covered:
        stress = run_stress_test({"ensemble": ens, "momentum": oos["momentum"]}, covered)
        assert len(stress) == len(covered)
        print(f"  stress test OK, {len(covered)} período(s) cubiertos: {list(covered)}")
    else:
        print("  (ningún crash cripto conocido cae en el rango sintético -- revisa fechas si esto no se esperaba)")

    print("\nCRYPTO SMOKE TEST OK: el pipeline cripto completo corre sin errores.")
    print("(Datos sintéticos -- corre run_crypto_backtest.py con datos reales de Binance para resultados válidos.)")


if __name__ == "__main__":
    main()
