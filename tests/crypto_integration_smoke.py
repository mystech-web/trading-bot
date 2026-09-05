"""Corre `scripts/run_crypto_backtest.py` completo con datos sintéticos
(monkeypatchea `src.crypto_data.download_prices`) -- mismo propósito que
`tests/integration_smoke.py` pero para el módulo cripto: detectar bugs de
integración en el script real (no solo en las piezas sueltas).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import src.crypto_data as crypto_data_mod

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports_crypto"


def make_synthetic_klines(n_days=6 * 365, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="D")
    universe = crypto_data_mod.load_crypto_universe()
    symbols = crypto_data_mod.all_crypto_symbols(universe)

    crash_windows = [("2020-02-19", "2020-03-13"), ("2021-05-12", "2021-05-23"),
                      ("2022-05-07", "2022-05-13"), ("2022-11-06", "2022-11-14")]
    out = {}
    for sym in symbols:
        mu = rng.uniform(0.0001, 0.0008)
        sigma = rng.uniform(0.025, 0.06)
        rets = rng.normal(mu, sigma, n_days)
        for start, end in crash_windows:
            mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            rets[mask] -= rng.uniform(0.02, 0.06)
        price = 10 * np.exp(np.cumsum(rets))
        volume = rng.uniform(1e6, 5e7, n_days)
        out[sym] = pd.DataFrame({"Close": price, "Volume": volume / price, "QuoteVolume": volume}, index=dates)
    return out


def main():
    synthetic = make_synthetic_klines()
    crypto_data_mod.download_prices = lambda symbols, years=11, force=False: {
        s: synthetic[s] for s in symbols if s in synthetic
    }

    import scripts.run_crypto_backtest as rcb
    rcb.download_prices = crypto_data_mod.download_prices
    rcb.MOMENTUM_GRID = [dict(fast=f, slow=s) for f in (10, 20) for s in (50, 100) if f < s]
    rcb.MEAN_REV_GRID = [dict(entry_rsi=e, exit_rsi=x) for e in (10.0, 15.0) for x in (60.0, 70.0)]
    rcb.ROTATION_GRID = [dict(top_n=n) for n in (2, 3)]

    print("Corriendo scripts/run_crypto_backtest.py completo con datos sintéticos...\n")
    rcb.main()

    expected_files = [
        "summary.csv", "equity_oos.png", "ensemble_monthly_returns.csv", "oos_returns.csv",
        "monte_carlo.json", "monte_carlo_hist.png",
        "param_stability_momentum.csv", "param_stability_mean_reversion.csv",
        "param_stability_sector_rotation.csv", "param_stability_scores.json",
        "param_drift.json", "ensemble_dynamic_allocations.json",
    ]
    missing = [f for f in expected_files if not (REPORTS_DIR / f).exists()]
    assert not missing, f"Faltan archivos esperados: {missing}"

    import json
    mc = json.loads((REPORTS_DIR / "monte_carlo.json").read_text())
    assert "avg_monthly_return" in mc and "p50" in mc["avg_monthly_return"]

    summary = pd.read_csv(REPORTS_DIR / "summary.csv", index_col=0)
    assert "ENSEMBLE_OOS_walkforward" in summary.index

    print("\nCRYPTO INTEGRATION SMOKE TEST OK: run_crypto_backtest.py corrió sin errores y generó todos los reportes.")


if __name__ == "__main__":
    main()
