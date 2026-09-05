"""Corre el script real `scripts/run_backtest.py` de punta a punta, pero con
`src.data.download_prices` reemplazado por datos sintéticos (10+ años, para que
haya suficientes folds y para que los crashes conocidos post-2015 caigan dentro
del rango). Objetivo: detectar bugs de integración en el script completo
(guardado de CSV/JSON/PNG, manejo de índices, etc.) que el smoke_test.py más
chico no ejercita porque arma el pipeline a mano en vez de llamar a main().

No usa red real. No valida que los números tengan sentido económico (son
datos sintéticos), solo que el script corre sin excepciones y produce los
archivos esperados.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import src.data as data_mod

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"


def make_synthetic_price_data(n_days=12 * 252, seed=123):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2013-01-02", periods=n_days)  # cubre 2015, 2018, 2020, 2022

    import yaml
    with open(ROOT / "config" / "universe.yaml") as f:
        universe = yaml.safe_load(f)
    tickers = data_mod.all_tickers(universe)

    out = {}
    for t in tickers:
        mu = rng.uniform(-0.0001, 0.0005)
        sigma = rng.uniform(0.008, 0.018)
        rets = rng.normal(mu, sigma, n_days)
        # Inyecta caídas reales en fechas de crashes conocidos para que el stress test tenga contenido.
        crash_windows = [("2018-10-01", "2018-12-24"), ("2020-02-19", "2020-03-23"), ("2022-01-03", "2022-10-12")]
        idx = pd.Series(range(n_days), index=dates)
        for start, end in crash_windows:
            mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            rets[mask] -= rng.uniform(0.0005, 0.002)
        price = 100 * np.exp(np.cumsum(rets))
        volume = rng.integers(200_000, 5_000_000, n_days).astype(float)
        out[t] = pd.DataFrame({"Close": price, "Volume": volume}, index=dates)
    return out


def _run_profile(rb, profile: str, reports_dir: pathlib.Path):
    old_argv = sys.argv
    sys.argv = ["run_backtest.py"] if profile == "conservative" else ["run_backtest.py", "--profile", "aggressive"]
    try:
        rb.main()
    finally:
        sys.argv = old_argv

    expected_files = [
        "summary.csv", "equity_oos.png", "ensemble_monthly_returns.csv", "oos_returns.csv",
        "monte_carlo.json", "monte_carlo_hist.png",
        "param_stability_momentum.csv", "param_stability_mean_reversion.csv",
        "param_stability_sector_rotation.csv", "param_stability_scores.json", "stress_test_insample.csv",
        "param_drift.json", "ensemble_dynamic_allocations.json",
    ]
    missing = [f for f in expected_files if not (reports_dir / f).exists()]
    assert not missing, f"[{profile}] Faltan archivos esperados: {missing}"

    import json
    mc = json.loads((reports_dir / "monte_carlo.json").read_text())
    assert "avg_monthly_return" in mc and "p50" in mc["avg_monthly_return"]

    summary = pd.read_csv(reports_dir / "summary.csv", index_col=0)
    assert "ENSEMBLE_OOS_walkforward" in summary.index
    print(f"  [{profile}] OK -- archivos generados en {reports_dir}/")


def main():
    synthetic = make_synthetic_price_data()
    data_mod.download_prices = lambda tickers, years=11, force=False: {
        t: synthetic[t] for t in tickers if t in synthetic
    }

    import scripts.run_backtest as rb
    rb.download_prices = data_mod.download_prices
    # Grids más chicos para que la corrida de integración sea rápida.
    rb.MOMENTUM_GRID = [dict(fast=f, slow=s) for f in (20, 50) for s in (100, 200) if f < s]
    rb.MEAN_REV_GRID = [dict(entry_rsi=e, exit_rsi=x) for e in (5.0, 10.0) for x in (60.0, 70.0)]
    rb.ROTATION_GRID = [dict(top_n=n) for n in (2, 3)]

    print("Corriendo scripts/run_backtest.py completo (perfil conservador) con datos sintéticos...\n")
    _run_profile(rb, "conservative", ROOT / "reports")

    print("\nCorriendo scripts/run_backtest.py completo (perfil AGRESIVO, con ETFs apalancados) "
          "con datos sintéticos...\n")
    _run_profile(rb, "aggressive", ROOT / "reports_aggressive")

    print("\nINTEGRATION SMOKE TEST OK: run_backtest.py completo corrió sin errores (ambos perfiles) "
          "y generó todos los reportes.")


if __name__ == "__main__":
    main()
