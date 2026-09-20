"""Valida el flag de diagnóstico `--vol-target-max-exposure` de
`scripts/run_crypto_backtest.py` -- misma idea que
`tests/vol_target_max_exposure_test.py` (bot de acciones), pero para el
módulo cripto. Contexto: un backtest real con datos de Binance mostró que
`ENSEMBLE_OOS_dynamic_alloc_vol_target` salía idéntico a
`ENSEMBLE_OOS_dynamic_alloc` -- la volatilidad realizada del ensamble corría
bien por debajo del `vol_target` de `config/crypto_live_params.yaml` (20%),
así que el overlay (`max_gross_exposure: 1.0` real) no tenía margen para
escalar. Este flag permite probar, solo en el backtest, si subir ese tope
ayuda -- sin tocar el config ni el bot en vivo.

No corre el backtest completo con datos reales (usa datos sintéticos) -- solo
verifica el cableado: que el flag llega hasta `apply_portfolio_vol_target` y
que, sin el flag, el comportamiento (tope real de config) no cambia.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import src.crypto_data as crypto_data_mod

ROOT = pathlib.Path(__file__).resolve().parent.parent


def make_synthetic_klines(n_days=6 * 365, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="D")
    universe = crypto_data_mod.load_crypto_universe()
    symbols = crypto_data_mod.all_crypto_symbols(universe)
    out = {}
    for sym in symbols:
        mu = rng.uniform(0.0001, 0.0008)
        sigma = rng.uniform(0.025, 0.06)
        price = 10 * np.exp(np.cumsum(rng.normal(mu, sigma, n_days)))
        volume = rng.uniform(1e6, 5e7, n_days)
        out[sym] = pd.DataFrame({"Close": price, "Volume": volume / price, "QuoteVolume": volume}, index=dates)
    return out


def _run_and_capture_max_exposure(rcb, extra_argv):
    synthetic = make_synthetic_klines()
    rcb.download_prices = lambda symbols, years=11, force=False: {
        s: synthetic[s] for s in symbols if s in synthetic
    }
    rcb.MOMENTUM_GRID = [dict(fast=20, slow=50)]
    rcb.MEAN_REV_GRID = [dict(entry_rsi=15.0, exit_rsi=70.0)]
    rcb.ROTATION_GRID = [dict(top_n=2)]

    captured = {}
    orig_apply = rcb.apply_portfolio_vol_target

    def spy_apply(returns, vol_target=0.20, vol_lookback=20, max_gross_exposure=1.0, periods_per_year=365):
        captured["max_gross_exposure"] = max_gross_exposure
        return orig_apply(returns, vol_target=vol_target, vol_lookback=vol_lookback,
                           max_gross_exposure=max_gross_exposure, periods_per_year=periods_per_year)

    rcb.apply_portfolio_vol_target = spy_apply
    old_argv = sys.argv
    sys.argv = ["run_crypto_backtest.py"] + extra_argv
    try:
        rcb.main()
    finally:
        sys.argv = old_argv
        rcb.apply_portfolio_vol_target = orig_apply
    return captured["max_gross_exposure"]


def test_default_run_uses_config_value():
    """Sin el flag, max_gross_exposure debe ser el de config/crypto_live_params.yaml (1.0)."""
    import scripts.run_crypto_backtest as rcb
    live_params = crypto_data_mod.load_crypto_live_params()
    expected = live_params.get("portfolio_vol_target", {}).get("max_gross_exposure", 1.0)
    got = _run_and_capture_max_exposure(rcb, [])
    assert got == expected, f"sin el flag debería usar el valor de config ({expected}), se usó {got}"


def test_flag_overrides_max_gross_exposure():
    """Con --vol-target-max-exposure 1.5, ese valor (no el de config) debe llegar
    hasta apply_portfolio_vol_target."""
    import scripts.run_crypto_backtest as rcb
    got = _run_and_capture_max_exposure(rcb, ["--vol-target-max-exposure", "1.5"])
    assert got == 1.5, f"el flag debería sobreescribir max_gross_exposure a 1.5, se usó {got}"


def test_monte_carlo_generated_for_all_three_ensemble_variants():
    """Misma regresión que en el bot de acciones (ver
    tests/vol_target_max_exposure_test.py): el Monte Carlo debe generarse por
    separado para las 3 variantes del ensamble, no solo para la de mezcla fija
    -- si no, decidir sobre --vol-target-max-exposure mirando monte_carlo.json
    estaría mirando una serie que ese flag nunca afecta."""
    import json
    import scripts.run_crypto_backtest as rcb
    _run_and_capture_max_exposure(rcb, ["--vol-target-max-exposure", "1.5"])

    mc_fixed = json.loads((rcb.REPORTS_DIR / "monte_carlo.json").read_text())
    mc_dynamic = json.loads((rcb.REPORTS_DIR / "monte_carlo_dynamic_alloc.json").read_text())
    mc_vol_target = json.loads((rcb.REPORTS_DIR / "monte_carlo_dynamic_alloc_vol_target.json").read_text())

    for name, mc in [("monte_carlo.json", mc_fixed), ("monte_carlo_dynamic_alloc.json", mc_dynamic),
                      ("monte_carlo_dynamic_alloc_vol_target.json", mc_vol_target)]:
        assert "avg_monthly_return" in mc and "p50" in mc["avg_monthly_return"], f"{name} incompleto: {mc}"

    assert mc_fixed["avg_monthly_return"]["p50"] != mc_dynamic["avg_monthly_return"]["p50"] \
        or mc_fixed["max_drawdown"]["p50"] != mc_dynamic["max_drawdown"]["p50"], \
        "monte_carlo.json y monte_carlo_dynamic_alloc.json no deberían ser idénticos (series distintas)"
    print("  las 3 variantes del Monte Carlo cripto se generaron por separado y con contenido propio")


def main():
    print("[1/3] Probando que sin el flag se usa el max_gross_exposure de config...")
    test_default_run_uses_config_value()
    print("\n[2/3] Probando que --vol-target-max-exposure sobreescribe ese tope...")
    test_flag_overrides_max_gross_exposure()
    print("\n[3/3] Probando que el Monte Carlo se genera para las 3 variantes del ensamble, no solo la fija...")
    test_monte_carlo_generated_for_all_three_ensemble_variants()
    print("\nCRYPTO VOL TARGET MAX EXPOSURE TEST OK: el flag de diagnóstico llega correctamente hasta el overlay.")


if __name__ == "__main__":
    main()
