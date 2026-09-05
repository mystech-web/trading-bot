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


def main():
    print("[1/2] Probando que sin el flag se usa el max_gross_exposure de config...")
    test_default_run_uses_config_value()
    print("\n[2/2] Probando que --vol-target-max-exposure sobreescribe ese tope...")
    test_flag_overrides_max_gross_exposure()
    print("\nCRYPTO VOL TARGET MAX EXPOSURE TEST OK: el flag de diagnóstico llega correctamente hasta el overlay.")


if __name__ == "__main__":
    main()
