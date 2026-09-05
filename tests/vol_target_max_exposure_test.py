"""Valida el flag de diagnóstico `--vol-target-max-exposure` de
`scripts/run_backtest.py`. Contexto: `src/ensemble.py -> apply_portfolio_vol_target`
usa `raw_scale = (vol_target / realized).clip(upper=max_gross_exposure)` -- con el
`max_gross_exposure` real de `config/live_params.yaml` (1.0), el overlay SOLO puede
REDUCIR exposición cuando la volatilidad realizada del ensamble está por encima del
objetivo, nunca aumentarla cuando está por debajo (que es el caso típico observado en
backtests reales: `ENSEMBLE_OOS_dynamic_alloc_vol_target` sale idéntico a
`ENSEMBLE_OOS_dynamic_alloc`). Este flag permite, solo para diagnóstico en el
backtest, sobreescribir ese tope y ver si usar ese margen de volatilidad no utilizado
ayuda a acercar el retorno mensual promedio al objetivo -- sin tocar
config/live_params.yaml ni el bot en vivo.

No corre el backtest completo con datos reales (usa datos sintéticos) -- solo
verifica el cableado: que el flag llega hasta `apply_portfolio_vol_target` y que,
sin el flag, el comportamiento (tope real de 1.0) no cambia.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import src.data as data_mod
from src.data import load_universe

ROOT = pathlib.Path(__file__).resolve().parent.parent


def make_synthetic_price_data(tickers, n_days=12 * 252, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2013-01-02", periods=n_days)
    out = {}
    for t in tickers:
        mu = rng.uniform(-0.0001, 0.0005)
        sigma = rng.uniform(0.008, 0.018)
        price = 100 * np.exp(np.cumsum(rng.normal(mu, sigma, n_days)))
        volume = rng.integers(200_000, 5_000_000, n_days).astype(float)
        out[t] = pd.DataFrame({"Close": price, "Volume": volume}, index=dates)
    return out


def _run_and_capture_max_exposure(rb, extra_argv):
    universe = load_universe()
    all_possible = data_mod.all_tickers(universe)
    synthetic = make_synthetic_price_data(all_possible)

    rb.download_prices = lambda tickers, years=11, force=False: {
        t: synthetic[t] for t in tickers if t in synthetic
    }
    rb.MOMENTUM_GRID = [dict(fast=20, slow=100)]
    rb.MEAN_REV_GRID = [dict(entry_rsi=10.0, exit_rsi=70.0)]
    rb.ROTATION_GRID = [dict(top_n=2)]

    captured = {}
    orig_apply = rb.apply_portfolio_vol_target

    def spy_apply(returns, vol_target=0.10, vol_lookback=20, max_gross_exposure=1.0, periods_per_year=252):
        captured["max_gross_exposure"] = max_gross_exposure
        return orig_apply(returns, vol_target=vol_target, vol_lookback=vol_lookback,
                           max_gross_exposure=max_gross_exposure, periods_per_year=periods_per_year)

    rb.apply_portfolio_vol_target = spy_apply
    old_argv = sys.argv
    sys.argv = ["run_backtest.py"] + extra_argv
    try:
        rb.main()
    finally:
        sys.argv = old_argv
        rb.apply_portfolio_vol_target = orig_apply
    return captured["max_gross_exposure"]


def test_default_run_uses_config_value():
    """Sin el flag, max_gross_exposure debe ser el de config/live_params.yaml (1.0),
    sin cambios de comportamiento respecto a antes de agregar este flag."""
    import scripts.run_backtest as rb
    live_params = rb.load_live_params(rb.resolve_profile("conservative")[0])
    expected = live_params.get("portfolio_vol_target", {}).get("max_gross_exposure", 1.0)
    got = _run_and_capture_max_exposure(rb, [])
    assert got == expected, f"sin el flag debería usar el valor de config ({expected}), se usó {got}"


def test_flag_overrides_max_gross_exposure():
    """Con --vol-target-max-exposure 1.3, ese valor (no el 1.0 de config) debe
    llegar hasta apply_portfolio_vol_target -- así el overlay puede escalar por
    encima de 100% cuando la volatilidad realizada del ensamble está por debajo
    del objetivo, en vez de quedar inerte."""
    import scripts.run_backtest as rb
    got = _run_and_capture_max_exposure(rb, ["--vol-target-max-exposure", "1.3"])
    assert got == 1.3, f"el flag debería sobreescribir max_gross_exposure a 1.3, se usó {got}"


def main():
    print("[1/2] Probando que sin el flag se usa el max_gross_exposure de config (1.0)...")
    test_default_run_uses_config_value()
    print("\n[2/2] Probando que --vol-target-max-exposure sobreescribe ese tope...")
    test_flag_overrides_max_gross_exposure()
    print("\nVOL TARGET MAX EXPOSURE TEST OK: el flag de diagnóstico llega correctamente hasta el overlay.")


if __name__ == "__main__":
    main()
