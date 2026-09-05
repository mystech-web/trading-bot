"""Valida los flags `--exclude-satellite`/`--include-satellite` de
`scripts/run_backtest.py`, que sobreescriben `include_satellite_etfs`
(`config/live_params*.yaml`, default `false`) SOLO para una corrida --
agregados para poder aislar, con un A/B real, si EFA/EEM/VNQ/DBC/IEF estaban
arrastrando el retorno mensual promedio del ensamble por debajo del 0.5%
objetivo (lo estaban -- por eso el default real quedó en `false`, ver
`config/live_params.yaml -> include_satellite_etfs`).

Es puramente un atajo de backtest para no tener que editar config/live_params.yaml
a mano para comparar: NO toca el archivo. `scripts/run_live_once.py` respeta el
mismo `include_satellite_etfs` de config (ver `tests/diversifier_etfs_test.py` y
`tests/international_diversification_test.py` para esa regresión).

Usa datos sintéticos (no red real); espía `src.data.download_prices` para
confirmar con qué tickers se llama realmente, que es la fuente de verdad de
qué universo usó la corrida.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import src.data as data_mod
from src.data import load_universe

ROOT = pathlib.Path(__file__).resolve().parent.parent


def make_synthetic_price_data(tickers, n_days=12 * 252, seed=99):
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


def _run_and_capture_tickers(rb, extra_argv):
    universe = load_universe()
    all_possible = data_mod.all_tickers(universe)
    synthetic = make_synthetic_price_data(all_possible)

    captured = {}

    def spy_download(tickers, years=11, force=False):
        # `main()` también llama a `download_prices` por separado para el VIX
        # (src/regime.py -> señal de régimen), con un solo ticker ('^VIX') --
        # nos interesa la descarga del universo completo, no esa, así que solo
        # guardamos la primera llamada (la del universo, siempre la más grande).
        if "tickers" not in captured:
            captured["tickers"] = list(tickers)
        return {t: synthetic[t] for t in tickers if t in synthetic}

    rb.download_prices = spy_download
    rb.MOMENTUM_GRID = [dict(fast=20, slow=100)]
    rb.MEAN_REV_GRID = [dict(entry_rsi=10.0, exit_rsi=70.0)]
    rb.ROTATION_GRID = [dict(top_n=2)]

    old_argv = sys.argv
    sys.argv = ["run_backtest.py"] + extra_argv
    try:
        rb.main()
    finally:
        sys.argv = old_argv
    return captured["tickers"]


def test_default_run_excludes_satellite_tickers():
    """Sin flags, debe reflejar el default real de config/live_params.yaml
    (include_satellite_etfs: false) -- EFA/EEM/VNQ/DBC/IEF NO se descargan."""
    import scripts.run_backtest as rb
    live_params = rb.load_live_params(rb.resolve_profile("conservative")[0])
    assert live_params.get("include_satellite_etfs", False) is False, (
        "este test asume el default real (false) en config/live_params.yaml -- "
        "si lo activaste a propósito, actualiza este test."
    )
    universe = load_universe()
    tickers = _run_and_capture_tickers(rb, [])
    excluded = universe.get("international_etfs", []) + universe.get("diversifier_etfs", [])
    for t in excluded:
        assert t not in tickers, f"{t} no debería descargarse por default (include_satellite_etfs: false)"


def test_exclude_satellite_flag_drops_the_five_tickers():
    """--exclude-satellite fuerza la exclusión sin importar config -- hoy es
    redundante con el default real, pero sigue sirviendo si algún día alguien
    activa include_satellite_etfs en su config y quiere una corrida puntual sin."""
    import scripts.run_backtest as rb
    universe = load_universe()
    tickers = _run_and_capture_tickers(rb, ["--exclude-satellite"])
    excluded = universe.get("international_etfs", []) + universe.get("diversifier_etfs", [])
    for t in excluded:
        assert t not in tickers, f"{t} no debería descargarse con --exclude-satellite"
    # El resto del universo (broad_etfs, sector_etfs, liquid_stocks, cash_proxy) se mantiene.
    for t in universe["broad_etfs"] + universe["liquid_stocks"] + [universe["cash_proxy"]]:
        assert t in tickers, f"{t} debería seguir en el universo -- --exclude-satellite solo saca los 5 satélite"


def test_include_satellite_flag_brings_them_back():
    """--include-satellite fuerza la inclusión sin importar config -- para
    comparar A/B contra el default (excluidos) sin tocar config/live_params.yaml."""
    import scripts.run_backtest as rb
    universe = load_universe()
    tickers = _run_and_capture_tickers(rb, ["--include-satellite"])
    included = universe.get("international_etfs", []) + universe.get("diversifier_etfs", [])
    for t in included:
        assert t in tickers, f"{t} debería descargarse con --include-satellite"


def main():
    print("[1/3] Probando que sin flags se respeta el default real (excluidos)...")
    test_default_run_excludes_satellite_tickers()
    print("\n[2/3] Probando que --exclude-satellite saca EFA/EEM/VNQ/DBC/IEF del backtest...")
    test_exclude_satellite_flag_drops_the_five_tickers()
    print("\n[3/3] Probando que --include-satellite los trae de vuelta...")
    test_include_satellite_flag_brings_them_back()
    print("\nEXCLUDE SATELLITE TEST OK: los flags de diagnóstico controlan correctamente los 5 tickers satélite.")


if __name__ == "__main__":
    main()
