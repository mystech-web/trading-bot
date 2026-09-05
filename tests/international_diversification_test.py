"""Valida la diversificación internacional (ver config/universe.yaml ->
`international_etfs`): que EFA/EEM entran al universo de descarga, que reciben
su propio tope de peso (`international_etf`, ver `build_position_caps` en
`src/data.py`) en vez de heredar por accidente el de otra categoría, y que el
flag `include_satellite_etfs` (`config/live_params*.yaml`, default `false`)
los excluye/incluye consistentemente en `scripts/run_backtest.py` Y
`scripts/run_live_once.py`.

`include_satellite_etfs` quedó en `false` por default después de que un
backtest real (2015-2026) mostrara que EFA/EEM (junto con VNQ/DBC/IEF, ver
`tests/diversifier_etfs_test.py`) diluían `avg_monthly_return` del ensamble
por debajo del objetivo -- ver el comentario en
`config/live_params.yaml -> include_satellite_etfs`.

No corre el backtest completo (eso ya lo hace `tests/integration_smoke.py`,
que lee `config/universe.yaml` de verdad y por lo tanto ya ejercita EFA/EEM
con datos sintéticos de punta a punta) -- este test aísla solo el cableado de
universo/topes/flag.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import load_universe, load_live_params, resolve_profile, all_tickers, build_position_caps
from src.strategies import momentum, mean_reversion, sector_rotation
import scripts.run_live_once as rlo


def test_universe_yaml_declares_international_etfs():
    universe = load_universe()
    assert "international_etfs" in universe, "falta la sección international_etfs en config/universe.yaml"
    intl = set(universe["international_etfs"])
    assert intl == {"EFA", "EEM"}, f"se esperaba exactamente {{EFA, EEM}}, se encontró {intl}"


def test_all_tickers_includes_international_etfs():
    """all_tickers() (usado para decidir qué se descarga) los incluye siempre,
    sin importar include_satellite_etfs -- ese flag solo controla si entran al
    universo de las ESTRATEGIAS, no si se descargan."""
    universe = load_universe()
    tickers = all_tickers(universe)
    for t in universe["international_etfs"]:
        assert t in tickers, f"{t} debería estar en all_tickers() para que se descargue su historial"


def test_position_caps_assigns_dedicated_international_cap():
    universe = load_universe()
    position_caps = {"broad_etf": 0.20, "sector_etf": 0.40, "individual_stock": 0.08, "international_etf": 0.15}
    caps = build_position_caps(universe, position_caps)
    for t in universe["international_etfs"]:
        assert caps[t] == 0.15, f"{t} debería recibir el tope 'international_etf' (0.15), recibió {caps.get(t)}"
    # No debería colarse en el tope de broad_etf (son valores distintos en este test a propósito).
    assert caps["EFA"] != position_caps["broad_etf"]


def test_position_caps_has_safe_default_when_key_missing():
    universe = load_universe()
    # Config sin la clave 'international_etf' (perfiles viejos, antes de este cambio) --
    # no debería explotar con KeyError, debe caer a un default conservador.
    position_caps = {"broad_etf": 0.20, "sector_etf": 0.40, "individual_stock": 0.08}
    caps = build_position_caps(universe, position_caps)
    for t in universe["international_etfs"]:
        assert t in caps and caps[t] > 0, f"{t} debería tener un tope por default aunque falte la clave en config"


def test_base_universe_construction_respects_include_satellite_etfs_flag():
    """Replica la lógica de scripts/run_backtest.py (base_universe se arma a
    partir de `universe`, que el script muta a categorías vacías cuando
    include_satellite_etfs es false): con el flag en false (default), EFA/EEM
    NO deberían estar en base_universe; con el flag en true, sí -- en ningún
    caso deberían estar en sector_universe (es rotación de sectores de EE.UU.)."""
    universe = load_universe()
    for include_satellite in (False, True):
        active_universe = universe if include_satellite else dict(universe, international_etfs=[], diversifier_etfs=[])
        base_universe = sorted(set(active_universe["broad_etfs"]) | set(active_universe["liquid_stocks"])
                                | set(active_universe.get("international_etfs", []))
                                | set(active_universe.get("diversifier_etfs", [])))
        for t in universe["international_etfs"]:
            if include_satellite:
                assert t in base_universe, f"{t} debería estar en base_universe con include_satellite_etfs=true"
            else:
                assert t not in base_universe, f"{t} NO debería estar en base_universe con include_satellite_etfs=false (default)"

        sector_universe = list(universe["sector_etfs"])
        for t in universe["international_etfs"]:
            assert t not in sector_universe, f"{t} NO debería estar en sector_universe (con o sin el flag)"


def _capture_live_universes(close, universe, live_params):
    """Corre rlo.compute_target_weights espiando generate_weights de cada
    estrategia, para confirmar con qué universo de tickers se llama realmente
    compute_target_weights -- la fuente de verdad de qué opera el bot en vivo."""
    captured = {}
    orig_mom, orig_mr, orig_rot = momentum.generate_weights, mean_reversion.generate_weights, \
        sector_rotation.generate_weights

    def spy_mom(close_, tickers, *a, **kw):
        captured["momentum"] = list(tickers)
        return orig_mom(close_, tickers, *a, **kw)

    def spy_mr(close_, tickers, *a, **kw):
        captured["mean_reversion"] = list(tickers)
        return orig_mr(close_, tickers, *a, **kw)

    def spy_rot(close_, sector_tickers, cash_ticker, *a, **kw):
        captured["sector_rotation"] = list(sector_tickers)
        return orig_rot(close_, sector_tickers, cash_ticker, *a, **kw)

    momentum.generate_weights = spy_mom
    mean_reversion.generate_weights = spy_mr
    sector_rotation.generate_weights = spy_rot
    try:
        rlo.compute_target_weights(close, universe, live_params)
    finally:
        momentum.generate_weights = orig_mom
        mean_reversion.generate_weights = orig_mr
        sector_rotation.generate_weights = orig_rot
    return captured


def test_run_live_once_respects_include_satellite_etfs_flag():
    """Regresión: run_backtest.py y run_live_once.py arman `base_universe` con
    su propia copia de la misma lógica (no comparten una función) -- que el
    flag apague/prenda EFA/EEM en un lado y se le olvide en el otro dejaría al
    bot en vivo operando (o no) algo distinto de lo que el backtest reportó."""
    universe = load_universe()
    live_params_path, _ = resolve_profile("conservative")
    live_params = load_live_params(live_params_path)
    assert live_params.get("include_satellite_etfs", False) is False, (
        "este test asume que config/live_params.yaml sigue con include_satellite_etfs: false "
        "(el default real) -- si lo cambiaste a propósito, actualiza este test."
    )

    needed = sorted(set(universe["broad_etfs"]) | set(universe["liquid_stocks"])
                     | set(universe.get("international_etfs", [])) | set(universe.get("diversifier_etfs", []))
                     | set(universe["sector_etfs"]) | {universe["cash_proxy"], universe["benchmark"]})
    dates = pd.bdate_range("2020-01-02", periods=300)
    rng = np.random.default_rng(7)
    close = pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(dates)))) for t in needed}, index=dates,
    )

    captured_default = _capture_live_universes(close, universe, live_params)
    for t in universe["international_etfs"]:
        assert t not in captured_default["momentum"], \
            f"{t} NO debería estar en el universo de momentum en vivo con include_satellite_etfs=false (default)"
        assert t not in captured_default["mean_reversion"], \
            f"{t} NO debería estar en el universo de mean_reversion en vivo con include_satellite_etfs=false (default)"

    live_params_on = dict(live_params, include_satellite_etfs=True)
    captured_on = _capture_live_universes(close, universe, live_params_on)
    for t in universe["international_etfs"]:
        assert t in captured_on["momentum"], f"{t} debería estar en momentum en vivo con include_satellite_etfs=true"
        assert t in captured_on["mean_reversion"], f"{t} debería estar en mean_reversion en vivo con include_satellite_etfs=true"
        assert t not in captured_on["sector_rotation"], f"{t} no debería estar en sector_rotation en ningún caso"


def main():
    print("[1/6] Probando que universe.yaml declara EFA/EEM...")
    test_universe_yaml_declares_international_etfs()
    print("\n[2/6] Probando que all_tickers() los incluye (se descargan)...")
    test_all_tickers_includes_international_etfs()
    print("\n[3/6] Probando que reciben su propio tope de peso...")
    test_position_caps_assigns_dedicated_international_cap()
    print("\n[4/6] Probando que hay un default seguro si falta la clave en config...")
    test_position_caps_has_safe_default_when_key_missing()
    print("\n[5/6] Probando que include_satellite_etfs controla base_universe (backtest), nunca sector_rotation...")
    test_base_universe_construction_respects_include_satellite_etfs_flag()
    print("\n[6/6] Probando que el bot EN VIVO (run_live_once.py) respeta el mismo flag que el backtest...")
    test_run_live_once_respects_include_satellite_etfs_flag()
    print("\nINTERNATIONAL DIVERSIFICATION TEST OK: EFA/EEM están correctamente cableados (backtest y en vivo, ambos casos del flag).")


if __name__ == "__main__":
    main()
