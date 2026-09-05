"""Valida tax-loss harvesting automático (ver src/tax_loss_harvesting.py):
detección de candidatos, la guardia de wash sale, el costo promedio ponderado
que ahora lleva `VirtualBroker`, el método equivalente en `AlpacaBroker` (con
un broker falso, sin red), y el flujo completo dentro de
`scripts/run_live_once.py` con el broker virtual -- que una posición con
pérdida se vende (aunque la señal normal quisiera mantenerla) y que no se
recompra automáticamente mientras dure la ventana de wash sale.
"""
import sys
import pathlib
import tempfile
import types
import json
import datetime as dt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.tax_loss_harvesting import find_harvest_candidates, block_recent_harvest_rebuys, prune_expired_harvests
from src.live.virtual_broker import VirtualBroker
from src.live.alpaca_broker import AlpacaBroker
from src.tracking import load_state
import scripts.run_live_once as rl


def test_find_harvest_candidates_detects_losses_past_threshold():
    positions = {
        "AAA": dict(qty=10.0, avg_entry_price=100.0, current_price=90.0),   # -10%, pasa el umbral -5%
        "BBB": dict(qty=5.0, avg_entry_price=100.0, current_price=97.0),    # -3%, NO pasa el umbral
        "CCC": dict(qty=8.0, avg_entry_price=50.0, current_price=60.0),     # ganancia, se ignora
    }
    candidates = find_harvest_candidates(positions, loss_threshold_pct=-0.05)
    tickers = {c["ticker"] for c in candidates}
    assert tickers == {"AAA"}, f"solo AAA debería calificar como candidato: {tickers}"
    assert candidates[0]["unrealized_plpc"] < -0.05


def test_find_harvest_candidates_orders_worst_first_and_ignores_missing_data():
    positions = {
        "AAA": dict(qty=10.0, avg_entry_price=100.0, current_price=90.0),   # -10%
        "DDD": dict(qty=10.0, avg_entry_price=100.0, current_price=70.0),   # -30%, la peor
        "EEE": dict(qty=0.0, avg_entry_price=100.0, current_price=50.0),    # qty=0, se ignora
        "FFF": dict(qty=10.0, avg_entry_price=None, current_price=50.0),    # sin cost basis, se ignora
    }
    candidates = find_harvest_candidates(positions, loss_threshold_pct=-0.05)
    assert [c["ticker"] for c in candidates] == ["DDD", "AAA"], \
        f"debe ordenar de peor a mejor pérdida, ignorando posiciones sin datos: {candidates}"


def test_block_recent_harvest_rebuys_zeroes_within_window_not_after():
    target_weights = {"AAA": 0.20, "BBB": 0.10}
    today = dt.date(2026, 1, 31)
    recently_harvested = {
        "AAA": (today - dt.timedelta(days=10)).isoformat(),   # dentro de la ventana de 31 días
        "BBB": (today - dt.timedelta(days=40)).isoformat(),   # ya pasó la ventana
    }
    blocked = block_recent_harvest_rebuys(target_weights, recently_harvested, today, wash_sale_days=31)
    assert blocked["AAA"] == 0.0, "AAA debería seguir bloqueado (10 días desde la cosecha, ventana de 31)"
    assert blocked["BBB"] == 0.10, "BBB ya debería estar libre (40 días desde la cosecha, pasó la ventana)"


def test_prune_expired_harvests_drops_old_entries():
    today = dt.date(2026, 1, 31)
    recently_harvested = {
        "AAA": (today - dt.timedelta(days=10)).isoformat(),
        "BBB": (today - dt.timedelta(days=40)).isoformat(),
    }
    pruned = prune_expired_harvests(recently_harvested, today, wash_sale_days=31)
    assert set(pruned) == {"AAA"}, f"solo AAA debería sobrevivir el pruning: {pruned}"


def test_virtual_broker_tracks_weighted_average_cost_basis():
    with tempfile.TemporaryDirectory() as tmp:
        state_file = pathlib.Path(tmp) / "state.json"
        broker = VirtualBroker(state_file, starting_cash=100_000.0)

        broker.rebalance_to_weights({"SPY": 0.20}, {"SPY": 100.0}, dry_run=False,
                                     min_weight_drift=0.0, min_order_usd=0.0)
        cb1 = broker.get_positions_with_cost_basis({"SPY": 100.0})
        assert abs(cb1["SPY"]["avg_entry_price"] - 100.0) < 0.01

        # Compra MÁS a un precio distinto ($150) -> el costo promedio debe quedar
        # PONDERADO entre ambas compras, ni en 100 ni en 150 puros.
        broker.rebalance_to_weights({"SPY": 0.40}, {"SPY": 150.0}, dry_run=False,
                                     min_weight_drift=0.0, min_order_usd=0.0)
        cb2 = broker.get_positions_with_cost_basis({"SPY": 150.0})
        avg = cb2["SPY"]["avg_entry_price"]
        assert 100.0 < avg < 150.0, f"el costo promedio ponderado debería quedar entre 100 y 150, quedó en {avg}"
        print(f"  costo promedio ponderado tras compras a $100 y $150: ${avg:.2f} (correcto: entre ambos)")


def test_virtual_broker_clears_cost_basis_when_position_closed():
    with tempfile.TemporaryDirectory() as tmp:
        state_file = pathlib.Path(tmp) / "state.json"
        broker = VirtualBroker(state_file, starting_cash=100_000.0)
        broker.rebalance_to_weights({"SPY": 0.20}, {"SPY": 100.0}, dry_run=False,
                                     min_weight_drift=0.0, min_order_usd=0.0)
        assert "SPY" in broker.state["cost_basis"]

        broker.rebalance_to_weights({"SPY": 0.0}, {"SPY": 100.0}, dry_run=False,
                                     min_weight_drift=0.0, min_order_usd=0.0)
        assert "SPY" not in broker.state.get("positions", {})
        assert "SPY" not in broker.state.get("cost_basis", {}), \
            "el costo base debería limpiarse al cerrar la posición, no quedar 'fantasma'"


def test_alpaca_broker_get_positions_with_cost_basis():
    class _FakePosition:
        def __init__(self, symbol, qty, avg_entry_price, current_price):
            self.symbol = symbol
            self.qty = qty
            self.avg_entry_price = avg_entry_price
            self.current_price = current_price

    class _FakeClient:
        def get_all_positions(self):
            return [
                _FakePosition("SPY", "10", "400.0", "380.0"),
                _FakePosition("QQQ", "5", "300.0", "320.0"),
            ]

    broker = AlpacaBroker.__new__(AlpacaBroker)  # evita __init__ (que exige API keys reales)
    broker.client = _FakeClient()
    positions = broker.get_positions_with_cost_basis()
    assert positions["SPY"] == dict(qty=10.0, avg_entry_price=400.0, current_price=380.0)
    assert positions["QQQ"]["current_price"] == 320.0
    print("  AlpacaBroker.get_positions_with_cost_basis OK: convierte los strings de la API a float correctamente")


def make_synthetic_close(n_days=3 * 252, seed=55):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLK", "XLF", "XLE", "XLV", "XLY",
               "XLP", "XLI", "XLU", "XLB", "BIL", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
               "JPM", "JNJ", "V", "PG", "HD"]
    data = {}
    for t in tickers:
        rets = rng.normal(0.0002, 0.01, n_days)
        data[t] = 100 * np.exp(np.cumsum(rets))
    close = pd.DataFrame(data, index=dates)
    # Fuerza que SPY cierre bien por debajo de los $500 sembrados como cost basis --
    # es la última fila, así que flag_and_clean_outliers no la puede confundir con un
    # bad tick (no hay días futuros para chequear reversión, ver src/data_quality.py).
    close.loc[close.index[-1], "SPY"] = 350.0
    return close


def make_universe():
    return dict(
        broad_etfs=["SPY", "QQQ", "IWM", "TLT", "GLD"],
        sector_etfs=["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"],
        liquid_stocks=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "JNJ", "V", "PG", "HD"],
        cash_proxy="BIL",
        benchmark="SPY",
    )


def make_live_params(harvest_enabled: bool):
    return dict(
        momentum=dict(fast=50, slow=200, vol_target=0.10, vol_lookback=20),
        mean_reversion=dict(trend_sma=200, entry_rsi=10.0, exit_rsi=70.0, max_hold_days=10,
                             stop_loss_pct=0.06, max_concurrent_positions=5, weight_per_position=0.10,
                             vol_lookback=20, reference_vol=0.15),
        sector_rotation=dict(top_n=3, max_weight_per_asset=0.40),
        allocation=dict(momentum=0.35, mean_reversion=0.25, sector_rotation=0.40),
        position_caps=dict(broad_etf=0.20, sector_etf=0.40, individual_stock=0.08),
        regime_filter=dict(enabled=False, sma_window=200, full_exposure_above_pct=0.0,
                            floor_below_pct=-0.15, min_scale=0.40),
        costs=dict(use_liquidity_costs=True, base_capital=100_000, base_spread_bps=3.0,
                   impact_coeff=15.0, max_cost_bps=50.0),
        tax=dict(estimate_enabled=True, short_term_rate=0.35, harvest_losses_enabled=harvest_enabled,
                 harvest_loss_threshold_pct=-0.05, wash_sale_days=31),
        execution=dict(max_slippage_pct=0.005, min_weight_drift=0.02),
        # earnings_blackout_enabled=False: evita que run() consulte yfinance real por
        # cada acción individual (ver el mismo comentario en tests/live_smoke.py).
        event_blackout=dict(earnings_blackout_enabled=False),
    )


def test_run_live_once_harvests_loss_and_blocks_rebuy_with_virtual_broker():
    close = make_synthetic_close()
    synthetic_raw = {t: pd.DataFrame({"Close": close[t]}) for t in close.columns}
    universe = make_universe()
    live_params = make_live_params(harvest_enabled=True)
    as_of = close.index[-1]

    with tempfile.TemporaryDirectory() as tmp:
        reports_dir = pathlib.Path(tmp) / "reports"
        tracking_dir = reports_dir / "virtual"
        tracking_dir.mkdir(parents=True)

        # Siembra una posición existente en SPY con costo base ALTO ($500) -- el
        # precio de "hoy" (forzado a $350) implica una pérdida no realizada de -30%,
        # muy por encima del umbral de cosecha (-5%).
        state_file = tracking_dir / "virtual_broker_state.json"
        state_file.write_text(json.dumps({
            "starting_cash": 1000.0, "cash": 500.0,
            "positions": {"SPY": 5.0}, "cost_basis": {"SPY": 500.0},
        }))

        orig = {}

        def patch(obj, name, value):
            orig[(id(obj), name)] = (obj, name, getattr(obj, name))
            setattr(obj, name, value)

        patch(rl, "load_universe", lambda: universe)
        patch(rl, "download_prices", lambda tickers, years=2, force=True: {
            t: synthetic_raw[t] for t in tickers if t in synthetic_raw
        })
        patch(rl, "resolve_profile", lambda profile: (None, reports_dir))
        patch(rl, "load_live_params", lambda path: live_params)

        try:
            args = types.SimpleNamespace(profile="conservative", broker="virtual", starting_cash=1000.0,
                                          execute=True, refresh_data=True)
            logger = rl.get_logger("tax-loss-harvest-test")

            rl.run(args, logger)

            broker_after = VirtualBroker(state_file, starting_cash=1000.0)
            assert broker_after.state["positions"].get("SPY", 0.0) == 0.0, \
                "SPY debería haberse vendido por completo (cosecha de pérdidas)"
            assert "SPY" not in broker_after.state.get("cost_basis", {})

            state = load_state(reports_dir=tracking_dir)
            assert "SPY" in state.get("recently_harvested", {}), \
                "SPY debería quedar registrado en recently_harvested tras la cosecha"
            print(f"  primera corrida OK: SPY vendido por pérdida y registrado "
                  f"(recently_harvested={state['recently_harvested']})")

            # Segunda corrida: fuerza a compute_target_weights a pedir SPY de nuevo
            # (simula que la señal normal quisiera recomprarlo) -- la guardia de
            # wash sale debe bloquearlo de todas formas.
            patch(rl, "compute_target_weights", lambda close_, universe_, live_params_, vix_close=None: ({"SPY": 0.25}, as_of))
            rl.run(args, logger)

            broker_after2 = VirtualBroker(state_file, starting_cash=1000.0)
            assert broker_after2.state["positions"].get("SPY", 0.0) == 0.0, \
                "SPY NO debería recomprarse mientras dure la ventana de wash sale, aunque la señal lo pida"
            print("  segunda corrida OK: guardia de wash sale bloquea la recompra pese a que la señal pide 25%")
        finally:
            for obj, name, value in orig.values():
                setattr(obj, name, value)


def test_run_live_once_does_nothing_when_harvest_disabled():
    """Con `harvest_losses_enabled: false` (el default), una posición perdedora
    NO debería venderse por esta lógica -- solo la señal normal decide."""
    close = make_synthetic_close()
    synthetic_raw = {t: pd.DataFrame({"Close": close[t]}) for t in close.columns}
    universe = make_universe()
    live_params = make_live_params(harvest_enabled=False)
    as_of = close.index[-1]

    with tempfile.TemporaryDirectory() as tmp:
        reports_dir = pathlib.Path(tmp) / "reports"
        tracking_dir = reports_dir / "virtual"
        tracking_dir.mkdir(parents=True)

        state_file = tracking_dir / "virtual_broker_state.json"
        state_file.write_text(json.dumps({
            "starting_cash": 1000.0, "cash": 500.0,
            "positions": {"SPY": 5.0}, "cost_basis": {"SPY": 500.0},
        }))

        orig = {}

        def patch(obj, name, value):
            orig[(id(obj), name)] = (obj, name, getattr(obj, name))
            setattr(obj, name, value)

        patch(rl, "load_universe", lambda: universe)
        patch(rl, "download_prices", lambda tickers, years=2, force=True: {
            t: synthetic_raw[t] for t in tickers if t in synthetic_raw
        })
        patch(rl, "resolve_profile", lambda profile: (None, reports_dir))
        patch(rl, "load_live_params", lambda path: live_params)
        # Aísla esta prueba de la señal real de las estrategias -- lo único que importa
        # es que, con la cosecha desactivada, nada fuerce a SPY a 0.
        patch(rl, "compute_target_weights", lambda close_, universe_, live_params_, vix_close=None: ({"SPY": 0.25}, as_of))

        try:
            args = types.SimpleNamespace(profile="conservative", broker="virtual", starting_cash=1000.0,
                                          execute=True, refresh_data=True)
            logger = rl.get_logger("tax-loss-harvest-disabled-test")
            rl.run(args, logger)

            broker_after = VirtualBroker(state_file, starting_cash=1000.0)
            assert broker_after.state["positions"].get("SPY", 0.0) > 0.0, \
                "con harvest_losses_enabled=false, SPY no debería venderse por esta lógica"
            state = load_state(reports_dir=tracking_dir)
            assert "recently_harvested" not in state or not state["recently_harvested"], \
                "no debería registrarse ninguna cosecha si la funcionalidad está desactivada"
            print("  OK: con harvest_losses_enabled=false, la posición perdedora NO se toca")
        finally:
            for obj, name, value in orig.values():
                setattr(obj, name, value)


def main():
    print("[1/9] Probando detección de candidatos a cosecha (umbral de pérdida)...")
    test_find_harvest_candidates_detects_losses_past_threshold()
    print("\n[2/9] Probando orden (peor a mejor) e ignorar posiciones sin datos...")
    test_find_harvest_candidates_orders_worst_first_and_ignores_missing_data()
    print("\n[3/9] Probando guardia de wash sale (bloquea dentro de la ventana, libera después)...")
    test_block_recent_harvest_rebuys_zeroes_within_window_not_after()
    print("\n[4/9] Probando limpieza de cosechas vencidas (housekeeping)...")
    test_prune_expired_harvests_drops_old_entries()
    print("\n[5/9] Probando costo promedio ponderado en VirtualBroker...")
    test_virtual_broker_tracks_weighted_average_cost_basis()
    print("\n[6/9] Probando que el costo base se limpia al cerrar una posición...")
    test_virtual_broker_clears_cost_basis_when_position_closed()
    print("\n[7/9] Probando AlpacaBroker.get_positions_with_cost_basis (broker falso)...")
    test_alpaca_broker_get_positions_with_cost_basis()
    print("\n[8/9] Probando el flujo completo en run_live_once.py (venta + bloqueo de recompra)...")
    test_run_live_once_harvests_loss_and_blocks_rebuy_with_virtual_broker()
    print("\n[9/9] Probando que con la funcionalidad desactivada no se toca nada...")
    test_run_live_once_does_nothing_when_harvest_disabled()
    print("\nTAX LOSS HARVESTING TEST OK: cosecha de pérdidas y guardia de wash sale funcionan correctamente.")


if __name__ == "__main__":
    main()
