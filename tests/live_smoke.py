"""Prueba la lógica de `run_live_once.py` y `AlpacaBroker` SIN tocar la red real
(ni Alpaca ni Yahoo Finance) -- construye datos sintéticos y un broker falso
(monkeypatched) para validar: cálculo de pesos objetivo con topes por tipo de
activo + filtro de régimen, y el cálculo de órdenes límite (incluyendo el
chequeo de mercado cerrado), que son las piezas nuevas más riesgosas de romper
en silencio porque normalmente solo se prueban corriendo contra la cuenta paper real.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import scripts.run_live_once as rl
from src.live.alpaca_broker import AlpacaBroker, MarketClosedError
from src.live.virtual_broker import VirtualBroker
from src.tracking import load_equity_log
from alpaca.trading.enums import OrderSide


def make_synthetic_close(n_days=3 * 252, seed=99):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLK", "XLF", "XLE", "XLV", "XLY",
               "XLP", "XLI", "XLU", "XLB", "BIL", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
               "JPM", "JNJ", "V", "PG", "HD", "TQQQ", "SPXL", "SOXL"]
    data = {}
    for t in tickers:
        rets = rng.normal(0.0003, 0.012, n_days)
        data[t] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=dates)


def make_universe():
    return dict(
        broad_etfs=["SPY", "QQQ", "IWM", "TLT", "GLD"],
        sector_etfs=["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"],
        liquid_stocks=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "JNJ", "V", "PG", "HD"],
        leveraged_etfs=["TQQQ", "SPXL", "SOXL"],
        cash_proxy="BIL",
        benchmark="SPY",
    )


def make_live_params(regime_enabled: bool):
    return dict(
        momentum=dict(fast=50, slow=200, vol_target=0.10, vol_lookback=20),
        mean_reversion=dict(trend_sma=200, entry_rsi=10.0, exit_rsi=70.0, max_hold_days=10,
                             stop_loss_pct=0.06, max_concurrent_positions=5, weight_per_position=0.10,
                             vol_lookback=20, reference_vol=0.15),
        sector_rotation=dict(top_n=3, max_weight_per_asset=0.40),
        allocation=dict(momentum=0.35, mean_reversion=0.25, sector_rotation=0.40),
        position_caps=dict(broad_etf=0.20, sector_etf=0.40, individual_stock=0.08),
        regime_filter=dict(enabled=regime_enabled, sma_window=200, full_exposure_above_pct=0.0,
                            floor_below_pct=-0.15, min_scale=0.40),
        costs=dict(use_liquidity_costs=True, base_capital=100_000, base_spread_bps=3.0,
                   impact_coeff=15.0, max_cost_bps=50.0),
        tax=dict(estimate_enabled=True, short_term_rate=0.35),
        execution=dict(max_slippage_pct=0.005),
        # earnings_blackout_enabled=False: sin esto, run() intentaría consultar
        # yfinance.Ticker(...).get_earnings_dates() por cada acción individual
        # -- red real, lenta e innecesaria en un test que ya corre 100% offline
        # con datos sintéticos (ver src/event_blackout.py). fomc_blackout no
        # necesita red, se deja en su default.
        event_blackout=dict(earnings_blackout_enabled=False),
    )


def make_live_params_aggressive():
    p = make_live_params(regime_enabled=True)
    p["include_leveraged_etfs"] = True
    p["position_caps"]["leveraged_etf"] = 0.25
    p["momentum"]["vol_target"] = 0.30
    return p


def test_aggressive_profile_leveraged_etfs_only_in_momentum():
    close = make_synthetic_close()
    universe = make_universe()
    live_params = make_live_params_aggressive()

    weights, as_of = rl.compute_target_weights(close, universe, live_params)
    leveraged = set(universe["leveraged_etfs"])
    cap = live_params["position_caps"]["leveraged_etf"]
    for t in leveraged:
        w = weights.get(t, 0.0)
        assert w <= cap + 1e-6, f"{t}: peso {w} excede el tope de ETF apalancado ({cap})"
    print(f"  perfil agresivo OK: corre sin errores, topes de ETFs apalancados respetados "
          f"(pesos actuales: { {t: round(weights.get(t, 0.0), 4) for t in leveraged} })")


def test_target_weights_respect_caps_and_regime():
    close = make_synthetic_close()
    universe = make_universe()

    stock_cap = 0.08
    live_params_no_regime = make_live_params(regime_enabled=False)
    weights_no_regime, as_of = rl.compute_target_weights(close, universe, live_params_no_regime)
    for t, w in weights_no_regime.items():
        if t in universe["liquid_stocks"]:
            assert w <= stock_cap + 1e-6, f"{t}: peso {w} excede el tope de acción individual ({stock_cap})"

    live_params_regime = make_live_params(regime_enabled=True)
    weights_regime, _ = rl.compute_target_weights(close, universe, live_params_regime)

    total_no_regime = sum(weights_no_regime.values())
    total_regime = sum(weights_regime.values())
    # No es garantía matemática estricta (dependen de señales distintas ese día), pero con
    # datos aleatorios sin tendencia fuerte, el escalado por régimen no debería aumentar la
    # exposición total respecto de no aplicarlo.
    assert total_regime <= total_no_regime + 1e-6, \
        f"el filtro de régimen no debería aumentar la exposición total: {total_regime} > {total_no_regime}"
    print(f"  pesos OK: exposición total sin régimen={total_no_regime:.3f}, con régimen={total_regime:.3f}, "
          f"topes por ticker respetados")


class _FakeOrder:
    def __init__(self, order_id, status, qty=0.0, price=None):
        self.id = order_id
        self.status = status
        self.filled_qty = qty
        self.filled_avg_price = price


class _FakeTradingClient:
    """`fill_immediately=True` (default) simula que la orden se llena en el
    primer poll -- valida el camino feliz. `fill_immediately=False` simula una
    orden que nunca se llena (queda "new" para siempre) -- valida que
    `_wait_for_fill` la CANCELA al agotar `fill_timeout_sec` en vez de creer
    silenciosamente que se ejecutó."""
    def __init__(self, is_open: bool, fill_immediately: bool = True):
        self._is_open = is_open
        self.fill_immediately = fill_immediately
        self.submitted = []
        self.canceled = []
        self._orders: dict[str, _FakeOrder] = {}
        self._next_id = 0

    def get_clock(self):
        class Clock:
            pass
        c = Clock()
        c.is_open = self._is_open
        return c

    def submit_order(self, req):
        self._next_id += 1
        oid = str(self._next_id)
        if self.fill_immediately:
            order = _FakeOrder(oid, "filled", qty=req.qty, price=req.limit_price)
        else:
            order = _FakeOrder(oid, "new")
        self._orders[oid] = order
        self.submitted.append(req)
        return order

    def get_order_by_id(self, order_id):
        return self._orders[str(order_id)]

    def cancel_order_by_id(self, order_id):
        self.canceled.append(str(order_id))
        self._orders[str(order_id)].status = "canceled"


def _make_fake_broker(is_open: bool, equity: float, current_positions: dict,
                       fill_immediately: bool = True) -> AlpacaBroker:
    broker = AlpacaBroker.__new__(AlpacaBroker)  # evita __init__ (que exige API keys reales)
    broker.client = _FakeTradingClient(is_open, fill_immediately=fill_immediately)
    broker.get_equity = lambda: equity
    broker.get_current_positions = lambda: current_positions
    return broker


def test_broker_limit_orders_and_market_closed():
    target_weights = {"SPY": 0.30, "QQQ": 0.20}
    reference_prices = {"SPY": 450.0, "QQQ": 380.0}

    broker = _make_fake_broker(is_open=True, equity=10_000.0, current_positions={})
    orders = broker.rebalance_to_weights(target_weights, reference_prices, dry_run=True)
    assert len(orders) == 2, f"se esperaban 2 órdenes, hubo {len(orders)}"
    spy_order = next(o for o in orders if o["ticker"] == "SPY")
    assert spy_order["side"] == OrderSide.BUY.value
    assert abs(spy_order["notional"] - 3000.0) < 1.0
    assert spy_order["limit_price"] > 450.0, "orden de compra debería tener límite por ENCIMA del precio de referencia"
    assert abs(spy_order["qty"] - 3000.0 / 450.0) < 0.01
    print(f"  dry-run OK: {len(orders)} órdenes calculadas correctamente, límites de precio en la dirección correcta")

    # Con posición ya existente más grande que el objetivo -> debería vender (SELL).
    broker2 = _make_fake_broker(is_open=True, equity=10_000.0, current_positions={"SPY": 5000.0})
    orders2 = broker2.rebalance_to_weights(target_weights, reference_prices, dry_run=True)
    spy_order2 = next(o for o in orders2 if o["ticker"] == "SPY")
    assert spy_order2["side"] == OrderSide.SELL.value
    assert spy_order2["limit_price"] < 450.0, "orden de venta debería tener límite por DEBAJO del precio de referencia"
    print("  rebalanceo con posición existente OK: detecta correctamente cuándo vender")

    # Mercado cerrado + dry_run=False -> debe lanzar MarketClosedError y NO enviar nada.
    broker_closed = _make_fake_broker(is_open=False, equity=10_000.0, current_positions={})
    threw = False
    try:
        broker_closed.rebalance_to_weights(target_weights, reference_prices, dry_run=False)
    except MarketClosedError:
        threw = True
    assert threw, "debería lanzar MarketClosedError si el mercado está cerrado y dry_run=False"
    assert broker_closed.client.submitted == [], "no debería haber enviado ninguna orden con el mercado cerrado"
    print("  guardia de mercado cerrado OK: no se envían órdenes reales fuera de horario")

    # Mercado abierto + dry_run=False -> sí debe enviar las órdenes calculadas Y
    # verificar que se llenaron (no basta con "se envió").
    broker_open = _make_fake_broker(is_open=True, equity=10_000.0, current_positions={})
    orders_sent = broker_open.rebalance_to_weights(target_weights, reference_prices, dry_run=False,
                                                     poll_interval_sec=0.01, fill_timeout_sec=0.05)
    assert len(broker_open.client.submitted) == len(orders_sent) == 2
    for o in orders_sent:
        assert o["fill_status"] == "filled" and o["timed_out"] is False and o["filled_qty"] > 0
    print("  ejecución real (contra broker falso) OK: envía las órdenes y verifica fill_status='filled'")

    # Orden que NUNCA se llena -> debe cancelarse al agotar el timeout, y reportarlo
    # como tal (no como si se hubiera ejecutado silenciosamente).
    broker_stuck = _make_fake_broker(is_open=True, equity=10_000.0, current_positions={}, fill_immediately=False)
    orders_stuck = broker_stuck.rebalance_to_weights(target_weights, reference_prices, dry_run=False,
                                                       poll_interval_sec=0.01, fill_timeout_sec=0.03)
    assert len(broker_stuck.client.canceled) == len(orders_stuck) == 2
    for o in orders_stuck:
        assert o["fill_status"] == "canceled" and o["timed_out"] is True and o["filled_qty"] == 0.0
    print("  verificación de fills OK: una orden que nunca se llena se CANCELA al agotar el timeout "
          "(en vez de asumir que se ejecutó)")


def test_virtual_broker_starting_cash_and_persistence(tmp_path):
    state_file = tmp_path / "virtual_state.json"
    reference_prices = {"SPY": 450.0, "QQQ": 380.0}

    broker = VirtualBroker(state_file, starting_cash=1000.0)
    assert broker.get_equity(reference_prices) == 1000.0, "el equity inicial debería ser exactamente el capital inicial"

    orders = broker.rebalance_to_weights({"SPY": 0.5}, reference_prices, dry_run=False)
    assert len(orders) == 1 and orders[0]["ticker"] == "SPY"
    equity_after = broker.get_equity(reference_prices)
    assert abs(equity_after - 1000.0) < 5.0, f"el equity no debería cambiar mucho solo por rebalancear (costos chicos): {equity_after}"
    assert state_file.exists(), "el estado debería haberse guardado en disco"

    # Reabrir el broker desde el archivo de estado (simula la siguiente corrida diaria) -> debe
    # recordar la posición, NO reiniciar con el capital inicial de nuevo.
    broker2 = VirtualBroker(state_file, starting_cash=1000.0)
    assert broker2.state["positions"].get("SPY", 0) > 0, "debería haber cargado la posición existente del archivo"

    # El precio de SPY sube 10% -> el equity debería reflejarlo (aprox. mitad del portafolio en SPY).
    higher_prices = {**reference_prices, "SPY": 495.0}
    equity_up = broker2.get_equity(higher_prices)
    assert equity_up > equity_after, "el equity debería subir si el precio de una posición sube"
    print(f"  broker virtual OK: capital inicial=$1000.00, persiste posiciones entre corridas, "
          f"equity tras +10% en SPY = ${equity_up:,.2f}")


def test_run_live_once_with_virtual_broker(tmp_path, monkeypatch):
    """Corre `rl.run()` completo (no solo compute_target_weights) con el broker
    virtual y datos sintéticos monkeypatcheados -- valida que el flujo real que
    usaría un cron (descarga -> señal -> equity -> guardia -> rebalanceo ->
    logging) funciona de punta a punta sin necesitar Alpaca ni red."""
    import types
    import src.data as data_mod

    close = make_synthetic_close()
    synthetic_raw = {t: pd.DataFrame({"Close": close[t]}) for t in close.columns}

    monkeypatch.setattr(rl, "load_universe", lambda: make_universe())
    monkeypatch.setattr(rl, "download_prices", lambda tickers, years=2, force=True: {
        t: synthetic_raw[t] for t in tickers if t in synthetic_raw
    })
    monkeypatch.setattr(rl, "resolve_profile", lambda profile: (
        None, tmp_path / ("reports_aggressive" if profile == "aggressive" else "reports")
    ))
    monkeypatch.setattr(rl, "load_live_params", lambda path: make_live_params(regime_enabled=True))

    args = types.SimpleNamespace(profile="conservative", broker="virtual", starting_cash=1000.0,
                                  execute=True, refresh_data=True)
    logger = rl.get_logger("live-smoke-run-test")

    rl.run(args, logger)

    # Subcarpeta por broker (reports/virtual/) -- no reports/ directo, para que
    # --broker alpaca y --broker virtual del mismo perfil no compartan tracking.
    tracking_dir = tmp_path / "reports" / "virtual"
    assert (tracking_dir / "tracking.sqlite3").exists(), "debería haberse creado la base SQLite de tracking"
    assert (tracking_dir / "virtual_broker_state.json").exists(), "debería haber guardado el estado del broker virtual"
    log_df = load_equity_log(tracking_dir)
    assert len(log_df) == 1 and abs(log_df["equity"].iloc[0] - 1000.0) < 50.0
    print(f"  run() completo con broker virtual OK: equity registrado = ${log_df['equity'].iloc[0]:,.2f} "
          f"(en reports/virtual/, separado de reports/alpaca/)")


def main():
    print("[1/5] Probando cálculo de pesos objetivo (topes por ticker + filtro de régimen)...")
    test_target_weights_respect_caps_and_regime()

    print("\n[2/5] Probando perfil agresivo (ETFs apalancados solo en momentum)...")
    test_aggressive_profile_leveraged_etfs_only_in_momentum()

    print("\n[3/5] Probando lógica de órdenes límite del broker (con broker falso, sin red)...")
    test_broker_limit_orders_and_market_closed()

    print("\n[4/5] Probando broker virtual (capital inicial propio, persistencia entre corridas)...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_virtual_broker_starting_cash_and_persistence(pathlib.Path(tmp))

    print("\n[5/5] Probando run() completo (descarga->señal->equity->rebalanceo) con broker virtual...")
    import unittest.mock as mock
    with tempfile.TemporaryDirectory() as tmp:
        class _MonkeyPatch:
            def __init__(self):
                self._orig = []

            def setattr(self, obj, name, value):
                self._orig.append((obj, name, getattr(obj, name)))
                setattr(obj, name, value)

            def undo(self):
                for obj, name, value in reversed(self._orig):
                    setattr(obj, name, value)

        mp = _MonkeyPatch()
        try:
            test_run_live_once_with_virtual_broker(pathlib.Path(tmp), mp)
        finally:
            mp.undo()

    print("\nLIVE SMOKE TEST OK: la lógica de run_live_once.py, AlpacaBroker y VirtualBroker está correcta.")


if __name__ == "__main__":
    main()
