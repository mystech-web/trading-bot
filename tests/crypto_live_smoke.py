"""Prueba `BinanceBroker` (con un cliente falso, sin red) y el flujo completo
de `run_crypto_live_once.py` con el broker virtual y datos sintéticos --
mismo patrón que `tests/live_smoke.py` para el módulo de acciones.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.live.binance_broker import BinanceBroker
from src.tracking import load_equity_log


def _make_fake_client(balances: dict[str, float], filters_by_symbol: dict[str, dict], fill_immediately: bool = True):
    """`fill_immediately=False` simula una orden que se queda en NEW para
    siempre -- valida que `_wait_for_fill` la cancela al agotar el timeout en
    vez de asumir silenciosamente que se ejecutó."""
    class FakeClient:
        def __init__(self):
            self.submitted = []
            self.canceled = []
            self._orders = {}
            self._next_id = 0

        def get_account(self):
            return {"balances": [{"asset": a, "free": str(q), "locked": "0"} for a, q in balances.items()]}

        def get_symbol_info(self, symbol):
            f = filters_by_symbol[symbol]
            return {"filters": [
                {"filterType": "LOT_SIZE", "stepSize": str(f["step_size"])},
                {"filterType": "PRICE_FILTER", "tickSize": str(f["tick_size"])},
                {"filterType": "MIN_NOTIONAL", "minNotional": str(f["min_notional"])},
            ]}

        def create_order(self, **kwargs):
            self.submitted.append(kwargs)
            self._next_id += 1
            oid = self._next_id
            if fill_immediately:
                qty = str(kwargs["quantity"])
                self._orders[oid] = {"status": "FILLED", "executedQty": qty,
                                      "cummulativeQuoteQty": str(float(qty) * float(kwargs["price"]))}
            else:
                self._orders[oid] = {"status": "NEW", "executedQty": "0", "cummulativeQuoteQty": "0"}
            return {"status": self._orders[oid]["status"], "orderId": oid}

        def get_order(self, symbol, orderId):
            return {"symbol": symbol, "orderId": orderId, **self._orders[orderId]}

        def cancel_order(self, symbol, orderId):
            self.canceled.append(orderId)
            self._orders[orderId]["status"] = "CANCELED"
            return self._orders[orderId]

    return FakeClient()


def _make_broker(balances, filters_by_symbol, fill_immediately: bool = True) -> BinanceBroker:
    broker = BinanceBroker.__new__(BinanceBroker)  # evita __init__ (que exige API keys reales)
    broker.client = _make_fake_client(balances, filters_by_symbol, fill_immediately=fill_immediately)
    broker.quote_currency = "USDT"
    broker._symbol_filters_cache = {}
    return broker


DEFAULT_FILTERS = {
    "BTCUSDT": dict(step_size=0.00001, tick_size=0.01, min_notional=10.0),
    "ETHUSDT": dict(step_size=0.0001, tick_size=0.01, min_notional=10.0),
}


def test_equity_and_positions_marked_to_market():
    broker = _make_broker({"USDT": 500.0, "BTC": 0.01}, DEFAULT_FILTERS)
    reference_prices = {"BTCUSDT": 50_000.0}
    equity = broker.get_equity(reference_prices)
    assert abs(equity - (500.0 + 0.01 * 50_000.0)) < 1e-6, f"equity mal calculado: {equity}"

    positions = broker.get_current_positions(reference_prices)
    assert abs(positions["BTCUSDT"] - 500.0) < 1e-6
    print(f"  equity/posiciones OK: equity=${equity:,.2f} (500 USDT + 0.01 BTC a $50,000)")


def test_rebalance_respects_min_notional_and_rounds_correctly():
    broker = _make_broker({"USDT": 1000.0}, DEFAULT_FILTERS)
    reference_prices = {"BTCUSDT": 50_000.0}

    # Orden grande -> debería calcular una cantidad razonable, redondeada al step_size.
    orders = broker.rebalance_to_weights({"BTCUSDT": 0.5}, reference_prices, dry_run=True)
    assert len(orders) == 1
    o = orders[0]
    assert o["side"] == "buy"
    assert o.get("error") is None, f"no debería haber error para una orden de $500: {o}"
    # 0.5 * 1000 = $500 a ~$50k -> ~0.01 BTC, redondeado a 5 decimales (step 0.00001)
    assert abs(o["qty"] - 0.01) < 0.001
    print(f"  redondeo OK: qty={o['qty']}, notional=${o['notional']:.2f}")

    # Orden muy chica -> debería marcarse con error de min_notional, sin lanzar excepción.
    # min_weight_drift=0.0 para aislar esta guardia de la banda de no-operación (ver
    # tests/no_trade_band_test.py): un peso objetivo de 0.1% ya cae por debajo del
    # default de la banda (2pp) y se filtraría ANTES de llegar al chequeo de min_notional.
    orders_small = broker.rebalance_to_weights({"BTCUSDT": 0.001}, reference_prices, min_order_usd=0.01,
                                                min_weight_drift=0.0, dry_run=True)
    small = next((o for o in orders_small if o["ticker"] == "BTCUSDT"), None)
    assert small is not None and small.get("error"), "una orden de $1 debería marcar error de min_notional"
    print(f"  guardia de min_notional OK: {small['error']}")


def test_dry_run_does_not_submit_orders():
    broker = _make_broker({"USDT": 1000.0}, DEFAULT_FILTERS)
    broker.rebalance_to_weights({"BTCUSDT": 0.5}, {"BTCUSDT": 50_000.0}, dry_run=True)
    assert broker.client.submitted == [], "dry_run=True no debería enviar ninguna orden"

    broker2 = _make_broker({"USDT": 1000.0}, DEFAULT_FILTERS)
    orders = broker2.rebalance_to_weights({"BTCUSDT": 0.5}, {"BTCUSDT": 50_000.0}, dry_run=False,
                                           poll_interval_sec=0.01, fill_timeout_sec=0.05)
    assert len(broker2.client.submitted) == len(orders) == 1
    assert broker2.client.submitted[0]["symbol"] == "BTCUSDT" and broker2.client.submitted[0]["side"] == "BUY"
    assert orders[0]["fill_status"] == "FILLED" and orders[0]["timed_out"] is False
    assert orders[0]["filled_qty"] > 0
    print("  dry_run OK: no envía nada; dry_run=False sí envía la orden y verifica fill_status='FILLED'")


def test_order_that_never_fills_gets_canceled():
    broker = _make_broker({"USDT": 1000.0}, DEFAULT_FILTERS, fill_immediately=False)
    orders = broker.rebalance_to_weights({"BTCUSDT": 0.5}, {"BTCUSDT": 50_000.0}, dry_run=False,
                                          poll_interval_sec=0.01, fill_timeout_sec=0.03)
    assert len(broker.client.canceled) == len(orders) == 1
    assert orders[0]["fill_status"] == "CANCELED" and orders[0]["timed_out"] is True
    assert orders[0]["filled_qty"] == 0.0
    print("  verificación de fills OK: una orden que nunca se llena se CANCELA al agotar el timeout")


def test_run_crypto_live_once_with_virtual_broker(tmp_path, monkeypatch):
    import types
    import scripts.run_crypto_live_once as rcl

    rng = np.random.default_rng(11)
    dates = pd.date_range("2021-01-01", periods=3 * 365, freq="D")
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT"]
    synthetic = {}
    for s in symbols:
        rets = rng.normal(0.0006, 0.04, len(dates))
        synthetic[s] = pd.DataFrame({"Close": 10 * np.exp(np.cumsum(rets)),
                                      "Volume": rng.uniform(1e5, 1e6, len(dates)),
                                      "QuoteVolume": rng.uniform(1e6, 5e7, len(dates))}, index=dates)

    universe = dict(majors=["BTCUSDT", "ETHUSDT"],
                     altcoins=["BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT"],
                     quote_currency="USDT", benchmark="BTCUSDT")
    live_params = dict(
        periods_per_year=365,
        momentum=dict(fast=20, slow=100, vol_target=0.20, vol_lookback=20, periods_per_year=365),
        mean_reversion=dict(trend_sma=100, entry_rsi=15.0, exit_rsi=70.0, max_hold_days=10, stop_loss_pct=0.10,
                             max_concurrent_positions=4, weight_per_position=0.08, vol_lookback=20,
                             reference_vol=0.50, periods_per_year=365),
        sector_rotation=dict(top_n=3, max_weight_per_asset=0.35, momentum_long=365, momentum_short=30,
                              rebalance_rule="ME"),
        allocation=dict(momentum=0.45, mean_reversion=0.20, sector_rotation=0.35),
        position_caps=dict(major_coin=0.30, altcoin=0.15),
        regime_filter=dict(enabled=True, sma_window=100, full_exposure_above_pct=0.0, floor_below_pct=-0.25,
                            min_scale=0.30),
        costs=dict(use_liquidity_costs=True, base_capital=1000, base_spread_bps=10.0, impact_coeff=25.0,
                   max_cost_bps=80.0),
        tax=dict(estimate_enabled=False),
        execution=dict(max_slippage_pct=0.01),
    )

    monkeypatch.setattr(rcl, "load_crypto_universe", lambda: universe)
    monkeypatch.setattr(rcl, "load_crypto_live_params", lambda: live_params)
    monkeypatch.setattr(rcl, "download_prices", lambda symbols, years=2, force=True: {
        s: synthetic[s] for s in symbols if s in synthetic
    })
    monkeypatch.setattr(rcl, "REPORTS_DIR", tmp_path)

    args = types.SimpleNamespace(broker="virtual", starting_cash=1000.0, execute=True, refresh_data=True)
    logger = rcl.get_logger("crypto-live-smoke-test")
    rcl.run(args, logger)

    # Subcarpeta por broker (tmp_path/virtual/) -- virtual/binance/bitso del mismo
    # perfil cripto no comparten tracking entre sí.
    tracking_dir = tmp_path / "virtual"
    assert (tracking_dir / "tracking.sqlite3").exists()
    assert (tracking_dir / "virtual_broker_state.json").exists()
    log_df = load_equity_log(tracking_dir)
    assert len(log_df) == 1 and abs(log_df["equity"].iloc[0] - 1000.0) < 50.0
    print(f"  run() completo (cripto, broker virtual) OK: equity registrado = ${log_df['equity'].iloc[0]:,.2f} "
          f"(en <reports_crypto>/virtual/)")


def main():
    print("[1/5] Probando equity/posiciones marcadas a mercado...")
    test_equity_and_positions_marked_to_market()

    print("\n[2/5] Probando redondeo a filtros del exchange (step_size, min_notional)...")
    test_rebalance_respects_min_notional_and_rounds_correctly()

    print("\n[3/5] Probando dry_run vs ejecución real (con cliente falso)...")
    test_dry_run_does_not_submit_orders()

    print("\n[4/5] Probando que una orden que nunca se llena se cancela al agotar el timeout...")
    test_order_that_never_fills_gets_canceled()

    print("\n[5/5] Probando run() completo de run_crypto_live_once.py con broker virtual...")

    class _MonkeyPatch:
        def __init__(self):
            self._orig = []

        def setattr(self, obj, name, value):
            self._orig.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def undo(self):
            for obj, name, value in reversed(self._orig):
                setattr(obj, name, value)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        mp = _MonkeyPatch()
        try:
            test_run_crypto_live_once_with_virtual_broker(pathlib.Path(tmp), mp)
        finally:
            mp.undo()

    print("\nCRYPTO LIVE SMOKE TEST OK: BinanceBroker y run_crypto_live_once.py funcionan correctamente.")


if __name__ == "__main__":
    main()
