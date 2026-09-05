"""Valida la división de órdenes grandes tipo TWAP en `AlpacaBroker` (ver
`src/live/alpaca_broker.py`): una orden por encima de `twap_threshold_usd` se
divide en varias órdenes límite más chicas enviadas en secuencia, y el
resultado se agrega en un solo registro por ticker -- sin red real, con un
broker (`TradingClient`) falso.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.live.alpaca_broker import AlpacaBroker


class _FakeOrder:
    def __init__(self, order_id, status, qty=0.0, price=None):
        self.id = order_id
        self.status = status
        self.filled_qty = qty
        self.filled_avg_price = price


class _FakeTradingClient:
    """`never_fill_after` (opcional): índice de envío (0-based, orden de
    llegada de `submit_order`) a partir del cual las órdenes NO se llenan
    (quedan "new" para siempre) -- para simular que una porción TWAP se cae a
    mitad de camino. Sin eso, todo se llena de inmediato (camino feliz)."""
    def __init__(self, is_open: bool = True, never_fill_after: int | None = None):
        self._is_open = is_open
        self.never_fill_after = never_fill_after
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
        idx = len(self.submitted)
        self.submitted.append(req)
        self._next_id += 1
        oid = str(self._next_id)
        will_fill = self.never_fill_after is None or idx < self.never_fill_after
        if will_fill:
            order = _FakeOrder(oid, "filled", qty=req.qty, price=req.limit_price)
        else:
            order = _FakeOrder(oid, "new")
        self._orders[oid] = order
        return order

    def get_order_by_id(self, order_id):
        return self._orders[str(order_id)]

    def cancel_order_by_id(self, order_id):
        self.canceled.append(str(order_id))
        self._orders[str(order_id)].status = "canceled"


def _make_broker(client: _FakeTradingClient, equity: float, current_positions: dict) -> AlpacaBroker:
    broker = AlpacaBroker.__new__(AlpacaBroker)  # evita __init__ (que exige API keys reales)
    broker.client = client
    broker.get_equity = lambda: equity
    broker.get_current_positions = lambda: current_positions
    return broker


def test_order_below_threshold_is_not_split():
    client = _FakeTradingClient()
    broker = _make_broker(client, equity=10_000.0, current_positions={})
    orders = broker.rebalance_to_weights(
        {"SPY": 0.30}, {"SPY": 450.0}, dry_run=False, poll_interval_sec=0.001, fill_timeout_sec=0.01,
        twap_threshold_usd=5000.0,
    )
    assert len(orders) == 1
    assert orders[0]["n_slices"] == 1, "una orden de $3000 (< $5000 de umbral) no debería dividirse"
    assert len(client.submitted) == 1, "debería haber una sola orden enviada al broker"
    assert orders[0]["fill_status"] == "filled"
    assert "slice_fills" not in orders[0], "una orden sin dividir no debería tener el detalle de porciones"


def test_order_above_threshold_is_split_into_slices():
    client = _FakeTradingClient()
    broker = _make_broker(client, equity=100_000.0, current_positions={})
    # Target 30% de $100k = $30,000 -- con umbral $5,000 y máximo 5 porciones,
    # debería dividirse en exactamente 5 (30000/5000 = 6, pero se topa en 5).
    orders = broker.rebalance_to_weights(
        {"SPY": 0.30}, {"SPY": 450.0}, dry_run=False, poll_interval_sec=0.001, fill_timeout_sec=0.01,
        twap_threshold_usd=5000.0, twap_max_slices=5, twap_slice_delay_sec=0.0,
    )
    assert len(orders) == 1
    o = orders[0]
    assert o["n_slices"] == 5, f"se esperaban 5 porciones (topado por twap_max_slices): {o['n_slices']}"
    assert len(client.submitted) == 5, f"deberían haberse enviado 5 órdenes reales al broker: {len(client.submitted)}"
    print(f"  OK: orden de ${o['notional']:.2f} dividida en {o['n_slices']} porciones "
          f"({len(client.submitted)} envíos reales al broker)")


def test_split_slices_sum_to_the_original_quantity():
    client = _FakeTradingClient()
    broker = _make_broker(client, equity=100_000.0, current_positions={})
    orders = broker.rebalance_to_weights(
        {"SPY": 0.30}, {"SPY": 450.0}, dry_run=False, poll_interval_sec=0.001, fill_timeout_sec=0.01,
        twap_threshold_usd=5000.0, twap_max_slices=5, twap_slice_delay_sec=0.0,
    )
    o = orders[0]
    total_submitted_qty = sum(float(req.qty) for req in client.submitted)
    assert abs(total_submitted_qty - o["qty"]) < 1e-6, \
        f"la suma de las porciones enviadas debería ser exactamente igual a qty original: " \
        f"{total_submitted_qty} vs {o['qty']}"
    assert abs(o["filled_qty"] - o["qty"]) < 1e-6, "con todas las porciones llenas, filled_qty debería == qty"
    assert o["fill_status"] == "filled", "con todas las porciones llenas, el estado agregado debería ser 'filled'"


def test_aggregate_reports_partial_fill_when_one_slice_never_fills():
    # Las primeras 3 porciones se llenan, la 4ta (índice 3) y la 5ta NO.
    client = _FakeTradingClient(never_fill_after=3)
    broker = _make_broker(client, equity=100_000.0, current_positions={})
    orders = broker.rebalance_to_weights(
        {"SPY": 0.30}, {"SPY": 450.0}, dry_run=False, poll_interval_sec=0.001, fill_timeout_sec=0.01,
        twap_threshold_usd=5000.0, twap_max_slices=5, twap_slice_delay_sec=0.0,
    )
    o = orders[0]
    assert o["fill_status"] != "filled", \
        f"con 2 de 5 porciones sin llenar, el estado agregado NO debería decir 'filled': {o['fill_status']}"
    assert 0 < o["filled_qty"] < o["qty"], \
        f"filled_qty debería reflejar un llenado PARCIAL (ni 0 ni el total): {o['filled_qty']} de {o['qty']}"
    assert len(o["slice_fills"]) == 5, "debería haber el detalle de las 5 porciones"
    assert sum(1 for s in o["slice_fills"] if s["fill_status"] == "filled") == 3
    print(f"  OK: llenado parcial detectado correctamente ({o['filled_qty']:.4f} de {o['qty']:.4f}), "
          f"fill_status='{o['fill_status']}' (no miente diciendo 'filled')")


def test_twap_disabled_with_none_threshold_sends_single_order():
    client = _FakeTradingClient()
    broker = _make_broker(client, equity=100_000.0, current_positions={})
    orders = broker.rebalance_to_weights(
        {"SPY": 0.30}, {"SPY": 450.0}, dry_run=False, poll_interval_sec=0.001, fill_timeout_sec=0.01,
        twap_threshold_usd=None,
    )
    assert orders[0]["n_slices"] == 1
    assert len(client.submitted) == 1, "twap_threshold_usd=None debería desactivar la división por completo"


def test_dry_run_computes_n_slices_without_submitting_anything():
    """El cálculo de `n_slices` (para mostrarlo en dry-run) no debería depender
    de si se ejecuta de verdad -- útil para que el usuario vea de antemano que
    una orden se dividiría, sin tener que pasar --execute."""
    client = _FakeTradingClient()
    broker = _make_broker(client, equity=100_000.0, current_positions={})
    orders = broker.rebalance_to_weights(
        {"SPY": 0.30}, {"SPY": 450.0}, dry_run=True,
        twap_threshold_usd=5000.0, twap_max_slices=5,
    )
    assert orders[0]["n_slices"] == 5
    assert client.submitted == [], "dry_run=True no debería enviar ninguna orden real"


def test_slice_delay_is_respected_between_submissions():
    """`twap_slice_delay_sec` debería pausar ENTRE envíos (n_slices - 1 pausas,
    nunca después del último) -- se verifica interceptando time.sleep en vez de
    esperar de verdad, para que el test siga siendo rápido."""
    import src.live.alpaca_broker as alpaca_broker_mod

    sleep_calls = []
    orig_sleep = alpaca_broker_mod.time.sleep
    alpaca_broker_mod.time.sleep = lambda secs: sleep_calls.append(secs)
    try:
        client = _FakeTradingClient()
        broker = _make_broker(client, equity=100_000.0, current_positions={})
        broker.rebalance_to_weights(
            {"SPY": 0.30}, {"SPY": 450.0}, dry_run=False, poll_interval_sec=0.001, fill_timeout_sec=0.01,
            twap_threshold_usd=5000.0, twap_max_slices=5, twap_slice_delay_sec=1.5,
        )
    finally:
        alpaca_broker_mod.time.sleep = orig_sleep

    assert sleep_calls.count(1.5) == 4, \
        f"con 5 porciones, se esperaban exactamente 4 pausas de 1.5s (nunca después de la última): {sleep_calls}"
    print(f"  OK: {sleep_calls.count(1.5)} pausas de 1.5s entre las 5 porciones (ninguna después de la última)")


def main():
    print("[1/7] Probando que una orden por debajo del umbral NO se divide...")
    test_order_below_threshold_is_not_split()
    print("\n[2/7] Probando que una orden grande se divide en varias porciones...")
    test_order_above_threshold_is_split_into_slices()
    print("\n[3/7] Probando que las porciones suman exactamente la cantidad original...")
    test_split_slices_sum_to_the_original_quantity()
    print("\n[4/7] Probando que un llenado parcial se reporta correctamente (no 'filled')...")
    test_aggregate_reports_partial_fill_when_one_slice_never_fills()
    print("\n[5/7] Probando que twap_threshold_usd=None desactiva la división...")
    test_twap_disabled_with_none_threshold_sends_single_order()
    print("\n[6/7] Probando que dry-run calcula n_slices sin enviar nada...")
    test_dry_run_computes_n_slices_without_submitting_anything()
    print("\n[7/7] Probando que twap_slice_delay_sec pausa entre envíos (no después del último)...")
    test_slice_delay_is_respected_between_submissions()
    print("\nTWAP ORDER SPLIT TEST OK: la división de órdenes grandes funciona correctamente.")


if __name__ == "__main__":
    main()
