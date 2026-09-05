"""Valida la banda de no-operación (`min_weight_drift`) en `rebalance_to_weights`
-- un ticker cuyo peso actual ya está cerca del objetivo NO debería generar una
orden, sin importar que la diferencia en dólares supere `min_order_usd` (el
caso real que motivó esto: una cuenta grande, donde hasta un drift de peso
irrelevante mueve más de los $10-50 del piso en dólares). Se prueba con
`VirtualBroker` (lógica directa, sin red) y `AlpacaBroker` (con broker falso,
mismo patrón que `tests/live_smoke.py`) -- Binance y Bitso comparten
literalmente el mismo patrón de 3 líneas, y sus propios smoke tests ya
ejercitan el método completo end-to-end.
"""
import sys
import pathlib
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.live.virtual_broker import VirtualBroker
from src.live.alpaca_broker import AlpacaBroker


def test_virtual_broker_skips_small_drift_even_if_dollars_are_large():
    with tempfile.TemporaryDirectory() as tmp:
        state_file = pathlib.Path(tmp) / "state.json"
        broker = VirtualBroker(state_file, starting_cash=100_000.0)  # cuenta grande a propósito
        reference_prices = {"SPY": 450.0}

        # Primer rebalanceo: sí debe comprar (no hay posición previa).
        orders1 = broker.rebalance_to_weights({"SPY": 0.30}, reference_prices, dry_run=False,
                                               min_weight_drift=0.02)
        assert len(orders1) == 1 and orders1[0]["ticker"] == "SPY"

        # Peso real ahora ~30% (menos los costos, prácticamente 30%). Pedir 30.3%
        # -- 0.3 puntos porcentuales de drift, MUY por debajo del 2% de la banda,
        # pero en una cuenta de $100k eso son ~$300, muy por encima de min_order_usd
        # (default $5-10). Sin la banda, esto generaría una orden; con ella, no.
        orders2 = broker.rebalance_to_weights({"SPY": 0.303}, reference_prices, dry_run=False,
                                               min_weight_drift=0.02, min_order_usd=5.0)
        assert orders2 == [], f"un drift de 0.3pp no debería generar orden con min_weight_drift=0.02: {orders2}"

        # Pero un drift GRANDE (target 40% vs ~30% actual, 10pp) sí debe generar orden,
        # aunque min_weight_drift siga en 0.02.
        orders3 = broker.rebalance_to_weights({"SPY": 0.40}, reference_prices, dry_run=False,
                                               min_weight_drift=0.02, min_order_usd=5.0)
        assert len(orders3) == 1 and orders3[0]["side"] == "buy", \
            f"un drift de 10pp SÍ debería generar orden: {orders3}"

    print("  VirtualBroker OK: ignora drift de 0.3pp (aunque sean ~$300), ejecuta drift de 10pp")


def test_virtual_broker_default_band_applies_without_explicit_arg():
    """El default (0.02) debe aplicarse aunque el caller no lo pase explícito --
    así es como lo llaman run_live_once.py/run_crypto_live_once.py hoy si el
    YAML no trae la clave (código viejo, o alguien la borró sin querer)."""
    with tempfile.TemporaryDirectory() as tmp:
        state_file = pathlib.Path(tmp) / "state.json"
        broker = VirtualBroker(state_file, starting_cash=50_000.0)
        reference_prices = {"QQQ": 400.0}
        broker.rebalance_to_weights({"QQQ": 0.20}, reference_prices, dry_run=False)
        orders = broker.rebalance_to_weights({"QQQ": 0.205}, reference_prices, dry_run=False)
        assert orders == [], "con el default (0.02) un drift de 0.5pp tampoco debería operar"
    print("  banda por default (sin pasar min_weight_drift explícito) OK")


class _FakeOrder:
    def __init__(self, order_id, status, qty=0.0, price=None):
        self.id = order_id
        self.status = status
        self.filled_qty = qty
        self.filled_avg_price = price


class _FakeTradingClient:
    def __init__(self):
        self.submitted = []
        self._orders = {}
        self._next_id = 0

    def submit_order(self, req):
        self._next_id += 1
        oid = str(self._next_id)
        order = _FakeOrder(oid, "filled", qty=req.qty, price=req.limit_price)
        self._orders[oid] = order
        self.submitted.append(req)
        return order

    def get_order_by_id(self, order_id):
        return self._orders[str(order_id)]

    def cancel_order_by_id(self, order_id):
        pass


def test_alpaca_broker_skips_small_drift():
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker.client = _FakeTradingClient()
    equity = 200_000.0
    # 29.8% actual vs 30% objetivo -> 0.2pp de drift, ~$400 en dólares.
    broker.get_equity = lambda: equity
    broker.get_current_positions = lambda: {"SPY": 0.298 * equity}

    orders = broker.rebalance_to_weights({"SPY": 0.30}, {"SPY": 450.0}, dry_run=True,
                                          min_weight_drift=0.02, min_order_usd=10.0)
    assert orders == [], f"0.2pp de drift no debería operar con min_weight_drift=0.02: {orders}"

    # Mismo escenario pero con un objetivo bien distinto (30% -> 45%, 15.2pp) SÍ debe operar.
    orders2 = broker.rebalance_to_weights({"SPY": 0.45}, {"SPY": 450.0}, dry_run=True,
                                           min_weight_drift=0.02, min_order_usd=10.0)
    assert len(orders2) == 1 and orders2[0]["side"] == "buy"
    print("  AlpacaBroker OK: ignora drift de 0.2pp, ejecuta drift de 15.2pp")


def main():
    print("[1/3] Probando que VirtualBroker ignora drift chico aunque sea caro en dólares...")
    test_virtual_broker_skips_small_drift_even_if_dollars_are_large()
    print("\n[2/3] Probando que la banda por default (0.02) aplica sin pasar el argumento...")
    test_virtual_broker_default_band_applies_without_explicit_arg()
    print("\n[3/3] Probando lo mismo con AlpacaBroker (broker falso, sin red)...")
    test_alpaca_broker_skips_small_drift()
    print("\nNO TRADE BAND TEST OK: la banda de no-operación funciona en ambos brokers probados.")


if __name__ == "__main__":
    main()
