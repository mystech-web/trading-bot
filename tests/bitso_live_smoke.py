"""Prueba `BitsoBroker` y el ticker público de Bitso SIN tocar la red real --
monkeypatchea `requests` y `_request` con respuestas falsas. Bitso no tiene
testnet, así que estas pruebas son la única validación posible fuera de una
cuenta real: verifican la lógica de firmado HMAC, el mapeo de símbolos, la
guardia de "confirmación de dinero real", y el cálculo de órdenes.
"""
import sys
import pathlib
import hashlib
import hmac as hmac_lib
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.live.bitso_broker import (
    BitsoBroker, RealMoneyNotConfirmedError, book_from_symbol, get_bitso_ticker_prices,
)
import src.live.bitso_broker as bitso_mod


def test_book_from_symbol_mapping():
    assert book_from_symbol("BTCUSDT") == "btc_mxn"
    assert book_from_symbol("ETHUSDT") == "eth_mxn"
    assert book_from_symbol("SOLUSDT", quote_currency="mxn") == "sol_mxn"
    print("  mapeo de símbolos OK: BTCUSDT -> btc_mxn, ETHUSDT -> eth_mxn, SOLUSDT -> sol_mxn")


def test_real_money_gate():
    threw = False
    try:
        BitsoBroker(api_key="x", secret_key="y", confirm_real_money=False)
    except RealMoneyNotConfirmedError:
        threw = True
    assert threw, "sin confirm_real_money=True debería lanzar RealMoneyNotConfirmedError"

    broker = BitsoBroker(api_key="x", secret_key="y", confirm_real_money=True)
    assert broker.quote_currency == "mxn"
    print("  guardia de dinero real OK: bloquea sin confirmación explícita, permite con ella")


def test_signature_matches_manual_hmac(monkeypatch):
    fixed_nonce_ms = 1_700_000_000_000
    monkeypatch.setattr(bitso_mod.time, "time", lambda: fixed_nonce_ms / 1000.0)

    broker = BitsoBroker(api_key="mykey", secret_key="mysecret", confirm_real_money=True)
    body = {"book": "btc_mxn", "side": "buy", "type": "limit", "major": "0.001", "price": "1000000"}
    headers = broker._signed_headers("POST", "/v3/orders", body)

    expected_message = str(fixed_nonce_ms) + "POST" + "/api/v3/orders" + json.dumps(body)
    expected_sig = hmac_lib.new(b"mysecret", expected_message.encode(), hashlib.sha256).hexdigest()
    expected_header = f"Bitso mykey:{fixed_nonce_ms}:{expected_sig}"

    assert headers["Authorization"] == expected_header, \
        f"firma no coincide:\n  got={headers['Authorization']}\n  want={expected_header}"
    print("  firma HMAC-SHA256 OK: coincide byte a byte con el cálculo manual esperado")


def test_ticker_prices_skip_unavailable_books(monkeypatch):
    class FakeResp:
        def __init__(self, book):
            self._book = book

        def raise_for_status(self):
            if self._book == "zzz_mxn":
                raise RuntimeError("404 -- libro no existe")

        def json(self):
            return {"payload": {"last": "123.45", "book": self._book}}

    def fake_get(url, params=None, timeout=10):
        book = params["book"]
        if book == "zzz_mxn":
            raise RuntimeError("libro no existe")
        return FakeResp(book)

    monkeypatch.setattr(bitso_mod.requests, "get", fake_get)

    prices = get_bitso_ticker_prices(["BTCUSDT", "ETHUSDT", "ZZZUSDT"], quote_currency="mxn")
    assert prices == {"BTCUSDT": 123.45, "ETHUSDT": 123.45}, prices
    print(f"  ticker público OK: {prices} -- ZZZUSDT (sin libro en Bitso) se omitió sin lanzar excepción")


def test_rebalance_computes_correct_orders(monkeypatch):
    broker = BitsoBroker(api_key="x", secret_key="y", confirm_real_money=True)

    balance_response = {"balances": [{"currency": "mxn", "available": "10000", "locked": "0"}]}
    submitted = []

    def fake_request(method, path, body=None):
        if path == "/v3/balance":
            return balance_response
        if path == "/v3/orders" and method == "POST":
            submitted.append(body)
            return {"oid": "fake-order-id"}
        if path == "/v3/orders/fake-order-id" and method == "GET":
            # se llena de inmediato en el primer poll -- valida el camino feliz.
            return [{"oid": "fake-order-id", "status": "completed",
                      "original_amount": body_major(submitted), "unfilled_amount": "0.00000000",
                      "price": "1000000.00"}]
        raise AssertionError(f"llamada inesperada: {method} {path}")

    def body_major(submitted_bodies):
        return submitted_bodies[-1]["major"]

    monkeypatch.setattr(broker, "_request", fake_request)

    reference_prices = {"BTCUSDT": 1_000_000.0}  # 1 BTC = $1,000,000 MXN (ficticio, solo para la prueba)
    orders = broker.rebalance_to_weights({"BTCUSDT": 0.3}, reference_prices, dry_run=True)
    assert len(orders) == 1 and orders[0]["side"] == "buy"
    assert abs(orders[0]["notional"] - 3000.0) < 1.0
    assert submitted == [], "dry_run=True no debería llamar a POST /v3/orders"

    orders2 = broker.rebalance_to_weights({"BTCUSDT": 0.3}, reference_prices, dry_run=False,
                                           poll_interval_sec=0.01, fill_timeout_sec=0.05)
    assert len(submitted) == 1
    assert submitted[0]["book"] == "btc_mxn" and submitted[0]["side"] == "buy" and submitted[0]["type"] == "limit"
    assert orders2[0]["fill_status"] == "completed" and orders2[0]["timed_out"] is False
    assert orders2[0]["filled_qty"] > 0
    print(f"  cálculo de órdenes OK: {orders2[0]['side']} ~${orders2[0]['notional']:.2f} MXN de BTCUSDT, "
          f"libro correcto (btc_mxn), dry_run respetado, fill_status='completed'")


def test_order_that_never_fills_gets_canceled(monkeypatch):
    """Bitso no tiene testnet -- esto mueve MXN real, así que verificar que una
    orden atorada se CANCELA (en vez de asumir que se llenó) es especialmente
    importante acá."""
    broker = BitsoBroker(api_key="x", secret_key="y", confirm_real_money=True)
    balance_response = {"balances": [{"currency": "mxn", "available": "10000", "locked": "0"}]}
    canceled = []

    def fake_request(method, path, body=None):
        if path == "/v3/balance":
            return balance_response
        if path == "/v3/orders" and method == "POST":
            return {"oid": "stuck-order-id"}
        if path == "/v3/orders/stuck-order-id" and method == "DELETE":
            canceled.append("stuck-order-id")
            return {}
        if path == "/v3/orders/stuck-order-id" and method == "GET":
            status = "cancelled" if canceled else "open"
            return [{"oid": "stuck-order-id", "status": status,
                      "original_amount": "0.003", "unfilled_amount": "0.003", "price": "1000000.00"}]
        raise AssertionError(f"llamada inesperada: {method} {path}")

    monkeypatch.setattr(broker, "_request", fake_request)

    reference_prices = {"BTCUSDT": 1_000_000.0}
    orders = broker.rebalance_to_weights({"BTCUSDT": 0.3}, reference_prices, dry_run=False,
                                          poll_interval_sec=0.01, fill_timeout_sec=0.03)
    assert len(canceled) == 1
    assert orders[0]["fill_status"] == "cancelled" and orders[0]["timed_out"] is True
    assert orders[0]["filled_qty"] == 0.0
    print("  verificación de fills OK: una orden que nunca se llena se CANCELA al agotar el timeout")


def main():
    print("[1/6] Probando mapeo de símbolos (Binance-style -> libro de Bitso)...")
    test_book_from_symbol_mapping()

    print("\n[2/6] Probando guardia de confirmación de dinero real (sin testnet)...")
    test_real_money_gate()

    print("\n[3/6] Probando firma HMAC-SHA256 contra un cálculo manual...")

    class _MP:
        def __init__(self):
            self._orig = []

        def setattr(self, obj, name, value):
            self._orig.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def undo(self):
            for obj, name, value in reversed(self._orig):
                setattr(obj, name, value)

    mp = _MP()
    try:
        test_signature_matches_manual_hmac(mp)
    finally:
        mp.undo()

    print("\n[4/6] Probando ticker público (símbolo sin libro en Bitso se omite, no rompe)...")
    mp2 = _MP()
    try:
        test_ticker_prices_skip_unavailable_books(mp2)
    finally:
        mp2.undo()

    print("\n[5/6] Probando cálculo de órdenes de rebalanceo (con _request falso)...")
    mp3 = _MP()
    try:
        test_rebalance_computes_correct_orders(mp3)
    finally:
        mp3.undo()

    print("\n[6/6] Probando que una orden que nunca se llena se cancela al agotar el timeout...")
    mp4 = _MP()
    try:
        test_order_that_never_fills_gets_canceled(mp4)
    finally:
        mp4.undo()

    print("\nBITSO LIVE SMOKE TEST OK: firmado, mapeo de símbolos, cálculo de órdenes y verificación de fills "
          "son correctos.")
    print("(No hay testnet de Bitso -- esto valida la LÓGICA, no una corrida real contra el exchange. "
          "La primera corrida real conviene hacerla con montos mínimos y --broker virtual como referencia.)")


if __name__ == "__main__":
    main()
