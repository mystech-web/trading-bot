"""Wrapper delgado sobre la API REST de Bitso (`requests` directo -- no hay SDK
oficial de Python mantenido) para trading spot contra MXN.

DIFERENCIA IMPORTANTE con Alpaca/Binance: Bitso NO tiene testnet/sandbox. No
existe una forma de "paper trading real" contra el exchange -- cualquier orden
que no sea dry-run mueve MXN de verdad. Por eso:

  1. El constructor exige `confirm_real_money=True` explícito (o la variable
     de entorno BITSO_CONFIRM_REAL_MONEY=true) -- sin eso, lanza una excepción
     en vez de operar. No hay "paper=True" al que caer por defecto como en los
     otros brokers, porque Bitso no lo ofrece.
  2. La recomendación por defecto en `run_crypto_live_once.py` sigue siendo
     `--broker virtual` (capital simulado, cero riesgo real) -- usa
     `--broker bitso` solo cuando ya validaste la estrategia y decidiste
     conscientemente arriesgar dinero real.

Las señales (momentum/RSI/rotación) se calculan con datos de Binance (ver
`run_crypto_live_once.py`) porque Bitso no expone un endpoint público de velas
históricas -- el movimiento relativo de precio de BTC/ETH/etc. es
prácticamente el mismo en cualquier exchange (arbitraje lo mantiene así), así
que la señal sigue siendo válida. Lo que sí se usa de Bitso es el precio de
referencia EN VIVO (MXN) al momento de calcular el tamaño de cada orden -- eso
sale del ticker público de Bitso, no de Binance.

Como esto mueve dinero real (sin testnet, ver arriba), después de enviar cada
orden hace POLLING de su estado real (GET /v3/orders/<oid>/) hasta que se
llene, se cancele, o se agote `fill_timeout_sec` -- en cuyo caso cancela
explícitamente lo pendiente (DELETE /v3/orders/<oid>/). Sin esto no había
forma de saber si una orden límite contra Bitso realmente se ejecutó.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

import requests

BITSO_BASE_URL = "https://api.bitso.com"
_TERMINAL_NO_FILL_STATUSES = {"cancelled"}


class RealMoneyNotConfirmedError(RuntimeError):
    pass


def book_from_symbol(symbol: str, base_suffix: str = "USDT", quote_currency: str = "mxn") -> str:
    """"BTCUSDT" -> "btc_mxn". Las señales usan símbolos estilo Binance
    (ver src/crypto_data.py); Bitso usa "libros" en minúsculas tipo btc_mxn."""
    base = symbol[: -len(base_suffix)] if symbol.endswith(base_suffix) else symbol
    return f"{base.lower()}_{quote_currency}"


def get_bitso_ticker_prices(symbols: list[str], quote_currency: str = "mxn") -> dict[str, float]:
    """Precios EN VIVO (público, sin autenticación) para cada símbolo, en el
    mismo formato de clave ("BTCUSDT") que usa el resto del motor -- para que
    `reference_prices` sea intercambiable entre brokers."""
    prices = {}
    for symbol in symbols:
        book = book_from_symbol(symbol, quote_currency=quote_currency)
        try:
            resp = requests.get(f"{BITSO_BASE_URL}/v3/ticker/", params={"book": book}, timeout=10)
            resp.raise_for_status()
            payload = resp.json()["payload"]
            prices[symbol] = float(payload["last"])
        except Exception:
            continue  # símbolo sin libro en Bitso (ej. una altcoin que no cotiza en MXN) -- se omite
    return prices


class BitsoBroker:
    def __init__(self, api_key: str | None = None, secret_key: str | None = None,
                 quote_currency: str = "mxn", confirm_real_money: bool | None = None):
        api_key = api_key or os.environ["BITSO_API_KEY"]
        secret_key = secret_key or os.environ["BITSO_SECRET_KEY"]
        if confirm_real_money is None:
            confirm_real_money = os.environ.get("BITSO_CONFIRM_REAL_MONEY", "false").lower() == "true"
        if not confirm_real_money:
            raise RealMoneyNotConfirmedError(
                "Bitso no tiene testnet -- cualquier orden real mueve MXN de verdad. "
                "Este broker exige confirmación explícita: pasa confirm_real_money=True o "
                "configura BITSO_CONFIRM_REAL_MONEY=true en .env SOLO si entendiste el riesgo. "
                "Para practicar sin riesgo, usa --broker virtual (default)."
            )
        self._api_key = api_key
        self._secret_key = secret_key
        self.quote_currency = quote_currency

    def _signed_headers(self, method: str, path: str, body: dict | None = None) -> dict:
        nonce = str(int(time.time() * 1000))
        json_body = json.dumps(body) if body else ""
        message = nonce + method.upper() + "/api" + path + json_body
        signature = hmac.new(self._secret_key.encode(), message.encode(), hashlib.sha256).hexdigest()
        return {
            "Authorization": f"Bitso {self._api_key}:{nonce}:{signature}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        headers = self._signed_headers(method, path, body)
        resp = requests.request(method, f"{BITSO_BASE_URL}{path}", headers=headers,
                                 json=body if body else None, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", False):
            raise RuntimeError(f"Bitso rechazó la solicitud: {data}")
        return data["payload"]

    def is_market_open(self) -> bool:
        return True  # cripto cotiza 24/7

    def get_balances(self) -> dict[str, float]:
        payload = self._request("GET", "/v3/balance")
        return {b["currency"].upper(): float(b["available"]) + float(b["locked"]) for b in payload["balances"]}

    def get_equity(self, reference_prices: dict[str, float] | None = None) -> float:
        reference_prices = reference_prices or {}
        balances = self.get_balances()
        quote = self.quote_currency.upper()
        equity = balances.get(quote, 0.0)
        for asset, qty in balances.items():
            if asset == quote:
                continue
            price = reference_prices.get(f"{asset}USDT")  # convención de símbolo compartida con Binance
            if price:
                equity += qty * price
        return equity

    def get_current_positions(self, reference_prices: dict[str, float] | None = None) -> dict[str, float]:
        reference_prices = reference_prices or {}
        balances = self.get_balances()
        quote = self.quote_currency.upper()
        positions = {}
        for asset, qty in balances.items():
            if asset == quote:
                continue
            symbol = f"{asset}USDT"
            price = reference_prices.get(symbol)
            if price:
                positions[symbol] = qty * price
        return positions

    def _wait_for_fill(self, order_id: str, poll_interval_sec: float, fill_timeout_sec: float) -> dict:
        """Poll del estado real de la orden en Bitso hasta que se llene
        (status="completed"), se cancele, o se agote el timeout -- en cuyo caso
        se cancela lo que quede pendiente. `fill_status` siempre es el estado
        real devuelto por Bitso ("open"/"partial-filled"/"completed"/"cancelled")."""
        def _fetch():
            payload = self._request("GET", f"/v3/orders/{order_id}")
            return payload[0] if isinstance(payload, list) else payload

        elapsed = 0.0
        timed_out = False
        order = _fetch()
        while elapsed < fill_timeout_sec:
            order = _fetch()
            status = order["status"]
            if status == "completed" or status in _TERMINAL_NO_FILL_STATUSES:
                break
            time.sleep(poll_interval_sec)
            elapsed += poll_interval_sec
        else:
            timed_out = True
            try:
                self._request("DELETE", f"/v3/orders/{order_id}")
            except Exception:
                pass  # puede haberse llenado justo entre el último poll y el cancel -- se refleja abajo
            order = _fetch()

        original_amount = float(order.get("original_amount", 0.0) or 0.0)
        unfilled_amount = float(order.get("unfilled_amount", 0.0) or 0.0)
        filled_qty = max(0.0, original_amount - unfilled_amount)
        return dict(
            order_id=str(order_id),
            fill_status=order["status"],
            timed_out=timed_out,
            filled_qty=filled_qty,
            filled_avg_price=float(order["price"]) if filled_qty > 0 and order.get("price") else None,
        )

    def rebalance_to_weights(
        self,
        target_weights: dict[str, float],
        reference_prices: dict[str, float],
        min_order_mxn: float = 50.0,
        min_weight_drift: float = 0.02,
        max_slippage_pct: float = 0.01,
        dry_run: bool = True,
        poll_interval_sec: float = 2.0,
        fill_timeout_sec: float = 60.0,
        **_ignored,
    ) -> list[dict]:
        """`min_weight_drift`: banda de no-operación -- ver docstring de la
        misma idea en `AlpacaBroker.rebalance_to_weights`. Especialmente
        relevante acá: Bitso mueve dinero real y no tiene testnet, así que
        evitar rebalanceos innecesarios por drift irrelevante también evita
        pagar comisiones reales sin ninguna razón."""
        equity = self.get_equity(reference_prices)
        current_value = self.get_current_positions(reference_prices)
        all_symbols = set(target_weights) | set(current_value)

        orders = []
        for symbol in sorted(all_symbols):
            price = reference_prices.get(symbol)
            if not price or price <= 0:
                continue
            target_w = target_weights.get(symbol, 0.0)
            target_val = target_w * equity
            current_val = current_value.get(symbol, 0.0)
            current_w = (current_val / equity) if equity > 0 else 0.0
            delta_val = target_val - current_val
            if abs(delta_val) < min_order_mxn or abs(target_w - current_w) < min_weight_drift:
                continue

            side = "buy" if delta_val > 0 else "sell"
            limit_price = price * (1 + max_slippage_pct) if side == "buy" else price * (1 - max_slippage_pct)
            major = abs(delta_val) / limit_price
            book = book_from_symbol(symbol, quote_currency=self.quote_currency)

            orders.append(dict(ticker=symbol, side=side, notional=round(abs(delta_val), 2),
                                qty=round(major, 8), limit_price=round(limit_price, 2), book=book))

            if not dry_run:
                try:
                    resp = self._request("POST", "/v3/orders", body={
                        "book": book, "side": side, "type": "limit",
                        "major": f"{major:.8f}", "price": f"{limit_price:.2f}",
                    })
                    orders[-1].update(self._wait_for_fill(resp["oid"], poll_interval_sec, fill_timeout_sec))
                except Exception as e:
                    orders[-1]["error"] = str(e)

        return orders
