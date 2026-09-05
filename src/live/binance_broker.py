"""Wrapper delgado sobre python-binance para trading spot -- por defecto contra
el TESTNET de Binance (paper trading real de Binance, no una simulación local
como `virtual_broker.py`), nunca contra la cuenta real sin un cambio explícito
y consciente en el código, igual que `alpaca_broker.py`.

Usa órdenes LÍMITE (no de mercado) con un slippage máximo configurable, y
redondea cantidad/precio a los "filtros" que exige cada símbolo en Binance
(LOT_SIZE, PRICE_FILTER, MIN_NOTIONAL) -- una orden que no respeta esos
filtros es rechazada por el exchange.

Después de enviar cada orden hace POLLING de su estado (NEW/PARTIALLY_FILLED
-> FILLED, o CANCELED/REJECTED/EXPIRED) hasta que se resuelva o se agote
`fill_timeout_sec`, momento en el que cancela lo que quede pendiente -- sin
esto el bot no tenía forma de saber si una orden límite realmente se llenó.
"""
from __future__ import annotations

import math
import os
import time

from binance.client import Client

_TERMINAL_NO_FILL_STATUSES = {"CANCELED", "REJECTED", "EXPIRED", "PENDING_CANCEL"}


class MinNotionalError(RuntimeError):
    """La orden calculada es más chica que el mínimo que Binance acepta para ese símbolo."""


def _round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    precision = max(0, -int(round(math.log10(step))))
    return math.floor(value / step) * step if precision == 0 else round(math.floor(value / step) * step, precision)


class BinanceBroker:
    def __init__(self, api_key: str | None = None, secret_key: str | None = None,
                 testnet: bool | None = None, quote_currency: str = "USDT"):
        api_key = api_key or os.environ["BINANCE_API_KEY"]
        secret_key = secret_key or os.environ["BINANCE_SECRET_KEY"]
        if testnet is None:
            testnet = os.environ.get("BINANCE_TESTNET", "true").lower() != "false"
        if not testnet:
            raise RuntimeError(
                "Este script está pensado para el testnet de Binance (paper trading). Para operar "
                "con dinero real, revisa el código y cambia esto de forma explícita y consciente."
            )
        self.client = Client(api_key, secret_key, testnet=testnet)
        self.quote_currency = quote_currency
        self._symbol_filters_cache: dict[str, dict] = {}

    def _base_asset(self, symbol: str) -> str:
        return symbol[: -len(self.quote_currency)] if symbol.endswith(self.quote_currency) else symbol

    def _get_filters(self, symbol: str) -> dict:
        if symbol not in self._symbol_filters_cache:
            info = self.client.get_symbol_info(symbol)
            filters = {f["filterType"]: f for f in info["filters"]}
            self._symbol_filters_cache[symbol] = dict(
                step_size=float(filters.get("LOT_SIZE", {}).get("stepSize", 0.0) or 0.0),
                tick_size=float(filters.get("PRICE_FILTER", {}).get("tickSize", 0.0) or 0.0),
                min_notional=float(
                    filters.get("MIN_NOTIONAL", filters.get("NOTIONAL", {})).get("minNotional", 0.0) or 0.0
                ),
            )
        return self._symbol_filters_cache[symbol]

    def is_market_open(self) -> bool:
        return True  # cripto cotiza 24/7 -- no aplica el concepto de "mercado cerrado"

    def get_balances(self) -> dict[str, float]:
        account = self.client.get_account()
        return {b["asset"]: float(b["free"]) + float(b["locked"]) for b in account["balances"]
                if float(b["free"]) + float(b["locked"]) > 0}

    def get_equity(self, reference_prices: dict[str, float] | None = None) -> float:
        reference_prices = reference_prices or {}
        balances = self.get_balances()
        equity = balances.get(self.quote_currency, 0.0)
        for asset, qty in balances.items():
            if asset == self.quote_currency:
                continue
            price = reference_prices.get(f"{asset}{self.quote_currency}")
            if price:
                equity += qty * price
        return equity

    def get_current_positions(self, reference_prices: dict[str, float] | None = None) -> dict[str, float]:
        reference_prices = reference_prices or {}
        balances = self.get_balances()
        positions = {}
        for asset, qty in balances.items():
            if asset == self.quote_currency:
                continue
            symbol = f"{asset}{self.quote_currency}"
            price = reference_prices.get(symbol)
            if price:
                positions[symbol] = qty * price
        return positions

    def _wait_for_fill(self, symbol: str, order_id, poll_interval_sec: float, fill_timeout_sec: float) -> dict:
        """Poll del estado real de la orden en Binance hasta que se llene, termine
        sin llenarse, o se agote el timeout -- en cuyo caso se cancela lo pendiente.
        `fill_status` siempre es el estado real devuelto por Binance."""
        elapsed = 0.0
        timed_out = False
        order = self.client.get_order(symbol=symbol, orderId=order_id)
        while elapsed < fill_timeout_sec:
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            status = order["status"]
            if status == "FILLED" or status in _TERMINAL_NO_FILL_STATUSES:
                break
            time.sleep(poll_interval_sec)
            elapsed += poll_interval_sec
        else:
            timed_out = True
            try:
                self.client.cancel_order(symbol=symbol, orderId=order_id)
            except Exception:
                pass  # puede haberse llenado justo entre el último poll y el cancel -- se refleja abajo
            order = self.client.get_order(symbol=symbol, orderId=order_id)

        return dict(
            order_id=str(order_id),
            fill_status=order["status"],
            timed_out=timed_out,
            filled_qty=float(order.get("executedQty", 0.0) or 0.0),
            filled_avg_price=(float(order["cummulativeQuoteQty"]) / float(order["executedQty"])
                               if float(order.get("executedQty", 0.0) or 0.0) > 0 else None),
        )

    def rebalance_to_weights(
        self,
        target_weights: dict[str, float],
        reference_prices: dict[str, float],
        min_order_usd: float = 10.0,
        min_weight_drift: float = 0.02,
        max_slippage_pct: float = 0.01,
        dry_run: bool = True,
        poll_interval_sec: float = 2.0,
        fill_timeout_sec: float = 60.0,
        **_ignored,
    ) -> list[dict]:
        """`min_weight_drift`: banda de no-operación -- ver docstring de la
        misma idea en `AlpacaBroker.rebalance_to_weights`."""
        equity = self.get_equity(reference_prices)
        current_value = self.get_current_positions(reference_prices)
        all_symbols = set(target_weights) | set(current_value)

        orders = []
        for symbol in sorted(all_symbols):
            price = reference_prices.get(symbol)
            if not price or price <= 0:
                continue
            target_w = target_weights.get(symbol, 0.0)
            target_usd = target_w * equity
            current_usd = current_value.get(symbol, 0.0)
            current_w = (current_usd / equity) if equity > 0 else 0.0
            delta_usd = target_usd - current_usd
            if abs(delta_usd) < min_order_usd or abs(target_w - current_w) < min_weight_drift:
                continue

            side = "BUY" if delta_usd > 0 else "SELL"
            limit_price = price * (1 + max_slippage_pct) if side == "BUY" else price * (1 - max_slippage_pct)

            try:
                filters = self._get_filters(symbol)
            except Exception as e:
                orders.append(dict(ticker=symbol, side=side.lower(), notional=round(abs(delta_usd), 2),
                                    qty=0.0, limit_price=round(limit_price, 8), error=f"sin filtros del símbolo: {e}"))
                continue

            qty = abs(delta_usd) / limit_price
            if filters["step_size"] > 0:
                qty = _round_step(qty, filters["step_size"])
            if filters["tick_size"] > 0:
                limit_price = _round_step(limit_price, filters["tick_size"])

            notional = qty * limit_price
            if qty <= 0 or notional < filters["min_notional"]:
                orders.append(dict(ticker=symbol, side=side.lower(), notional=round(abs(delta_usd), 2),
                                    qty=qty, limit_price=round(limit_price, 8),
                                    error=f"por debajo del mínimo del exchange (min_notional=${filters['min_notional']:.2f})"))
                continue

            orders.append(dict(ticker=symbol, side=side.lower(), notional=round(notional, 2),
                                qty=qty, limit_price=round(limit_price, 8)))

            if not dry_run:
                try:
                    resp = self.client.create_order(
                        symbol=symbol, side=side, type="LIMIT",
                        timeInForce="GTC", quantity=qty, price=f"{limit_price:.8f}".rstrip("0").rstrip("."),
                    )
                    orders[-1].update(self._wait_for_fill(symbol, resp["orderId"], poll_interval_sec, fill_timeout_sec))
                except Exception as e:
                    orders[-1]["error"] = str(e)

        return orders
