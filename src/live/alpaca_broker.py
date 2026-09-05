"""Wrapper delgado sobre alpaca-py para paper trading (por defecto) o real.

Toma un vector de pesos objetivo (ticker -> fracción del equity de la cuenta)
y emite las órdenes necesarias para que la cuenta converja a esos pesos.

Usa órdenes LÍMITE (no de mercado): sin límite de precio, en un gap o un
momento de alta volatilidad podrías comprar mucho más caro (o vender mucho
más barato) de lo esperado. El límite se calcula a partir del último cierre
conocido +/- `max_slippage_pct` -- si el precio se mueve más que eso antes de
que la orden se llene, la orden simplemente no se ejecuta ese día (mejor
perder una entrada que ejecutar a un precio absurdo).

También verifica que el mercado esté abierto antes de enviar órdenes reales.

Después de enviar cada orden, hace POLLING de su estado hasta que se llene,
se cancele/rechace, o se agote `fill_timeout_sec` -- una orden límite NO
garantiza ejecución (el precio puede no tocar el límite en todo el día). Sin
esto, el bot creía haber rebalanceado cuando en realidad la orden se quedó
pendiente (o parcialmente llena) silenciosamente. Si se agota el timeout, la
orden pendiente se CANCELA explícitamente (mejor una posición que se queda
"a medias" y se corrige al día siguiente, que una orden límite vieja
ejecutándose sola horas después a un precio que ya no tiene sentido).

División de órdenes grandes tipo TWAP (`twap_threshold_usd`, opt-in con
default): una orden cuyo monto supera el umbral se divide en varias órdenes
límite más chicas, enviadas en SECUENCIA con una pequeña pausa entre cada una
(`twap_slice_delay_sec`), en vez de una sola orden grande de una vez -- reduce
el impacto de mercado de una orden grande golpeando el libro de una sola vez.
Cada porción se llena (o se cancela por timeout) de forma independiente, y el
resultado se agrega en un solo registro por ticker (mismas claves que una
orden sin dividir, más `n_slices` y `slice_fills` con el detalle de cada
porción) -- el resto del bot (`scripts/run_live_once.py`) no necesita saber
que una orden se dividió. Solo aplica a `AlpacaBroker` (es un detalle de
EJECUCIÓN real -- `VirtualBroker` simula fills instantáneos, no tiene libro de
órdenes que impactar)."""
from __future__ import annotations

import math
import os
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

_TERMINAL_NO_FILL_STATUSES = {"canceled", "expired", "rejected", "done_for_day", "replaced", "suspended"}

DEFAULT_TWAP_THRESHOLD_USD = 5000.0
DEFAULT_TWAP_MAX_SLICES = 5
DEFAULT_TWAP_SLICE_DELAY_SEC = 3.0


class MarketClosedError(RuntimeError):
    pass


class AlpacaBroker:
    def __init__(self, api_key: str | None = None, secret_key: str | None = None, paper: bool | None = None):
        api_key = api_key or os.environ["ALPACA_API_KEY"]
        secret_key = secret_key or os.environ["ALPACA_SECRET_KEY"]
        if paper is None:
            paper = os.environ.get("ALPACA_PAPER", "true").lower() != "false"
        if not paper:
            raise RuntimeError(
                "Este script está pensado para paper trading. Para operar con dinero real, "
                "revisa el código y cambia esto de forma explícita y consciente."
            )
        self.client = TradingClient(api_key, secret_key, paper=paper)

    def get_equity(self, reference_prices: dict[str, float] | None = None) -> float:
        # `reference_prices` no se usa acá (el equity viene directo de la cuenta de Alpaca) --
        # el parámetro existe solo para que la firma sea intercambiable con `VirtualBroker`.
        account = self.client.get_account()
        return float(account.equity)

    def get_current_positions(self) -> dict[str, float]:
        positions = self.client.get_all_positions()
        return {p.symbol: float(p.market_value) for p in positions}

    def get_positions_with_cost_basis(self, reference_prices: dict[str, float] | None = None) -> dict[str, dict]:
        """ticker -> {"qty", "avg_entry_price", "current_price"}, directo de la
        cuenta de Alpaca (`avg_entry_price` viene nativo por posición -- no hace
        falta llevarlo nosotros como en `VirtualBroker`). `reference_prices` se
        ignora, solo existe para que la firma sea intercambiable entre brokers
        (ver `src/tax_loss_harvesting.py`)."""
        positions = self.client.get_all_positions()
        out = {}
        for p in positions:
            current = float(p.current_price) if p.current_price is not None else None
            if current is None:
                continue
            out[p.symbol] = dict(qty=float(p.qty), avg_entry_price=float(p.avg_entry_price), current_price=current)
        return out

    def is_market_open(self) -> bool:
        return bool(self.client.get_clock().is_open)

    @staticmethod
    def _order_status(order) -> str:
        s = order.status
        return str(s.value if hasattr(s, "value") else s).lower()

    def _wait_for_fill(self, order_id, poll_interval_sec: float, fill_timeout_sec: float) -> dict:
        """Poll del estado de la orden hasta que se llene, termine sin llenarse
        (cancelada/rechazada/expirada), o se agote el timeout -- en cuyo caso se
        cancela explícitamente lo que quede pendiente. `fill_status` en el resultado
        es siempre el estado REAL devuelto por Alpaca (nunca se inventa un estado
        sintético) -- `timed_out=True` indica aparte si se llegó a esa cancelación
        por timeout, para que el caller distinga "se rechazó de entrada" de
        "nunca llegó a tocar el límite y se canceló"."""
        elapsed = 0.0
        timed_out = False
        order = self.client.get_order_by_id(order_id)
        while elapsed < fill_timeout_sec:
            order = self.client.get_order_by_id(order_id)
            status = self._order_status(order)
            if status == "filled" or status in _TERMINAL_NO_FILL_STATUSES:
                break
            time.sleep(poll_interval_sec)
            elapsed += poll_interval_sec
        else:
            timed_out = True
            try:
                self.client.cancel_order_by_id(order_id)
            except Exception:
                pass  # puede haberse llenado justo entre el último poll y el cancel -- se refleja abajo
            order = self.client.get_order_by_id(order_id)

        return dict(
            order_id=str(order_id),
            fill_status=self._order_status(order),
            timed_out=timed_out,
            filled_qty=float(order.filled_qty) if order.filled_qty else 0.0,
            filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
        )

    def _submit_and_wait(self, ticker: str, side: str, qty: float, limit_price: float,
                          poll_interval_sec: float, fill_timeout_sec: float) -> dict:
        req = LimitOrderRequest(
            symbol=ticker, qty=qty, limit_price=limit_price,
            side=OrderSide(side), time_in_force=TimeInForce.DAY,
        )
        submitted = self.client.submit_order(req)
        return self._wait_for_fill(submitted.id, poll_interval_sec, fill_timeout_sec)

    @staticmethod
    def _aggregate_slice_fills(slice_fills: list[dict]) -> dict:
        """Combina el resultado de varias porciones TWAP de UN MISMO ticker en un
        solo registro -- `filled_avg_price` es el promedio ponderado por cantidad
        llenada (no el promedio simple de precios), y `fill_status` solo dice
        "filled" si TODAS las porciones se llenaron por completo; si alguna quedó
        pendiente/cancelada, se reporta el estado de esa porción en vez de mentir
        diciendo "filled" cuando el rebalanceo quedó incompleto."""
        total_filled_qty = sum(s["filled_qty"] for s in slice_fills)
        filled_notional = sum(s["filled_qty"] * (s["filled_avg_price"] or 0.0) for s in slice_fills)
        avg_price = (filled_notional / total_filled_qty) if total_filled_qty > 0 else None
        all_filled = all(s["fill_status"] == "filled" for s in slice_fills)
        agg_status = "filled" if all_filled else next(
            (s["fill_status"] for s in slice_fills if s["fill_status"] != "filled"), "unknown")
        return dict(
            fill_status=agg_status,
            timed_out=any(s["timed_out"] for s in slice_fills),
            filled_qty=total_filled_qty,
            filled_avg_price=avg_price,
            slice_fills=slice_fills,
        )

    def rebalance_to_weights(
        self,
        target_weights: dict[str, float],
        reference_prices: dict[str, float],
        min_order_usd: float = 10.0,
        min_weight_drift: float = 0.02,
        max_slippage_pct: float = 0.005,
        dry_run: bool = True,
        require_market_open: bool = True,
        poll_interval_sec: float = 2.0,
        fill_timeout_sec: float = 60.0,
        twap_threshold_usd: float | None = DEFAULT_TWAP_THRESHOLD_USD,
        twap_max_slices: int = DEFAULT_TWAP_MAX_SLICES,
        twap_slice_delay_sec: float = DEFAULT_TWAP_SLICE_DELAY_SEC,
    ) -> list[dict]:
        """Calcula y (si dry_run=False) envía órdenes LÍMITE para acercar la cuenta a
        `target_weights` (ticker -> fracción del equity). `reference_prices` (ticker ->
        último cierre conocido) se usa para calcular el precio límite y el tamaño en
        acciones/fracciones -- Alpaca no permite fijar límite de precio Y monto en
        dólares (`notional`) a la vez, así que la cantidad se deriva del precio.

        `min_weight_drift` (banda de no-operación): no rebalancea un ticker si su
        peso actual ya está a menos de esa fracción del objetivo, sin importar
        cuántos dólares represente esa diferencia -- en una cuenta grande, un
        drift de peso irrelevante (30.0% vs 30.3%) puede superar fácilmente
        `min_order_usd` en dólares y generar turnover que solo paga costos y,
        en cuenta gravable, impuestos de corto plazo sin beneficio real.

        `twap_threshold_usd` (división de órdenes grandes tipo TWAP, ver
        docstring del módulo): una orden con monto por encima de este umbral se
        divide en hasta `twap_max_slices` porciones, enviadas en secuencia con
        `twap_slice_delay_sec` de pausa entre cada una. `twap_threshold_usd=None`
        desactiva la división por completo (una sola orden, como antes de este
        cambio)."""
        equity = self.get_equity()
        current_value = self.get_current_positions()
        all_tickers = set(target_weights) | set(current_value)

        orders = []
        for ticker in sorted(all_tickers):
            target_w = target_weights.get(ticker, 0.0)
            target_usd = target_w * equity
            current_usd = current_value.get(ticker, 0.0)
            current_w = (current_usd / equity) if equity > 0 else 0.0
            delta_usd = target_usd - current_usd

            if abs(delta_usd) < min_order_usd or abs(target_w - current_w) < min_weight_drift:
                continue

            price = reference_prices.get(ticker)
            if not price or price <= 0:
                continue

            side = OrderSide.BUY if delta_usd > 0 else OrderSide.SELL
            limit_price = price * (1 + max_slippage_pct) if side == OrderSide.BUY else price * (1 - max_slippage_pct)
            qty = round(abs(delta_usd) / price, 4)
            if qty <= 0:
                continue

            n_slices = 1
            if twap_threshold_usd and abs(delta_usd) > twap_threshold_usd:
                n_slices = min(twap_max_slices, max(1, math.ceil(abs(delta_usd) / twap_threshold_usd)))

            orders.append(dict(
                ticker=ticker, side=side.value, notional=round(abs(delta_usd), 2),
                qty=qty, limit_price=round(limit_price, 2), n_slices=n_slices,
            ))

        if not dry_run:
            if require_market_open and not self.is_market_open():
                raise MarketClosedError(
                    "El mercado está cerrado -- no se envían órdenes. Corre este script "
                    "durante el horario de mercado (ver README, sección de automatización)."
                )
            for o in orders:
                if o["n_slices"] <= 1:
                    o.update(self._submit_and_wait(o["ticker"], o["side"], o["qty"], o["limit_price"],
                                                     poll_interval_sec, fill_timeout_sec))
                    continue

                # TWAP: divide en n_slices porciones casi iguales (la última se
                # lleva el resto, para no perder cantidad por redondeo) y las manda
                # en secuencia, con una pausa entre cada envío.
                slice_qty = round(o["qty"] / o["n_slices"], 4)
                remaining_qty = o["qty"]
                slice_fills = []
                for i in range(o["n_slices"]):
                    this_qty = slice_qty if i < o["n_slices"] - 1 else round(remaining_qty, 4)
                    remaining_qty -= this_qty
                    if this_qty <= 0:
                        continue
                    slice_fills.append(self._submit_and_wait(o["ticker"], o["side"], this_qty, o["limit_price"],
                                                               poll_interval_sec, fill_timeout_sec))
                    if i < o["n_slices"] - 1 and twap_slice_delay_sec > 0:
                        time.sleep(twap_slice_delay_sec)
                o.update(self._aggregate_slice_fills(slice_fills))

        return orders
