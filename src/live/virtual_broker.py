"""Broker "virtual": simula una cuenta de paper trading con SU PROPIO capital
inicial, sin pasar por Alpaca ni necesitar ninguna cuenta/API key.

Para qué sirve: correr los 2 perfiles (conservador y agresivo) EN PARALELO, cada
uno con su propio capital (ej. $1000), sin tener que crear dos cuentas de Alpaca
separadas -- Alpaca solo te da un balance de paper trading por cuenta. Cada
perfil ya escribe en su propia carpeta de reports (`reports/` vs.
`reports_aggressive/`), así que el estado de cada uno queda completamente
aislado del otro automáticamente.

No es una ejecución "real" en el sentido de que una orden llegue a un exchange
-- es una simulación día a día con precios de cierre reales (los mismos que
descarga `run_live_once.py`), marcando posiciones a mercado y aplicando el
mismo modelo de costos que el backtest. Es la forma más simple de responder
"¿cómo le va de verdad a esto, empezando con $1000, mes a mes?" sin fricción
de configurar brokers.
"""
from __future__ import annotations

import json
from pathlib import Path


class VirtualBroker:
    def __init__(self, state_file: Path, starting_cash: float = 1000.0):
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        if self.state_file.exists():
            self.state = json.loads(self.state_file.read_text())
            self.state.setdefault("cost_basis", {})  # compatibilidad con estado guardado antes de este campo
        else:
            self.state = {"starting_cash": starting_cash, "cash": starting_cash, "positions": {}, "cost_basis": {}}
            self._save()

    def _save(self) -> None:
        self.state_file.write_text(json.dumps(self.state))

    def mark_to_market(self, reference_prices: dict[str, float]) -> float:
        equity = self.state["cash"]
        for ticker, shares in self.state["positions"].items():
            price = reference_prices.get(ticker)
            if price:
                equity += shares * price
        return equity

    def get_equity(self, reference_prices: dict[str, float] | None = None) -> float:
        return self.mark_to_market(reference_prices or {})

    def get_current_positions(self, reference_prices: dict[str, float] | None = None) -> dict[str, float]:
        reference_prices = reference_prices or {}
        return {t: shares * reference_prices.get(t, 0.0) for t, shares in self.state["positions"].items()}

    def get_positions_with_cost_basis(self, reference_prices: dict[str, float] | None = None) -> dict[str, dict]:
        """ticker -> {"qty", "avg_entry_price", "current_price"} para cada posición
        abierta con costo promedio conocido -- ver `src/tax_loss_harvesting.py`.
        Firma compatible con `AlpacaBroker.get_positions_with_cost_basis` (que
        ignora `reference_prices`, ya que Alpaca ya trae el precio actual en cada
        posición) para que `run_live_once.py` no necesite ramas por broker."""
        reference_prices = reference_prices or {}
        out = {}
        for ticker, shares in self.state["positions"].items():
            avg_entry = self.state.get("cost_basis", {}).get(ticker)
            price = reference_prices.get(ticker)
            if not avg_entry or not price or shares == 0:
                continue
            out[ticker] = dict(qty=shares, avg_entry_price=avg_entry, current_price=price)
        return out

    def is_market_open(self) -> bool:
        # No aplica -- no hay un exchange real de por medio. Siempre "abierto" para la simulación.
        return True

    def rebalance_to_weights(
        self,
        target_weights: dict[str, float],
        reference_prices: dict[str, float],
        min_order_usd: float = 5.0,
        min_weight_drift: float = 0.02,
        cost_bps: float = 5.0,
        dry_run: bool = True,
        **_ignored,
    ) -> list[dict]:
        """Firma compatible con `AlpacaBroker.rebalance_to_weights` (acepta y
        descarta kwargs como `max_slippage_pct`/`require_market_open` que no
        aplican acá) para que `run_live_once.py` pueda usar cualquiera de los
        dos brokers sin ramas de código separadas.

        `min_weight_drift` (banda de no-operación): además del piso en dólares
        de `min_order_usd`, no rebalancea un ticker si su peso actual ya está a
        menos de esa fracción del objetivo (ej. 0.02 = 2 puntos porcentuales).
        Sin esto, una cuenta grande rebalancea posiciones cada corrida por
        diferencias de peso irrelevantes (ej. 30.0% vs 30.3%) solo porque el
        monto en dólares de ese ajuste ya supera `min_order_usd` -- puro
        turnover innecesario que paga costos de transacción y, en cuenta
        gravable, impuestos de corto plazo sin ningún beneficio real."""
        equity = self.mark_to_market(reference_prices)
        current_value = self.get_current_positions(reference_prices)
        all_tickers = set(target_weights) | set(current_value)

        orders = []
        for ticker in sorted(all_tickers):
            price = reference_prices.get(ticker)
            if not price or price <= 0:
                continue
            target_w = target_weights.get(ticker, 0.0)
            target_usd = target_w * equity
            current_usd = current_value.get(ticker, 0.0)
            current_w = (current_usd / equity) if equity > 0 else 0.0
            delta_usd = target_usd - current_usd
            if abs(delta_usd) < min_order_usd or abs(target_w - current_w) < min_weight_drift:
                continue

            delta_shares = delta_usd / price
            cost = abs(delta_usd) * (cost_bps / 10_000.0)
            side = "buy" if delta_usd > 0 else "sell"
            orders.append(dict(ticker=ticker, side=side, notional=round(abs(delta_usd), 2),
                                qty=round(abs(delta_shares), 4), limit_price=round(price, 2)))

            if not dry_run:
                old_shares = self.state["positions"].get(ticker, 0.0)
                new_shares = old_shares + delta_shares
                if delta_shares > 0:
                    # Costo promedio ponderado: si ya había posición (long), pondera el
                    # promedio existente con esta compra; si no había posición (o venía de
                    # cero/negativa), el costo base arranca de nuevo en el precio de hoy.
                    old_avg = self.state["cost_basis"].get(ticker)
                    if old_shares > 0 and old_avg:
                        self.state["cost_basis"][ticker] = (
                            (old_avg * old_shares + price * delta_shares) / new_shares
                        )
                    else:
                        self.state["cost_basis"][ticker] = price
                if abs(new_shares) < 1e-9:
                    self.state["positions"].pop(ticker, None)
                    self.state["cost_basis"].pop(ticker, None)
                else:
                    self.state["positions"][ticker] = new_shares
                self.state["cash"] -= (delta_usd + cost)

        if not dry_run:
            self._save()
        return orders
