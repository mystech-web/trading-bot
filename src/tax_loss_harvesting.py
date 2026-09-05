"""Tax-loss harvesting automático (opt-in -- ver `tax.harvest_losses_enabled`
en config/live_params.yaml). Identifica posiciones con una pérdida NO REALIZADA
que supera un umbral y las vende para "cosechar" (realizar) esa pérdida, que
luego se puede usar para compensar ganancias de otras posiciones en tu
declaración de impuestos (consulta a tu contador -- esto NO es asesoría fiscal).

ALCANCE, honestamente limitado: solo funciona con `VirtualBroker` y
`AlpacaBroker`, los únicos brokers de este proyecto que exponen cost basis por
posición (`VirtualBroker` lo calcula él mismo con costo promedio ponderado, ver
`get_positions_with_cost_basis`; Alpaca lo expone nativamente vía
`avg_entry_price`). `BinanceBroker`/`BitsoBroker` NO lo soportan: llevar cost
basis ahí requeriría un ledger propio (fills, fees, conversiones spot) fuera de
alcance de este proyecto, y el tratamiento fiscal de cripto varía demasiado por
país (ver `src/tax.py`) para que valga la pena aproximarlo mal.

Guardia de "wash sale" -- APROXIMADA, no garantiza cumplimiento al 100%: la
regla de EE.UU. desconoce la pérdida fiscal si recompras el MISMO activo (o uno
"sustancialmente idéntico") dentro de los 30 días antes o después de la venta.
Este módulo no lleva un libro contable por lote -- lo que sí hace es bloquear,
durante `wash_sale_days` después de cada cosecha, que la estrategia normal
vuelva a comprar ese mismo ticker (ver `block_recent_harvest_rebuys`), que es
el caso que SÍ puede pasar solo con este bot corriendo día a día. Si tú (u otra
cuenta tuya, ej. un IRA) compras el mismo ticker por fuera del bot dentro de esa
ventana, este módulo no puede saberlo -- trátalo como una salvaguarda razonable
para conversar con tu contador, no como garantía de cumplimiento con el IRS.
"""
from __future__ import annotations

import datetime as dt

DEFAULT_LOSS_THRESHOLD_PCT = -0.05
DEFAULT_WASH_SALE_DAYS = 31


def find_harvest_candidates(
    positions: dict[str, dict],
    loss_threshold_pct: float = DEFAULT_LOSS_THRESHOLD_PCT,
) -> list[dict]:
    """`positions`: ticker -> {"qty", "avg_entry_price", "current_price"}.

    Devuelve los candidatos a cosechar (pérdida no realizada <= umbral, ej. -5%),
    ordenados de PEOR a mejor pérdida -- útil si algún día se quiere limitar
    cuántas posiciones cosechar en una sola corrida."""
    candidates = []
    for ticker, pos in positions.items():
        avg_entry = pos.get("avg_entry_price")
        current = pos.get("current_price")
        qty = pos.get("qty", 0.0)
        if not avg_entry or avg_entry <= 0 or not current or current <= 0 or qty == 0:
            continue
        plpc = (current - avg_entry) / avg_entry
        if plpc <= loss_threshold_pct:
            candidates.append(dict(ticker=ticker, qty=qty, avg_entry_price=avg_entry,
                                    current_price=current, unrealized_plpc=plpc))
    candidates.sort(key=lambda c: c["unrealized_plpc"])
    return candidates


def block_recent_harvest_rebuys(
    target_weights: dict[str, float],
    recently_harvested: dict[str, str],
    as_of: dt.date,
    wash_sale_days: int = DEFAULT_WASH_SALE_DAYS,
) -> dict[str, float]:
    """Pone a 0 el peso objetivo de cualquier ticker vendido por cosecha de
    pérdidas dentro de los últimos `wash_sale_days` días -- si la estrategia
    normal pidiera recomprarlo antes de que pase la ventana, hacerlo invalidaría
    la pérdida fiscal ya cosechada (wash sale). `recently_harvested`: ticker ->
    fecha de la venta en formato ISO ("YYYY-MM-DD"). El resto de los pesos NO
    se renormaliza acá -- ese ticker simplemente queda en 0 hasta que pase la
    ventana; en la próxima corrida sin bloqueo, la señal normal retoma el control."""
    blocked = dict(target_weights)
    for ticker, sold_at_iso in recently_harvested.items():
        if ticker not in blocked:
            continue
        sold_at = dt.date.fromisoformat(sold_at_iso)
        if (as_of - sold_at).days <= wash_sale_days:
            blocked[ticker] = 0.0
    return blocked


def prune_expired_harvests(
    recently_harvested: dict[str, str],
    as_of: dt.date,
    wash_sale_days: int = DEFAULT_WASH_SALE_DAYS,
) -> dict[str, str]:
    """Housekeeping: descarta entradas cuya ventana de wash sale ya pasó, para
    que el estado persistido (ver `src/tracking.py` -> `load_state`/`save_state`)
    no crezca sin límite."""
    return {
        t: sold_at_iso for t, sold_at_iso in recently_harvested.items()
        if (as_of - dt.date.fromisoformat(sold_at_iso)).days <= wash_sale_days
    }
