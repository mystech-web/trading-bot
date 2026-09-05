"""Estimación (aproximada, NO un cálculo fiscal exacto) del drag de impuestos.

Estas estrategias tienen turnover alto -> la gran mayoría de las ganancias que
generan se realizan en menos de un año -> en EE.UU. eso normalmente se grava
como ganancia de corto plazo, a la tasa marginal ordinaria (más alta que la de
largo plazo). El backtest, al usar `auto_adjust=True` de yfinance, asume
reinversión perfecta sin fricción fiscal -- optimista para una cuenta gravable
(no aplica si operas dentro de una cuenta con ventajas fiscales, ej. un 401k/IRA
en EE.UU. o el equivalente en tu país).

Este módulo NO intenta rastrear cost basis por lote ni pérdidas compensables
entre años (eso requeriría un motor contable completo). Usa la aproximación
estándar para estrategias de turnover alto: como casi toda la ganancia se
realiza como corto plazo cada año, el CAGR después de impuestos es
aproximadamente CAGR_antes_de_impuestos * (1 - tasa_marginal). Trátalo como
una cota conservadora para conversar con tu contador, no como un número final.
"""
from __future__ import annotations

DEFAULT_SHORT_TERM_TAX_RATE = 0.35


def estimate_after_tax_cagr(pretax_cagr: float, tax_rate: float = DEFAULT_SHORT_TERM_TAX_RATE) -> float:
    if pretax_cagr <= 0:
        return pretax_cagr  # no hay ganancia que gravar (y las pérdidas no se modelan aquí)
    return pretax_cagr * (1 - tax_rate)


def estimate_after_tax_monthly(pretax_avg_monthly: float, tax_rate: float = DEFAULT_SHORT_TERM_TAX_RATE) -> float:
    if pretax_avg_monthly <= 0:
        return pretax_avg_monthly
    return pretax_avg_monthly * (1 - tax_rate)
