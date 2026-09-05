"""Comportamiento de cada estrategia durante caídas de mercado conocidas.

Las métricas agregadas (Sharpe, CAGR) pueden esconder un mal comportamiento en
la cola: una estrategia con buen Sharpe promedio puede igual perder 30% en un
crash puntual. Esto mide, período por período, qué tan mal (o bien) le fue a
cada estrategia exactamente cuando más importaba.

Nota: estos períodos son un subconjunto conocido de crisis dentro de la
ventana de datos típica (~10-11 años). Si tu rango de datos no cubre alguno,
simplemente se omite del reporte.
"""
from __future__ import annotations

import pandas as pd

CRISIS_PERIODS = {
    "Selloff_Ago_2015": ("2015-08-17", "2015-08-25"),
    "Selloff_Q4_2018": ("2018-10-01", "2018-12-24"),
    "Crash_COVID_2020": ("2020-02-19", "2020-03-23"),
    "Bear_market_2022": ("2022-01-03", "2022-10-12"),
}

# Crashes específicos de cripto -- distintos a los de acciones (aunque COVID-2020
# también aplica a ambos mercados). Cotiza fines de semana incluidos.
CRYPTO_CRISIS_PERIODS = {
    "Crash_COVID_2020": ("2020-02-19", "2020-03-13"),
    "Crash_Mayo_2021": ("2021-05-12", "2021-05-23"),        # primer gran crash del ciclo 2021
    "Colapso_Terra_Luna_2022": ("2022-05-07", "2022-05-13"),
    "Colapso_FTX_2022": ("2022-11-06", "2022-11-14"),
    "Bear_market_2022": ("2022-01-01", "2022-12-31"),        # todo el año, no solo el crash puntual
}


def periods_covered(index: pd.DatetimeIndex, periods: dict = CRISIS_PERIODS) -> dict:
    if len(index) == 0:
        return {}
    lo, hi = index.min(), index.max()
    return {
        name: (start, end) for name, (start, end) in periods.items()
        if pd.Timestamp(start) >= lo and pd.Timestamp(end) <= hi
    }


def run_stress_test(returns_by_strategy: dict[str, pd.Series], periods: dict) -> pd.DataFrame:
    rows = []
    for period_name, (start, end) in periods.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        row = {"period": period_name, "start": start.date(), "end": end.date()}
        for strat_name, rets in returns_by_strategy.items():
            window = rets.loc[(rets.index >= start) & (rets.index <= end)]
            if window.empty:
                row[f"{strat_name}_return_%"] = None
                row[f"{strat_name}_max_dd_%"] = None
                continue
            total_return = (1 + window).prod() - 1
            equity = (1 + window).cumprod()
            dd = (equity / equity.cummax() - 1).min()
            row[f"{strat_name}_return_%"] = round(total_return * 100, 2)
            row[f"{strat_name}_max_dd_%"] = round(dd * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows)
