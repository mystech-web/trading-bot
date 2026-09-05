"""Métricas de desempeño calculadas sobre una serie de retornos diarios.

`periods_per_year` default 252 (días hábiles de bolsa de EE.UU.) -- el módulo
cripto lo pasa en 365 (BTC/USDT etc. cotizan los 365 días del año, fines de
semana incluidos). Sin este parámetro, el CAGR/Sharpe/vol anualizados de cripto
saldrían ~45% subestimados (252/365) -- no es un detalle cosmético.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # default para acciones/ETFs -- ver `periods_per_year` en cada función


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    equity = (1 + returns).cumprod()
    if len(equity) == 0 or equity.iloc[-1] <= 0:
        return float("nan")
    years = len(returns) / periods_per_year
    if years <= 0:
        return float("nan")
    return equity.iloc[-1] ** (1 / years) - 1


def annualized_vol(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return returns.std() * np.sqrt(periods_per_year)


def sharpe(returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    excess = returns - rf_annual / periods_per_year
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return excess.mean() / vol * np.sqrt(periods_per_year)


def sortino(returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    excess = returns - rf_annual / periods_per_year
    downside = excess[excess < 0]
    dd_std = downside.std()
    if dd_std == 0 or np.isnan(dd_std):
        return float("nan")
    return excess.mean() / dd_std * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    return dd.min()


def calmar(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return cagr(returns, periods_per_year) / abs(mdd)


def monthly_returns(returns: pd.Series) -> pd.Series:
    equity = (1 + returns).cumprod()
    monthly_equity = equity.resample("ME").last()
    monthly_ret = monthly_equity.pct_change()
    first_month_ret = monthly_equity.iloc[0] - 1 if len(monthly_equity) else np.nan
    if len(monthly_ret) > 0:
        monthly_ret.iloc[0] = first_month_ret
    return monthly_ret


def summary(returns: pd.Series, label: str = "", periods_per_year: int = TRADING_DAYS) -> dict:
    m = monthly_returns(returns)
    return dict(
        strategy=label,
        cagr=cagr(returns, periods_per_year),
        ann_vol=annualized_vol(returns, periods_per_year),
        sharpe=sharpe(returns, periods_per_year=periods_per_year),
        sortino=sortino(returns, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(returns),
        calmar=calmar(returns, periods_per_year),
        avg_monthly_return=m.mean(),
        median_monthly_return=m.median(),
        pct_positive_months=(m > 0).mean(),
        worst_month=m.min(),
        best_month=m.max(),
        n_months=len(m),
    )


def summary_table(results: dict[str, pd.Series], periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    rows = [summary(returns, label, periods_per_year) for label, returns in results.items()]
    df = pd.DataFrame(rows).set_index("strategy")
    pct_cols = ["cagr", "ann_vol", "max_drawdown", "avg_monthly_return",
                "median_monthly_return", "pct_positive_months", "worst_month", "best_month"]
    for c in pct_cols:
        df[c] = (df[c] * 100).round(2)
    df["sharpe"] = df["sharpe"].round(2)
    df["sortino"] = df["sortino"].round(2)
    df["calmar"] = df["calmar"].round(2)
    return df
