"""Simulación Monte Carlo por block bootstrap sobre los retornos out-of-sample.

El walk-forward te da UN número de retorno mensual promedio. Pero ese número
salió de una sola trayectoria histórica -- si la historia hubiera ocurrido en
otro orden (misma distribución de retornos, distinta secuencia), el resultado
pudo haber sido mejor o peor. El block bootstrap remuestrea bloques de ~1 mes
de retornos (con reemplazo, preservando la autocorrelación/clustering de
volatilidad dentro de cada bloque) para generar cientos de historias
alternativas igual de plausibles, y así reportar un RANGO (percentiles) en vez
de un solo punto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import cagr, max_drawdown, monthly_returns

TARGET_LOW = 0.005
TARGET_HIGH = 0.02


def block_bootstrap_paths(returns: pd.Series, n_sims: int = 1000, block_size: int = 21,
                           seed: int | None = 7) -> list[pd.Series]:
    rng = np.random.default_rng(seed)
    values = returns.dropna().to_numpy()
    n = len(values)
    if n < block_size:
        block_size = max(1, n)
    n_blocks = int(np.ceil(n / block_size))

    paths = []
    for _ in range(n_sims):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sim = np.concatenate([values[s:s + block_size] for s in starts])[:n]
        paths.append(pd.Series(sim, index=pd.bdate_range("2000-01-03", periods=len(sim))))
    return paths


def _percentiles(arr: np.ndarray) -> dict:
    return {f"p{q}": float(np.nanpercentile(arr, q)) for q in (5, 25, 50, 75, 95)}


def monte_carlo_summary(returns: pd.Series, n_sims: int = 1000, block_size: int = 21,
                         seed: int | None = 7, periods_per_year: int = 252) -> dict:
    paths = block_bootstrap_paths(returns, n_sims=n_sims, block_size=block_size, seed=seed)

    avg_monthly = np.array([monthly_returns(p).mean() for p in paths])
    cagrs = np.array([cagr(p, periods_per_year) for p in paths])
    max_dds = np.array([max_drawdown(p) for p in paths])

    prob_in_target = float(((avg_monthly >= TARGET_LOW) & (avg_monthly <= TARGET_HIGH)).mean())
    prob_negative_month_avg = float((avg_monthly < 0).mean())

    return {
        "n_sims": n_sims,
        "block_size_days": block_size,
        "avg_monthly_return": _percentiles(avg_monthly),
        "cagr": _percentiles(cagrs),
        "max_drawdown": _percentiles(max_dds),
        "prob_avg_monthly_in_target_0.5_2pct": prob_in_target,
        "prob_avg_monthly_negative": prob_negative_month_avg,
        "_raw_avg_monthly": avg_monthly,  # para graficar el histograma, no serializar tal cual a JSON
    }


def summary_for_json(summary: dict) -> dict:
    """Copia del summary sin el array crudo, lista para json.dump."""
    return {k: v for k, v in summary.items() if not k.startswith("_")}


def project_forward(returns: pd.Series, months: int = 24, n_sims: int = 500, block_size: int = 21,
                     start_capital: float = 10_000.0, seed: int | None = 11, days_per_month: int = 21) -> pd.DataFrame:
    """Simula `months` meses hacia adelante por block bootstrap de los retornos
    diarios históricos (OOS), partiendo de `start_capital`. Devuelve percentiles
    (p5/p25/p50/p75/p95) del capital proyectado, mes a mes -- un "cono de
    incertidumbre" para mostrar en el dashboard, no una única línea de
    proyección (que sería falsamente precisa). `days_per_month`: ~21 días
    hábiles/mes para acciones, ~30 días de calendario/mes para cripto (cotiza
    todos los días)."""
    rng = np.random.default_rng(seed)
    total_days = months * days_per_month

    values = returns.dropna().to_numpy()
    n = len(values)
    if n < block_size:
        block_size = max(1, n)
    n_blocks = int(np.ceil(total_days / block_size))

    paths = np.empty((n_sims, months + 1))
    paths[:, 0] = start_capital
    for s in range(n_sims):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sim_rets = np.concatenate([values[st:st + block_size] for st in starts])[:total_days]
        capital = start_capital
        for m in range(months):
            month_rets = sim_rets[m * days_per_month:(m + 1) * days_per_month]
            capital *= float(np.prod(1 + month_rets))
            paths[s, m + 1] = capital

    pct = {f"p{q}": np.percentile(paths, q, axis=0) for q in (5, 25, 50, 75, 95)}
    df = pd.DataFrame(pct, index=range(months + 1))
    df.index.name = "month"
    return df
