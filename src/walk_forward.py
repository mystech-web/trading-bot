"""Validación walk-forward: la única forma honesta de estimar el retorno esperado.

Un backtest normal optimiza parámetros sobre TODO el histórico y los evalúa sobre
ESE MISMO histórico -> sobreajuste garantizado. Walk-forward en cambio:

  1. Divide el histórico en ventanas rodantes de train/test.
  2. En cada ventana, elige los mejores parámetros SOLO con datos de train.
  3. Aplica esos parámetros (sin volver a tocarlos) al período de test, que el
     optimizador nunca vio.
  4. Concatena todos los tramos de test (que son contiguos y no se solapan) para
     formar una curva de equity "fuera de muestra" que es la mejor estimación
     realista de cómo se hubiera comportado la estrategia operándose en vivo.
"""
from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.backtest import run_backtest
from src.metrics import calmar, sharpe


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_folds(index: pd.DatetimeIndex, analysis_start: pd.Timestamp,
                 train_years: float = 3, test_years: float = 1, step_years: float = 1,
                 embargo_days: int = 5) -> list[Fold]:
    """`embargo_days` deja un hueco (en días de calendario) entre el fin de train y
    el inicio de test. No hay fuga de información hacia adelante en estas estrategias
    (los indicadores solo miran hacia atrás), pero sin el embargo el último tramo de
    train y el primer tramo de test pueden compartir el mismo evento de mercado
    (ej. una racha que cruza justo el límite), inflando la aparente independencia
    entre "elegí los parámetros aquí" y "los probé aquí". El costo es perder unos
    pocos días de cobertura OOS en cada fold -- vale la pena por el rigor."""
    end = index.max()
    folds = []
    train_start = pd.Timestamp(analysis_start)
    while True:
        train_end = train_start + pd.DateOffset(months=int(train_years * 12))
        test_start = train_end + pd.Timedelta(days=embargo_days)
        test_end = test_start + pd.DateOffset(months=int(test_years * 12))
        if test_end > end:
            break
        folds.append(Fold(train_start, train_end, test_start, test_end))
        train_start = train_start + pd.DateOffset(months=int(step_years * 12))
    return folds


def _score(returns: pd.Series, periods_per_year: int = 252) -> float:
    c = calmar(returns, periods_per_year)
    s = sharpe(returns, periods_per_year=periods_per_year)
    if c != c:  # NaN check
        c = -10.0
    if s != s:
        s = -10.0
    return c + 0.1 * s


def _param_key(params: dict) -> tuple:
    return tuple(sorted(params.items()))


def _precompute_weights(weights_fn: Callable[[dict], pd.DataFrame], param_grid: list[dict],
                         n_jobs: int) -> dict[tuple, pd.DataFrame]:
    """Calcula el DataFrame de pesos de cada combinación ÚNICA de parámetros del
    grid -- una sola vez cada una (no una vez por fold), y en paralelo si
    `n_jobs > 1`. Es el paso caro para mean_reversion (loop por ticker) y el que
    más se beneficia de repartirse entre núcleos.

    `weights_fn` debe ser "picklable" para que esto funcione con n_jobs>1 -- es
    decir, `functools.partial(alguna_función_de_módulo, argumentos_fijos...)`,
    NUNCA un lambda ni una función anidada (esas no se pueden enviar a otro
    proceso). Ver `scripts/run_backtest.py` para el patrón correcto.
    """
    unique = {_param_key(p): p for p in param_grid}

    if n_jobs <= 1 or len(unique) <= 1:
        return {key: weights_fn(p) for key, p in unique.items()}

    # Contexto "spawn" explícito: es el default en macOS/Windows y no permite
    # lambdas ni closures (fork sí, por eso probar solo en Linux con el default
    # ocultaría este tipo de bug hasta correrlo en un Mac).
    ctx = multiprocessing.get_context("spawn")
    results = {}
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as executor:
        futures = {executor.submit(weights_fn, p): key for key, p in unique.items()}
        for future in futures:
            key = futures[future]
            results[key] = future.result()
    return results


def walk_forward_backtest(
    close: pd.DataFrame,
    weights_fn: Callable[[dict], pd.DataFrame],
    param_grid: list[dict],
    folds: list[Fold],
    backtest_kwargs: dict | None = None,
    periods_per_year: int = 252,
    n_jobs: int = 1,
) -> dict:
    backtest_kwargs = backtest_kwargs or {}
    oos_returns_segments = []
    fold_reports = []

    weights_cache = _precompute_weights(weights_fn, param_grid, n_jobs)
    bt_cache: dict[tuple, dict] = {}  # (params_key, fold.test_end) -> resultado de run_backtest

    def get_bt(params_key, weights, fold, full_close):
        cache_key = (params_key, fold.test_end)
        if cache_key not in bt_cache:
            bt_cache[cache_key] = run_backtest(full_close, weights.loc[:fold.test_end], **backtest_kwargs)
        return bt_cache[cache_key]

    for fold in folds:
        best_score = -1e9
        best_params = None
        best_key = None
        full_close = close.loc[:fold.test_end]

        for params in param_grid:
            key = _param_key(params)
            weights = weights_cache[key]
            bt = get_bt(key, weights, fold, full_close)
            train_rets = bt["returns"].loc[fold.train_start:fold.train_end]
            score = _score(train_rets, periods_per_year)
            if score > best_score:
                best_score = score
                best_params = params
                best_key = key

        # Reusa el resultado ya calculado arriba para `best_key` en este fold --
        # antes se volvía a correr run_backtest acá, duplicando el cómputo del
        # ganador (una vez para el score de train, otra para el tramo de test).
        bt = bt_cache[(best_key, fold.test_end)]
        test_rets = bt["returns"].loc[fold.test_start:fold.test_end]
        oos_returns_segments.append(test_rets)
        fold_reports.append(dict(
            train_start=fold.train_start, train_end=fold.train_end,
            test_start=fold.test_start, test_end=fold.test_end,
            best_params=best_params, train_score=best_score,
        ))

    oos_returns = pd.concat(oos_returns_segments).sort_index()
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    return dict(returns=oos_returns, fold_reports=fold_reports)


def param_stability_table(fold_reports: list[dict]) -> pd.DataFrame:
    """Qué parámetros ganó cada fold, uno por fila -- para inspeccionar a ojo si
    saltan de un extremo a otro (mala señal) o son consistentes (buena señal)."""
    rows = []
    for i, fr in enumerate(fold_reports):
        row = {
            "fold": i,
            "test_start": fr["test_start"].date(),
            "test_end": fr["test_end"].date(),
            "train_score": round(fr["train_score"], 3),
        }
        row.update(fr["best_params"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("fold")


def param_stability_score(fold_reports: list[dict]) -> dict[str, float]:
    """Por cada parámetro: fracción de folds que eligió el valor más frecuente (moda).
    1.0 = siempre el mismo valor (estable). Cerca de 1/n_valores_posibles = el óptimo
    salta de fold a fold -> señal de que el "mejor" parámetro es solo ruido de ese
    tramo de historia, no algo real -> baja la confianza en el resultado OOS."""
    param_names: set[str] = set()
    for fr in fold_reports:
        param_names |= set(fr["best_params"].keys())

    scores = {}
    for name in param_names:
        values = [fr["best_params"].get(name) for fr in fold_reports]
        if not values:
            continue
        mode_count = max(values.count(v) for v in set(values))
        scores[name] = round(mode_count / len(values), 2)
    return scores
