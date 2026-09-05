"""Overlays que se aplican DESPUÉS de que cada estrategia genera sus pesos, pero
ANTES de correr el backtest (o de combinarlos en vivo) -- tres mejoras
relacionadas que se componen en el mismo punto del pipeline, en este orden
(ver `apply_weight_overlays`):

  1. `tighten_caps_by_correlation`: cuando el universo entero se mueve muy
     correlacionado (todo sube o baja junto -- poca diversificación real pese a
     tener muchos activos), aprieta los topes de posición por ticker
     (`position_caps`) más allá de lo estático -- fuerza más diversificación
     justo quando más importa. El capital recortado NO se redistribuye a otros
     activos -- queda "libre".

  2. `ramp_in_new_positions`: limita cuánto puede SUBIR el peso de un ticker en
     un solo día -- entra a una posición nueva o creciente en varios días en
     vez de todo de una vez (las bajadas nunca se limitan, siempre instantáneas).
     El capital todavía "sin ramp-ear" ese día también queda libre.

  3. `sweep_idle_cash`: el capital que quedó libre en los pasos anteriores, más
     el que la estrategia ya dejaba sin asignar por su cuenta (menos activos
     "en tendencia" de los que caben, vol-targeting que reduce exposición a
     propósito, o `sector_rotation` cuando nada le gana al cash), por defecto
     ganaba 0% -- ni siquiera la tasa libre de riesgo. Este overlay lo "barre"
     hacia el proxy de cash (BIL en acciones, USDT en cripto) para que gane su
     retorno real en vez de quedar fuera del portafolio ganando nada. Por eso
     va DESPUÉS de los dos anteriores: recoge todo lo que ellos liberaron.

Un cuarto overlay relacionado, `freeze_weights_on_blackout_days` (blackout de
eventos macro), vive en `src/event_blackout.py` mismo pero se compone acá al
final, en `apply_weight_overlays` -- ver ese módulo para los detalles.

Alcance deliberado: NINGUNO de los tres overlays de este archivo se aplica
cuando el filtro de régimen (`src/regime.py`) o la guardia de drawdown
(`src/backtest.py`) reducen exposición -- esa reducción es una decisión de
RIESGO explícita (salir de activos riesgosos ante señales de estrés de
mercado), no una ineficiencia de cash, concentración, o velocidad de entrada.
Mezclarlas asumiría, por ejemplo, que el proxy de cash es "seguro" incluso en
el escenario de estrés que motivó la guardia, lo cual no siempre es cierto
(ej. estrés de liquidez sistémico) -- se deja como una decisión explícita del
filtro de régimen/guardia, no automática.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.event_blackout import freeze_weights_on_blackout_days


def sweep_idle_cash(weights: pd.DataFrame, cash_ticker: str | None) -> pd.DataFrame:
    """Agrega (o incrementa) la columna `cash_ticker` con lo que falte para
    sumar 1.0 cada día, sin tocar el resto de los pesos ni redistribuir nada
    entre ellos. Si `cash_ticker` es `None`, o el remanente de algún día es
    <= 0 (la estrategia ya invirtió el 100% o más), no hace nada ese día --
    nunca resta peso a otros activos ni deja el remanente en negativo."""
    if not cash_ticker:
        return weights
    out = weights.copy()
    existing_cash = out[cash_ticker] if cash_ticker in out.columns else 0.0
    invested_other = out.drop(columns=[cash_ticker], errors="ignore").sum(axis=1)
    idle = (1.0 - invested_other - existing_cash).clip(lower=0.0)
    out[cash_ticker] = existing_cash + idle
    return out


def compute_aggregate_correlation(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Correlación promedio, rodante, entre TODOS los activos de `returns` --
    proxy barato de qué tan "concentrado" está el riesgo del portafolio en un
    solo factor de mercado en vez de estar genuinamente diversificado. Mismo
    proxy que ya usa `src/strategies/momentum.py::_rolling_corr_to_basket`
    (correlación de cada activo contra la canasta equal-weight de los demás),
    ahora agregado a un solo número por día (el promedio entre activos) en vez
    de uno por activo."""
    basket = returns.mean(axis=1)
    min_periods = max(5, window // 2)
    per_asset_corr = pd.DataFrame({
        col: returns[col].rolling(window, min_periods=min_periods).corr(basket) for col in returns.columns
    })
    return per_asset_corr.mean(axis=1)


def correlation_based_cap_scale(avg_corr: pd.Series, full_cap_below: float = 0.3,
                                 floor_above: float = 0.7, min_scale: float = 0.6) -> pd.Series:
    """Traduce la correlación promedio del portafolio (ver
    `compute_aggregate_correlation`) a un multiplicador de los topes de
    posición (`position_caps`): por debajo de `full_cap_below`, sin ajuste
    (multiplicador 1.0 -- activos genuinamente diversificados entre sí); por
    encima de `floor_above`, el multiplicador mínimo (`min_scale`) --
    concentración de riesgo alta, aprieta los topes por posición para forzar
    más diversificación aunque cada activo individualmente parezca atractivo.
    Interpolación lineal entre los dos umbrales, mismo mecanismo que el filtro
    de régimen (`src/regime.py`)."""
    lo, hi = full_cap_below, floor_above
    scale = (hi - avg_corr) / (hi - lo)
    scale = min_scale + scale.clip(lower=0.0, upper=1.0) * (1.0 - min_scale)
    return scale.fillna(1.0)


def tighten_caps_by_correlation(weights: pd.DataFrame, position_caps: dict[str, float],
                                 cap_scale: pd.Series) -> pd.DataFrame:
    """Recorta (nunca aumenta) cada peso al tope original (`position_caps`)
    multiplicado por `cap_scale` ese día -- el remanente recortado NO se
    redistribuye a otros activos (ver `sweep_idle_cash` para qué pasa con eso
    después). Tickers sin tope conocido en `position_caps` no se tocan."""
    cols = [c for c in weights.columns if c in position_caps]
    if not cols:
        return weights
    cap_row = pd.Series({c: position_caps[c] for c in cols})
    scale_aligned = cap_scale.reindex(weights.index).fillna(1.0)
    dynamic_cap = pd.DataFrame(
        np.outer(scale_aligned.to_numpy(), cap_row.to_numpy()), index=weights.index, columns=cols,
    )
    out = weights.copy()
    out[cols] = out[cols].clip(upper=dynamic_cap)
    return out


DEFAULT_RAMP_MAX_DAILY_INCREASE = 0.02  # 2 puntos porcentuales por día


def ramp_in_new_positions(weights: pd.DataFrame, max_daily_increase: float = DEFAULT_RAMP_MAX_DAILY_INCREASE,
                           cash_ticker: str | None = None) -> pd.DataFrame:
    """Limita cuánto puede SUBIR el peso de un ticker en un solo día
    (`max_daily_increase`, en puntos porcentuales -- default 2%) -- entra a una
    posición nueva o creciente en varios días en vez de todo de una vez,
    reduciendo el riesgo de "comprar justo el techo" de un movimiento de corto
    plazo. Las BAJADAS de peso (salir o reducir una posición) NUNCA se limitan
    -- salir de una posición de riesgo es siempre instantáneo, solo ENTRAR se
    escalona (mismo principio que la reentrada gradual de la guardia de
    drawdown en `src/backtest.py`: bajar riesgo rápido, subirlo despacio).

    `cash_ticker` (si se pasa) queda completamente excluido de este límite --
    moverse HACIA cash es una acción de seguridad, nunca debería frenarse (ver
    `sweep_idle_cash`, que agrega esa columna DESPUÉS de este overlay, así que
    normalmente ni siquiera está presente todavía -- el parámetro existe por
    si `cash_ticker` ya trae peso propio, ej. la salida a cash de
    `sector_rotation.py`).

    `max_daily_increase <= 0` (o `None`) desactiva el límite por completo."""
    if not max_daily_increase or max_daily_increase <= 0:
        return weights
    cols = [c for c in weights.columns if c != cash_ticker]
    if not cols:
        return weights

    target = weights[cols].to_numpy(dtype=float)
    n, k = target.shape
    out = np.empty_like(target)
    prev = np.zeros(k)
    for i in range(n):
        row_target = target[i]
        capped = np.minimum(row_target, prev + max_daily_increase)
        row_out = np.where(row_target >= prev, capped, row_target)
        out[i] = row_out
        prev = row_out

    result = weights.copy()
    result[cols] = out
    return result


def apply_weight_overlays(weights_fn: Callable[..., pd.DataFrame], cash_ticker: str | None,
                           position_caps: dict[str, float] | None, cap_scale: pd.Series | None,
                           blackout_dates: set | None, ramp_max_daily_increase: float | None,
                           *args, **kwargs) -> pd.DataFrame:
    """Compone `tighten_caps_by_correlation` + `ramp_in_new_positions` +
    `sweep_idle_cash` + `freeze_weights_on_blackout_days` (ver
    `src/event_blackout.py`) sobre el resultado de `weights_fn(*args, **kwargs)`
    -- pensada para usarse con `functools.partial` en el walk-forward paralelo
    (`src/walk_forward.py`), que exige funciones "picklables" (nada de lambdas
    ni closures anidadas, ver el comentario en `scripts/run_backtest.py`).
    `position_caps`/`cap_scale` en `None` desactiva el ajuste de topes
    dinámico; `blackout_dates` vacío o `None` desactiva el blackout;
    `ramp_max_daily_increase` en `None` o <= 0 desactiva la entrada
    escalonada -- ninguno de los tres afecta al barrido de cash (que solo
    depende de `cash_ticker`)."""
    w = weights_fn(*args, **kwargs)
    if position_caps is not None and cap_scale is not None:
        w = tighten_caps_by_correlation(w, position_caps, cap_scale)
    if ramp_max_daily_increase is not None and ramp_max_daily_increase > 0:
        w = ramp_in_new_positions(w, max_daily_increase=ramp_max_daily_increase, cash_ticker=cash_ticker)
    w = sweep_idle_cash(w, cash_ticker)
    return freeze_weights_on_blackout_days(w, blackout_dates or set())
