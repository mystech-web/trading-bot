"""Motor de backtest diario, vectorizado salvo por el overlay de riesgo (que es secuencial
por naturaleza: depende del drawdown acumulado hasta ese día).

Reglas clave para que el resultado sea realista y no "de laboratorio":
  1. Los pesos se aplican con 1 día de rezago (`shift(1)`) -> la señal calculada con el
     cierre de hoy se ejecuta en el retorno de mañana. Sin esto hay look-ahead bias.
  2. Costos de transacción proporcionales al turnover (comisión + slippage estimado).
  3. Guardia de drawdown a nivel portafolio: si el drawdown acumulado cruza un umbral,
     se reduce la exposición a la mitad hasta que el drawdown se recupere -> esto es lo
     que evita que una mala racha te deje fuera del objetivo mensual por meses. La
     REACTIVACIÓN de exposición completa es GRADUAL (`guard_ramp_days`, default 5), no
     instantánea: una vez que se cumplen las condiciones de recuperación, la exposición
     sube en pasos iguales día a día hasta volver a 100%, en vez de saltar de golpe de
     50% a 100% el mismo día que se "libera" la guardia -- eso evita quedar totalmente
     expuesto de nuevo justo cuando la recuperación todavía podría ser un rebote falso
     dentro de una racha volátil. `guard_ramp_days=0` recupera el salto instantáneo.
  4. Guardia PROACTIVA por aceleración de volatilidad: la guardia de drawdown de arriba
     reacciona DESPUÉS de que el drawdown ya ocurrió -- para cuando cruza -15%, el daño
     ya está hecho. Esta guardia mira la volatilidad realizada de corto plazo del
     portafolio (últimos `vol_accel_short_window` días) contra su nivel "normal"
     (`vol_accel_long_window` días) -- un salto brusco (ej. 2x) suele preceder a los
     drawdowns grandes (así empiezan la mayoría de los crashes: la volatilidad se
     dispara ANTES de que el precio caiga en serio), así que reduce exposición un poco
     antes en vez de solo después.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.costs import liquidity_adjusted_cost, average_dollar_volume


def run_backtest(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = 5.0,
    volume: pd.DataFrame | None = None,
    use_liquidity_costs: bool = False,
    liquidity_cost_kwargs: dict | None = None,
    regime_scale: pd.Series | None = None,
    dd_guard_threshold: float = -0.15,
    dd_guard_recover: float = -0.07,
    dd_guard_scale: float = 0.5,
    guard_ramp_days: int = 5,
    vol_accel_enabled: bool = True,
    vol_accel_short_window: int = 5,
    vol_accel_long_window: int = 60,
    vol_accel_threshold: float = 2.0,
    vol_accel_reset: float = 1.2,
) -> dict:
    """`use_liquidity_costs=True` (con `volume` provisto) reemplaza el costo plano
    `cost_bps` por un modelo sensible a liquidez (ver `src/costs.py`): más caro
    mover una posición grande en un activo poco líquido, más barato en uno muy
    líquido como SPY. `regime_scale` (opcional, ver `src/regime.py`) multiplica
    los pesos ANTES del rezago de 1 día -- es el filtro macro proactivo, que baja
    exposición antes de que el drawdown reactivo tenga que hacerlo.

    `vol_accel_*`: guardia proactiva adicional (ver docstring del módulo). Se
    activa (reduce exposición a `dd_guard_scale`) si el drawdown cruza
    `dd_guard_threshold` O si la razón vol_corta/vol_larga supera
    `vol_accel_threshold` -- lo que ocurra primero. Se libera solo cuando AMBAS
    condiciones se normalizan (drawdown >= `dd_guard_recover` Y razón de
    volatilidad < `vol_accel_reset`) -- exige que el riesgo baje en ambos
    frentes antes de volver a exposición completa, no solo en uno.
    `vol_accel_enabled=False` recupera el comportamiento anterior (solo drawdown).

    `guard_ramp_days`: una vez que se cumplen las condiciones de liberación, la
    exposición sube en pasos iguales de `(1.0 - dd_guard_scale) / guard_ramp_days`
    por día hasta volver a 1.0, en vez de saltar directo. Si las condiciones dejan
    de cumplirse a mitad de la rampa, se PAUSA (histéresis, mantiene el `scale`
    actual) hasta que se vuelvan a cumplir -- no retrocede, pero tampoco avanza. Si
    la guardia se vuelve a activar (nuevo cruce de umbral) en cualquier punto de la
    rampa, `scale` vuelve de golpe a `dd_guard_scale`, sin conservar el progreso de
    la rampa anterior. `guard_ramp_days=0` recupera el salto instantáneo."""
    tickers = [c for c in weights.columns if c in close.columns]
    close = close[tickers].copy()
    weights = weights[tickers].reindex(close.index).fillna(0.0)

    if regime_scale is not None:
        weights = weights.mul(regime_scale.reindex(weights.index).fillna(1.0), axis=0)

    asset_returns = close.pct_change().fillna(0.0)
    weights_shifted = weights.shift(1).fillna(0.0)  # decisión de hoy se ejecuta mañana
    delta_weights = weights_shifted.diff().fillna(weights_shifted)

    turnover = delta_weights.abs().sum(axis=1)
    if use_liquidity_costs and volume is not None:
        adv = average_dollar_volume(close, volume)
        cost = liquidity_adjusted_cost(delta_weights, adv, **(liquidity_cost_kwargs or {}))
        cost = cost.reindex(close.index).fillna(0.0)
    else:
        cost = turnover * (cost_bps / 10_000.0)

    # Razón de volatilidad realizada corta/larga, causal (el valor del día i solo usa
    # retornos hasta el día i, nunca futuros) -- se calcula sobre el retorno "gross" ya
    # rezagado un día (weights_shifted), así que no agrega ninguna fuga de información
    # adicional más allá del rezago que ya tiene todo el backtest.
    gross_returns = (weights_shifted * asset_returns).sum(axis=1)
    if vol_accel_enabled:
        short_vol = gross_returns.rolling(vol_accel_short_window, min_periods=vol_accel_short_window).std()
        long_vol = gross_returns.rolling(vol_accel_long_window, min_periods=vol_accel_long_window).std()
        vol_ratio = (short_vol / long_vol.replace(0, np.nan)).fillna(1.0)
        vol_ratio_arr = vol_ratio.to_numpy()
    else:
        vol_ratio_arr = np.ones(len(close.index))

    n = len(close.index)
    equity = np.empty(n)
    port_ret = np.empty(n)
    exposure_scale = np.empty(n)

    eq = 1.0
    peak = 1.0
    scale = 1.0

    w_arr = weights_shifted.to_numpy()
    r_arr = asset_returns.to_numpy()
    cost_arr = cost.to_numpy()

    for i in range(n):
        gross = float(np.dot(w_arr[i], r_arr[i]))
        ret = gross * scale - cost_arr[i]
        eq *= (1.0 + ret)
        peak = max(peak, eq)
        dd = eq / peak - 1.0

        port_ret[i] = ret
        equity[i] = eq
        exposure_scale[i] = scale

        vol_spiked = vol_accel_enabled and vol_ratio_arr[i] >= vol_accel_threshold
        vol_normalized = (not vol_accel_enabled) or vol_ratio_arr[i] < vol_accel_reset
        if dd <= dd_guard_threshold or vol_spiked:
            scale = dd_guard_scale
        elif dd >= dd_guard_recover and vol_normalized:
            if guard_ramp_days > 0:
                step = (1.0 - dd_guard_scale) / guard_ramp_days
                scale = min(1.0, scale + step)
            else:
                scale = 1.0
        # si no se cumple ninguna de las dos, mantiene el scale actual (histéresis) --
        # incluye el caso de una rampa a medio camino: se pausa, no retrocede ni avanza.

    idx = close.index
    return dict(
        returns=pd.Series(port_ret, index=idx, name="return"),
        equity=pd.Series(equity, index=idx, name="equity"),
        exposure_scale=pd.Series(exposure_scale, index=idx, name="exposure_scale"),
        turnover=turnover,
        weights_used=weights_shifted,
    )
