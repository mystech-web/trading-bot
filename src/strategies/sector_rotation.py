"""Rotación sectorial con "dual momentum" (Antonacci simplificado):

  - Rebalanceo mensual (fin de mes).
  - Momentum absoluto (filtro de cash): un sector solo CALIFICA si su momentum
    supera al del proxy de cash/T-Bills (BIL) -> si nada le gana al cash, el
    portafolio queda en cash esa ventana. Esto es lo que recorta los grandes
    drawdowns de las estrategias de rotación pura.
  - Momentum relativo AJUSTADO A RIESGO (`risk_adjusted_ranking=True`, default):
    entre los sectores que califican, se ordenan por momentum/volatilidad
    (una razón tipo Sharpe, ver `vol_lookback`), no por momentum crudo -- rankear
    por retorno puro favorece sistemáticamente a sectores más volátiles (suben
    más rápido, pero también caen más rápido), y termina persiguiendo el sector
    más "ruidoso" del momento en vez del que mejor recompensa el riesgo que
    toma. `risk_adjusted_ranking=False` recupera el ranking por momentum crudo
    (ver `src/indicators.py::momentum_n_m` -- 252/21 días hábiles para
    acciones, 365/30 días de calendario para cripto).
  - Se sostienen los `top_n` mejores sectores que pasan el filtro, con peso
    AJUSTADO POR CORRELACIÓN entre ellos (mismo principio que
    `src/strategies/momentum.py`): si dos de los sectores elegidos suelen
    moverse juntos (ej. XLK y XLY en un rally amplio), son redundantes entre sí
    -- se les amortigua el peso relativo a favor del que más diversifica dentro
    de la canasta elegida. Peso igual (`1/top_n`) es el caso particular cuando
    `corr_dampening=0`.
"""
import numpy as np
import pandas as pd

from src.indicators import momentum_n_m

DEFAULT_PARAMS = dict(
    top_n=3,
    max_weight_per_asset=0.40,
    momentum_long=252,
    momentum_short=21,
    rebalance_rule="BME",  # "BME" (fin de mes hábil) para acciones, "ME" (fin de mes calendario) para cripto
    corr_lookback=60,      # ventana (días) para estimar correlación entre los sectores elegidos
    corr_dampening=0.5,    # 0 = peso igual puro (sin ajuste); más alto = penaliza más la redundancia
    risk_adjusted_ranking=True,  # rankear por momentum/volatilidad en vez de momentum crudo
    vol_lookback=60,             # ventana (días) de volatilidad usada en el ranking ajustado a riesgo
)


def _correlation_weights(rets: pd.DataFrame, as_of, picks: list[str], lookback: int, dampening: float) -> pd.Series:
    """Pesos (suman 1) entre los `picks` elegidos ese rebalanceo, ponderados
    inversamente a qué tan correlacionado está cada uno con los DEMÁS picks
    (no con el universo completo -- lo que importa es la redundancia DENTRO de
    la canasta ya elegida). Con un solo pick, o sin suficiente historia, cae de
    vuelta a peso igual."""
    n = len(picks)
    if n <= 1 or dampening <= 0:
        return pd.Series(1.0 / n, index=picks)

    hist = rets.loc[:as_of, picks].tail(lookback)
    if len(hist) < 5:
        return pd.Series(1.0 / n, index=picks)

    corr_mat = hist.corr()
    raw = {}
    for t in picks:
        others = [o for o in picks if o != t]
        avg_corr = corr_mat.loc[t, others].mean() if others else 0.0
        avg_corr = 0.0 if pd.isna(avg_corr) else max(-1.0, min(1.0, avg_corr))
        penalty = max(1.0 + dampening * avg_corr, 0.2)
        raw[t] = 1.0 / penalty
    total = sum(raw.values())
    return pd.Series({t: v / total for t, v in raw.items()}) if total > 0 else pd.Series(1.0 / n, index=picks)


def generate_weights(close: pd.DataFrame, sector_tickers: list[str], cash_ticker: str,
                      params: dict | None = None) -> pd.DataFrame:
    p = {**DEFAULT_PARAMS, **(params or {})}
    all_cols = sector_tickers + ([cash_ticker] if cash_ticker not in sector_tickers else [])
    sub = close[all_cols]
    rets = sub.pct_change()

    mom = sub.apply(lambda s: momentum_n_m(s, p["momentum_long"], p["momentum_short"]))
    vol_min_periods = max(5, p["vol_lookback"] // 2)
    vol = rets.rolling(p["vol_lookback"], min_periods=vol_min_periods).std()

    rebal_dates = sub.resample(p["rebalance_rule"]).last().index
    rebal_dates = [d for d in rebal_dates if d in mom.index] or \
        [mom.index[mom.index.get_indexer([d], method="nearest")[0]] for d in rebal_dates if len(mom.index)]

    rebal_rows = {}
    for d in rebal_dates:
        if d not in mom.index:
            continue
        row = mom.loc[d, sector_tickers].dropna()
        cash_mom = mom.loc[d, cash_ticker] if cash_ticker in mom.columns else 0.0
        target = pd.Series(0.0, index=sector_tickers)
        if not row.empty:
            qualifying = row[row > cash_mom]  # filtro de momentum ABSOLUTO -- sin cambios, sigue siendo binario
            if not qualifying.empty:
                if p["risk_adjusted_ranking"]:
                    vol_row = vol.loc[d, qualifying.index].replace(0, np.nan)
                    # activos sin suficiente historia de volatilidad (NaN, calentamiento) se
                    # excluyen del ranking ajustado -- conservador, nunca elige "a ciegas" un
                    # activo cuyo riesgo todavía no se puede medir.
                    rank_score = (qualifying / vol_row).dropna()
                else:
                    rank_score = qualifying
                picks_index = rank_score.sort_values(ascending=False).head(p["top_n"]).index
                if len(picks_index) > 0:
                    pick_weights = _correlation_weights(rets, d, list(picks_index), p["corr_lookback"],
                                                          p["corr_dampening"])
                    # el cap por posición se aplica SIN redistribuir el sobrante a los demás
                    # picks -- si se recorta, ese remanente queda implícitamente en cash, igual
                    # que se comportaba el peso igual original cuando 1/top_n > max_weight_per_asset.
                    target.loc[picks_index] = pick_weights.clip(upper=p["max_weight_per_asset"])
        rebal_rows[d] = target

    # DataFrame solo con las filas de rebalanceo (incluye explícitamente los rebalanceos
    # que resultaron en 100% cash, es decir, todo en cero) y luego se propaga hacia
    # adelante (buy & hold) hasta el siguiente rebalanceo.
    rebal_df = pd.DataFrame(rebal_rows).T
    rebal_df.index.name = "date"
    weights = rebal_df.reindex(sub.index).ffill().fillna(0.0)
    return weights
