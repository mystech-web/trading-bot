"""Estrategia de momentum/tendencia: cruce de medias móviles por activo,
CONFIRMADO por momentum MULTI-HORIZONTE, con tamaño de posición por PARIDAD DE
RIESGO (inverse-vol), AJUSTADA POR CORRELACIÓN, en vez de peso igual -- luego
escalado por volatilidad objetivo del portafolio completo.

Regla base: activo está "en tendencia" si SMA_fast > SMA_slow -- pero un solo
cruce de medias es sensible a "whipsaws" (falsas señales cuando el precio
oscila justo alrededor del cruce). Para filtrar ruido, se exige además que el
momentum esté mayormente de acuerdo en VARIOS horizontes (por defecto ~1, 3, 6
y 12 meses): si al menos `multi_horizon_agreement` de esos horizontes muestran
retorno positivo, se confirma la tendencia; si no, aunque el cruce de medias
diga que sí, no se entra -- un cruce que no viene acompañado de momentum real
en la mayoría de los plazos suele ser ruido de corto plazo, no una tendencia
real. `use_multi_horizon=False` recupera el comportamiento anterior (solo el
cruce de medias).

Entre los activos en tendencia (confirmada) se reparte peso INVERSAMENTE
PROPORCIONAL a su volatilidad reciente (un activo el doble de volátil que otro
recibe la mitad de peso relativo) -- así ningún activo ruidoso domina el
riesgo del portafolio solo por estar "en tendencia".

Paridad de riesgo pura (solo por volatilidad) subestima el riesgo real cuando
los activos "en tendencia" están muy correlacionados entre sí -- ej. si 5 ETFs
sectoriales suben todos juntos en un rally amplio, pesarlos por vol individual
ignora que en la práctica se mueven como un solo activo grande (poco
diversificado). Para corregirlo, se calcula la correlación rodante de cada
activo contra la canasta equal-weight del universo (proxy barato de "qué tan
redundante es este activo" -- calcular la matriz de correlación N x N completa
sería más preciso pero mucho más caro sin cambiar la conclusión práctica) y se
usa para AMORTIGUAR el peso de activos muy correlacionados (redundantes) y
premiar levemente a los que diversifican (correlación negativa o baja).

Los pesos resultantes se capan por posición (ETFs con un tope más alto que
acciones individuales, para no concentrar en nombres con riesgo de
supervivencia -- ver `max_weight_by_ticker`), y luego se escala la exposición
total para apuntar a una volatilidad anualizada objetivo.
"""
import numpy as np
import pandas as pd

from src.indicators import sma, realized_vol

DEFAULT_PARAMS = dict(
    fast=50,
    slow=200,
    max_weight_per_asset=0.20,
    vol_target=0.10,       # 10% anualizado de volatilidad objetivo del portafolio
    vol_lookback=20,
    max_gross_exposure=1.0,
    periods_per_year=252,   # 252 (días hábiles, acciones) o 365 (cripto, cotiza todos los días)
    corr_lookback=60,      # ventana rodante para estimar correlación de cada activo vs. la canasta
    corr_dampening=0.5,    # 0 = paridad de riesgo pura (sin ajuste); más alto = penaliza más la redundancia
    use_multi_horizon=True,
    # ~1, 3, 6 y 12 meses en días HÁBILES (acciones) -- para cripto (cotiza todos
    # los días) usa días de calendario, ej. [30, 91, 182, 365] (ver crypto_live_params.yaml).
    multi_horizon_windows=(21, 63, 126, 252),
    multi_horizon_agreement=0.75,  # fracción de horizontes que deben tener retorno positivo para confirmar
)


def _multi_horizon_confirmation(sub: pd.DataFrame, windows: tuple, agreement: float) -> pd.DataFrame:
    """True el día/activo donde al menos `agreement` de los horizontes en
    `windows` tienen retorno acumulado positivo -- un cruce de medias que no
    viene acompañado de momentum real en la mayoría de los plazos suele ser
    ruido, no tendencia real. Durante el "calentamiento" (sin historia
    suficiente para un horizonte todavía) ese horizonte cuenta como "no
    confirma" -- falla del lado conservador, nunca genera una señal falsa por
    falta de datos."""
    votes = None
    for w in windows:
        mom = sub / sub.shift(w) - 1.0
        vote = (mom > 0).astype(float)
        votes = vote if votes is None else votes + vote
    frac = votes / len(windows)
    return frac >= agreement


def _rolling_corr_to_basket(rets: pd.DataFrame, window: int) -> pd.DataFrame:
    """Correlación rodante de cada columna contra el promedio equal-weight de
    TODAS las columnas (la "canasta") -- barato de calcular (una pasada por
    columna, no una matriz N x N por día) y suficiente como proxy de qué tan
    redundante es cada activo frente al resto del universo en ese momento."""
    basket = rets.mean(axis=1)
    min_periods = max(5, window // 2)
    corr = {col: rets[col].rolling(window, min_periods=min_periods).corr(basket) for col in rets.columns}
    return pd.DataFrame(corr, index=rets.index)


def generate_weights(close: pd.DataFrame, tickers: list[str], params: dict | None = None,
                      max_weight_by_ticker: dict[str, float] | None = None) -> pd.DataFrame:
    p = {**DEFAULT_PARAMS, **(params or {})}
    sub = close[tickers]
    rets = sub.pct_change()

    fast_ma = sub.apply(lambda s: sma(s, p["fast"]))
    slow_ma = sub.apply(lambda s: sma(s, p["slow"]))
    in_trend = fast_ma > slow_ma
    if p["use_multi_horizon"]:
        in_trend = in_trend & _multi_horizon_confirmation(sub, p["multi_horizon_windows"],
                                                            p["multi_horizon_agreement"])
    in_trend = in_trend.astype(float)

    asset_vol = rets.rolling(p["vol_lookback"], min_periods=p["vol_lookback"]).std()
    inv_vol = (1.0 / asset_vol.replace(0, np.nan)).where(in_trend.astype(bool))

    if p["corr_dampening"] > 0:
        corr_to_basket = _rolling_corr_to_basket(rets, p["corr_lookback"]).clip(lower=-1.0, upper=1.0)
        # penalty > 1 amortigua (activo redundante, muy correlacionado con la canasta);
        # penalty < 1 premia (activo diversificador, correlación baja o negativa). El piso
        # evita que una correlación muy negativa dispare el peso de forma extrema.
        penalty = (1.0 + p["corr_dampening"] * corr_to_basket).clip(lower=0.2)
        inv_vol = inv_vol / penalty

    inv_vol_sum = inv_vol.sum(axis=1).replace(0, np.nan)
    risk_parity_weight = inv_vol.div(inv_vol_sum, axis=0).fillna(0.0)

    if max_weight_by_ticker:
        cap = pd.Series({t: max_weight_by_ticker.get(t, p["max_weight_per_asset"]) for t in tickers})
    else:
        cap = pd.Series(p["max_weight_per_asset"], index=tickers)
    raw_weights = risk_parity_weight.clip(upper=cap, axis=1)

    # Volatilidad realizada del portafolio "sin escalar" para dimensionar la exposición total.
    port_rets_unscaled = (raw_weights.shift(1) * rets).sum(axis=1)
    realized = realized_vol(port_rets_unscaled, window=p["vol_lookback"], periods_per_year=p["periods_per_year"])
    scale = (p["vol_target"] / realized).clip(upper=p["max_gross_exposure"])
    scale = scale.fillna(0.0)

    weights = raw_weights.mul(scale, axis=0)
    weights = weights.clip(lower=0.0)
    return weights
