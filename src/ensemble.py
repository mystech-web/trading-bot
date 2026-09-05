"""Combina estrategias en un solo portafolio, repartiendo capital entre ellas.

Asignación por defecto (ajustable): favorece un poco la rotación sectorial porque
en el walk-forward suele ser la de menor drawdown, y reparte el resto entre
momentum (capta tendencias fuertes) y reversión a la media (capta rebotes, no
correlacionada con las otras dos).
"""
import pandas as pd

from src.indicators import realized_vol

DEFAULT_ALLOCATION = dict(
    momentum=0.35,
    mean_reversion=0.25,
    sector_rotation=0.40,
)


def combine_returns(returns_by_strategy: dict[str, pd.Series],
                     allocation: dict[str, float] | None = None) -> pd.Series:
    alloc = allocation or DEFAULT_ALLOCATION
    aligned = pd.DataFrame(returns_by_strategy).fillna(0.0)
    weights = pd.Series({k: alloc.get(k, 0.0) for k in aligned.columns})
    return (aligned * weights).sum(axis=1)


def combine_weights(weights_by_strategy: dict[str, pd.DataFrame],
                     allocation: dict[str, float] | None = None) -> pd.DataFrame:
    alloc = allocation or DEFAULT_ALLOCATION
    parts = []
    for name, w in weights_by_strategy.items():
        parts.append(w * alloc.get(name, 0.0))
    combined = parts[0]
    for p in parts[1:]:
        combined = combined.add(p, fill_value=0.0)
    return combined.fillna(0.0)


def _apply_weight_floor(raw_weights: dict[str, float], min_weight: float) -> dict[str, float]:
    """Normaliza `raw_weights` a que sumen 1 con un piso `min_weight` por
    entrada, GARANTIZADO (no una sola pasada de "clip y renormalizar", que
    puede empujar una entrada recién clippeada otra vez por debajo del piso --
    ver el bug que este helper corrige). Water-filling: fija en el piso las
    entradas que quedarían debajo, reparte el resto proporcionalmente entre las
    que quedan libres, y repite hasta que ninguna libre quede por debajo."""
    n = len(raw_weights)
    if n == 0:
        return {}
    if min_weight * n > 1.0 + 1e-9:
        return {k: 1.0 / n for k in raw_weights}  # piso inalcanzable para todas -> reparto igual

    alloc = dict(raw_weights)
    fixed: set[str] = set()
    for _ in range(n):
        free = [k for k in alloc if k not in fixed]
        if not free:
            break
        avail = 1.0 - min_weight * len(fixed)
        free_sum = sum(alloc[k] for k in free)
        if free_sum > 0:
            for k in free:
                alloc[k] = alloc[k] / free_sum * avail
        else:
            for k in free:
                alloc[k] = avail / len(free)
        newly_below = [k for k in free if alloc[k] < min_weight - 1e-9]
        if not newly_below:
            break
        for k in newly_below:
            alloc[k] = min_weight
            fixed.add(k)
    return alloc


def _correlation_penalty(hist_by_strategy: dict[str, pd.Series], strategies: list[str],
                          dampening: float) -> dict[str, float]:
    """Factor (>=0.2) por el que se DIVIDE el peso inverse-vol de cada
    estrategia -- >1 amortigua (correlacionada con las demás, redundante),
    <1 premia (diversificadora, correlación baja o negativa). Mismo principio
    que `src/strategies/momentum.py` y `src/strategies/sector_rotation.py`,
    aplicado ahora entre ESTRATEGIAS: si momentum y rotación sectorial suelen
    ganar/perder juntas en el mismo tramo (ej. un rally amplio de mercado), el
    inverse-vol solo no lo detecta -- dos estrategias con la misma volatilidad
    individual pueden aportar diversificación muy distinta al ensamble."""
    if dampening <= 0 or len(strategies) <= 1:
        return {name: 1.0 for name in strategies}
    combined = pd.DataFrame(hist_by_strategy)[strategies].dropna()
    if len(combined) < 5:
        return {name: 1.0 for name in strategies}
    corr_mat = combined.corr()
    penalty = {}
    for name in strategies:
        others = [o for o in strategies if o != name]
        avg_corr = corr_mat.loc[name, others].mean() if others else 0.0
        avg_corr = 0.0 if pd.isna(avg_corr) else max(-1.0, min(1.0, avg_corr))
        penalty[name] = max(1.0 + dampening * avg_corr, 0.2)
    return penalty


def optimize_ensemble_weights(oos_returns: dict[str, pd.Series], folds: list,
                               default_allocation: dict[str, float] | None = None,
                               method: str = "inverse_vol", min_weight: float = 0.05,
                               corr_dampening: float = 0.5) -> dict:
    """Asigna capital entre estrategias FOLD A FOLD en vez de con una mezcla fija
    (`DEFAULT_ALLOCATION` / `combine_returns`) -- la idea es que la estrategia que
    viene rindiendo mejor (ajustado a riesgo, y que menos se solapa con las
    demás) últimamente reciba más capital, igual que `momentum.py` ya hace
    entre ACTIVOS (paridad de riesgo consciente de correlación), aplicado
    ahora un nivel arriba, entre ESTRATEGIAS.

    Honesto sobre fuga de información: para el fold `i`, la asignación se calcula
    SOLO con el desempeño OOS ya conocido de los folds `0..i-1` (ventana
    expandible) -- nunca mira el propio fold `i` ni folds futuros. El fold 0 no
    tiene historial OOS previo, así que usa `default_allocation` (la mezcla
    estática) como punto de partida.

    `oos_returns`: dict estrategia -> Serie de retornos OOS ya calculada por
    `walk_forward_backtest` (una por estrategia, mismos `folds` para las tres --
    ver `scripts/run_backtest.py`).

    method="inverse_vol": peso proporcional a 1/volatilidad reciente de cada
    estrategia (no hace falta anualizar -- el factor de anualización es el mismo
    para las tres estrategias en una corrida, así que se cancela en la
    comparación relativa), AJUSTADO por `corr_dampening` (0 = paridad de riesgo
    pura, sin ajuste de correlación).

    `min_weight`: piso de asignación por estrategia -- sin esto, una mala racha
    corta podría dejar a una estrategia en ~0% de por vida (nunca hay folds
    futuros para "probar de nuevo" y recuperarse), perdiendo toda la
    diversificación. Se aplica recortando hacia arriba y renormalizando.

    Devuelve dict con:
      - "returns": Serie de retornos del ensamble dinámico (mismo formato que
        `combine_returns`, para meterla directo a `summary_table`).
      - "fold_allocations": lista de dicts (uno por fold) con la asignación
        usada en ese fold, para transparencia/depuración.
    """
    if method != "inverse_vol":
        raise ValueError(f"método desconocido: {method!r} (soportado: 'inverse_vol')")

    default_allocation = default_allocation or DEFAULT_ALLOCATION
    strategies = list(oos_returns.keys())

    segments = []
    fold_allocations = []

    for i, fold in enumerate(folds):
        if i == 0:
            alloc = dict(default_allocation)
        else:
            # oos_returns[name] ya contiene SOLO tramos de test (walk-forward) --
            # cortar hasta el fin del fold anterior es la ventana expandible.
            hist_by_strategy = {name: oos_returns[name].loc[:folds[i - 1].test_end] for name in strategies}
            vols = {name: (h.std() if len(h) > 1 else float("nan")) for name, h in hist_by_strategy.items()}
            penalty = _correlation_penalty(hist_by_strategy, strategies, corr_dampening)
            raw = {}
            for name in strategies:
                v = vols[name]
                inv_vol = 1.0 / v if v and v > 1e-9 else 0.0
                raw[name] = inv_vol / penalty.get(name, 1.0)
            total = sum(raw.values())
            if total <= 0:
                alloc = dict(default_allocation)
            else:
                norm = {k: v / total for k, v in raw.items()}
                alloc = _apply_weight_floor(norm, min_weight)

        fold_allocations.append(dict(
            fold=i, test_start=str(fold.test_start.date()), test_end=str(fold.test_end.date()),
            allocation={k: round(v, 4) for k, v in alloc.items()},
        ))

        fold_returns = {name: oos_returns[name].loc[fold.test_start:fold.test_end] for name in strategies}
        aligned = pd.DataFrame(fold_returns).fillna(0.0)
        weights = pd.Series({k: alloc.get(k, 0.0) for k in aligned.columns})
        segments.append((aligned * weights).sum(axis=1))

    combined = pd.concat(segments).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return dict(returns=combined, fold_allocations=fold_allocations)


def apply_portfolio_vol_target(returns: pd.Series, vol_target: float = 0.10, vol_lookback: int = 20,
                                max_gross_exposure: float = 1.0, periods_per_year: int = 252) -> pd.Series:
    """Escala la exposición del ENSAMBLE YA COMBINADO para apuntar a una
    volatilidad anualizada objetivo -- cada estrategia individual ya hace su
    propio vol-targeting internamente (ver `src/strategies/momentum.py`), pero
    la combinación de las 3 no tenía ningún control de volatilidad propio: si
    coincidentemente las 3 entran en modo agresivo al mismo tiempo (ej. un
    rally amplio), el ensamble puede terminar con más volatilidad de la que
    cualquiera de ellas apunta individualmente. Este overlay corrige eso.

    Causal a propósito: la escala del día `i` se decide con la volatilidad
    realizada conocida HASTA el día `i-1` (`shift(1)`) -- nunca usa el propio
    retorno del día que está escalando, para no meter fuga de información.

    `max_gross_exposure=1.0` por defecto: esto escala el RETORNO ya combinado,
    no compra apalancamiento real -- solo baja exposición cuando la volatilidad
    del ensamble se dispara, nunca la sube por encima de "sin escalar" salvo
    que subas `max_gross_exposure` a propósito (simulando apalancamiento, con
    todos los riesgos que eso implica -- ver la sección de perfil agresivo)."""
    realized = realized_vol(returns, window=vol_lookback, periods_per_year=periods_per_year)
    raw_scale = (vol_target / realized).clip(upper=max_gross_exposure)
    scale = raw_scale.shift(1).fillna(1.0)
    return returns * scale
