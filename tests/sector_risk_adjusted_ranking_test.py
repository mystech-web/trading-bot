"""Valida el ranking AJUSTADO A RIESGO de `src.strategies.sector_rotation`
(`risk_adjusted_ranking`, default `True`): entre los sectores que califican
(le ganan al cash), debería preferir el que mejor recompensa el riesgo que
toma (momentum/volatilidad), no el de mayor retorno crudo -- que puede ser
solo el más ruidoso, no el mejor.

Escenario sintético con 3 sectores donde se conoce de antemano cuál "debería"
ganar bajo cada criterio: HIGH (retorno alto pero MUY volátil, alternando +6%/
-3% cada día) vs. LOW (retorno más bajo pero MUY estable, +0.6%/+0.4% cada
día) vs. LOSER (momentum negativo, nunca califica). Verificado numéricamente
que con estos parámetros, el ranking crudo elige SIEMPRE a HIGH y el ajustado
a riesgo elige SIEMPRE a LOW, en cada rebalanceo activo.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.strategies import sector_rotation


def _make_close(n=200):
    dates = pd.bdate_range("2020-01-02", periods=n)
    high_ret = np.tile([0.06, -0.03], n // 2 + 1)[:n]     # alto retorno, MUY volátil
    low_ret = np.tile([0.006, 0.004], n // 2 + 1)[:n]     # retorno más bajo, MUY estable
    loser_ret = np.full(n, -0.001)                         # nunca le gana al cash
    cash_ret = np.full(n, 0.0001)

    close = pd.DataFrame({
        "HIGH": 100 * np.cumprod(1 + high_ret),
        "LOW": 100 * np.cumprod(1 + low_ret),
        "LOSER": 100 * np.cumprod(1 + loser_ret),
        "CASH": 100 * np.cumprod(1 + cash_ret),
    }, index=dates)
    return close


def test_raw_ranking_picks_high_volatility_high_return_sector():
    close = _make_close()
    tickers = ["HIGH", "LOW", "LOSER"]
    w = sector_rotation.generate_weights(close, tickers, "CASH",
                                          params=dict(top_n=1, momentum_long=40, momentum_short=5,
                                                      vol_lookback=20, risk_adjusted_ranking=False,
                                                      corr_dampening=0.0, max_weight_per_asset=1.0))
    active = (w[tickers] > 0).any(axis=1)
    assert active.sum() > 20, "se esperaban varios rebalanceos con algún pick activo"
    picks = set(w.loc[active, tickers].idxmax(axis=1).unique())
    assert picks == {"HIGH"}, f"con ranking crudo, siempre debería elegir HIGH (mayor retorno): {picks}"
    print("  OK: el ranking por momentum crudo elige siempre al sector de mayor retorno (HIGH), pese a su volatilidad")


def test_risk_adjusted_ranking_picks_low_volatility_sector_instead():
    close = _make_close()
    tickers = ["HIGH", "LOW", "LOSER"]
    w = sector_rotation.generate_weights(close, tickers, "CASH",
                                          params=dict(top_n=1, momentum_long=40, momentum_short=5,
                                                      vol_lookback=20, risk_adjusted_ranking=True,
                                                      corr_dampening=0.0, max_weight_per_asset=1.0))
    active = (w[tickers] > 0).any(axis=1)
    assert active.sum() > 20, "se esperaban varios rebalanceos con algún pick activo"
    picks = set(w.loc[active, tickers].idxmax(axis=1).unique())
    assert picks == {"LOW"}, \
        f"con ranking ajustado a riesgo, debería elegir LOW (mejor momentum/volatilidad), no HIGH: {picks}"
    print("  OK: el ranking ajustado a riesgo elige al sector con mejor momentum/volatilidad (LOW), no al más volátil")


def test_risk_adjusted_ranking_is_the_default():
    close = _make_close()
    tickers = ["HIGH", "LOW", "LOSER"]
    w_default = sector_rotation.generate_weights(close, tickers, "CASH",
                                                  params=dict(top_n=1, momentum_long=40, momentum_short=5,
                                                              vol_lookback=20, corr_dampening=0.0,
                                                              max_weight_per_asset=1.0))
    active = (w_default[tickers] > 0).any(axis=1)
    picks = set(w_default.loc[active, tickers].idxmax(axis=1).unique())
    assert picks == {"LOW"}, f"sin especificar risk_adjusted_ranking, el default debería comportarse como True: {picks}"
    print("  OK: risk_adjusted_ranking=True es el comportamiento por default (sin pasarlo explícitamente)")


def test_never_picks_a_sector_with_negative_momentum_regardless_of_ranking():
    """El filtro de momentum ABSOLUTO (le gana al cash) sigue siendo binario --
    ni el ranking crudo ni el ajustado a riesgo deberían elegir jamás a LOSER,
    sin importar cuán "estable" sea su caída."""
    close = _make_close()
    tickers = ["HIGH", "LOW", "LOSER"]
    for adj in (True, False):
        w = sector_rotation.generate_weights(close, tickers, "CASH",
                                              params=dict(top_n=3, momentum_long=40, momentum_short=5,
                                                          vol_lookback=20, risk_adjusted_ranking=adj,
                                                          corr_dampening=0.0, max_weight_per_asset=1.0))
        assert (w["LOSER"] == 0).all(), f"LOSER nunca debería recibir peso (risk_adjusted_ranking={adj})"
    print("  OK: el filtro de momentum absoluto sigue excluyendo a LOSER en ambos modos de ranking")


def main():
    print("[1/4] Probando que el ranking crudo elige el sector más volátil (mayor retorno)...")
    test_raw_ranking_picks_high_volatility_high_return_sector()
    print("\n[2/4] Probando que el ranking ajustado a riesgo elige el sector más estable...")
    test_risk_adjusted_ranking_picks_low_volatility_sector_instead()
    print("\n[3/4] Probando que el ranking ajustado a riesgo es el default...")
    test_risk_adjusted_ranking_is_the_default()
    print("\n[4/4] Probando que el filtro de momentum absoluto sigue excluyendo perdedores en ambos modos...")
    test_never_picks_a_sector_with_negative_momentum_regardless_of_ranking()
    print("\nSECTOR RISK-ADJUSTED RANKING TEST OK: prioriza sectores que recompensan mejor el riesgo, no solo el ruido.")


if __name__ == "__main__":
    main()
