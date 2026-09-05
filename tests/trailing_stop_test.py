"""Valida el stop TRAILING de `src/strategies/mean_reversion.py` (`use_trailing_stop`,
default `True`): debería salir de una posición ANTES que el stop fijo cuando el
precio subió y después retrocede -- protegiendo la ganancia ya generada -- pero
comportarse EXACTAMENTE igual que el stop fijo cuando el precio nunca sube por
encima de la entrada (el máximo desde la entrada == el precio de entrada, así que
ambos stops caen en el mismo nivel).

Prueba directamente `_state_machine_loop` (el loop numba-jitted) con arrays
armados a mano -- permite razonar exactamente qué debería pasar día a día, sin
depender de que SMA/RSI produzcan una entrada en un día específico.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.strategies.mean_reversion import _state_machine_loop, generate_weights


def test_trailing_stop_exits_earlier_than_fixed_stop_after_a_rally():
    # Entra día 0 (único día con RSI<entry_rsi), sube fuerte hasta el día 2 (peak=130),
    # retrocede a 122 el día 3 -- un retroceso del 6.15% desde el peak (> 6% de
    # trailing_stop_pct), pero SOLO un 22% por ENCIMA de la entrada (100), muy lejos
    # del stop fijo (94 = 100 * 0.94).
    close = np.array([100.0, 110.0, 130.0, 122.0, 90.0, 90.0])
    trend_ok = np.array([True] * 6)
    rsi_vals = np.array([5.0, 40.0, 40.0, 40.0, 40.0, 40.0])  # solo el día 0 califica para entrar (RSI<10)
    vol_vals = np.array([0.15] * 6)  # == reference_vol -> size_mult=1.0, no afecta esta prueba

    sig_trailing, _ = _state_machine_loop(close, trend_ok, rsi_vals, vol_vals,
                                           entry_rsi=10.0, exit_rsi=70.0, max_hold_days=100,
                                           stop_loss_pct=0.06, reference_vol=0.15,
                                           use_trailing_stop=True, trailing_stop_pct=0.06)
    sig_fixed, _ = _state_machine_loop(close, trend_ok, rsi_vals, vol_vals,
                                        entry_rsi=10.0, exit_rsi=70.0, max_hold_days=100,
                                        stop_loss_pct=0.06, reference_vol=0.15,
                                        use_trailing_stop=False, trailing_stop_pct=0.06)

    assert list(sig_trailing) == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0], \
        f"el stop trailing debería salir el día 3 (retroceso de 6.15% desde el peak): {sig_trailing}"
    assert list(sig_fixed) == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0], \
        f"el stop fijo NO debería salir hasta el día 4 (recién ahí cruza 94 desde la entrada): {sig_fixed}"
    print("  OK: el stop trailing protege la ganancia y sale un día antes que el stop fijo tras el retroceso")


def test_trailing_stop_matches_fixed_stop_when_price_never_rises_above_entry():
    """Sin ninguna subida desde la entrada, el máximo == precio de entrada todo el
    tiempo -- el nivel de stop trailing y el fijo coinciden exactamente, así que
    ambos modos deberían dar la MISMA señal día a día."""
    close = np.array([100.0, 99.0, 98.0, 97.0, 93.0, 90.0])  # cae monótono, nunca sube de 100
    trend_ok = np.array([True] * 6)
    rsi_vals = np.array([5.0, 40.0, 40.0, 40.0, 40.0, 40.0])
    vol_vals = np.array([0.15] * 6)

    sig_trailing, _ = _state_machine_loop(close, trend_ok, rsi_vals, vol_vals,
                                           entry_rsi=10.0, exit_rsi=70.0, max_hold_days=100,
                                           stop_loss_pct=0.06, reference_vol=0.15,
                                           use_trailing_stop=True, trailing_stop_pct=0.06)
    sig_fixed, _ = _state_machine_loop(close, trend_ok, rsi_vals, vol_vals,
                                        entry_rsi=10.0, exit_rsi=70.0, max_hold_days=100,
                                        stop_loss_pct=0.06, reference_vol=0.15,
                                        use_trailing_stop=False, trailing_stop_pct=0.06)

    assert list(sig_trailing) == list(sig_fixed), \
        f"sin subida desde la entrada, trailing y fijo deberían coincidir exactamente: " \
        f"trailing={sig_trailing}, fijo={sig_fixed}"
    print("  OK: sin subida desde la entrada, el stop trailing coincide exactamente con el fijo")


def test_trailing_stop_does_not_trigger_on_strong_sustained_uptrend():
    """Un rally sostenido, con retrocesos siempre menores al 6%, no debería
    disparar el stop trailing en ningún día -- solo confirma que no hay falsos
    positivos por ruido normal de un uptrend."""
    close = np.array([100.0, 105.0, 103.0, 110.0, 108.0, 118.0, 116.0, 125.0])
    n = len(close)
    trend_ok = np.array([True] * n)
    rsi_vals = np.array([5.0] + [40.0] * (n - 1))
    vol_vals = np.array([0.15] * n)

    sig, _ = _state_machine_loop(close, trend_ok, rsi_vals, vol_vals,
                                  entry_rsi=10.0, exit_rsi=70.0, max_hold_days=100,
                                  stop_loss_pct=0.06, reference_vol=0.15,
                                  use_trailing_stop=True, trailing_stop_pct=0.06)
    assert list(sig) == [1.0] * n, f"un rally con retrocesos chicos no debería disparar el stop trailing: {sig}"
    print("  OK: retrocesos normales dentro de un rally sostenido no disparan el stop trailing")


def test_generate_weights_default_uses_trailing_stop():
    """A nivel de `generate_weights` (no solo el loop interno): el default
    (`use_trailing_stop=True`, sin pasar params) debería salir de una posición en
    rally-y-retroceso antes que pasar `use_trailing_stop=False` explícitamente.

    Escenario verificado numéricamente (ver `src/indicators.py` -> `sma`/`rsi`):
    tendencia alcista sostenida y pronunciada (para que el precio quede bien por
    encima de su SMA20 incluso durante el dip de un día), un dip de un día que
    dispara RSI(2)=20 (< entry_rsi=50, entra), un rally fuerte hasta 250, y un
    retroceso a 235 -- exactamente 6% desde el peak (== trailing_stop_pct),
    pero muy por encima del stop fijo (198 * 0.94 = 186.12). `exit_rsi=99.9`
    (en vez del default 70) para aislar el stop como motivo de salida -- un
    rally tan fuerte también dispara RSI(2) muy alto por sí solo, y con el
    `exit_rsi` default esa salida por RSI ocurriría ANTES que cualquier stop,
    para ambas variantes por igual, sin diferenciarlas."""
    dates = pd.bdate_range("2021-01-04", periods=214)
    n = len(dates)
    prices = np.concatenate([
        np.linspace(100, 200, 200),          # tendencia alcista sostenida, calienta SMA/RSI
        [198, 210, 230, 250, 235],            # dip de 1 día (RSI bajo) + rally fuerte + retroceso
        np.full(n - 205, 235.0),
    ])
    close = pd.DataFrame({"TICK": prices}, index=dates)

    params_default = dict(trend_sma=20, entry_rsi=50.0, exit_rsi=99.9, max_hold_days=100,
                           stop_loss_pct=0.06, max_concurrent_positions=5, weight_per_position=0.10,
                           vol_lookback=10, reference_vol=0.15, trailing_stop_pct=0.06)
    params_fixed = {**params_default, "use_trailing_stop": False}

    w_trailing = generate_weights(close, ["TICK"], params_default)
    w_fixed = generate_weights(close, ["TICK"], params_fixed)

    days_held_trailing = int((w_trailing["TICK"] > 0).sum())
    days_held_fixed = int((w_fixed["TICK"] > 0).sum())
    assert days_held_trailing > 0, "la entrada debería haber ocurrido en ambos escenarios"
    assert days_held_trailing < days_held_fixed, \
        f"con trailing stop (default) se esperaba salir ANTES que con stop fijo tras el retroceso: " \
        f"trailing={days_held_trailing} días, fijo={days_held_fixed} días"
    print(f"  OK: generate_weights usa trailing stop por default (días en posición: "
          f"trailing={days_held_trailing}, fijo={days_held_fixed})")


def main():
    print("[1/4] Probando que el stop trailing sale ANTES que el fijo tras un rally y retroceso...")
    test_trailing_stop_exits_earlier_than_fixed_stop_after_a_rally()
    print("\n[2/4] Probando que sin subida desde la entrada, trailing == fijo...")
    test_trailing_stop_matches_fixed_stop_when_price_never_rises_above_entry()
    print("\n[3/4] Probando que un rally sostenido no dispara falsos positivos...")
    test_trailing_stop_does_not_trigger_on_strong_sustained_uptrend()
    print("\n[4/4] Probando que generate_weights usa trailing stop por default...")
    test_generate_weights_default_uses_trailing_stop()
    print("\nTRAILING STOP TEST OK: mean_reversion.py protege ganancias correctamente.")


if __name__ == "__main__":
    main()
