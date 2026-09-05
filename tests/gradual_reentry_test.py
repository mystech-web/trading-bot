"""Valida la reentrada GRADUAL de la guardia de drawdown (`guard_ramp_days`, ver
`src/backtest.py::run_backtest` y `src/tracking.py::update_drawdown_guard`):
una vez que se cumplen las condiciones de recuperación, la exposición debería
subir en pasos iguales hasta 1.0 en vez de saltar de golpe, pausarse si la
recuperación se interrumpe a mitad de camino (sin volver a cruzar el umbral de
activación), y resetearse por completo si SÍ se vuelve a cruzar ese umbral.

Todos los escenarios de `src.backtest` están verificados numéricamente (los
valores exactos de drawdown/escala se calcularon corriendo el escenario y
leyendo la salida real antes de fijar las aserciones -- no son adivinados).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import tempfile

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.tracking import update_drawdown_guard


def _run(returns: list[float], guard_ramp_days: int) -> pd.Series:
    dates = pd.bdate_range("2021-01-04", periods=len(returns))
    price = 100 * np.cumprod(1 + np.array(returns))
    close = pd.DataFrame({"X": price}, index=dates)
    weights = pd.DataFrame({"X": 1.0}, index=dates)
    res = run_backtest(close, weights, cost_bps=0.0, dd_guard_threshold=-0.15, dd_guard_recover=-0.07,
                        dd_guard_scale=0.5, guard_ramp_days=guard_ramp_days, vol_accel_enabled=False)
    return res["exposure_scale"]


def test_ramp_increases_gradually_instead_of_jumping_to_full_exposure():
    calm = [0.0] * 40
    crash = [-0.03] * 8      # cruza el umbral de -15% el día 46
    recover = [0.02] * 20    # se recupera por encima de -7% alrededor del día 62
    tail = [0.001] * 30

    returns = calm + crash + recover + tail
    scale_instant = _run(returns, guard_ramp_days=0)
    scale_ramp = _run(returns, guard_ramp_days=5)

    assert scale_instant.iloc[46] == 0.5 and scale_ramp.iloc[46] == 0.5, \
        "ambas variantes deberían activar la guardia al cruzar el umbral, igual que antes"
    # Con el salto instantáneo, en cuanto se recupera pasa directo a 1.0.
    assert scale_instant.iloc[63] == 1.0, f"salto instantáneo: se esperaba 1.0 en el día 63, fue {scale_instant.iloc[63]}"
    # Con la rampa, sube en pasos de (1.0-0.5)/5 = 0.1 por día, 5 días.
    ramp_values = [round(v, 6) for v in scale_ramp.iloc[63:68]]
    assert ramp_values == [0.6, 0.7, 0.8, 0.9, 1.0], f"se esperaba una rampa de 0.6 a 1.0 en 5 pasos: {ramp_values}"
    print(f"  OK: salto instantáneo día 63 = {scale_instant.iloc[63]}, rampa días 63-67 = {ramp_values}")


def test_guard_ramp_days_zero_matches_old_instant_behavior_exactly():
    """Regresión de compatibilidad: `guard_ramp_days=0` debe reproducir EXACTAMENTE
    el comportamiento de antes de este cambio (salto instantáneo)."""
    calm = [0.0] * 40
    crash = [-0.03] * 8
    recover = [0.02] * 20
    tail = [0.001] * 30
    returns = calm + crash + recover + tail

    scale = _run(returns, guard_ramp_days=0)
    # Antes de la recuperación: 0.5 apenas se activa; en cuanto se cumple la
    # condición de recuperación, salta directo a 1.0 el mismo día (sin pasos intermedios).
    assert scale.iloc[62] == 0.5
    assert scale.iloc[63] == 1.0
    assert (scale.iloc[63:] == 1.0).all(), "con guard_ramp_days=0 no debería haber ningún paso intermedio"
    print("  OK: guard_ramp_days=0 reproduce el salto instantáneo exacto de antes")


def test_ramp_pauses_if_recovery_is_interrupted_without_retriggering():
    """Un retroceso a mitad de la rampa que vuelve a la zona intermedia (entre
    dd_guard_recover y dd_guard_threshold) SIN cruzar el umbral de activación de
    nuevo debería PAUSAR la rampa (mantener la escala actual), no resetearla ni
    seguir avanzando ese día."""
    calm = [0.0] * 40
    crash = [-0.03] * 8
    recover_part1 = [0.02] * 15
    pause_dip = [-0.03] * 1     # empuja el drawdown de vuelta bajo -7%, pero SIN cruzar -15%
    recover_part2 = [0.02] * 10
    tail = [0.001] * 20
    returns = calm + crash + recover_part1 + pause_dip + recover_part2 + tail

    scale = _run(returns, guard_ramp_days=5)
    # La rampa llega a 0.6 el día 63; el dip cae ESE mismo día (índice 63 en la
    # lista de retornos), y como el día 64 todavía no se recuperó (dd < -7%), la
    # escala se mantiene en 0.6 -- no avanza a 0.7 hasta el día 65.
    assert round(scale.iloc[63], 6) == 0.6
    assert round(scale.iloc[64], 6) == 0.6, \
        f"la rampa debería pausarse (mantenerse en 0.6) durante el retroceso, no avanzar: {scale.iloc[64]}"
    assert [round(v, 6) for v in scale.iloc[65:69]] == [0.7, 0.8, 0.9, 1.0], \
        "tras el retroceso, la rampa debería REANUDAR desde donde se pausó, no reiniciar desde 0.5"
    print(f"  OK: la rampa se pausó en 0.6 durante el retroceso y reanudó normalmente después "
          f"(no perdió el progreso)")


def test_ramp_resets_if_threshold_is_crossed_again_mid_ramp():
    """Un retroceso FUERTE a mitad de la rampa que SÍ vuelve a cruzar
    dd_guard_threshold debería resetear la escala a `dd_guard_scale` de golpe,
    perdiendo el progreso de la rampa anterior -- un nuevo evento de estrés
    real no debería recibir "crédito parcial" por la recuperación previa."""
    calm = [0.0] * 40
    crash = [-0.03] * 8
    recover_part1 = [0.02] * 17   # deja la rampa avanzando (0.6, 0.7...)
    retrigger = [-0.20] * 1       # caída fuerte -- vuelve a cruzar -15%
    recover_part2 = [0.02] * 20
    tail = [0.001] * 10
    returns = calm + crash + recover_part1 + retrigger + recover_part2 + tail

    scale = _run(returns, guard_ramp_days=5)
    assert round(scale.iloc[64], 6) == 0.7, f"la rampa debería estar en 0.7 antes de la recaída: {scale.iloc[64]}"
    assert round(scale.iloc[65], 6) == 0.8, f"un paso más antes de la recaída: {scale.iloc[65]}"
    assert round(scale.iloc[66], 6) == 0.5, \
        f"tras volver a cruzar el umbral de activación, la escala debería resetear a 0.5, no seguir en 0.8: " \
        f"{scale.iloc[66]}"
    print("  OK: un nuevo cruce del umbral a mitad de la rampa resetea la escala a 0.5, sin crédito parcial")


def test_live_guard_ramps_gradually_across_daily_calls():
    """Mismo mecanismo, pero a través de `update_drawdown_guard` (el que usa
    `run_live_once.py`/`run_crypto_live_once.py`) -- cada llamada simula una
    corrida diaria distinta."""
    with tempfile.TemporaryDirectory() as tmp:
        tracking_dir = pathlib.Path(tmp)

        # Pico inicial y luego una caída que activa la guardia.
        dd0, scale0, active0, changed0 = update_drawdown_guard(10_000.0, threshold=-0.15, recover=-0.07,
                                                                 dd_guard_scale=0.5, guard_ramp_days=5,
                                                                 reports_dir=tracking_dir)
        assert scale0 == 1.0 and active0 is False, "el primer día (nuevo pico) no debería activar la guardia"

        dd1, scale1, active1, changed1 = update_drawdown_guard(8_000.0, threshold=-0.15, recover=-0.07,
                                                                 dd_guard_scale=0.5, guard_ramp_days=5,
                                                                 reports_dir=tracking_dir)
        assert active1 is True and scale1 == 0.5 and changed1 is True, \
            f"con -20% de drawdown la guardia debería activarse en escala 0.5: {scale1}"

        # Recuperación: equity vuelve a estar por encima del piso de recover (-7%
        # del pico de 10,000 = 9,300) en corridas sucesivas -- la escala debería
        # subir de a 0.1 (= (1.0-0.5)/5) por corrida, no saltar directo a 1.0.
        expected_scale = [0.6, 0.7, 0.8, 0.9, 1.0]
        for expected in expected_scale:
            dd, scale, active, changed = update_drawdown_guard(9_500.0, threshold=-0.15, recover=-0.07,
                                                                 dd_guard_scale=0.5, guard_ramp_days=5,
                                                                 reports_dir=tracking_dir)
            assert round(scale, 6) == expected, f"se esperaba escala {expected}, fue {scale}"

        # Recién en la ÚLTIMA corrida (escala llega a 1.0) debería reportarse el cambio de estado.
        assert active is False and changed is True, \
            "la guardia debería quedar desactivada (y reportar el cambio) solo al completar la rampa"
        print(f"  OK: update_drawdown_guard rampa gradualmente 0.5 -> {expected_scale} a través de corridas diarias")


def test_live_guard_ramp_zero_matches_old_instant_behavior():
    with tempfile.TemporaryDirectory() as tmp:
        tracking_dir = pathlib.Path(tmp)
        update_drawdown_guard(10_000.0, reports_dir=tracking_dir, guard_ramp_days=0)
        dd1, scale1, active1, changed1 = update_drawdown_guard(8_000.0, threshold=-0.15, recover=-0.07,
                                                                 guard_ramp_days=0, reports_dir=tracking_dir)
        assert scale1 == 0.5 and active1 is True

        dd2, scale2, active2, changed2 = update_drawdown_guard(9_500.0, threshold=-0.15, recover=-0.07,
                                                                 guard_ramp_days=0, reports_dir=tracking_dir)
        assert scale2 == 1.0 and active2 is False and changed2 is True, \
            "con guard_ramp_days=0 debería volver a 100% de exposición en la primera corrida recuperada"
        print("  OK: update_drawdown_guard con guard_ramp_days=0 reproduce el salto instantáneo de antes")


def main():
    print("[1/6] Probando que la rampa sube gradualmente en vez de saltar a 100%...")
    test_ramp_increases_gradually_instead_of_jumping_to_full_exposure()
    print("\n[2/6] Probando que guard_ramp_days=0 reproduce el comportamiento anterior EXACTO (backtest)...")
    test_guard_ramp_days_zero_matches_old_instant_behavior_exactly()
    print("\n[3/6] Probando que la rampa se PAUSA si la recuperación se interrumpe sin reactivar la guardia...")
    test_ramp_pauses_if_recovery_is_interrupted_without_retriggering()
    print("\n[4/6] Probando que la rampa se RESETEA si se vuelve a cruzar el umbral de activación...")
    test_ramp_resets_if_threshold_is_crossed_again_mid_ramp()
    print("\n[5/6] Probando la rampa gradual en update_drawdown_guard (bot en vivo, corrida a corrida)...")
    test_live_guard_ramps_gradually_across_daily_calls()
    print("\n[6/6] Probando que guard_ramp_days=0 reproduce el comportamiento anterior EXACTO (en vivo)...")
    test_live_guard_ramp_zero_matches_old_instant_behavior()
    print("\nGRADUAL REENTRY TEST OK: la reentrada gradual tras la guardia de drawdown funciona correctamente.")


if __name__ == "__main__":
    main()
