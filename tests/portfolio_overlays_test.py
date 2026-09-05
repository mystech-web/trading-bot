"""Valida `src/portfolio_overlays.py`: barrido de cash ocioso hacia el proxy de
cash (`sweep_idle_cash`), el ajuste dinámico de topes de posición por
correlación agregada del portafolio (`compute_aggregate_correlation`,
`correlation_based_cap_scale`, `tighten_caps_by_correlation`), la entrada
escalonada de posiciones nuevas (`ramp_in_new_positions`), y la función de
composición usada en el walk-forward paralelo (`apply_weight_overlays`).
"""
import sys
import pathlib
import pickle
import functools

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.portfolio_overlays import (
    sweep_idle_cash, compute_aggregate_correlation, correlation_based_cap_scale,
    tighten_caps_by_correlation, ramp_in_new_positions, apply_weight_overlays,
)


def test_sweep_idle_cash_fills_remainder_into_cash_ticker():
    dates = pd.bdate_range("2024-01-02", periods=3)
    weights = pd.DataFrame({"SPY": [0.3, 0.6, 1.0], "QQQ": [0.2, 0.1, 0.0]}, index=dates)
    out = sweep_idle_cash(weights, "BIL")
    assert list(out["BIL"].round(6)) == [0.5, 0.3, 0.0], f"el remanente debería ir a BIL: {out['BIL'].tolist()}"
    assert (out[["SPY", "QQQ"]] == weights[["SPY", "QQQ"]]).all().all(), \
        "el barrido de cash no debería tocar los pesos de los demás activos"
    assert (out.sum(axis=1).round(6) == 1.0).all(), "cada día debería sumar exactamente 1.0 tras el barrido"


def test_sweep_idle_cash_increments_existing_cash_weight():
    dates = pd.bdate_range("2024-01-02", periods=2)
    weights = pd.DataFrame({"SPY": [0.5, 0.5], "BIL": [0.1, 0.3]}, index=dates)
    out = sweep_idle_cash(weights, "BIL")
    assert list(out["BIL"].round(6)) == [0.5, 0.5], f"debería sumar al peso de cash ya existente, no reemplazarlo: {out['BIL'].tolist()}"


def test_sweep_idle_cash_never_goes_negative_or_reduces_other_weights():
    dates = pd.bdate_range("2024-01-02", periods=1)
    weights = pd.DataFrame({"SPY": [0.7], "QQQ": [0.5]}, index=dates)  # ya suma 1.2 (más de 1.0)
    out = sweep_idle_cash(weights, "BIL")
    assert out["BIL"].iloc[0] == 0.0, "sin remanente (suma ya >= 1.0), no debería agregar cash"
    assert out["SPY"].iloc[0] == 0.7 and out["QQQ"].iloc[0] == 0.5, \
        "el barrido de cash nunca debería RECORTAR los pesos existentes, solo rellenar el remanente"


def test_sweep_idle_cash_none_ticker_is_a_no_op():
    dates = pd.bdate_range("2024-01-02", periods=2)
    weights = pd.DataFrame({"SPY": [0.3, 0.6]}, index=dates)
    out = sweep_idle_cash(weights, None)
    assert out.equals(weights), "cash_ticker=None no debería modificar nada"


def test_aggregate_correlation_high_when_assets_move_together():
    n = 300
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2021-01-04", periods=n)
    common = rng.normal(0, 0.01, n)
    # 4 activos que comparten casi todo su movimiento (factor común domina el idiosincrático).
    rets = pd.DataFrame({
        f"A{i}": common + rng.normal(0, 0.001, n) for i in range(4)
    }, index=dates)
    avg_corr = compute_aggregate_correlation(rets, window=60)
    tail_avg = avg_corr.iloc[100:].mean()
    assert tail_avg > 0.9, f"con un factor común dominante, la correlación agregada debería ser muy alta: {tail_avg}"


def test_aggregate_correlation_low_when_assets_are_independent():
    n = 300
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2021-01-04", periods=n)
    rets = pd.DataFrame({
        f"A{i}": rng.normal(0, 0.01, n) for i in range(6)
    }, index=dates)
    avg_corr = compute_aggregate_correlation(rets, window=60)
    tail_avg = avg_corr.iloc[100:].mean()
    assert tail_avg < 0.4, f"con activos independientes, la correlación agregada debería ser baja: {tail_avg}"


def test_correlation_based_cap_scale_thresholds_and_interpolation():
    avg_corr = pd.Series([0.0, 0.3, 0.5, 0.7, 1.0])
    scale = correlation_based_cap_scale(avg_corr, full_cap_below=0.3, floor_above=0.7, min_scale=0.6)
    assert scale.iloc[0] == 1.0 and scale.iloc[1] == 1.0, "por debajo (o en) full_cap_below, sin ajuste (1.0)"
    assert scale.iloc[3] == 0.6 and scale.iloc[4] == 0.6, "por encima (o en) floor_above, el mínimo (min_scale)"
    assert 0.6 < scale.iloc[2] < 1.0, f"a mitad de camino (0.5), debería interpolar entre 1.0 y 0.6: {scale.iloc[2]}"


def test_tighten_caps_by_correlation_only_clips_known_tickers():
    dates = pd.bdate_range("2024-01-02", periods=2)
    weights = pd.DataFrame({"SPY": [0.30, 0.30], "UNKNOWN": [0.20, 0.20]}, index=dates)
    position_caps = {"SPY": 0.20}
    cap_scale = pd.Series([1.0, 0.5], index=dates)  # día 1 sin ajuste, día 2 tope a la mitad

    out = tighten_caps_by_correlation(weights, position_caps, cap_scale)
    assert out["SPY"].iloc[0] == 0.20, "día 1: tope sin ajustar (0.20 * 1.0) debería recortar 0.30 -> 0.20"
    assert out["SPY"].iloc[1] == 0.10, "día 2: tope apretado a la mitad (0.20 * 0.5 = 0.10) debería recortar más"
    assert (out["UNKNOWN"] == weights["UNKNOWN"]).all(), \
        "un ticker sin tope conocido en position_caps no debería tocarse"


def test_ramp_in_new_position_takes_several_days_to_reach_target():
    dates = pd.bdate_range("2024-01-02", periods=6)
    # Entra de golpe el día 0: target salta de 0% a 10% y se queda ahí.
    weights = pd.DataFrame({"SPY": [0.10] * 6}, index=dates)
    out = ramp_in_new_positions(weights, max_daily_increase=0.02)
    assert list(out["SPY"].round(6)) == [0.02, 0.04, 0.06, 0.08, 0.10, 0.10], \
        f"debería subir en pasos de 2 puntos porcentuales hasta llegar al 10%: {out['SPY'].tolist()}"


def test_ramp_in_position_decrease_is_always_instant():
    dates = pd.bdate_range("2024-01-02", periods=3)
    # Día 0: entra directo al 10% (se supone ya ramp-eada, prev=0 así que ramp-ea
    # igual, pero lo que importa es el día 1: cae a 2% de golpe).
    weights = pd.DataFrame({"SPY": [0.02, 0.02, 0.02]}, index=dates)
    out = ramp_in_new_positions(weights, max_daily_increase=0.02)
    # Sube 0->0.02 en el día 0 (un solo paso, cabe exacto), se mantiene después.
    assert list(out["SPY"].round(6)) == [0.02, 0.02, 0.02]

    # Ahora una posición YA abierta que sube fuerte y después cae de golpe.
    dates2 = pd.bdate_range("2024-01-02", periods=4)
    weights2 = pd.DataFrame({"SPY": [0.02, 0.10, 0.10, 0.01]}, index=dates2)
    out2 = ramp_in_new_positions(weights2, max_daily_increase=0.02)
    # día 0: 0->0.02 (un paso). día 1: target 0.10, sube solo a 0.04 (limitado).
    # día 2: sigue subiendo a 0.06. día 3: CAE a 0.01 -- instantáneo, sin límite.
    assert list(out2["SPY"].round(6)) == [0.02, 0.04, 0.06, 0.01], \
        f"la bajada del día 3 debería ser instantánea (sin importar la magnitud): {out2['SPY'].tolist()}"


def test_ramp_in_excludes_cash_ticker_entirely():
    dates = pd.bdate_range("2024-01-02", periods=2)
    weights = pd.DataFrame({"SPY": [0.0, 0.10], "BIL": [0.0, 0.90]}, index=dates)
    out = ramp_in_new_positions(weights, max_daily_increase=0.02, cash_ticker="BIL")
    assert out["SPY"].iloc[1] == 0.02, "SPY (no es cash) debería seguir limitado por el paso diario"
    assert out["BIL"].iloc[1] == 0.90, "BIL (cash_ticker) debería quedar sin ningún límite, salto directo"


def test_ramp_in_disabled_with_none_or_zero_is_a_no_op():
    dates = pd.bdate_range("2024-01-02", periods=2)
    weights = pd.DataFrame({"SPY": [0.0, 0.20]}, index=dates)
    out_none = ramp_in_new_positions(weights, max_daily_increase=None)
    out_zero = ramp_in_new_positions(weights, max_daily_increase=0.0)
    pd.testing.assert_frame_equal(out_none, weights)
    pd.testing.assert_frame_equal(out_zero, weights)


def _fake_generate_weights(close: pd.DataFrame, tickers: list, params: dict | None = None,
                            max_weight_by_ticker: dict | None = None) -> pd.DataFrame:
    """Función de módulo (no un closure) -- imita la firma real de
    `momentum.generate_weights` para probar `apply_weight_overlays` con el mismo
    patrón `functools.partial` que usa `scripts/run_backtest.py` en el
    walk-forward paralelo."""
    dates = close.index
    return pd.DataFrame({t: 0.3 for t in tickers}, index=dates)


def test_apply_weight_overlays_composes_cap_tightening_and_cash_sweep():
    dates = pd.bdate_range("2024-01-02", periods=2)
    close = pd.DataFrame({"SPY": 100.0, "QQQ": 100.0}, index=dates)
    position_caps = {"SPY": 0.20, "QQQ": 0.20}
    cap_scale = pd.Series([1.0, 1.0], index=dates)

    inner = functools.partial(_fake_generate_weights, close, ["SPY", "QQQ"])
    wrapped = functools.partial(apply_weight_overlays, inner, "BIL", position_caps, cap_scale, None, None)
    out = wrapped(params=None)

    # Cada activo entra con 0.30 (fake), se recorta al tope 0.20, y el remanente (0.60) va a BIL.
    assert (out["SPY"] == 0.20).all() and (out["QQQ"] == 0.20).all()
    assert (out["BIL"].round(6) == 0.60).all()


def _fake_generate_weights_varying_by_day(close: pd.DataFrame, tickers: list, params: dict | None = None,
                                           max_weight_by_ticker: dict | None = None) -> pd.DataFrame:
    """Como `_fake_generate_weights`, pero el peso cambia día a día -- necesario
    para poder distinguir "se congeló en el valor de ayer" de "coincidencia"."""
    dates = close.index
    return pd.DataFrame({t: [0.05 * (i + 1) for i in range(len(dates))] for t in tickers}, index=dates)


def test_apply_weight_overlays_composes_blackout_freeze_too():
    dates = pd.bdate_range("2024-01-02", periods=3)
    close = pd.DataFrame({"SPY": 100.0, "QQQ": 100.0}, index=dates)
    cap_scale = pd.Series([1.0, 1.0, 1.0], index=dates)
    blackout_dates = {dates[1]}

    inner = functools.partial(_fake_generate_weights_varying_by_day, close, ["SPY", "QQQ"])
    wrapped = functools.partial(apply_weight_overlays, inner, None, None, None, blackout_dates, None)
    out = wrapped(params=None)

    assert (out.loc[dates[1], ["SPY", "QQQ"]] == out.loc[dates[0], ["SPY", "QQQ"]]).all(), \
        "el día de blackout debería congelarse en el valor del día anterior, no seguir la señal 'cruda' de ese día"
    assert (out.loc[dates[2], ["SPY", "QQQ"]] > out.loc[dates[0], ["SPY", "QQQ"]]).all(), \
        "el día después del blackout debería volver a seguir la señal normal (más alta que el día 0)"


def test_apply_weight_overlays_composes_ramp_in_too():
    dates = pd.bdate_range("2024-01-02", periods=2)
    close = pd.DataFrame({"SPY": 100.0, "QQQ": 100.0}, index=dates)

    inner = functools.partial(_fake_generate_weights, close, ["SPY", "QQQ"])  # fake siempre pide 0.30
    wrapped = functools.partial(apply_weight_overlays, inner, "BIL", None, None, None, 0.02)
    out = wrapped(params=None)

    # Sin ramp-in, ambos entrarían directo a 0.30 desde el día 0. Con
    # ramp_max_daily_increase=0.02, el día 0 (prev=0) queda limitado a 0.02 cada uno
    # -- el resto (0.96) va a BIL. El día 1 sigue subiendo (0.04), todavía lejos de 0.30.
    assert out["SPY"].iloc[0] == 0.02 and out["QQQ"].iloc[0] == 0.02, \
        f"el ramp-in debería limitar la entrada del día 0 a 0.02: SPY={out['SPY'].tolist()}, QQQ={out['QQQ'].tolist()}"
    assert out["SPY"].iloc[1] == 0.04 and out["QQQ"].iloc[1] == 0.04, \
        f"el día 1 debería seguir subiendo de a 0.02, todavía lejos del target 0.30: {out['SPY'].tolist()}"
    assert out["BIL"].round(6).iloc[0] == 0.96 and out["BIL"].round(6).iloc[1] == 0.92


def test_apply_weight_overlays_is_picklable_for_multiprocessing():
    """Requisito real del walk-forward paralelo (`--jobs > 1`, contexto
    'spawn'): la composición armada con functools.partial tiene que poder
    mandarse a otro proceso -- si alguna pieza fuera un lambda o un closure,
    esto fallaría con un PicklingError."""
    dates = pd.bdate_range("2024-01-02", periods=2)
    close = pd.DataFrame({"SPY": 100.0, "QQQ": 100.0}, index=dates)
    position_caps = {"SPY": 0.20, "QQQ": 0.20}
    cap_scale = pd.Series([1.0, 1.0], index=dates)

    inner = functools.partial(_fake_generate_weights, close, ["SPY", "QQQ"])
    wrapped = functools.partial(apply_weight_overlays, inner, "BIL", position_caps, cap_scale, None, None)

    roundtripped = pickle.loads(pickle.dumps(wrapped))
    out = roundtripped(params=None)
    assert (out["BIL"].round(6) == 0.60).all(), "la función reconstruida tras el pickle debería dar el mismo resultado"


def main():
    print("[1/16] Probando que sweep_idle_cash rellena el remanente en el proxy de cash...")
    test_sweep_idle_cash_fills_remainder_into_cash_ticker()
    print("\n[2/16] Probando que suma al peso de cash ya existente, no lo reemplaza...")
    test_sweep_idle_cash_increments_existing_cash_weight()
    print("\n[3/16] Probando que nunca va negativo ni recorta pesos existentes...")
    test_sweep_idle_cash_never_goes_negative_or_reduces_other_weights()
    print("\n[4/16] Probando que cash_ticker=None es un no-op...")
    test_sweep_idle_cash_none_ticker_is_a_no_op()
    print("\n[5/16] Probando correlación agregada ALTA con un factor común dominante...")
    test_aggregate_correlation_high_when_assets_move_together()
    print("\n[6/16] Probando correlación agregada BAJA con activos independientes...")
    test_aggregate_correlation_low_when_assets_are_independent()
    print("\n[7/16] Probando los umbrales e interpolación de correlation_based_cap_scale...")
    test_correlation_based_cap_scale_thresholds_and_interpolation()
    print("\n[8/16] Probando que tighten_caps_by_correlation solo recorta tickers conocidos...")
    test_tighten_caps_by_correlation_only_clips_known_tickers()
    print("\n[9/16] Probando que una posición nueva tarda varios días en llegar al target (ramp-in)...")
    test_ramp_in_new_position_takes_several_days_to_reach_target()
    print("\n[10/16] Probando que las bajadas de peso siempre son instantáneas...")
    test_ramp_in_position_decrease_is_always_instant()
    print("\n[11/16] Probando que cash_ticker queda excluido del ramp-in...")
    test_ramp_in_excludes_cash_ticker_entirely()
    print("\n[12/16] Probando que max_daily_increase=None/0 desactiva el ramp-in...")
    test_ramp_in_disabled_with_none_or_zero_is_a_no_op()
    print("\n[13/16] Probando que apply_weight_overlays compone cap tightening y cash sweep correctamente...")
    test_apply_weight_overlays_composes_cap_tightening_and_cash_sweep()
    print("\n[14/16] Probando que apply_weight_overlays también compone el freeze de blackout...")
    test_apply_weight_overlays_composes_blackout_freeze_too()
    print("\n[15/16] Probando que apply_weight_overlays también compone el ramp-in...")
    test_apply_weight_overlays_composes_ramp_in_too()
    print("\n[16/16] Probando que apply_weight_overlays es 'picklable' (requisito del walk-forward paralelo)...")
    test_apply_weight_overlays_is_picklable_for_multiprocessing()
    print("\nPORTFOLIO OVERLAYS TEST OK: barrido de cash, topes dinámicos, ramp-in y blackout de eventos "
          "funcionan correctamente.")


if __name__ == "__main__":
    main()
