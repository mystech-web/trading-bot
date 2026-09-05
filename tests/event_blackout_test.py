"""Valida `src/event_blackout.py`: el blackout de FOMC (congela pesos el día
del evento, backtest y en vivo) y el blackout de earnings (bloquea acciones
individuales cerca de su reporte, solo en vivo, "best effort" contra yfinance).
"""
import sys
import pathlib
import tempfile
import datetime as dt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.event_blackout import (
    load_macro_calendar, freeze_weights_on_blackout_days,
    get_earnings_blackout_tickers, apply_earnings_blackout,
)


def test_load_macro_calendar_reads_dates_from_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "macro_calendar.yaml"
        path.write_text('fomc_decision_dates:\n  - "2024-01-31"\n  - "2024-03-20"\n')
        dates = load_macro_calendar(path)
        assert dates == {pd.Timestamp("2024-01-31"), pd.Timestamp("2024-03-20")}


def test_load_macro_calendar_missing_file_returns_empty_set():
    dates = load_macro_calendar(pathlib.Path("/tmp/does_not_exist_macro_calendar.yaml"))
    assert dates == set()


def test_real_config_file_loads_without_error():
    """El archivo real del proyecto (config/macro_calendar.yaml) debería cargar
    sin errores y tener al menos algunas fechas."""
    dates = load_macro_calendar()
    assert len(dates) > 0, "config/macro_calendar.yaml debería tener al menos algunas fechas de FOMC"
    assert all(isinstance(d, pd.Timestamp) for d in dates)


def _make_weights(n=10):
    dates = pd.bdate_range("2024-01-25", periods=n)  # cubre 2024-01-31 (miércoles)
    rng = np.random.default_rng(1)
    return pd.DataFrame({"SPY": rng.uniform(0.2, 0.8, n), "QQQ": rng.uniform(0.1, 0.5, n)}, index=dates)


def test_freeze_weights_on_blackout_day_keeps_previous_day_values():
    weights = _make_weights()
    blackout_date = pd.Timestamp("2024-01-31")
    assert blackout_date in weights.index, "la fecha de blackout debe estar en el rango de prueba"

    out = freeze_weights_on_blackout_days(weights, {blackout_date})
    prev_date = weights.index[weights.index.get_loc(blackout_date) - 1]

    assert (out.loc[blackout_date] == weights.loc[prev_date]).all(), \
        "el día de blackout debería tener los MISMOS pesos que el día hábil anterior"
    assert not (out.loc[blackout_date] == weights.loc[blackout_date]).all(), \
        "el día de blackout NO debería quedarse con sus propios pesos originales (generados al azar, distintos)"

    other_dates = [d for d in weights.index if d != blackout_date]
    assert (out.loc[other_dates] == weights.loc[other_dates]).all().all(), \
        "ningún otro día debería tocarse"


def test_freeze_weights_consecutive_blackout_days_chain_from_last_good_day():
    dates = pd.bdate_range("2024-01-25", periods=10)
    weights = pd.DataFrame({"SPY": np.arange(10, dtype=float) / 10}, index=dates)
    # Dos días de blackout consecutivos (índices 3 y 4).
    blackout = {dates[3], dates[4]}
    out = freeze_weights_on_blackout_days(weights, blackout)
    last_good = weights["SPY"].iloc[2]
    assert out["SPY"].iloc[3] == last_good and out["SPY"].iloc[4] == last_good, \
        f"ambos días de blackout consecutivos deberían heredar del último día bueno ({last_good}): " \
        f"{out['SPY'].iloc[3]}, {out['SPY'].iloc[4]}"
    assert out["SPY"].iloc[5] == weights["SPY"].iloc[5], "el día después del blackout debería rebalancear normal"


def test_freeze_weights_no_blackout_in_range_is_a_no_op():
    weights = _make_weights()
    out = freeze_weights_on_blackout_days(weights, {pd.Timestamp("2019-01-01")})
    pd.testing.assert_frame_equal(out, weights)


def test_freeze_weights_empty_blackout_set_is_a_no_op():
    weights = _make_weights()
    out = freeze_weights_on_blackout_days(weights, set())
    pd.testing.assert_frame_equal(out, weights)


def test_freeze_weights_blackout_on_first_day_falls_back_to_zero():
    weights = _make_weights()
    first_date = weights.index[0]
    out = freeze_weights_on_blackout_days(weights, {first_date})
    assert (out.loc[first_date] == 0.0).all(), \
        "sin ningún día previo del que heredar, el primer día debería caer a 0, no quedar NaN"


class _FakeTicker:
    """Simula yfinance.Ticker sin tocar la red -- `get_earnings_dates` devuelve
    un DataFrame con las fechas configuradas para ese símbolo."""
    _EARNINGS_BY_SYMBOL = {}

    def __init__(self, symbol):
        self.symbol = symbol

    def get_earnings_dates(self, limit=4):
        dates = self._EARNINGS_BY_SYMBOL.get(self.symbol)
        if dates is None:
            return pd.DataFrame()
        # yfinance real trae varias columnas (EPS estimate, etc.) -- acá alcanza con
        # una columna cualquiera para que el DataFrame NO cuente como "empty" (un
        # DataFrame con índice pero CERO columnas se considera empty en pandas,
        # sin importar cuántas filas tenga el índice).
        return pd.DataFrame({"EPS Estimate": [0.0] * len(dates)}, index=pd.DatetimeIndex(dates))


class _FailingTicker:
    def __init__(self, symbol):
        self.symbol = symbol

    def get_earnings_dates(self, limit=4):
        raise RuntimeError("simulated network failure")


def test_get_earnings_blackout_tickers_flags_tickers_near_their_report_date():
    import yfinance

    _FakeTicker._EARNINGS_BY_SYMBOL = {
        "AAPL": ["2024-02-01"],   # dentro de la ventana de "as_of"
        "MSFT": ["2024-03-15"],   # lejos de "as_of"
    }
    orig_ticker = yfinance.Ticker
    yfinance.Ticker = _FakeTicker
    try:
        blocked = get_earnings_blackout_tickers(["AAPL", "MSFT"], as_of=dt.date(2024, 1, 31),
                                                  days_before=1, days_after=1)
        assert blocked == {"AAPL"}, f"solo AAPL debería estar en blackout (reporta el 2024-02-01): {blocked}"
    finally:
        yfinance.Ticker = orig_ticker


def test_get_earnings_blackout_tickers_fails_safe_on_error():
    import yfinance
    orig_ticker = yfinance.Ticker
    yfinance.Ticker = _FailingTicker
    try:
        blocked = get_earnings_blackout_tickers(["AAPL", "MSFT"], as_of=dt.date(2024, 1, 31))
        assert blocked == set(), \
            "si la consulta a yfinance falla, ningún ticker debería bloquearse (fail-safe, no fail-paranoid)"
    finally:
        yfinance.Ticker = orig_ticker


def test_apply_earnings_blackout_zeroes_blocked_tickers_only():
    target_weights = {"AAPL": 0.10, "MSFT": 0.15, "SPY": 0.20}
    out = apply_earnings_blackout(target_weights, {"AAPL"})
    assert out == {"AAPL": 0.0, "MSFT": 0.15, "SPY": 0.20}


def main():
    print("[1/10] Probando que load_macro_calendar lee fechas del YAML...")
    test_load_macro_calendar_reads_dates_from_yaml()
    print("\n[2/10] Probando que un archivo faltante devuelve un set vacío...")
    test_load_macro_calendar_missing_file_returns_empty_set()
    print("\n[3/10] Probando que el archivo real del proyecto carga sin errores...")
    test_real_config_file_loads_without_error()
    print("\n[4/10] Probando que el día de blackout congela los pesos del día anterior...")
    test_freeze_weights_on_blackout_day_keeps_previous_day_values()
    print("\n[5/10] Probando que blackouts consecutivos encadenan desde el último día bueno...")
    test_freeze_weights_consecutive_blackout_days_chain_from_last_good_day()
    print("\n[6/10] Probando que sin blackout en rango, no hace nada...")
    test_freeze_weights_no_blackout_in_range_is_a_no_op()
    print("\n[7/10] Probando que un set de blackout vacío no hace nada...")
    test_freeze_weights_empty_blackout_set_is_a_no_op()
    print("\n[8/10] Probando que un blackout el primer día cae a 0 (no NaN)...")
    test_freeze_weights_blackout_on_first_day_falls_back_to_zero()
    print("\n[9/10] Probando el blackout de earnings (broker falso, sin red)...")
    test_get_earnings_blackout_tickers_flags_tickers_near_their_report_date()
    test_get_earnings_blackout_tickers_fails_safe_on_error()
    test_apply_earnings_blackout_zeroes_blocked_tickers_only()
    print("\n[10/10] OK")
    print("\nEVENT BLACKOUT TEST OK: blackout de FOMC y de earnings funcionan correctamente.")


if __name__ == "__main__":
    main()
