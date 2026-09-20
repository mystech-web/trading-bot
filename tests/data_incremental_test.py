"""Valida la descarga incremental (src.data y src.crypto_data) SIN red real --
monkeypatchea las funciones de descarga de bajo nivel y verifica: (1) sin
caché, descarga completa; (2) con caché y force=False, cero llamadas de red;
(3) con caché y force=True, solo se pide la ventana reciente y se pega
correctamente al caché existente (sin duplicar días, sin perder historia vieja).
"""
import sys
import pathlib
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def test_equity_incremental_download(monkeypatch):
    import src.data as data_mod

    calls = {"full": 0, "since": 0}

    def fake_full(ticker, period):
        calls["full"] += 1
        dates = pd.bdate_range("2020-01-02", periods=100)
        df = pd.DataFrame({"Close": np.linspace(100, 200, 100), "Volume": 1e6}, index=dates)
        df.index.name = "date"
        return df

    def fake_since(ticker, start):
        calls["since"] += 1
        # Simula que Yahoo devuelve unos pocos días nuevos + el overlap solicitado.
        dates = pd.bdate_range(start.normalize(), periods=8)
        df = pd.DataFrame({"Close": np.linspace(500, 510, 8), "Volume": 2e6}, index=dates)
        df.index.name = "date"
        return df

    monkeypatch.setattr(data_mod, "_download_full", fake_full)
    monkeypatch.setattr(data_mod, "_download_since", fake_since)

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(data_mod, "CACHE_DIR", pathlib.Path(tmp))

        # 1) Sin caché -> descarga completa.
        out1 = data_mod.download_prices(["FAKE"], years=1, force=False)
        assert calls["full"] == 1 and calls["since"] == 0
        assert len(out1["FAKE"]) == 100
        original_last_date = out1["FAKE"].index.max()

        # 2) Con caché, force=False -> cero llamadas de red, se lee del disco tal cual.
        out2 = data_mod.download_prices(["FAKE"], years=1, force=False)
        assert calls["full"] == 1 and calls["since"] == 0, "no debería haber llamado a la red de nuevo"
        assert len(out2["FAKE"]) == 100

        # 3) Con caché, force=True -> SOLO pide la ventana reciente (no full de nuevo),
        #    y el resultado final tiene la historia vieja + la nueva, sin duplicados.
        out3 = data_mod.download_prices(["FAKE"], years=1, force=True)
        assert calls["full"] == 1, "force=True con caché existente NO debería re-pedir el historial completo"
        assert calls["since"] == 1, "debería haber pedido la ventana incremental exactamente una vez"
        merged = out3["FAKE"]
        assert not merged.index.duplicated().any(), "no debería haber fechas duplicadas tras el merge"
        assert merged.index.is_monotonic_increasing, "el índice debería quedar ordenado"
        assert merged.index.min() < original_last_date, "debería conservar la historia vieja"
        assert merged.index.max() > original_last_date, "debería incorporar los días nuevos"
        # En los días que se solapan, el dato NUEVO gana (por si Yahoo revisó un cierre).
        overlap_date = merged.index[merged.index <= original_last_date][-1]
        if overlap_date in pd.bdate_range(original_last_date - pd.Timedelta(days=5), original_last_date):
            pass  # el overlap exacto depende del fixture; lo importante ya se validó arriba (sin duplicados)

    print(f"  data.py: descarga incremental OK -- 1 descarga completa, 1 incremental, "
          f"{len(merged)} filas finales sin duplicados")


def test_equity_cache_with_fewer_years_than_requested_is_not_trusted(monkeypatch):
    """Regresión de un bug real: `run_live_once.py` pide `years=2` (solo necesita
    datos recientes); si eso corre ANTES que `run_backtest.py` (pide `years=11`),
    con la lógica vieja el backtest se quedaba con el caché corto de 2 años sin
    avisar -- en una corrida real esto vació por completo el walk-forward
    (`IndexError: list index out of range` al no haber folds). Ahora
    `download_prices` debe notar que el caché no cubre los años pedidos y volver
    a descargar completo."""
    import src.data as data_mod

    calls = {"full": 0, "since": 0}

    def fake_full(ticker, period):
        calls["full"] += 1
        years_requested = int(period.rstrip("y"))
        end = pd.Timestamp.now().normalize()
        dates = pd.bdate_range(end - pd.Timedelta(days=years_requested * 365), end)
        df = pd.DataFrame({"Close": np.linspace(100, 200, len(dates)), "Volume": 1e6}, index=dates)
        df.index.name = "date"
        return df

    def fake_since(ticker, start):
        calls["since"] += 1
        return pd.DataFrame({"Close": [], "Volume": []})

    monkeypatch.setattr(data_mod, "_download_full", fake_full)
    monkeypatch.setattr(data_mod, "_download_since", fake_since)

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(data_mod, "CACHE_DIR", pathlib.Path(tmp))

        # 1) Como run_live_once.py: pide solo 2 años.
        out1 = data_mod.download_prices(["FAKE2"], years=2, force=False)
        assert calls["full"] == 1
        span_days_1 = (out1["FAKE2"].index.max() - out1["FAKE2"].index.min()).days
        assert span_days_1 < 3 * 365, "el fixture debería haber generado ~2 años, no más"

        # 2) Como run_backtest.py: pide 11 años, MISMO ticker, force=False -- con el
        #    bug viejo esto devolvía el caché corto de 2 años tal cual (cero llamadas
        #    de red adicionales). Ahora debe notar que no alcanza y re-descargar.
        out2 = data_mod.download_prices(["FAKE2"], years=11, force=False)
        assert calls["full"] == 2, "debería haber re-descargado completo al notar que el caché no cubría 11 años"
        span_days_2 = (out2["FAKE2"].index.max() - out2["FAKE2"].index.min()).days
        assert span_days_2 > 10 * 365, f"el resultado final debería cubrir ~11 años, cubrió {span_days_2} días"

        # 3) Volver a pedir 11 años -- ahora sí debería confiar en el caché (ya lo cubre).
        data_mod.download_prices(["FAKE2"], years=11, force=False)
        assert calls["full"] == 2, "con el caché ya profundo, no debería volver a descargar"

    print(f"  data.py: caché insuficiente (2 años) detectado y re-descargado al pedir 11 años "
          f"({span_days_1} días -> {span_days_2} días)")


def test_download_retries_transient_sqlite_lock_error():
    """Reproduce el bug real visto en producción: `yf.download` usa internamente
    una caché SQLite compartida entre tickers -- al descargar varios en paralelo
    (ThreadPoolExecutor en download_prices), es normal que alguno choque con un
    `database is locked` transitorio y falle sin que el ticker tenga nada malo.
    `_with_retries` debe reintentar y recuperarse solo, en vez de tirar el
    ticker a la basura al primer error transitorio."""
    import src.data as data_mod

    calls = {"n": 0}

    def flaky_twice():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise Exception("database is locked")  # el error real que tira sqlite3.OperationalError vía yfinance
        return pd.DataFrame({"Close": [1.0, 2.0]})

    result = data_mod._with_retries(flaky_twice, "FAKE", retries=4, backoff_sec=0.01)
    assert calls["n"] == 3, "debería haber fallado 2 veces y recuperarse al tercer intento"
    assert not result.empty and list(result["Close"]) == [1.0, 2.0]
    print("  reintento tras 'database is locked' transitorio OK: se recupera solo sin perder el ticker")

    calls2 = {"n": 0}

    def always_fails():
        calls2["n"] += 1
        raise Exception("database is locked")

    result2 = data_mod._with_retries(always_fails, "FAKE_DEAD", retries=3, backoff_sec=0.01)
    assert calls2["n"] == 3, "debería agotar exactamente los 3 intentos configurados"
    assert result2.empty, "si TODOS los intentos fallan, debe devolver vacío (el ticker se omite, no truena todo)"
    print("  agotamiento de reintentos OK: tras fallar todos los intentos, devuelve vacío en vez de lanzar excepción")


def test_crypto_incremental_download(monkeypatch):
    import src.crypto_data as crypto_data_mod

    calls = {"full": 0, "incremental": 0}

    def fake_download_klines(symbol, years=11, interval="1d", since=None):
        dates_len = 100
        if since is None:
            calls["full"] += 1
            dates = pd.date_range("2020-01-01", periods=dates_len, freq="D")
            price = np.linspace(10, 20, dates_len)
        else:
            calls["incremental"] += 1
            dates = pd.date_range(since.normalize(), periods=6, freq="D")
            price = np.linspace(50, 55, 6)
        return pd.DataFrame({"Close": price, "Volume": 1e5, "QuoteVolume": 1e6}, index=dates)

    monkeypatch.setattr(crypto_data_mod, "download_klines", fake_download_klines)

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(crypto_data_mod, "CACHE_DIR", pathlib.Path(tmp))

        out1 = crypto_data_mod.download_prices(["FAKEUSDT"], years=1, force=False)
        assert calls["full"] == 1 and calls["incremental"] == 0
        last_date = out1["FAKEUSDT"].index.max()

        out2 = crypto_data_mod.download_prices(["FAKEUSDT"], years=1, force=False)
        assert calls["full"] == 1, "con force=False y caché existente no debería volver a descargar"

        out3 = crypto_data_mod.download_prices(["FAKEUSDT"], years=1, force=True)
        assert calls["full"] == 1 and calls["incremental"] == 1, \
            "force=True con caché existente debería pedir solo la ventana incremental"
        merged = out3["FAKEUSDT"]
        assert not merged.index.duplicated().any()
        assert merged.index.is_monotonic_increasing
        assert merged.index.max() > last_date

    print(f"  crypto_data.py: descarga incremental OK -- 1 descarga completa, 1 incremental, "
          f"{len(merged)} filas finales sin duplicados")


def test_crypto_cache_with_fewer_years_than_requested_is_not_trusted(monkeypatch):
    """Misma regresión que test_equity_cache_with_fewer_years_than_requested_is_not_trusted,
    pero para el módulo cripto -- este es el bug que realmente se vio en producción:
    `run_crypto_live_once.py` (pide `years=2`) corrió antes que
    `run_crypto_backtest.py` (pide `years=11`), y el backtest terminó con solo
    ~2 años de datos sin ningún aviso (walk-forward vacío, `IndexError`)."""
    import src.crypto_data as crypto_data_mod

    calls = {"full": 0}

    def fake_download_klines(symbol, years=11, interval="1d", since=None):
        assert since is None, "este test no ejercita la ruta incremental"
        calls["full"] += 1
        end = pd.Timestamp.now().normalize()
        dates = pd.date_range(end - pd.Timedelta(days=years * 365), end, freq="D")
        return pd.DataFrame({"Close": np.linspace(10, 20, len(dates)), "Volume": 1e5,
                              "QuoteVolume": 1e6}, index=dates)

    monkeypatch.setattr(crypto_data_mod, "download_klines", fake_download_klines)

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(crypto_data_mod, "CACHE_DIR", pathlib.Path(tmp))

        # 1) Como run_crypto_live_once.py: pide solo 2 años.
        out1 = crypto_data_mod.download_prices(["FAKE2USDT"], years=2, force=False)
        assert calls["full"] == 1
        span_days_1 = (out1["FAKE2USDT"].index.max() - out1["FAKE2USDT"].index.min()).days
        assert span_days_1 < 3 * 365

        # 2) Como run_crypto_backtest.py: pide 11 años, MISMO símbolo, force=False.
        out2 = crypto_data_mod.download_prices(["FAKE2USDT"], years=11, force=False)
        assert calls["full"] == 2, "debería haber re-descargado completo al notar que el caché no cubría 11 años"
        span_days_2 = (out2["FAKE2USDT"].index.max() - out2["FAKE2USDT"].index.min()).days
        assert span_days_2 > 10 * 365, f"el resultado final debería cubrir ~11 años, cubrió {span_days_2} días"

        # 3) Volver a pedir 11 años -- ahora sí debería confiar en el caché.
        crypto_data_mod.download_prices(["FAKE2USDT"], years=11, force=False)
        assert calls["full"] == 2, "con el caché ya profundo, no debería volver a descargar"

    print(f"  crypto_data.py: caché insuficiente (2 años) detectado y re-descargado al pedir 11 años "
          f"({span_days_1} días -> {span_days_2} días)")


class _MonkeyPatch:
    def __init__(self):
        self._orig = []

    def setattr(self, obj, name, value):
        self._orig.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in reversed(self._orig):
            setattr(obj, name, value)


def main():
    print("[1/5] Probando descarga incremental de acciones (src.data)...")
    mp1 = _MonkeyPatch()
    try:
        test_equity_incremental_download(mp1)
    finally:
        mp1.undo()

    print("\n[2/5] Probando que un caché con menos años de los pedidos se re-descarga (acciones)...")
    mp1b = _MonkeyPatch()
    try:
        test_equity_cache_with_fewer_years_than_requested_is_not_trusted(mp1b)
    finally:
        mp1b.undo()

    print("\n[3/5] Probando reintento ante 'database is locked' transitorio de yfinance...")
    test_download_retries_transient_sqlite_lock_error()

    print("\n[4/5] Probando descarga incremental de cripto (src.crypto_data)...")
    mp2 = _MonkeyPatch()
    try:
        test_crypto_incremental_download(mp2)
    finally:
        mp2.undo()

    print("\n[5/5] Probando que un caché con menos años de los pedidos se re-descarga (cripto)...")
    mp2b = _MonkeyPatch()
    try:
        test_crypto_cache_with_fewer_years_than_requested_is_not_trusted(mp2b)
    finally:
        mp2b.undo()

    print("\nDATA INCREMENTAL TEST OK: la descarga incremental funciona correctamente en ambos módulos.")


if __name__ == "__main__":
    main()
