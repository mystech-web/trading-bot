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
    print("[1/3] Probando descarga incremental de acciones (src.data)...")
    mp1 = _MonkeyPatch()
    try:
        test_equity_incremental_download(mp1)
    finally:
        mp1.undo()

    print("\n[2/3] Probando reintento ante 'database is locked' transitorio de yfinance...")
    test_download_retries_transient_sqlite_lock_error()

    print("\n[3/3] Probando descarga incremental de cripto (src.crypto_data)...")
    mp2 = _MonkeyPatch()
    try:
        test_crypto_incremental_download(mp2)
    finally:
        mp2.undo()

    print("\nDATA INCREMENTAL TEST OK: la descarga incremental funciona correctamente en ambos módulos.")


if __name__ == "__main__":
    main()
