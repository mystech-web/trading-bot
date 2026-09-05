"""Descarga y cachea velas diarias (klines) de Binance -- el equivalente de
`src/data.py` (Yahoo Finance) para el universo cripto.

Usa el endpoint público `/api/v3/klines` (sin API key -- es dato de mercado
público) vía `requests` directo, paginando hacia atrás porque Binance limita
cada respuesta a 1000 velas.
"""
from __future__ import annotations

import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import pandas as pd
import requests
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache_crypto"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BINANCE_BASE_URL = "https://api.binance.com"
KLINES_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def load_crypto_universe(path: pathlib.Path | None = None) -> dict:
    path = path or (ROOT / "config" / "crypto_universe.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def load_crypto_live_params(path: pathlib.Path | None = None) -> dict:
    path = path or (ROOT / "config" / "crypto_live_params.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def all_crypto_symbols(universe: dict) -> list[str]:
    return sorted(set(universe["majors"]) | set(universe["altcoins"]))


def _fetch_klines_page(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> list:
    resp = requests.get(
        f"{BINANCE_BASE_URL}/api/v3/klines",
        params=dict(symbol=symbol, interval=interval, startTime=start_ms, endTime=end_ms, limit=limit),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def download_klines(symbol: str, years: int = 11, interval: str = "1d",
                     since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Descarga velas diarias hasta hoy, paginando en bloques de 1000 (el máximo
    por request de Binance). Por defecto arranca `years` atrás; si se pasa
    `since`, arranca ahí en cambio (para actualizaciones incrementales -- ver
    `download_prices`). Si el símbolo se listó después de esa fecha (la mayoría
    de altcoins), simplemente empieza desde su primer dato disponible -- no es
    un error, Binance lo maneja devolviendo lo que tenga."""
    end_ms = int(time.time() * 1000)
    start_ms = int(since.timestamp() * 1000) if since is not None else end_ms - years * 365 * 24 * 60 * 60 * 1000

    all_rows = []
    cursor = start_ms
    while cursor < end_ms:
        page = _fetch_klines_page(symbol, interval, cursor, end_ms)
        if not page:
            break
        all_rows.extend(page)
        last_open_time = page[-1][0]
        if last_open_time <= cursor:
            break
        cursor = last_open_time + 1
        if len(page) < 1000:
            break
        time.sleep(0.2)  # cortesía con el rate limit público de Binance

    if not all_rows:
        return pd.DataFrame(columns=["Close", "Volume", "QuoteVolume"])

    df = pd.DataFrame(all_rows, columns=KLINES_COLUMNS)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    df = df.drop_duplicates(subset="date").set_index("date").sort_index()
    out = pd.DataFrame({
        "Close": df["close"].astype(float),
        "Volume": df["volume"].astype(float),
        "QuoteVolume": df["quote_volume"].astype(float),  # volumen en USDT -- más preciso que Close*Volume
    })
    return out


def download_prices(symbols: Iterable[str], years: int = 11, force: bool = False,
                     max_workers: int = 5, incremental_overlap_days: int = 3) -> dict[str, pd.DataFrame]:
    """Igual que `src.data.download_prices`: caché en disco si `force=False`; si
    `force=True` y ya hay caché, NO se re-baja todo el historial -- solo las
    velas desde el último día cacheado (con `incremental_overlap_days` de
    margen), pegadas al caché existente. Lo que sí hace falta bajar (completo o
    incremental) se pide en paralelo (`max_workers` hilos, moderado a propósito:
    cada símbolo ya pagina internamente con cortesía de rate-limit)."""
    out: dict[str, pd.DataFrame] = {}
    need_full: list[str] = []
    need_incremental: list[tuple[str, pd.DataFrame, pd.Timestamp]] = []

    for symbol in symbols:
        cache_file = CACHE_DIR / f"{symbol}.parquet"
        if not cache_file.exists():
            need_full.append(symbol)
            continue
        existing = pd.read_parquet(cache_file)
        if not force:
            out[symbol] = existing
            continue
        if existing.empty:
            need_full.append(symbol)
            continue
        start = existing.index.max() - pd.Timedelta(days=incremental_overlap_days)
        need_incremental.append((symbol, existing, start))

    if need_full or need_incremental:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for s in need_full:
                futures[executor.submit(download_klines, s, years)] = ("full", s, None)
            for s, existing, start in need_incremental:
                futures[executor.submit(download_klines, s, years, "1d", start)] = ("incremental", s, existing)

            for future in as_completed(futures):
                kind, symbol, existing = futures[future]
                new_df = future.result()

                if kind == "full":
                    if new_df.empty:
                        print(f"[WARN] sin datos para {symbol}, se omite")
                        continue
                    final_df = new_df
                else:
                    if new_df.empty:
                        final_df = existing
                    else:
                        combined = pd.concat([existing, new_df])
                        final_df = combined[~combined.index.duplicated(keep="last")].sort_index()

                final_df.to_parquet(CACHE_DIR / f"{symbol}.parquet")
                out[symbol] = final_df

    return out


def build_close_matrix(price_data: dict[str, pd.DataFrame], quote_currency: str = "USDT") -> pd.DataFrame:
    """Igual que `src.data.build_close_matrix`, pero además agrega una columna
    sintética plana para `quote_currency` (USDT) -- no se descarga (no es un
    par tradeable contra sí mismo), representa "cash" a precio constante $1
    para que el filtro de momentum absoluto de `sector_rotation.py` funcione
    igual que con el proxy de T-Bills (BIL) del universo de acciones."""
    closes = {t: df["Close"] for t, df in price_data.items()}
    matrix = pd.DataFrame(closes).sort_index()
    matrix[quote_currency] = 1.0
    return matrix


def build_crypto_position_caps(universe: dict, position_caps: dict) -> dict[str, float]:
    caps = {}
    for t in universe.get("majors", []):
        caps[t] = position_caps.get("major_coin", 0.30)
    for t in universe.get("altcoins", []):
        caps[t] = position_caps.get("altcoin", 0.15)
    return caps


def build_quote_volume_matrix(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Volumen en USDT (más preciso que Close*Volume para el modelo de costos
    por liquidez, ver src/costs.py) -- Binance ya lo reporta directamente."""
    data = {t: df["QuoteVolume"] for t, df in price_data.items() if "QuoteVolume" in df.columns}
    return pd.DataFrame(data).sort_index()


if __name__ == "__main__":
    universe = load_crypto_universe()
    symbols = all_crypto_symbols(universe)
    print(f"Descargando {len(symbols)} símbolos: {symbols}")
    data = download_prices(symbols)
    close = build_close_matrix(data, universe["quote_currency"])
    print(close.tail())
    print(f"Rango de fechas: {close.index.min()} -> {close.index.max()}")
