"""Descarga y cachea datos históricos OHLCV con yfinance."""
from __future__ import annotations

import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable

import pandas as pd
import yaml
import yfinance as yf

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_universe(path: pathlib.Path | None = None) -> dict:
    path = path or (ROOT / "config" / "universe.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def load_live_params(path: pathlib.Path | None = None) -> dict:
    """Único lugar de donde `run_backtest.py` y `run_live_once.py` leen los
    parámetros de riesgo (topes por tipo de activo, régimen macro, costos,
    impuestos) -- si estuvieran hardcodeados por separado en cada script,
    backtest y ejecución en vivo terminarían divergiendo silenciosamente."""
    path = path or (ROOT / "config" / "live_params.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_profile(profile: str = "conservative") -> tuple[pathlib.Path, pathlib.Path]:
    """`profile`: "conservative" (default, `live_params.yaml`) o "aggressive"
    (`live_params_aggressive.yaml`, ver README -- usa apalancamiento y es
    mucho más riesgoso). Devuelve (ruta_a_live_params, carpeta_de_reports) --
    cada perfil escribe en su propia carpeta de reports para no mezclar
    resultados de riesgos muy distintos en el mismo dashboard sin querer."""
    if profile == "aggressive":
        return ROOT / "config" / "live_params_aggressive.yaml", ROOT / "reports_aggressive"
    if profile != "conservative":
        raise ValueError(f"Perfil desconocido: {profile!r} (usa 'conservative' o 'aggressive')")
    return ROOT / "config" / "live_params.yaml", ROOT / "reports"


def all_tickers(universe: dict) -> list[str]:
    tickers = (set(universe["broad_etfs"]) | set(universe["sector_etfs"]) | set(universe["liquid_stocks"])
               | set(universe.get("leveraged_etfs", [])) | set(universe.get("international_etfs", []))
               | set(universe.get("diversifier_etfs", [])))
    tickers.add(universe["cash_proxy"])
    return sorted(tickers)


def _with_retries(fn: Callable[[], pd.DataFrame], ticker: str, retries: int = 4,
                   backoff_sec: float = 1.5) -> pd.DataFrame:
    """`yf.download` usa internamente una caché SQLite compartida (timezone/moneda
    por ticker) que no maneja bien escrituras concurrentes -- con varios tickers
    descargándose en paralelo (`ThreadPoolExecutor`, ver `download_prices`), es
    normal que alguno choque con un `database is locked` transitorio de SQLite y
    falle sin motivo real (no es un problema del ticker ni de la red). Es
    justamente el tipo de error donde reintentar con una pequeña espera casi
    siempre resuelve -- así que se reintenta unas pocas veces antes de rendirse
    de verdad y devolver vacío (lo que hace que el ticker se omita, como antes)."""
    last_error = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(backoff_sec * (attempt + 1))
    print(f"[WARN] {ticker}: falló tras {retries} intentos ({last_error}), se omite")
    return pd.DataFrame()


def _download_full(ticker: str, period: str) -> pd.DataFrame:
    def _do():
        df = yf.download(
            ticker, period=period, interval="1d", auto_adjust=True,
            progress=False, multi_level_index=False,
        )
        if not df.empty:
            df.index.name = "date"
        return df
    return _with_retries(_do, ticker)


def _download_since(ticker: str, start: pd.Timestamp) -> pd.DataFrame:
    def _do():
        df = yf.download(
            ticker, start=start.strftime("%Y-%m-%d"), interval="1d", auto_adjust=True,
            progress=False, multi_level_index=False,
        )
        if not df.empty:
            df.index.name = "date"
        return df
    return _with_retries(_do, ticker)


def download_prices(tickers: Iterable[str], years: int = 11, force: bool = False,
                     max_workers: int = 8, incremental_overlap_days: int = 5) -> dict[str, pd.DataFrame]:
    """Descarga OHLCV ajustado por dividendos/splits para cada ticker. Cachea en parquet.

    `years` incluye un buffer extra sobre los 10 años de análisis para que los
    indicadores (SMA200, momentum 12m) tengan suficiente historia de calentamiento
    desde el primer día evaluado.

    Tres casos, todos en paralelo entre sí (`max_workers` hilos, son llamadas de
    red no de CPU):
      - Ticker sin caché -> descarga completa (`years` atrás).
      - Ticker en caché y `force=False` -> se lee del disco, cero llamadas de red.
      - Ticker en caché y `force=True` (el caso de todos los días en vivo) -> NO se
        vuelve a descargar la historia completa. Solo se piden los últimos días
        (desde el último dato en caché, con `incremental_overlap_days` de margen
        por si Yahoo revisa un cierre reciente) y se pegan al caché existente. Antes
        de esto, cada corrida diaria re-bajaba años de historia por cada ticker --
        de minutos a segundos.
    """
    period = f"{years}y"
    out: dict[str, pd.DataFrame] = {}
    need_full: list[str] = []
    need_incremental: list[tuple[str, pd.DataFrame, pd.Timestamp]] = []

    for ticker in tickers:
        cache_file = CACHE_DIR / f"{ticker}.parquet"
        if not cache_file.exists():
            need_full.append(ticker)
            continue
        existing = pd.read_parquet(cache_file)
        if not force:
            out[ticker] = existing
            continue
        if existing.empty:
            need_full.append(ticker)
            continue
        start = existing.index.max() - pd.Timedelta(days=incremental_overlap_days)
        need_incremental.append((ticker, existing, start))

    if need_full or need_incremental:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for t in need_full:
                futures[executor.submit(_download_full, t, period)] = ("full", t, None)
            for t, existing, start in need_incremental:
                futures[executor.submit(_download_since, t, start)] = ("incremental", t, existing)

            for future in as_completed(futures):
                kind, ticker, existing = futures[future]
                new_df = future.result()

                if kind == "full":
                    if new_df.empty:
                        print(f"[WARN] sin datos para {ticker}, se omite")
                        continue
                    final_df = new_df
                else:
                    if new_df.empty:
                        final_df = existing  # sin datos nuevos (fin de semana, feriado) -- usa lo que ya había
                    else:
                        combined = pd.concat([existing, new_df])
                        final_df = combined[~combined.index.duplicated(keep="last")].sort_index()

                cache_file = CACHE_DIR / f"{ticker}.parquet"
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                final_df.to_parquet(cache_file)
                out[ticker] = final_df

    return out


def build_close_matrix(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = {t: df["Close"] for t, df in price_data.items()}
    matrix = pd.DataFrame(closes).sort_index()
    return matrix


def build_field_matrix(price_data: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    data = {t: df[field] for t, df in price_data.items() if field in df.columns}
    return pd.DataFrame(data).sort_index()


def build_position_caps(universe: dict, position_caps: dict) -> dict[str, float]:
    """Mapa ticker -> tope de peso máximo, según el tipo de activo (ETF amplio,
    ETF sectorial, o acción individual). Ver `config/live_params.yaml` ->
    `position_caps` y el comentario sobre sesgo de supervivencia en el README."""
    caps = {}
    for t in universe.get("broad_etfs", []):
        caps[t] = position_caps.get("broad_etf", 0.20)
    for t in universe.get("sector_etfs", []):
        caps[t] = position_caps.get("sector_etf", 0.40)
    for t in universe.get("liquid_stocks", []):
        caps[t] = position_caps.get("individual_stock", 0.08)
    for t in universe.get("leveraged_etfs", []):
        caps[t] = position_caps.get("leveraged_etf", 0.15)
    for t in universe.get("international_etfs", []):
        caps[t] = position_caps.get("international_etf", 0.15)
    for t in universe.get("diversifier_etfs", []):
        caps[t] = position_caps.get("diversifier_etf", 0.15)
    return caps


if __name__ == "__main__":
    universe = load_universe()
    tickers = all_tickers(universe)
    print(f"Descargando {len(tickers)} tickers: {tickers}")
    data = download_prices(tickers)
    close = build_close_matrix(data)
    print(close.tail())
    print(f"Rango de fechas: {close.index.min()} -> {close.index.max()}")
