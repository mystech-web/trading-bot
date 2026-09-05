"""Corre UNA vez el cálculo de señales del día para el módulo cripto y
rebalancea -- vía Binance testnet o el broker virtual local. Mismo patrón que
`run_live_once.py` (acciones): guardia de drawdown persistente, comparación
contra la banda del Monte Carlo, alertas.

Tres "brokers" posibles:
  --broker virtual (default)  -- simulación local con su propio capital inicial
                                   (--starting-cash, default $1000), sin ninguna
                                   cuenta ni API key. Recomendado para empezar.
  --broker binance              -- paper trading real contra el TESTNET de Binance
                                     (requiere BINANCE_API_KEY/SECRET_KEY en .env).
  --broker bitso                 -- Bitso NO TIENE TESTNET: esto mueve MXN real.
                                     Requiere BITSO_API_KEY/SECRET_KEY y
                                     BITSO_CONFIRM_REAL_MONEY=true en .env. Las
                                     señales se calculan igual (con datos de
                                     Binance), pero el precio de referencia para
                                     dimensionar cada orden se toma EN VIVO del
                                     ticker público de Bitso (MXN), y la cuenta/
                                     equity que reporta también está en MXN.

Uso:
    python scripts/run_crypto_live_once.py --execute                     # virtual, $1000, dry-run si falta --execute
    python scripts/run_crypto_live_once.py --broker binance --execute    # Binance testnet
    python scripts/run_crypto_live_once.py --broker bitso --execute      # Bitso real -- lee el README primero
"""
import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

from src.crypto_data import (
    load_crypto_universe, load_crypto_live_params, all_crypto_symbols, download_prices,
    build_close_matrix, build_crypto_position_caps,
)
from src.data_quality import flag_and_clean_outliers
from src.strategies import momentum, mean_reversion, sector_rotation
from src.ensemble import combine_weights
from src.portfolio_overlays import (
    sweep_idle_cash, compute_aggregate_correlation, correlation_based_cap_scale, tighten_caps_by_correlation,
    ramp_in_new_positions,
)
from src.event_blackout import load_macro_calendar, freeze_weights_on_blackout_days
from src.regime import compute_regime_scale
from src.live.virtual_broker import VirtualBroker
from src.notify import get_logger, send_alert
from src.tracking import append_equity, update_drawdown_guard, check_drift

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports_crypto"


def compute_target_weights(close, universe, live_params):
    momentum_universe = all_crypto_symbols(universe)
    sector_universe = list(universe["altcoins"])
    cash_ticker = universe["quote_currency"]
    benchmark = universe["benchmark"]

    position_caps = build_crypto_position_caps(universe, live_params["position_caps"])

    # Overlays de portafolio (ver src/portfolio_overlays.py) -- mismo mecanismo y
    # misma config que scripts/run_crypto_backtest.py.
    overlay_cfg = live_params.get("portfolio_overlays", {})
    sweep_cash_ticker = cash_ticker if overlay_cfg.get("cash_sweep_enabled", True) else None
    cap_scale = None
    if overlay_cfg.get("dynamic_caps_enabled", True):
        corr_universe = [t for t in momentum_universe if t in close.columns]
        agg_corr = compute_aggregate_correlation(close[corr_universe].pct_change(),
                                                   window=overlay_cfg.get("corr_window", 60))
        cap_scale = correlation_based_cap_scale(
            agg_corr, full_cap_below=overlay_cfg.get("full_cap_below", 0.3),
            floor_above=overlay_cfg.get("floor_above", 0.7), min_scale=overlay_cfg.get("min_scale", 0.6),
        )

    ramp_max_daily_increase = overlay_cfg.get("ramp_max_daily_increase", 0.02) \
        if overlay_cfg.get("ramp_in_enabled", True) else None

    # Blackout de eventos macro (FOMC, ver src/event_blackout.py) -- mismo
    # mecanismo y misma config que scripts/run_crypto_backtest.py.
    blackout_cfg = live_params.get("event_blackout", {})
    blackout_dates = load_macro_calendar() if blackout_cfg.get("fomc_blackout_enabled", True) else set()

    def _finish_weights(w):
        if cap_scale is not None:
            w = tighten_caps_by_correlation(w, position_caps, cap_scale)
        if ramp_max_daily_increase:
            w = ramp_in_new_positions(w, max_daily_increase=ramp_max_daily_increase, cash_ticker=sweep_cash_ticker)
        w = sweep_idle_cash(w, sweep_cash_ticker)
        return freeze_weights_on_blackout_days(w, blackout_dates)

    w_mom = _finish_weights(momentum.generate_weights(close, momentum_universe, live_params["momentum"], position_caps))
    w_mr = _finish_weights(mean_reversion.generate_weights(close, momentum_universe, live_params["mean_reversion"],
                                                              position_caps))
    w_rot = _finish_weights(sector_rotation.generate_weights(close, sector_universe, cash_ticker,
                                                                live_params["sector_rotation"]))

    combined = combine_weights(
        {"momentum": w_mom, "mean_reversion": w_mr, "sector_rotation": w_rot},
        live_params["allocation"],
    )

    regime_cfg = live_params["regime_filter"]
    if regime_cfg.get("enabled") and benchmark in close.columns:
        regime_scale = compute_regime_scale(close[benchmark], regime_cfg)
        combined = combined.mul(regime_scale.reindex(combined.index).fillna(1.0), axis=0)

    return combined.iloc[-1].to_dict(), combined.index[-1]


def run(args, logger) -> None:
    load_dotenv(ROOT / ".env")
    REPORTS_DIR.mkdir(exist_ok=True)
    # Subcarpeta por broker: virtual/binance/bitso NO comparten guardia de drawdown
    # ni historial de equity entre sí, aunque sean del mismo perfil cripto -- antes
    # de esto, correr dos brokers en paralelo mezclaba su tracking sin avisar.
    tracking_dir = REPORTS_DIR / args.broker
    tracking_dir.mkdir(exist_ok=True)

    universe = load_crypto_universe()
    live_params = load_crypto_live_params()
    symbols = all_crypto_symbols(universe)

    logger.info(f"Descargando/actualizando datos de {len(symbols)} símbolos de Binance...")
    data = download_prices(symbols, years=2, force=args.refresh_data)
    close = build_close_matrix(data, universe["quote_currency"])
    close, outlier_report = flag_and_clean_outliers(close, max_daily_return=0.60, verbose=False)
    if not outlier_report.empty:
        logger.warning(f"{len(outlier_report)} outlier(s) de precio detectados y limpiados antes de calcular "
                        f"la señal (ver src/data_quality.py): {outlier_report.to_dict('records')}")

    target_weights, as_of = compute_target_weights(close, universe, live_params)
    target_weights = {t: w for t, w in target_weights.items() if abs(w) > 1e-6}
    reference_prices = close.loc[as_of].to_dict()
    logger.info(f"Señal calculada con cierre de: {as_of.date()} (UTC)")

    if args.broker == "binance":
        from src.live.binance_broker import BinanceBroker
        broker = BinanceBroker(quote_currency=universe["quote_currency"])
    elif args.broker == "bitso":
        from src.live.bitso_broker import BitsoBroker, get_bitso_ticker_prices
        broker = BitsoBroker(quote_currency="mxn")
        reference_prices = get_bitso_ticker_prices(symbols, quote_currency="mxn")
        logger.info(f"Precios de referencia tomados EN VIVO del ticker de Bitso (MXN), no de Binance. "
                    f"{len(reference_prices)}/{len(symbols)} símbolos tienen libro en Bitso.")
    else:
        state_file = tracking_dir / "virtual_broker_state.json"
        is_first_run = not state_file.exists()
        broker = VirtualBroker(state_file, starting_cash=args.starting_cash)
        if is_first_run:
            logger.info(f"Broker virtual: primera corrida, capital inicial ${args.starting_cash:,.2f}.")
    equity = broker.get_equity(reference_prices)

    append_equity(pd.Timestamp.today(), equity, reports_dir=tracking_dir)
    dd, exposure_scale, guard_active, changed = update_drawdown_guard(equity, reports_dir=tracking_dir)
    logger.info(f"Equity actual: ${equity:,.2f} | drawdown desde el pico: {dd * 100:.2f}% | "
                f"guardia de drawdown activa: {guard_active} (escala de exposición: {exposure_scale * 100:.0f}%)")

    if changed:
        estado = "ACTIVADA (exposición reducida)" if guard_active else "DESACTIVADA (exposición 100% de vuelta)"
        send_alert(f"Bot cripto: guardia de drawdown {estado}",
                   f"Drawdown actual desde el pico: {dd * 100:.2f}%. Equity: ${equity:,.2f}. "
                   f"Escala de exposición: {exposure_scale * 100:.0f}%.", logger)

    if exposure_scale < 1.0:
        target_weights = {t: w * exposure_scale for t, w in target_weights.items()}
        logger.info(f"Guardia de drawdown activa: exposición escalada a {exposure_scale * 100:.0f}% "
                    f"(reentrada gradual, no instantánea).")

    drift_msg = check_drift(reports_dir=tracking_dir, band_dir=REPORTS_DIR)
    if drift_msg:
        logger.warning(drift_msg)
        send_alert("Bot cripto: posible decay de la estrategia", drift_msg, logger)

    total = sum(target_weights.values())
    logger.info("Pesos objetivo (fracción del equity):")
    for t, w in sorted(target_weights.items(), key=lambda kv: -kv[1]):
        logger.info(f"  {t:8s} {w * 100:6.2f}%")
    logger.info(f"  {'CASH':8s} {(1 - total) * 100:6.2f}%")

    cost_bps = live_params["costs"]["base_spread_bps"]
    max_slippage_pct = live_params["execution"]["max_slippage_pct"]
    min_weight_drift = live_params["execution"].get("min_weight_drift", 0.02)
    kwargs = dict(cost_bps=cost_bps) if args.broker == "virtual" else dict(max_slippage_pct=max_slippage_pct)
    kwargs["min_weight_drift"] = min_weight_drift
    orders = broker.rebalance_to_weights(target_weights, reference_prices, dry_run=not args.execute, **kwargs)

    modo = "Órdenes ejecutadas" if args.execute else "Órdenes que se enviarían (dry-run, nada se ejecutó)"
    logger.info(f"{modo}:")
    if not orders:
        logger.info("  (ninguna -- la cuenta ya está alineada con los pesos objetivo, o los deltas son muy chicos)")
    for o in orders:
        err = f" [ERROR: {o['error']}]" if o.get("error") else ""
        logger.info(f"  {o['side'].upper():4s} {o['qty']:>14.6f} de {o['ticker']} @ ${o['limit_price']:.4f} "
                    f"(~${o['notional']:.2f}){err}")

    # Verificación de fills: broker.rebalance_to_weights ya hizo polling del estado
    # real de cada orden y canceló lo que no se llenó a tiempo (ver
    # src/live/binance_broker.py y src/live/bitso_broker.py). Acá solo se avisa si
    # algo quedó sin llenar -- "FILLED" (Binance) / "completed" (Bitso) son el único
    # estado de éxito, cualquier otro (incluida una orden cancelada por timeout) se reporta.
    if args.execute and args.broker != "virtual":
        filled_status = {"binance": "FILLED", "bitso": "completed"}.get(args.broker)
        unfilled = [o for o in orders if o.get("fill_status") != filled_status and not o.get("error")]
        if unfilled:
            detail = "; ".join(f"{o['ticker']} {o['side']}: {o.get('fill_status')}" for o in unfilled)
            msg = f"{len(unfilled)} orden(es) NO se llenaron por completo y se cancelaron: {detail}"
            logger.warning(msg)
            send_alert(f"Bot cripto ({args.broker}): órdenes sin llenar", msg, logger)
        elif orders:
            logger.info(f"Verificación de fills OK: las {len(orders)} orden(es) se llenaron por completo.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", choices=["virtual", "binance", "bitso"], default="virtual",
                         help="'virtual' (default, sin cuenta, capital propio), 'binance' (testnet real), "
                              "o 'bitso' (SIN testnet -- mueve MXN real, requiere confirmación explícita).")
    parser.add_argument("--starting-cash", type=float, default=1000.0,
                         help="Capital inicial del broker virtual (default $1000, ignorado con --broker binance).")
    parser.add_argument("--execute", action="store_true",
                         help="Ejecuta de verdad (Binance testnet real, o persiste el estado virtual).")
    parser.add_argument("--refresh-data", action="store_true", default=True,
                         help="Re-descarga precios antes de calcular señales (default: sí).")
    args = parser.parse_args()

    # Log separado por broker -- si no, virtual/binance/bitso terminan escribiendo
    # el mismo reports_crypto/live.log mezclado entre sí.
    logger = get_logger("crypto-bot", log_dir=REPORTS_DIR / args.broker)
    try:
        run(args, logger)
    except Exception as e:
        logger.exception("Error en la corrida diaria del bot cripto")
        send_alert("Bot cripto: ERROR en la corrida diaria", f"{type(e).__name__}: {e}", logger)
        raise


if __name__ == "__main__":
    main()
