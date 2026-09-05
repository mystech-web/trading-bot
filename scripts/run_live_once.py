"""Corre UNA vez el cálculo de señales del día y rebalancea la cuenta paper de Alpaca.

Pensado para ejecutarse una vez por día hábil vía cron (Mac/Linux) o Task
Scheduler (Windows) -- ver README.md para configurarlo.

Por defecto corre en modo "dry-run" (solo imprime qué órdenes haría, no las
envía). Pasa --execute para que efectivamente rebalancee la cuenta PAPER.

Cada corrida:
  - registra el equity actual en reports/<broker>/tracking.sqlite3 (una subcarpeta
    por broker -- alpaca y virtual del mismo perfil NO comparten guardia de drawdown)
  - actualiza una guardia de drawdown PERSISTENTE (sobrevive entre corridas) y
    escala la exposición a la mitad si el drawdown pasa -15%, igual que en el backtest
  - compara el retorno realizado en vivo contra la banda esperada del Monte Carlo
    del backtest (reports/monte_carlo.json, compartido entre brokers del mismo
    perfil -- lo genera run_backtest.py) y alerta si hay señales de decay
  - manda una alerta (email/Telegram, si están configurados en .env) si algo falla,
    si la guardia de drawdown cambia de estado, o si detecta drift

Dos "brokers" posibles:
  --broker alpaca (default)  -- paper trading real contra tu cuenta de Alpaca
                                 (requiere ALPACA_API_KEY/SECRET_KEY en .env).
  --broker virtual            -- simulación local con SU PROPIO capital inicial
                                  (--starting-cash, default $1000), sin ninguna
                                  cuenta ni API key. Útil para correr los 2
                                  perfiles EN PARALELO cada uno con $1000 y
                                  comparar cómo avanzan (ver README).

Uso:
    python scripts/run_live_once.py                              # perfil conservador, Alpaca, dry-run
    python scripts/run_live_once.py --execute                     # perfil conservador, Alpaca, ejecuta en paper
    python scripts/run_live_once.py --profile aggressive --execute  # perfil agresivo -- ver README primero

    # Los dos perfiles en paralelo, cada uno con $1000, sin Alpaca:
    python scripts/run_live_once.py --broker virtual --starting-cash 1000 --execute
    python scripts/run_live_once.py --broker virtual --starting-cash 1000 --profile aggressive --execute
"""
import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

from src.data import load_universe, load_live_params, resolve_profile, all_tickers, download_prices, \
    build_close_matrix, build_position_caps
from src.data_quality import flag_and_clean_outliers
from src.strategies import momentum, mean_reversion, sector_rotation
from src.ensemble import combine_weights
from src.regime import compute_regime_scale
from src.live.alpaca_broker import AlpacaBroker, MarketClosedError
from src.live.virtual_broker import VirtualBroker
from src.notify import get_logger, send_alert
from src.tracking import append_equity, update_drawdown_guard, check_drift, load_state, save_state
from src.tax_loss_harvesting import find_harvest_candidates, block_recent_harvest_rebuys, prune_expired_harvests, \
    DEFAULT_LOSS_THRESHOLD_PCT, DEFAULT_WASH_SALE_DAYS
from src.portfolio_overlays import (
    sweep_idle_cash, compute_aggregate_correlation, correlation_based_cap_scale, tighten_caps_by_correlation,
    ramp_in_new_positions,
)
from src.event_blackout import (
    load_macro_calendar, freeze_weights_on_blackout_days, get_earnings_blackout_tickers, apply_earnings_blackout,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def compute_target_weights(close, universe, live_params, vix_close=None):
    # include_satellite_etfs (default False, ver config/live_params.yaml): un backtest
    # real mostró que EFA/EEM/VNQ/DBC/IEF diluían el retorno mensual promedio del
    # ensamble por debajo del objetivo -- quedan fuera de momentum/mean_reversion por
    # default. MISMA lógica que scripts/run_backtest.py -- si un cambio futuro agrega
    # otra categoría al universo, actualiza los DOS lugares (ver tests/exclude_satellite_test.py
    # y tests/diversifier_etfs_test.py, que existen justo para detectar ese tipo de bug).
    if live_params.get("include_satellite_etfs", False):
        base_universe = sorted(set(universe["broad_etfs"]) | set(universe["liquid_stocks"])
                                | set(universe.get("international_etfs", []))
                                | set(universe.get("diversifier_etfs", [])))
    else:
        base_universe = sorted(set(universe["broad_etfs"]) | set(universe["liquid_stocks"]))
    base_universe = [t for t in base_universe if t in close.columns]
    mean_reversion_universe = base_universe
    momentum_universe = base_universe
    if live_params.get("include_leveraged_etfs"):
        leveraged = [t for t in universe.get("leveraged_etfs", []) if t in close.columns]
        momentum_universe = sorted(set(base_universe) | set(leveraged))
    sector_universe = [t for t in universe["sector_etfs"] if t in close.columns]
    cash_ticker = universe["cash_proxy"]
    benchmark = universe["benchmark"]

    position_caps = build_position_caps(universe, live_params["position_caps"])

    # Overlays de portafolio (ver src/portfolio_overlays.py): barrido de cash ocioso
    # hacia el proxy de cash, y (opcional) topes de posición más estrictos en
    # momentos de correlación agregada alta entre todo el universo -- mismo
    # mecanismo y misma config que scripts/run_backtest.py.
    overlay_cfg = live_params.get("portfolio_overlays", {})
    sweep_cash_ticker = cash_ticker if overlay_cfg.get("cash_sweep_enabled", True) else None
    cap_scale = None
    if overlay_cfg.get("dynamic_caps_enabled", True):
        corr_universe = sorted(set(universe["broad_etfs"]) | set(universe["sector_etfs"])
                                | set(universe["liquid_stocks"]))
        if live_params.get("include_satellite_etfs", False):
            corr_universe = sorted(set(corr_universe) | set(universe.get("international_etfs", []))
                                    | set(universe.get("diversifier_etfs", [])))
        corr_universe = [t for t in corr_universe if t in close.columns]
        agg_corr = compute_aggregate_correlation(close[corr_universe].pct_change(),
                                                   window=overlay_cfg.get("corr_window", 60))
        cap_scale = correlation_based_cap_scale(
            agg_corr, full_cap_below=overlay_cfg.get("full_cap_below", 0.3),
            floor_above=overlay_cfg.get("floor_above", 0.7), min_scale=overlay_cfg.get("min_scale", 0.6),
        )

    ramp_max_daily_increase = overlay_cfg.get("ramp_max_daily_increase", 0.02) \
        if overlay_cfg.get("ramp_in_enabled", True) else None

    # Blackout de eventos macro (FOMC, ver src/event_blackout.py) -- mismo
    # mecanismo y misma config que scripts/run_backtest.py.
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
    w_mr = _finish_weights(mean_reversion.generate_weights(close, mean_reversion_universe, live_params["mean_reversion"],
                                                             position_caps))
    w_rot = _finish_weights(sector_rotation.generate_weights(close, sector_universe, cash_ticker,
                                                               live_params["sector_rotation"]))

    combined = combine_weights(
        {"momentum": w_mom, "mean_reversion": w_mr, "sector_rotation": w_rot},
        live_params["allocation"],
    )

    regime_cfg = live_params["regime_filter"]
    if regime_cfg.get("enabled") and benchmark in close.columns:
        regime_scale = compute_regime_scale(close[benchmark], regime_cfg, vix_close=vix_close)
        combined = combined.mul(regime_scale.reindex(combined.index).fillna(1.0), axis=0)

    return combined.iloc[-1].to_dict(), combined.index[-1]


def run(args, logger) -> None:
    load_dotenv(ROOT / ".env")

    live_params_path, reports_dir = resolve_profile(args.profile)
    reports_dir.mkdir(exist_ok=True)
    # Subcarpeta por broker: si corres --broker alpaca Y --broker virtual del MISMO
    # perfil (ej. para comparar), cada uno lleva su propia guardia de drawdown y su
    # propio historial de equity -- antes de esto, compartían el mismo archivo y se
    # mezclaban entre sí sin que nada lo avisara.
    tracking_dir = reports_dir / args.broker
    tracking_dir.mkdir(exist_ok=True)
    if args.profile == "aggressive":
        logger.warning("PERFIL AGRESIVO activo (apalancamiento, más riesgo) -- ver README antes de usar --execute.")

    universe = load_universe()
    live_params = load_live_params(live_params_path)
    tickers = all_tickers(universe)

    logger.info(f"Descargando/actualizando datos de {len(tickers)} tickers...")
    data = download_prices(tickers, years=2, force=args.refresh_data)
    close = build_close_matrix(data)
    close, outlier_report = flag_and_clean_outliers(close, verbose=False)
    if not outlier_report.empty:
        logger.warning(f"{len(outlier_report)} outlier(s) de precio detectados y limpiados antes de calcular "
                        f"la señal (ver src/data_quality.py): {outlier_report.to_dict('records')}")

    regime_cfg_check = live_params["regime_filter"]
    vix_close = None
    if regime_cfg_check.get("enabled") and regime_cfg_check.get("vix_enabled", True):
        vix_raw = download_prices(["^VIX"], years=2, force=args.refresh_data)
        vix_matrix = build_close_matrix(vix_raw)
        if "^VIX" in vix_matrix.columns and not vix_matrix["^VIX"].dropna().empty:
            vix_close = vix_matrix["^VIX"]
        else:
            logger.warning("No se pudo descargar VIX -- el filtro de régimen sigue funcionando solo con "
                            "tendencia de precio + volatilidad realizada.")

    target_weights, as_of = compute_target_weights(close, universe, live_params, vix_close=vix_close)
    target_weights = {t: w for t, w in target_weights.items() if abs(w) > 1e-6}
    reference_prices = close.loc[as_of].to_dict()
    logger.info(f"Señal calculada con cierre de: {as_of.date()}")

    # Blackout de earnings (ver src/event_blackout.py) -- SOLO acciones individuales,
    # SOLO en vivo (yfinance no expone historial point-in-time, así que esto nunca
    # aparece en ningún backtest -- ver el docstring del módulo). "Best effort": si
    # la consulta a yfinance falla, no bloquea nada (falla del lado seguro).
    earnings_cfg = live_params.get("event_blackout", {})
    if earnings_cfg.get("earnings_blackout_enabled", True):
        stock_tickers = [t for t in universe.get("liquid_stocks", []) if t in target_weights]
        if stock_tickers:
            try:
                blackout_tickers = get_earnings_blackout_tickers(
                    stock_tickers, as_of.date(),
                    days_before=earnings_cfg.get("earnings_blackout_days_before", 1),
                    days_after=earnings_cfg.get("earnings_blackout_days_after", 1),
                )
            except Exception as e:
                logger.warning(f"Blackout de earnings: falló la consulta a yfinance ({e}) -- no se bloquea "
                                f"ningún ticker esta corrida (falla del lado seguro).")
                blackout_tickers = set()
            if blackout_tickers:
                logger.info(f"Blackout de earnings: se excluyen {sorted(blackout_tickers)} "
                             f"(reporte de resultados cerca de hoy).")
                target_weights = apply_earnings_blackout(target_weights, blackout_tickers)

    if args.broker == "virtual":
        state_file = tracking_dir / "virtual_broker_state.json"
        is_first_run = not state_file.exists()
        broker = VirtualBroker(state_file, starting_cash=args.starting_cash)
        if is_first_run:
            logger.info(f"Broker virtual: primera corrida para el perfil '{args.profile}', "
                        f"capital inicial ${args.starting_cash:,.2f} (sin cuenta real de por medio).")
    else:
        broker = AlpacaBroker()
    equity = broker.get_equity(reference_prices)

    append_equity(pd.Timestamp.today(), equity, reports_dir=tracking_dir)
    dd, exposure_scale, guard_active, changed = update_drawdown_guard(equity, reports_dir=tracking_dir)
    logger.info(f"Equity actual: ${equity:,.2f} | drawdown desde el pico: {dd * 100:.2f}% | "
                f"guardia de drawdown activa: {guard_active} (escala de exposición: {exposure_scale * 100:.0f}%)")

    if changed:
        estado = "ACTIVADA (exposición reducida)" if guard_active else "DESACTIVADA (exposición 100% de vuelta)"
        send_alert(
            f"Bot de trading ({args.profile}): guardia de drawdown {estado}",
            f"Drawdown actual desde el pico: {dd * 100:.2f}%. Equity: ${equity:,.2f}. "
            f"Escala de exposición: {exposure_scale * 100:.0f}%.",
            logger,
        )

    if exposure_scale < 1.0:
        target_weights = {t: w * exposure_scale for t, w in target_weights.items()}
        logger.info(f"Guardia de drawdown activa: exposición de todas las posiciones escalada a "
                    f"{exposure_scale * 100:.0f}% (reentrada gradual, no instantánea).")

    # ---------- Tax-loss harvesting (opt-in, ver src/tax_loss_harvesting.py) ----------
    # Alcance: solo funciona con --broker virtual o alpaca (los únicos que exponen
    # cost basis por posición) -- ver el docstring del módulo para el porqué.
    tax_cfg = live_params.get("tax", {})
    if tax_cfg.get("harvest_losses_enabled", False) and hasattr(broker, "get_positions_with_cost_basis"):
        wash_sale_days = tax_cfg.get("wash_sale_days", DEFAULT_WASH_SALE_DAYS)
        today = pd.Timestamp.today().date()
        positions = broker.get_positions_with_cost_basis(reference_prices)
        candidates = find_harvest_candidates(
            positions, loss_threshold_pct=tax_cfg.get("harvest_loss_threshold_pct", DEFAULT_LOSS_THRESHOLD_PCT),
        )

        recently_harvested = load_state(reports_dir=tracking_dir).get("recently_harvested", {})
        if candidates:
            lines = []
            for c in candidates:
                target_weights[c["ticker"]] = 0.0
                recently_harvested[c["ticker"]] = today.isoformat()
                est_loss_usd = (c["current_price"] - c["avg_entry_price"]) * c["qty"]
                lines.append(f"{c['ticker']}: {c['unrealized_plpc'] * 100:.1f}% (~${est_loss_usd:,.2f})")
                logger.info(f"Tax-loss harvesting: se vende {c['ticker']} para realizar una pérdida de "
                            f"{c['unrealized_plpc'] * 100:.1f}% (~${est_loss_usd:,.2f}).")
            send_alert(
                f"Bot de trading ({args.profile}, {args.broker}): tax-loss harvesting",
                "Se venden las siguientes posiciones para realizar pérdidas fiscales:\n" + "\n".join(lines) +
                f"\n\nNo se recomprarán automáticamente por {wash_sale_days} días (regla de wash sale). "
                "Esto NO es asesoría fiscal -- confirma el tratamiento con tu contador.",
                logger,
            )

        recently_harvested = prune_expired_harvests(recently_harvested, today, wash_sale_days)
        target_weights = block_recent_harvest_rebuys(target_weights, recently_harvested, today, wash_sale_days)
        save_state({"recently_harvested": recently_harvested}, reports_dir=tracking_dir)

    drift_msg = check_drift(reports_dir=tracking_dir, band_dir=reports_dir)
    if drift_msg:
        logger.warning(drift_msg)
        send_alert(f"Bot de trading ({args.profile}): posible decay de la estrategia", drift_msg, logger)

    total = sum(target_weights.values())
    logger.info("Pesos objetivo (fracción del equity de la cuenta):")
    for t, w in sorted(target_weights.items(), key=lambda kv: -kv[1]):
        logger.info(f"  {t:6s} {w * 100:6.2f}%")
    logger.info(f"  {'CASH':6s} {(1 - total) * 100:6.2f}%")

    min_weight_drift = live_params["execution"].get("min_weight_drift", 0.02)
    if args.broker == "virtual":
        cost_bps = live_params["costs"]["base_spread_bps"]
        orders = broker.rebalance_to_weights(target_weights, reference_prices, cost_bps=cost_bps,
                                              min_weight_drift=min_weight_drift, dry_run=not args.execute)
        precio_label = "precio simulado"
    else:
        max_slippage_pct = live_params["execution"]["max_slippage_pct"]
        exec_cfg = live_params["execution"]
        try:
            orders = broker.rebalance_to_weights(
                target_weights, reference_prices, max_slippage_pct=max_slippage_pct,
                min_weight_drift=min_weight_drift, dry_run=not args.execute,
                twap_threshold_usd=exec_cfg.get("twap_threshold_usd", 5000.0),
                twap_max_slices=exec_cfg.get("twap_max_slices", 5),
                twap_slice_delay_sec=exec_cfg.get("twap_slice_delay_sec", 3.0),
            )
        except MarketClosedError as e:
            logger.warning(str(e))
            return
        precio_label = "límite"

    if args.execute:
        modo = "Órdenes ejecutadas (virtual)" if args.broker == "virtual" else "Órdenes LÍMITE enviadas"
    else:
        modo = "Órdenes que se enviarían (dry-run, nada se ejecutó)"
    logger.info(f"{modo}:")
    if not orders:
        logger.info("  (ninguna -- la cuenta ya está alineada con los pesos objetivo, o los deltas son muy chicos)")
    for o in orders:
        twap_note = f" [dividida en {o['n_slices']} porciones TWAP]" if o.get("n_slices", 1) > 1 else ""
        logger.info(f"  {o['side'].upper():4s} {o['qty']:>10.4f} de {o['ticker']} @ {precio_label} "
                    f"${o['limit_price']:.2f} (~${o['notional']:.2f}){twap_note}")

    # Verificación de fills: una orden LÍMITE enviada no es lo mismo que una orden
    # EJECUTADA -- broker.rebalance_to_weights ya hizo polling del estado real y
    # canceló lo que no se llenó a tiempo (ver src/live/alpaca_broker.py). Acá solo
    # se avisa si algo quedó sin llenar, para que no pase desapercibido.
    if args.execute and args.broker != "virtual":
        unfilled = [o for o in orders if o.get("fill_status") not in ("filled",) and not o.get("error")]
        if unfilled:
            detail = "; ".join(f"{o['ticker']} {o['side']}: {o.get('fill_status')}" for o in unfilled)
            msg = f"{len(unfilled)} orden(es) NO se llenaron por completo y se cancelaron: {detail}"
            logger.warning(msg)
            send_alert(f"Bot de trading ({args.profile}, {args.broker}): órdenes sin llenar", msg, logger)
        elif orders:
            logger.info(f"Verificación de fills OK: las {len(orders)} orden(es) se llenaron por completo.")

    if args.broker == "virtual":
        equity_after = broker.get_equity(reference_prices)
        logger.info(f"Equity {'después de rebalancear' if args.execute else 'estimado si se ejecutara'}: "
                    f"${equity_after:,.2f} (capital inicial: ${broker.state['starting_cash']:,.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["conservative", "aggressive"], default="conservative",
                         help="'conservative' (default) o 'aggressive' (apalancado -- ver README antes de usarlo).")
    parser.add_argument("--broker", choices=["alpaca", "virtual"], default="alpaca",
                         help="'alpaca' (default, requiere .env) o 'virtual' (simulación local, sin cuenta, "
                              "con su propio capital inicial -- ver --starting-cash).")
    parser.add_argument("--starting-cash", type=float, default=1000.0,
                         help="Capital inicial del broker virtual (solo aplica la primera vez que corre "
                              "para un perfil dado; ignorado con --broker alpaca). Default: $1000.")
    parser.add_argument("--execute", action="store_true",
                         help="Ejecuta de verdad (paper real con Alpaca, o persiste el estado con el broker "
                              "virtual). Sin esto, solo dry-run: muestra qué haría, no cambia nada.")
    parser.add_argument("--refresh-data", action="store_true", default=True,
                         help="Re-descarga precios antes de calcular señales (default: sí).")
    args = parser.parse_args()

    # Log separado por perfil+broker (mismo criterio que el tracking de equity) --
    # si no, conservador/agresivo y alpaca/virtual terminan escribiendo el mismo
    # reports/live.log mezclado entre sí.
    _, reports_dir = resolve_profile(args.profile)
    logger = get_logger(log_dir=reports_dir / args.broker)
    try:
        run(args, logger)
    except Exception as e:
        logger.exception("Error en la corrida diaria del bot")
        send_alert(f"Bot de trading ({args.profile}): ERROR en la corrida diaria", f"{type(e).__name__}: {e}", logger)
        raise


if __name__ == "__main__":
    main()
