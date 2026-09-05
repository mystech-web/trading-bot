#!/bin/bash
# Instala 2 tareas programadas en tu Mac usando launchd para el MÓDULO CRIPTO
# (BTC/ETH/altcoins contra USDT, ver config/crypto_universe.yaml):
#   1) Diaria, TODOS LOS DÍAS (a diferencia del bot de acciones, cripto cotiza
#      365 días/año -- no hay "días hábiles" ni horario de mercado que respetar).
#      Por defecto en modo SIMULACIÓN (dry-run) -- no manda órdenes reales.
#   2) Semanal (sábados 8:15am): re-corre el backtest cripto completo para
#      refrescar el dashboard y las bandas de alerta.
#
# Doble clic para correr, o "bash install_crypto_automation_mac.command" en Terminal.
set -e
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

if [ ! -d venv ]; then
  echo "Primero corre start_mac.command al menos una vez (instala el entorno)."
  read -p "Presiona Enter para cerrar..."
  exit 1
fi

PLIST_DAILY="$HOME/Library/LaunchAgents/com.tradingbot.crypto.daily.plist"
PLIST_WEEKLY="$HOME/Library/LaunchAgents/com.tradingbot.crypto.weekly.plist"

echo "=== Instalar automatización 24/7 del módulo cripto (BTC/USDT y demás) ==="
echo
echo "Esto va a instalar 2 tareas programadas:"
echo "  1) Diaria, TODOS los días (cripto no cierra -- sin restricción de día hábil)."
echo "     Por defecto en modo SIMULACIÓN -- NO manda órdenes reales todavía, y usa"
echo "     el broker VIRTUAL (\$1000 simulados, sin API key)."
echo "  2) Semanal (sábados 8:15am): re-corre el backtest cripto completo."
echo
echo "Nota sobre el horario: a diferencia de acciones, cripto no tiene 'apertura de"
echo "mercado' -- la hora de abajo (00:10) es arbitraria, elegida por estar cerca del"
echo "cierre de la vela diaria de Binance (00:00 UTC). Ajusta si prefieres otra hora."
echo
read -p "¿Continuar? (s/n) " confirm
if [ "$confirm" != "s" ]; then echo "Cancelado."; exit 0; fi

mkdir -p "$HOME/Library/LaunchAgents"

cat > "$PLIST_DAILY" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.tradingbot.crypto.daily</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PROJECT_DIR/venv/bin/python</string>
    <string>$PROJECT_DIR/scripts/run_crypto_live_once.py</string>
  </array>
  <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>0</integer><key>Minute</key><integer>10</integer></dict>
  <key>StandardOutPath</key><string>$PROJECT_DIR/reports_crypto/live_cron.log</string>
  <key>StandardErrorPath</key><string>$PROJECT_DIR/reports_crypto/live_cron_error.log</string>
</dict>
</plist>
EOF

cat > "$PLIST_WEEKLY" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.tradingbot.crypto.weekly</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PROJECT_DIR/venv/bin/python</string>
    <string>$PROJECT_DIR/scripts/run_crypto_backtest.py</string>
  </array>
  <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
  <key>StartCalendarInterval</key>
  <dict><key>Weekday</key><integer>6</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>15</integer></dict>
  <key>StandardOutPath</key><string>$PROJECT_DIR/reports_crypto/backtest_cron.log</string>
  <key>StandardErrorPath</key><string>$PROJECT_DIR/reports_crypto/backtest_cron_error.log</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST_DAILY" 2>/dev/null || true
launchctl load "$PLIST_DAILY"
launchctl unload "$PLIST_WEEKLY" 2>/dev/null || true
launchctl load "$PLIST_WEEKLY"

echo
echo "Listo. Tareas instaladas en modo SIMULACIÓN (broker virtual, no se envía dinero"
echo "real ni siquiera en paper trading de un exchange real)."
echo
echo "Para pasar a paper trading real (Binance testnet), edita este archivo:"
echo "  $PLIST_DAILY"
echo "y agrega, dentro del <array> de ProgramArguments (después de la línea"
echo "run_crypto_live_once.py), estas dos líneas:"
echo "  <string>--broker</string>"
echo "  <string>binance</string>"
echo "y esta para que además envíe las órdenes (si no, se queda en dry-run):"
echo "  <string>--execute</string>"
echo "(requiere BINANCE_API_KEY/BINANCE_SECRET_KEY de testnet en tu .env -- ver README)."
echo "Luego: launchctl unload $PLIST_DAILY && launchctl load $PLIST_DAILY"
echo
echo "Para desinstalar ambas tareas más adelante:"
echo "  launchctl unload $PLIST_DAILY && rm $PLIST_DAILY"
echo "  launchctl unload $PLIST_WEEKLY && rm $PLIST_WEEKLY"
echo
read -p "Presiona Enter para cerrar..."
