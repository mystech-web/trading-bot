#!/bin/bash
# Instala 2 tareas programadas en tu Mac usando launchd:
#   1) Diaria (días hábiles, 9:35am hora de Nueva York): calcula señales del
#      bot. Por defecto en modo SIMULACIÓN (dry-run) -- no manda órdenes reales.
#   2) Semanal (sábados 8am): re-corre el backtest completo para refrescar el
#      dashboard y las bandas de alerta.
#
# Doble clic para correr, o "bash install_daily_automation_mac.command" en Terminal.
set -e
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

if [ ! -d venv ]; then
  echo "Primero corre start_mac.command al menos una vez (instala el entorno)."
  read -p "Presiona Enter para cerrar..."
  exit 1
fi

PLIST_DAILY="$HOME/Library/LaunchAgents/com.tradingbot.daily.plist"
PLIST_WEEKLY="$HOME/Library/LaunchAgents/com.tradingbot.weekly.plist"

echo "=== Instalar automatización diaria/semanal del bot ==="
echo
echo "Esto va a instalar 2 tareas programadas:"
echo "  1) Diaria (días hábiles, 9:35am hora de Nueva York): calcula la señal del día."
echo "     Por defecto en modo SIMULACIÓN -- NO manda órdenes reales todavía."
echo "  2) Semanal (sábados 8am): re-corre el backtest completo (perfil conservador)."
echo
echo "Nota: '9:35am hora de Nueva York' es la hora del MERCADO -- si tu Mac está en"
echo "otra zona horaria, ajusta manualmente la hora/minuto en el .plist después."
echo
read -p "¿Continuar? (s/n) " confirm
if [ "$confirm" != "s" ]; then echo "Cancelado."; exit 0; fi

mkdir -p "$HOME/Library/LaunchAgents"

cat > "$PLIST_DAILY" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.tradingbot.daily</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PROJECT_DIR/venv/bin/python</string>
    <string>$PROJECT_DIR/scripts/run_live_once.py</string>
  </array>
  <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>35</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>35</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>35</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>35</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>35</integer></dict>
  </array>
  <key>StandardOutPath</key><string>$PROJECT_DIR/reports/live_cron.log</string>
  <key>StandardErrorPath</key><string>$PROJECT_DIR/reports/live_cron_error.log</string>
</dict>
</plist>
EOF

cat > "$PLIST_WEEKLY" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.tradingbot.weekly</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PROJECT_DIR/venv/bin/python</string>
    <string>$PROJECT_DIR/scripts/run_backtest.py</string>
  </array>
  <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
  <key>StartCalendarInterval</key>
  <dict><key>Weekday</key><integer>6</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>$PROJECT_DIR/reports/backtest_cron.log</string>
  <key>StandardErrorPath</key><string>$PROJECT_DIR/reports/backtest_cron_error.log</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST_DAILY" 2>/dev/null || true
launchctl load "$PLIST_DAILY"
launchctl unload "$PLIST_WEEKLY" 2>/dev/null || true
launchctl load "$PLIST_WEEKLY"

echo
echo "Listo. Tareas instaladas en modo SIMULACIÓN (no se envía dinero, ni siquiera en paper)."
echo
echo "Para activar las órdenes reales de paper trading, edita este archivo:"
echo "  $PLIST_DAILY"
echo "y agrega una línea <string>--execute</string> dentro del <array> de ProgramArguments"
echo "(justo después de la línea que dice run_live_once.py), luego corre:"
echo "  launchctl unload $PLIST_DAILY && launchctl load $PLIST_DAILY"
echo
echo "Para desinstalar ambas tareas más adelante:"
echo "  launchctl unload $PLIST_DAILY && rm $PLIST_DAILY"
echo "  launchctl unload $PLIST_WEEKLY && rm $PLIST_WEEKLY"
echo
read -p "Presiona Enter para cerrar..."
