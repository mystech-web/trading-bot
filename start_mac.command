#!/bin/bash
# Doble clic en este archivo (en Finder) para instalar todo (la primera vez)
# y abrir el dashboard. Las siguientes veces solo abre el dashboard.
#
# La primera vez que lo abras, macOS probablemente bloquee el script por venir
# de "un desarrollador no identificado". Si eso pasa: clic derecho sobre este
# archivo -> Abrir -> confirmar "Abrir" en el diálogo. Solo hace falta una vez.
set -e
cd "$(dirname "$0")"

echo "=== Bot de inversión: instalación / arranque ==="
echo

if ! command -v python3 &>/dev/null; then
  echo "No se encontró Python 3 en este Mac."
  echo "Instálalo desde https://www.python.org/downloads/ y vuelve a abrir este script."
  read -p "Presiona Enter para cerrar..."
  exit 1
fi

if [ ! -d venv ]; then
  echo "Primera vez: creando entorno virtual e instalando dependencias."
  echo "Esto puede tardar varios minutos -- no cierres esta ventana."
  echo
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip -q
  pip install -r requirements.txt -q
  echo "Dependencias instaladas."
else
  source venv/bin/activate
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo
  echo "Se creó el archivo .env (con tus claves de Alpaca en blanco)."
  echo "Solo lo necesitas para paper trading en vivo (scripts/run_live_once.py)."
  echo "El dashboard y el backtest funcionan sin él."
fi

echo
echo "Abriendo el dashboard en tu navegador (http://localhost:8501)..."
echo "Para cerrarlo: vuelve a esta ventana de Terminal y presiona Ctrl+C."
echo
streamlit run dashboard.py
