@echo off
setlocal
REM Doble clic en este archivo para instalar todo (la primera vez) y abrir el
REM dashboard. Las siguientes veces solo abre el dashboard.
cd /d "%~dp0"

echo === Bot de inversion: instalacion / arranque ===
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo No se encontro Python en este PC.
    echo Instalalo desde https://www.python.org/downloads/
    echo IMPORTANTE: marca la casilla "Add python.exe to PATH" durante la instalacion.
    echo Luego vuelve a abrir este archivo.
    pause
    exit /b 1
)

if not exist venv (
    echo Primera vez: creando entorno virtual e instalando dependencias.
    echo Esto puede tardar varios minutos -- no cierres esta ventana.
    echo.
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo Dependencias instaladas.
) else (
    call venv\Scripts\activate.bat
)

if not exist .env (
    copy .env.example .env >nul
    echo.
    echo Se creo el archivo .env ^(con tus claves de Alpaca en blanco^).
    echo Solo lo necesitas para paper trading en vivo ^(scripts\run_live_once.py^).
    echo El dashboard y el backtest funcionan sin el.
)

echo.
echo Abriendo el dashboard en tu navegador ^(http://localhost:8501^)...
echo Para cerrarlo: vuelve a esta ventana y presiona Ctrl+C.
echo.
streamlit run dashboard.py
pause
