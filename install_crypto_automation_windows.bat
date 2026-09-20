@echo off
setlocal
REM Instala 2 tareas programadas en el Task Scheduler de Windows para el
REM MODULO CRIPTO (BTC/ETH/altcoins contra USDT, ver config/crypto_universe.yaml):
REM   1) Diaria, TODOS LOS DIAS (cripto cotiza 365 dias/ano, sin dias habiles).
REM      Usa el broker VIRTUAL -- portafolio 100%% ficticio ($1000 simulados),
REM      sin cuenta ni API key real. Corre 100%% sola desde el dia uno.
REM   2) Semanal (sabados 8:15am): re-corre el backtest cripto completo.
cd /d "%~dp0"
set PROJECT_DIR=%~dp0

if not exist venv (
    echo Primero corre start_windows.bat al menos una vez ^(instala el entorno^).
    pause
    exit /b 1
)

echo === Instalar automatizacion 24/7 del modulo cripto (broker VIRTUAL, dinero ficticio) ===
echo.
echo Esto va a instalar 2 tareas programadas:
echo   1^) Diaria, TODOS los dias ^(cripto no cierra^): calcula la senal del dia y la
echo      aplica a un portafolio FICTICIO de $1000 -- corre 100%% sola, sin que hagas nada mas.
echo   2^) Semanal ^(sabados 8:15am^): re-corre el backtest cripto completo.
echo.
echo Nota sobre el horario: a diferencia de acciones, cripto no tiene "apertura de
echo mercado" -- la hora de abajo (00:10) es arbitraria, elegida por estar cerca del
echo cierre de la vela diaria de Binance (00:00 UTC). Ajusta si prefieres otra hora.
echo.
set /p CONFIRM=Continuar? (s/n):
if /i not "%CONFIRM%"=="s" exit /b 0

schtasks /create /tn "TradingBotCryptoDaily" /tr "\"%PROJECT_DIR%venv\Scripts\python.exe\" \"%PROJECT_DIR%scripts\run_crypto_live_once.py\" --broker virtual --starting-cash 1000 --execute" /sc daily /st 00:10 /f
schtasks /create /tn "TradingBotCryptoWeekly" /tr "\"%PROJECT_DIR%venv\Scripts\python.exe\" \"%PROJECT_DIR%scripts\run_crypto_backtest.py\"" /sc weekly /d SAT /st 08:15 /f

echo.
echo Listo. La tarea diaria ya corre 100%% sola con el broker VIRTUAL ($1000 ficticios,
echo sin ninguna cuenta real) -- revisa el progreso con "streamlit run dashboard.py"
echo (perfil "Cripto") o en %PROJECT_DIR%reports_crypto\virtual\.
echo.
echo Cuando quieras pasar al testnet real de Binance (con las claves que ya generaste,
echo ver .env):
echo   1. Abre el Task Scheduler de Windows.
echo   2. Busca la tarea "TradingBotCryptoDaily" -^> click derecho -^> Properties -^> pestana Actions.
echo   3. Edita la accion: reemplaza "--broker virtual --starting-cash 1000" por
echo      "--broker binance" en el campo "Add arguments".
echo.
echo Para desinstalar ambas tareas mas adelante:
echo   schtasks /delete /tn "TradingBotCryptoDaily" /f
echo   schtasks /delete /tn "TradingBotCryptoWeekly" /f
echo.
pause
