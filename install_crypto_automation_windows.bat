@echo off
setlocal
REM Instala 2 tareas programadas en el Task Scheduler de Windows para el
REM MODULO CRIPTO (BTC/ETH/altcoins contra USDT, ver config/crypto_universe.yaml):
REM   1) Diaria, TODOS LOS DIAS (cripto cotiza 365 dias/ano, sin dias habiles).
REM      Por defecto en modo SIMULACION (dry-run), broker VIRTUAL.
REM   2) Semanal (sabados 8:15am): re-corre el backtest cripto completo.
cd /d "%~dp0"
set PROJECT_DIR=%~dp0

if not exist venv (
    echo Primero corre start_windows.bat al menos una vez ^(instala el entorno^).
    pause
    exit /b 1
)

echo === Instalar automatizacion 24/7 del modulo cripto (BTC/USDT y demas) ===
echo.
echo Esto va a instalar 2 tareas programadas:
echo   1^) Diaria, TODOS los dias ^(cripto no cierra^): calcula la senal del dia.
echo      Por defecto en modo SIMULACION -- NO manda ordenes reales todavia, y usa
echo      el broker VIRTUAL ^($1000 simulados, sin API key^).
echo   2^) Semanal ^(sabados 8:15am^): re-corre el backtest cripto completo.
echo.
echo Nota sobre el horario: a diferencia de acciones, cripto no tiene "apertura de
echo mercado" -- la hora de abajo (00:10) es arbitraria, elegida por estar cerca del
echo cierre de la vela diaria de Binance (00:00 UTC). Ajusta si prefieres otra hora.
echo.
set /p CONFIRM=Continuar? (s/n):
if /i not "%CONFIRM%"=="s" exit /b 0

schtasks /create /tn "TradingBotCryptoDaily" /tr "\"%PROJECT_DIR%venv\Scripts\python.exe\" \"%PROJECT_DIR%scripts\run_crypto_live_once.py\"" /sc daily /st 00:10 /f
schtasks /create /tn "TradingBotCryptoWeekly" /tr "\"%PROJECT_DIR%venv\Scripts\python.exe\" \"%PROJECT_DIR%scripts\run_crypto_backtest.py\"" /sc weekly /d SAT /st 08:15 /f

echo.
echo Listo. Tareas instaladas en modo SIMULACION (broker virtual, no se envia dinero
echo real ni siquiera en paper trading de un exchange real).
echo.
echo Para pasar a paper trading real (Binance testnet):
echo   1. Abre el Task Scheduler de Windows.
echo   2. Busca la tarea "TradingBotCryptoDaily" -^> click derecho -^> Properties -^> pestana Actions.
echo   3. Edita la accion y agrega " --broker binance --execute" al final del campo
echo      "Add arguments" (requiere BINANCE_API_KEY/BINANCE_SECRET_KEY de testnet en
echo      tu .env -- ver README).
echo.
echo Para desinstalar ambas tareas mas adelante:
echo   schtasks /delete /tn "TradingBotCryptoDaily" /f
echo   schtasks /delete /tn "TradingBotCryptoWeekly" /f
echo.
pause
