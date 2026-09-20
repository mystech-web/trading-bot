@echo off
setlocal
REM Instala 2 tareas programadas en el Task Scheduler de Windows:
REM   1) Diaria (dias habiles, 9:35am -- ver nota de zona horaria abajo): calcula
REM      senales del bot y las "ejecuta" contra el BROKER VIRTUAL -- un portafolio
REM      100%% ficticio ($1000 simulados, sin cuenta ni API key real). Como no toca
REM      ningun exchange/broker real, corre 100%% sola desde el primer dia.
REM   2) Semanal (sabados 8am): re-corre el backtest completo.
cd /d "%~dp0"
set PROJECT_DIR=%~dp0

if not exist venv (
    echo Primero corre start_windows.bat al menos una vez ^(instala el entorno^).
    pause
    exit /b 1
)

echo === Instalar automatizacion diaria/semanal del bot (broker VIRTUAL, dinero ficticio) ===
echo.
echo Esto va a instalar 2 tareas programadas:
echo   1^) Diaria ^(dias habiles, 9:35am hora de este PC^): calcula la senal del dia y la
echo      aplica a un portafolio FICTICIO de $1000 -- corre 100%% sola, sin que hagas nada mas.
echo   2^) Semanal ^(sabados 8am^): re-corre el backtest completo ^(perfil conservador^).
echo.
echo NOTA: el mercado de EE.UU. abre 9:30am hora de Nueva York. Si este PC esta en
echo otra zona horaria, ajusta la hora "09:35" mas abajo o edita la tarea despues
echo en el Task Scheduler para que corresponda a 9:35am hora de Nueva York.
echo.
set /p CONFIRM=Continuar? (s/n):
if /i not "%CONFIRM%"=="s" exit /b 0

schtasks /create /tn "TradingBotDaily" /tr "\"%PROJECT_DIR%venv\Scripts\python.exe\" \"%PROJECT_DIR%scripts\run_live_once.py\" --broker virtual --starting-cash 1000 --execute" /sc weekly /d MON,TUE,WED,THU,FRI /st 09:35 /f
schtasks /create /tn "TradingBotWeekly" /tr "\"%PROJECT_DIR%venv\Scripts\python.exe\" \"%PROJECT_DIR%scripts\run_backtest.py\"" /sc weekly /d SAT /st 08:00 /f

echo.
echo Listo. La tarea diaria ya corre 100%% sola con el broker VIRTUAL ($1000 ficticios,
echo sin ninguna cuenta real) -- revisa el progreso con "streamlit run dashboard.py" o
echo en %PROJECT_DIR%reports\live_cron.log (si rediriges la salida; por default
echo Task Scheduler no guarda un log de texto, revisa el dashboard).
echo.
echo Cuando quieras pasar a paper trading real contra Alpaca (con tu cuenta y API key,
echo ver .env):
echo   1. Abre el Task Scheduler de Windows.
echo   2. Busca la tarea "TradingBotDaily" -^> click derecho -^> Properties -^> pestana Actions.
echo   3. Edita la accion: reemplaza "--broker virtual --starting-cash 1000" por
echo      "--broker alpaca" en el campo "Add arguments" (deja --execute si quieres
echo      que mande ordenes reales de paper trading; quitalo para ver primero
echo      que haria, en dry-run).
echo.
echo Para desinstalar ambas tareas mas adelante:
echo   schtasks /delete /tn "TradingBotDaily" /f
echo   schtasks /delete /tn "TradingBotWeekly" /f
echo.
pause
