@echo off
setlocal
REM Instala 2 tareas programadas en el Task Scheduler de Windows:
REM   1) Diaria (dias habiles, 9:35am -- ver nota de zona horaria abajo): calcula
REM      senales del bot. Por defecto en modo SIMULACION (dry-run).
REM   2) Semanal (sabados 8am): re-corre el backtest completo.
cd /d "%~dp0"
set PROJECT_DIR=%~dp0

if not exist venv (
    echo Primero corre start_windows.bat al menos una vez ^(instala el entorno^).
    pause
    exit /b 1
)

echo === Instalar automatizacion diaria/semanal del bot ===
echo.
echo Esto va a instalar 2 tareas programadas:
echo   1^) Diaria ^(dias habiles, 9:35am hora de este PC^): calcula la senal del dia.
echo      Por defecto en modo SIMULACION -- NO manda ordenes reales todavia.
echo   2^) Semanal ^(sabados 8am^): re-corre el backtest completo ^(perfil conservador^).
echo.
echo NOTA: el mercado de EE.UU. abre 9:30am hora de Nueva York. Si este PC esta en
echo otra zona horaria, ajusta la hora "09:35" mas abajo o edita la tarea despues
echo en el Task Scheduler para que corresponda a 9:35am hora de Nueva York.
echo.
set /p CONFIRM=Continuar? (s/n):
if /i not "%CONFIRM%"=="s" exit /b 0

schtasks /create /tn "TradingBotDaily" /tr "\"%PROJECT_DIR%venv\Scripts\python.exe\" \"%PROJECT_DIR%scripts\run_live_once.py\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 09:35 /f
schtasks /create /tn "TradingBotWeekly" /tr "\"%PROJECT_DIR%venv\Scripts\python.exe\" \"%PROJECT_DIR%scripts\run_backtest.py\"" /sc weekly /d SAT /st 08:00 /f

echo.
echo Listo. Tareas instaladas en modo SIMULACION (no se envia dinero, ni siquiera en paper).
echo.
echo Para activar las ordenes reales de paper trading:
echo   1. Abre el Task Scheduler de Windows.
echo   2. Busca la tarea "TradingBotDaily" -^> click derecho -^> Properties -^> pestana Actions.
echo   3. Edita la accion y agrega " --execute" al final del campo "Add arguments".
echo.
echo Para desinstalar ambas tareas mas adelante:
echo   schtasks /delete /tn "TradingBotDaily" /f
echo   schtasks /delete /tn "TradingBotWeekly" /f
echo.
pause
