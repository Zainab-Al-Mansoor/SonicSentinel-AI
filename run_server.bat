@echo off
REM SonicSentinel AI - production-style server for the demo / evaluation day (Windows).
REM  * waitress (multi-threaded WSGI server) instead of the Flask development server
REM  * restarts automatically if the process stops (availability requirement)
REM  * health check: http://127.0.0.1:5000/healthz
REM Usage:  run_server.bat            (port 5000, this PC only)
REM         run_server.bat 0.0.0.0    (reachable from the LAN; the microphone needs https or localhost)
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set HOST=%1
if "%HOST%"=="" set HOST=127.0.0.1
if "%PORT%"=="" set PORT=5000
if not exist logs mkdir logs
python -m database.init_db
:loop
echo [%date% %time%] starting SonicSentinel AI on http://%HOST%:%PORT%  (Ctrl+C twice to stop)
echo [%date% %time%] start >> logs\server.log
waitress-serve --host=%HOST% --port=%PORT% --threads=8 --channel-timeout=180 run:app
echo [%date% %time%] server stopped (exit code %errorlevel%) - restarting in 5 s >> logs\server.log
echo Server stopped - restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto loop
