@echo off
REM One-time setup on Windows (run from the project folder)
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m database.init_db
echo.
echo Setup finished. Start the app with run_windows.bat
pause
