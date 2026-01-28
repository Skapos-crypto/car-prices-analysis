@echo off
echo Starting Car Prices Analysis Dashboard...
echo.
cd /d "%~dp0"
.venv\Scripts\streamlit.exe run dashboard_app.py
pause
