@echo off
cd /d "%~dp0"
python pc_voice_app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application crashed or failed to start.
    pause
)