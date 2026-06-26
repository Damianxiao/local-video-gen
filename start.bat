@echo off
cd /d "%~dp0"
title Local Video Gen

echo ============================================
echo          Local Video Generator
echo ============================================
echo.

REM Check Python
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ and add to PATH.
    echo         Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create virtual env on first run
if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Creating virtual environment .venv ...
    python -m venv .venv
)

REM Install deps on first run (marker file avoids reinstalling every time)
if not exist ".venv\.installed" (
    echo [SETUP] Installing dependencies, please wait ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies. Check your network.
        pause
        exit /b 1
    )
    echo done > ".venv\.installed"
)

set PORT=5321

REM Free the port if an old (possibly hung) server is still holding it.
echo [CHECK] Releasing port %PORT% if occupied ...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT% " ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>nul
)

echo.
echo [RUN] Server: http://127.0.0.1:%PORT%
echo       Auto-reload is ON: code changes apply automatically, just refresh.
echo       Close this window to stop the server.
echo.

REM Open browser after 3 seconds
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:%PORT%"

".venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port %PORT% --reload

pause
