@echo off
cd /d "%~dp0"
"D:\work2\penguinchuan\Scripts\python.exe" "%~dp0main.py"
if errorlevel 1 (
    echo.
    echo [Error on exit - press any key to close]
    pause >nul
)
