@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo The trainer environment is missing.
    echo Open README.md for setup instructions.
    pause
    exit /b 1
)

start "LoRA Video Trainer" ".venv\Scripts\pythonw.exe" "%~dp0app.py"
endlocal
