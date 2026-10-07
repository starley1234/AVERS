@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Windows environment not found. Run: py -3.12 -m venv venv
    echo Then run: venv\Scripts\python.exe -m pip install -e ".[web]"
    exit /b 1
)
"venv\Scripts\python.exe" -m avers web --port 8030
