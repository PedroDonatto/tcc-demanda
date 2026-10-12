@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
start "" http://localhost:8765
python -m http.server 8765 -d app\web
