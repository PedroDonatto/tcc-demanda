@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
call .venv\Scripts\activate.bat
python -m src.exportar_web > outputs\log_web.txt 2>&1
echo %ERRORLEVEL% > outputs\log_web_fim.txt
