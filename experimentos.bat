@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
call .venv\Scripts\activate.bat
echo Rodando o laboratorio de experimentos (alguns minutos)...
python -m src.experimentos > outputs\log_experimentos.txt 2>&1
python -m src.exportar_web >> outputs\log_experimentos.txt 2>&1
echo FIM >> outputs\log_experimentos.txt
