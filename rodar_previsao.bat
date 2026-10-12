@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
call .venv\Scripts\activate.bat
echo [1/3] Instalando streamlit e plotly...
pip install streamlit plotly > outputs\log_instalacao.txt 2>&1
echo [2/3] Rodando testes...
python -m tests.test_pipeline > outputs\log_testes.txt 2>&1
echo [3/3] Treinando e prevendo 2026 (alguns minutos)...
python -m src.prever 6 18 > outputs\log_prever.txt 2>&1
echo %ERRORLEVEL% > outputs\log_fim.txt
echo.
echo ===== PRONTO! Resultado em outputs\log_prever.txt =====
