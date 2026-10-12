@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat || goto :erro
streamlit run app\dashboard.py || goto :erro
exit /b 0

:erro
echo.
echo ===== Deu erro. Copie a mensagem acima e mande pro Claude =====
pause
exit /b 1
