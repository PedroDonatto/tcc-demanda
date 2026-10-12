@echo off
cd /d "%~dp0"
REM Roda o pipeline completo, na ordem. Pode rodar de novo se cair:
REM os downloads ja feitos ficam em cache.
call .venv\Scripts\activate.bat || goto :erro

python -m src.ingest_sih    || goto :erro
python -m src.ingest_cnes   || goto :erro
python -m src.ingest_ibge   || goto :erro
python -m src.build_dataset regiao || goto :erro
python -m src.build_dataset estado || goto :erro
python -m src.train regiao 12 || goto :erro
python -m src.prever 6 18    || goto :erro
python -m src.db            || goto :erro

echo.
echo ===== PRONTO! Resultados na pasta outputs =====
pause
exit /b 0

:erro
echo.
echo ===== Deu erro. Copie a mensagem acima e mande pro Claude =====
pause
exit /b 1
