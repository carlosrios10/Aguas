@echo off
set PROYECTO=D:\2024\BID\Aguas\Empresa2-Queretaro\proyecto\queretaro_poc
cd /d "%PROYECTO%"
call qenv\Scripts\activate.bat
python -u scripts/run_etl.py %*
pause
