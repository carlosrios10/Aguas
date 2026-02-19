@echo off
set PROYECTO=D:\2024\BID\Aguas\Empresa4-Cali\proyecto\emcali_poc
cd /d "%PROYECTO%"
call qenv\Scripts\activate.bat
python scripts/run_train.py %*
pause
