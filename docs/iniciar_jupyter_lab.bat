@echo off
echo Cerrando Jupyter si está en ejecución...
taskkill /F /IM jupyter-lab.exe 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq*jupyter*" 2>nul
timeout /t 2 /nobreak >nul

echo Activando entorno qenv...
call "C:\Users\cea\AquaData\proyecto\Aguas\qenv\Scripts\activate.bat"

echo Iniciando Jupyter Lab en el proyecto...
cd /d "C:\Users\cea\AquaData\proyecto\Aguas"
jupyter lab

pause
