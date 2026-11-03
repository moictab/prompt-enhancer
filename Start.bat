@echo off
chcp 65001 >nul
pushd %~dp0
set "URL=http://127.0.0.1:8090"
set "STOP_MARK=%TEMP%\prompt-enhancer.stop"

rem Relanzado minimizado: aquí corre el servidor (ver más abajo).
if "%~1"=="servidor" goto :servidor

rem Si ya está en marcha, solo se abre en el navegador.
netstat -ano -p tcp | findstr /c:":8090 " | findstr LISTENING >nul
if not errorlevel 1 (
    start "" %URL%/conversar
    popd
    exit /b 0
)

rem El entorno de Python, en la propia carpeta (como comfy-gateway): se prepara la primera vez y
rem se pone al día cuando cambian las dependencias (requirements.txt distinto del último instalado).
fc /b requirements.txt .venv\requirements.installed >nul 2>&1
if errorlevel 1 (
    echo Preparando el entorno de Python...
    python -m venv .venv || goto :error
    .venv\Scripts\python -m pip install -q --disable-pip-version-check -r requirements.txt || goto :error
    copy /y requirements.txt .venv\requirements.installed >nul
)
if not exist .env (
    echo Falta el archivo .env: copia .env.example a .env y rellénalo.
    pause
    popd
    exit /b 1
)

rem El servidor, en una ventana minimizada; el navegador se abre en cuanto responde.
start "prompt-enhancer (servidor)" /min cmd /c ""%~f0" servidor"
powershell -NoProfile -Command "$n = 0; do { Start-Sleep -Milliseconds 500; curl.exe -s -o NUL %URL%/; $n++ } until ($LASTEXITCODE -eq 0 -or $n -ge 60); Start-Process '%URL%/conversar'"
popd
exit /b 0

:servidor
title prompt-enhancer (servidor)
echo prompt-enhancer en %URL%
echo Para pararlo: Stop.bat, o cerrar esta ventana.
echo.
del "%STOP_MARK%" 2>nul
.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8090 --no-access-log --log-level warning
rem Parado con Stop.bat: la ventana se cierra sola. Si no, se queda abierta para ver por qué se paró.
if exist "%STOP_MARK%" (
    del "%STOP_MARK%"
    exit
)
echo.
echo El servidor se ha parado.
pause
exit

:error
echo No he podido preparar el entorno de Python.
pause
popd
exit /b 1
