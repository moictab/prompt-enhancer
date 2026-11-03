@echo off
chcp 65001 >nul
set "FOUND="

rem Lo que escucha en el puerto de prompt-enhancer, y nada más. La marca le dice a la ventana
rem del servidor que se ha parado a propósito, para que se cierre sola.
for /f "tokens=5" %%p in ('netstat -ano -p tcp ^| findstr /c:":8090 " ^| findstr LISTENING') do (
    echo.>"%TEMP%\prompt-enhancer.stop"
    taskkill /PID %%p /F >nul 2>&1
    set "FOUND=1"
)

if not defined FOUND (
    echo prompt-enhancer no estaba en marcha.
    timeout /t 2 /nobreak >nul 2>&1
    exit /b 0
)
timeout /t 1 /nobreak >nul 2>&1
netstat -ano -p tcp | findstr /c:":8090 " | findstr LISTENING >nul
if errorlevel 1 (
    echo prompt-enhancer parado.
    timeout /t 2 /nobreak >nul 2>&1
    exit /b 0
)
echo No he podido pararlo: sigue escuchando en el puerto 8090.
pause
exit /b 1
