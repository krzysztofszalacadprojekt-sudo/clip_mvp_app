@echo off
title MyApp Launcher

echo ==============================
echo Starting FastAPI backend...
echo ==============================

start "" /D "%~dp0clip_fast_api\dist\server\" server.exe

echo Waiting for backend to start...

REM Wait for a fixed amount of time (e.g., 5 seconds).
echo Waiting 5 seconds for the server to initialize...
timeout /t 5 /nobreak > nul

echo Backend is assumed to be up!
echo ==============================
echo Starting C++ application...
echo ==============================

cd /d "%~dp0search_client\build\Release"
start "" TestBackend.exe

echo All started.
pause

cd ..
cd ..