@echo off
setlocal
pushd %~dp0

if not exist "clip_env\Scripts\activate.bat" (
    echo ERROR: Virtual environment activation script not found: clip_env\Scripts\activate.bat
    echo Make sure the virtual environment exists in the project root.
    pause
    exit /b 1
)

set "NGROK_CMD="
for %%G in (ngrok.exe ngrok) do (
    if defined NGROK_CMD goto :skip_ngrok_search
    where %%G >nul 2>&1 && set "NGROK_CMD=%%G"
)
:skip_ngrok_search

if defined NGROK_CMD (
    echo Starting FastAPI server and ngrok tunnel...
    start "FastAPI Server" cmd /k "cd /d %~dp0 && call clip_env\Scripts\activate.bat && python run.py"
    start "ngrok Tunnel" cmd /k "cd /d %~dp0 && %NGROK_CMD% http 8000"
) else (
    echo WARNING: ngrok.exe not found in the project folder or PATH.
    echo Download ngrok from https://ngrok.com/download, place ngrok.exe here, or add ngrok to PATH.
    echo Starting only the FastAPI server...
    start "FastAPI Server" cmd /k "cd /d %~dp0 && call clip_env\Scripts\activate.bat && python run.py"
)

popd