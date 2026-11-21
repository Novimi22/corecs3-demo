@echo off
REM CORECS3 system demo on the HPC. Double-click to start. Close this window to stop.
REM Restart the HPC first on the day (free vGPU licence), so the GPU accepts work.
cd /d "%~dp0"
set DEMO_MODE=live
set DEMO_THESIS=C:\corecs3-dual-stream
set DEMO_GPU_MODELS=4
set DEMO_UPLOAD_BATCH=reference
start "" /b cmd /c "timeout /t 8 >nul & start http://localhost:7860"
C:\corecs3-dual-stream\venv\Scripts\python.exe app\server.py
pause
