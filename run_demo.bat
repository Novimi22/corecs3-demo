@echo off
REM CORECS3 system demo. Double-click to start. Close this window to stop.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo The demo is not installed yet. Follow README steps 4 and 5 first.
  pause
  exit /b 1
)
set DEMO_MODE=live
if /I "%1"=="saved" set DEMO_MODE=replay
set DEMO_THESIS=%~dp0thesis
set MODEL_DOWNLOAD_ROOT=%~dp0bundle\clip_weights
set LANDMARK_PREDICTOR=%~dp0bundle\shape_predictor_81_face_landmarks.dat
set DEMO_GPU_MODELS=2
start "" /b cmd /c "timeout /t 8 >nul & start http://localhost:7860"
.venv\Scripts\python.exe app\server.py
pause
