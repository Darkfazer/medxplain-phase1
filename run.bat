@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  run.bat  –  Start FastAPI backend (port 8000) + Gradio UI (port 7860)
REM  Run from the medxplain-simple project root.
REM ─────────────────────────────────────────────────────────────────────────────

echo.
set MEDXPLAIN_DEVICE=cuda
set MEDXPLAIN_REQUIRE_CUDA=1
set CUDA_VISIBLE_DEVICES=0

echo [MedXplain] Starting FastAPI backend on http://localhost:8000 ...
start "MedXplain API" cmd /k "python backend_api.py"

REM Wait 3 seconds for the backend to initialise before opening the UI
timeout /t 3 /nobreak >nul

echo [MedXplain] Starting Gradio UI on http://localhost:7860 ...
start "MedXplain Gradio" cmd /k "python app.py"

echo.
echo Both services are starting.
echo   Backend : http://localhost:8000
echo   Gradio  : http://localhost:7860
echo   HTML UI : Open medxplain_ui.html in your browser (or http://localhost:8000)
echo.
