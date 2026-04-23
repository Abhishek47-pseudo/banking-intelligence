@echo off
REM ── AI Banking Intelligence Platform — Quick Start ──

echo.
echo  ======================================================
echo   AI Banking Client Intelligence Platform
echo  ======================================================
echo.

REM Check for .env
if not exist ".env" (
    echo  [!] .env not found. Copying from .env.example...
    copy .env.example .env
    echo  [!] Please edit .env and add your OPENAI_API_KEY, then re-run.
    pause
    exit /b 1
)

REM Generate mock data if needed
if not exist "data\mock\transactions.csv" (
    echo  [*] Generating mock data...
    python data\mock\generate_mock_data.py
)

echo  [*] Starting FastAPI backend on http://localhost:8000 ...
start "Banking Backend" cmd /k "set PYTHONIOENCODING=utf-8 && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 >nul

echo  [*] Starting Streamlit frontend on http://localhost:8501 ...
start "Banking Frontend" cmd /k "set PYTHONIOENCODING=utf-8 && set API_BASE=http://localhost:8000 && python -m streamlit run frontend/app.py --server.port 8501"

echo.
echo  ======================================================
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo   Frontend: http://localhost:8501
echo  ======================================================
echo.
pause
