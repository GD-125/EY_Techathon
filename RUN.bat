@echo off
echo ============================================
echo   Loan ERP System - Quick Start
echo ============================================
echo.

:: Check if .env exists
if not exist "backend\.env" (
    echo [ERROR] .env file not found!
    echo Please run: cd backend ^&^& copy .env.example .env
    echo Then update the keys in .env file
    pause
    exit /b 1
)

:: Check if dependencies are installed
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    cd backend
    pip install -r requirements.txt
    cd ..
)

echo [INFO] Starting backend server...
echo.
echo Backend API: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

cd backend
python main.py
