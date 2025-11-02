@echo off
echo ============================================
echo   Starting Loan ERP Backend Server
echo ============================================
echo.

cd F:\Techathon\loan-erp-system\backend

echo Activating virtual environment...
call ..\venv\Scripts\activate.bat

echo.
echo Starting FastAPI server...
python main.py

pause
