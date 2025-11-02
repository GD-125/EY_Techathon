#!/bin/bash

echo "============================================"
echo "  Loan ERP System - Quick Start"
echo "============================================"
echo ""

# Check if .env exists
if [ ! -f "backend/.env" ]; then
    echo "[ERROR] .env file not found!"
    echo "Please run: cd backend && cp .env.example .env"
    echo "Then update the keys in .env file"
    exit 1
fi

# Check if dependencies are installed
python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[INFO] Installing dependencies..."
    cd backend
    pip install -r requirements.txt
    cd ..
fi

echo "[INFO] Starting backend server..."
echo ""
echo "Backend API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd backend
python main.py
