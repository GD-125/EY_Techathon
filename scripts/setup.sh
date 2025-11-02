#!/bin/bash

# Loan ERP System Setup Script

echo "============================================"
echo "  Loan ERP System - Setup"
echo "============================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
cd backend
python -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if not exists
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please update .env file with your configuration!"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p ../data/mock
mkdir -p ../data/storage/uploads
mkdir -p ../data/storage/sanction_letters
mkdir -p ../logs

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "To start the application:"
echo "  1. Update the .env file with your configuration"
echo "  2. Run: python main.py"
echo ""
echo "API will be available at: http://localhost:8000"
echo "Documentation at: http://localhost:8000/docs"
echo ""
