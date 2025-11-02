# Quick Startup Instructions

## ✅ Prerequisites Check

Make sure you have:
- ✅ Python 3.8+ installed
- ✅ Node.js 16+ installed
- ✅ Virtual environment created at `venv/`
- ✅ Backend dependencies installed (`pip install -r backend/requirements.txt`)
- ✅ Frontend dependencies installed (`npm install` in frontend/)

## 🚀 Starting the System

### Option 1: Using Batch Files (Easiest - Windows)

1. **Start Backend:**
   - Double-click `START_BACKEND.bat`
   - Server will start at http://localhost:8000

2. **Start Frontend** (in another window):
   - Double-click `START_FRONTEND.bat`
   - App will open at http://localhost:3000

### Option 2: Manual Start

#### Backend:
```bash
# Navigate to backend
cd F:\Techathon\loan-erp-system\backend

# Activate virtual environment
..\venv\Scripts\activate

# Start server
python main.py
```

#### Frontend (new terminal):
```bash
# Navigate to frontend
cd F:\Techathon\loan-erp-system\frontend

# Start React app
npm start
```

## 🔍 Verify Installation

### Check Python packages:
```bash
cd F:\Techathon\loan-erp-system
venv\Scripts\activate
pip list | findstr "fastapi tinydb pandas numpy"
```

Should show:
- fastapi
- tinydb
- pandas
- numpy
- uvicorn

### Check Node packages:
```bash
cd frontend
npm list react react-dom axios recharts
```

## ❗ Common Issues

### Issue 1: ModuleNotFoundError
**Solution:** Install dependencies in virtual environment
```bash
cd F:\Techathon\loan-erp-system
venv\Scripts\activate
pip install -r backend\requirements.txt
```

### Issue 2: Virtual environment not found
**Solution:** Create virtual environment
```bash
cd F:\Techathon\loan-erp-system
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
```

### Issue 3: npm dependencies not installed
**Solution:**
```bash
cd F:\Techathon\loan-erp-system\frontend
npm install
```

### Issue 4: Port already in use
**Solution:** Kill the process using the port
```bash
# For backend (port 8000)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# For frontend (port 3000)
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

## 🎯 Quick Test

Once both servers are running:

1. **Test Backend API:**
   - Open browser: http://localhost:8000/docs
   - You should see the FastAPI documentation

2. **Test Frontend:**
   - Open browser: http://localhost:3000
   - You should see the login page

3. **Login:**
   - Email: `admin@loan.com`
   - Password: `admin123`

## 📊 Full Installation (If starting fresh)

```bash
# 1. Create virtual environment
cd F:\Techathon\loan-erp-system
python -m venv venv

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Install backend dependencies
pip install -r backend\requirements.txt

# 4. Install frontend dependencies
cd frontend
npm install

# 5. Start backend (keep this terminal open)
cd ..\backend
python main.py

# 6. Start frontend (open NEW terminal)
cd F:\Techathon\loan-erp-system\frontend
npm start
```

## ✅ Success Indicators

**Backend running successfully:**
```
======================================================================
  🚀 Starting Loan-ERP-System v1.0.0
  Environment: development
======================================================================

📦 Initializing services...
✅ Encryption service initialized
✅ Database initialized
✅ JWT service initialized
✅ Session manager initialized

🌐 Server running on http://0.0.0.0:8000
📚 API Documentation: http://0.0.0.0:8000/docs
======================================================================
```

**Frontend running successfully:**
```
Compiled successfully!

You can now view loan-erp-frontend in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000
```

## 🎉 You're Ready!

Navigate to http://localhost:3000 and login with:
- **Admin:** admin@loan.com / admin123
- **User:** user@loan.com / user123

---

**Need help?** Check:
- `IMPLEMENTATION_GUIDE.md` - Detailed implementation guide
- `QUICKSTART.md` - Quick start guide
- `PROJECT_STATUS.md` - Project status and features
