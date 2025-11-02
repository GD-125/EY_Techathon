# 🚀 QUICK START GUIDE
## Loan ERP System - Complete Setup in 5 Minutes

---

## ✅ Prerequisites Check

```bash
# Check Python (3.10+)
python --version

# Check pip
pip --version
```

---

## 📦 Installation (3 Steps)

### Step 1: Setup Environment
```bash
# Navigate to backend
cd backend

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Security Keys
```bash
# Copy environment template
cp .env.example .env

# Generate secure keys (run these commands)
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('ENCRYPTION_KEY=' + secrets.token_urlsafe(24)[:32])"
python -c "import secrets; print('SALT=' + secrets.token_hex(16))"
```

**Update `.env` file with the generated keys!**

### Step 3: Run Application
```bash
# Start backend server
python main.py
```

✅ **Backend Running**: http://localhost:8000
📚 **API Docs**: http://localhost:8000/docs

---

## 🌐 Frontend Setup (Optional)

```bash
# Open in browser
# Navigate to: frontend/public/index.html
# Or serve with Python:
cd frontend/public
python -m http.server 3000
```

✅ **Frontend**: http://localhost:3000

---

## 🎮 Test the System

### Method 1: Using Frontend
1. Open http://localhost:3000
2. Type your phone: `9876543210`
3. Request amount: `500000`
4. Choose tenure: `36`
5. Accept offer: `yes`

### Method 2: Using API (curl)
```bash
# Start session
curl -X POST http://localhost:8000/api/chat/start

# Send message
curl -X POST http://localhost:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "SESSION_xxx",
    "message": "9876543210",
    "response_time": 2.0
  }'
```

### Method 3: Using Python Script
```python
import requests

# Start session
r = requests.post('http://localhost:8000/api/chat/start')
session_id = r.json()['session_id']

# Chat
for msg in ["9876543210", "500000", "36", "yes"]:
    r = requests.post('http://localhost:8000/api/chat/message',
        json={"session_id": session_id, "message": msg, "response_time": 2.0})
    print(r.json()['message'])
```

---

## 📊 Pre-loaded Demo Data

The system comes with **3 demo customers**:

| Phone | Name | Credit Score | Status |
|-------|------|--------------|--------|
| 9876543210 | Raj Kumar | 780 (Excellent) | ✅ Will Approve |
| 9123456789 | Priya Sharma | 720 (Good) | ✅ Will Approve |
| 9988776655 | Amit Patel | 620 (Low) | ❌ May Reject |

**Try these phone numbers for instant demo!**

---

## 🔧 Common Issues & Solutions

### Issue: "Module not found"
```bash
# Solution: Ensure you're in backend directory
cd backend
pip install -r requirements.txt
```

### Issue: "Encryption key must be 32 characters"
```bash
# Solution: Regenerate key
python -c "import secrets; key = secrets.token_urlsafe(24); print(key[:32])"
# Copy to .env ENCRYPTION_KEY
```

### Issue: "Port 8000 already in use"
```bash
# Solution: Change port in .env
PORT=8001
```

### Issue: "Cannot connect to backend"
```bash
# Check if backend is running
curl http://localhost:8000/health

# If not running, start it:
cd backend
python main.py
```

---

## 📁 Project Structure Overview

```
loan-erp-system/
├── backend/
│   ├── src/
│   │   ├── agents/              # AI Agents
│   │   │   ├── master/          # Master Orchestrator
│   │   │   └── workers/         # Sales, Verify, Underwriting, Sanction
│   │   ├── api/                 # REST API
│   │   │   ├── routes/          # Endpoints
│   │   │   └── middleware/      # Auth, Audit
│   │   ├── services/            # Business Logic
│   │   │   ├── auth/            # JWT Authentication
│   │   │   ├── encryption/      # AES-256 Encryption
│   │   │   ├── behavioral_analyzer.py
│   │   │   └── explainability_engine.py
│   │   ├── models/              # Data Schemas
│   │   ├── database/            # DB Manager
│   │   └── config/              # Settings
│   ├── main.py                  # ⭐ START HERE
│   ├── requirements.txt         # Dependencies
│   └── .env                     # Configuration
├── frontend/
│   └── public/
│       └── index.html           # Chat Interface
├── data/                        # Database Storage
│   ├── mock/                    # JSON Database
│   └── storage/                 # Files
├── docs/                        # Documentation
└── QUICKSTART.md               # This file
```

---

## 🎯 Next Steps

### For Development:
1. ✅ System is running
2. 📖 Read: `docs/COMPLETE_DOCUMENTATION.md`
3. 🧪 Test with demo data
4. 🛠️ Customize loan products in database
5. 🔐 Update security keys for production

### For Production:
1. Replace TinyDB with PostgreSQL
2. Add Redis for sessions
3. Setup SSL/TLS certificates
4. Configure load balancer
5. Enable monitoring (Prometheus)

---

## 📚 Important Files

| File | Purpose |
|------|---------|
| `backend/main.py` | Start application |
| `backend/.env` | Configuration |
| `docs/COMPLETE_DOCUMENTATION.md` | Full documentation |
| `frontend/public/index.html` | Chat UI |

---

## 🆘 Need Help?

1. **API Documentation**: http://localhost:8000/docs
2. **System Health**: http://localhost:8000/health
3. **Full Docs**: `docs/COMPLETE_DOCUMENTATION.md`
4. **Issues**: Check logs in `logs/` directory

---

## 🎉 Success Checklist

- [ ] Backend running on port 8000
- [ ] Can access API docs at /docs
- [ ] Frontend loads successfully
- [ ] Can send messages in chat
- [ ] Demo customer data loads
- [ ] Loan application completes

**All checked? Congratulations! Your system is ready! 🚀**

---

**Time to Full Setup**: ~5 minutes
**Complexity**: Low (just 3 commands)
**Support**: enterprise-grade
