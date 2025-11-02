# Enterprise Loan Processing ERP System

## 🏢 Overview

A comprehensive, enterprise-grade ERP system for NBFC loan processing with AI-driven multi-agent architecture, featuring end-to-end encryption, JWT authentication, behavioral analysis, and explainable AI.

## 📁 Project Structure

```
loan-erp-system/
├── backend/                          # Python Backend (FastAPI)
│   ├── src/
│   │   ├── agents/                   # AI Agent Layer
│   │   │   ├── master/              # Master Agent (Orchestrator)
│   │   │   └── workers/             # Worker Agents
│   │   ├── api/                     # REST API Layer
│   │   │   ├── routes/              # API Route Handlers
│   │   │   └── middleware/          # Authentication, Logging
│   │   ├── models/                  # Data Models (SQLAlchemy)
│   │   ├── services/                # Business Logic Layer
│   │   │   ├── auth/                # Authentication Service
│   │   │   ├── encryption/          # AES-256 Encryption
│   │   │   ├── verification/        # KYC Verification
│   │   │   ├── credit/              # Credit Scoring
│   │   │   └── notification/        # Email/SMS Service
│   │   ├── database/                # Database Layer
│   │   ├── config/                  # Configuration
│   │   ├── utils/                   # Utility Functions
│   │   └── core/                    # Core Functionality
│   ├── requirements.txt             # Python Dependencies
│   └── main.py                      # Application Entry Point
│
├── frontend/                         # React Frontend
│   ├── src/
│   │   ├── components/              # React Components
│   │   │   ├── chat/                # Chatbot Interface
│   │   │   ├── dashboard/           # User Dashboard
│   │   │   └── admin/               # Admin Panel
│   │   ├── services/                # API Services
│   │   ├── store/                   # Redux Store
│   │   ├── utils/                   # Helper Functions
│   │   └── assets/                  # Static Assets
│   ├── package.json
│   └── public/
│
├── infrastructure/                   # DevOps & Deployment
│   ├── docker/                      # Docker Configurations
│   ├── kubernetes/                  # K8s Manifests
│   └── nginx/                       # Nginx Configs
│
├── tests/                           # Test Suite
│   ├── unit/                        # Unit Tests
│   ├── integration/                 # Integration Tests
│   └── e2e/                         # End-to-End Tests
│
├── docs/                            # Documentation
├── scripts/                         # Deployment Scripts
├── logs/                            # Application Logs
└── data/                            # Mock Data & Storage
    ├── mock/                        # Mock Databases
    └── storage/                     # File Storage
```

## 🔐 Security Features

### 1. **JWT Authentication (HS256/RS256)**
- Secure session management
- Token-based authentication
- Automatic token refresh
- Role-based access control (RBAC)

### 2. **AES-256 Encryption**
- KYC data encryption at rest
- Credit score encryption
- Salary slip encryption
- PII data protection

### 3. **TLS 1.3**
- End-to-end encrypted communication
- Secure data transmission
- Certificate-based authentication

### 4. **Audit Logging**
- Complete activity tracking
- Compliance reporting
- GDPR/PCI-DSS compliance
- Tamper-proof logs

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Redis (optional)

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## 🏗️ Architecture

### Multi-Agent System
- **Master Agent**: Orchestrator managing workflow
- **Sales Agent**: Product offering and negotiation
- **Verification Agent**: KYC and fraud detection
- **Underwriting Agent**: Credit evaluation
- **Sanction Agent**: Letter generation

### API Architecture
- RESTful API design
- Microservices-ready
- Event-driven architecture
- Async processing support

## 📊 Features

✅ AI-driven conversational interface
✅ Behavioral trust scoring
✅ Explainable AI decisions
✅ Real-time credit evaluation
✅ Automated sanction letters
✅ Multi-factor authentication
✅ End-to-end encryption
✅ Audit trail
✅ Admin dashboard
✅ Analytics & reporting

## 📖 Documentation

See `/docs` for comprehensive documentation:
- API Documentation
- Architecture Guide
- Security Best Practices
- Deployment Guide
- User Manual

## 🔧 Configuration

See `.env.example` for configuration options

## 📄 License

Proprietary - Tata Capital
