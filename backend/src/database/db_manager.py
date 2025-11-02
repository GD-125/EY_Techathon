"""
Database Manager
TinyDB-based JSON database for prototype
Supports encryption for sensitive fields
"""

import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from tinydb import TinyDB, Query, where
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware
from pathlib import Path


class DatabaseManager:
    """
    Database Manager using TinyDB (JSON-based)
    Provides CRUD operations with encryption support
    """

    def __init__(self, db_path: str, crypto_service=None):
        """
        Initialize database manager

        Args:
            db_path: Path to JSON database file
            crypto_service: Encryption service for sensitive data
        """
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize TinyDB with caching
        self.db = TinyDB(db_path, storage=CachingMiddleware(JSONStorage))
        self.crypto = crypto_service

        # Define tables
        self.users = self.db.table('users')
        self.customers = self.db.table('customers')
        self.applications = self.db.table('applications')
        self.conversations = self.db.table('conversations')
        self.sessions = self.db.table('sessions')
        self.audit_logs = self.db.table('audit_logs')
        self.loan_products = self.db.table('loan_products')
        self.credit_profiles = self.db.table('credit_profiles')
        self.employment_records = self.db.table('employment_records')

        # Initialize default data
        self._initialize_default_data()

    def _initialize_default_data(self):
        """Initialize database with default mock data"""
        # Initialize loan products if empty
        if len(self.loan_products) == 0:
            self._initialize_loan_products()

        # Initialize mock customers
        if len(self.customers) == 0:
            self._initialize_mock_customers()

        # Initialize mock credit profiles
        if len(self.credit_profiles) == 0:
            self._initialize_mock_credit_profiles()

        # Initialize mock employment records
        if len(self.employment_records) == 0:
            self._initialize_mock_employment_records()

    def _initialize_loan_products(self):
        """Initialize loan product catalog"""
        products = [
            {
                "product_id": "PL_PREMIUM",
                "name": "Premium Personal Loan",
                "min_amount": 100000,
                "max_amount": 2000000,
                "tenure_months": [12, 24, 36, 48, 60],
                "interest_rate": 10.5,
                "processing_fee": 1.0,
                "min_credit_score": 750,
                "min_monthly_income": 50000,
                "features": ["Lowest interest rate", "Quick approval", "Flexible tenure"]
            },
            {
                "product_id": "PL_STANDARD",
                "name": "Standard Personal Loan",
                "min_amount": 50000,
                "max_amount": 1000000,
                "tenure_months": [12, 24, 36, 48],
                "interest_rate": 12.5,
                "processing_fee": 1.5,
                "min_credit_score": 650,
                "min_monthly_income": 30000,
                "features": ["Competitive rates", "Standard processing", "Good flexibility"]
            },
            {
                "product_id": "PL_BASIC",
                "name": "Basic Personal Loan",
                "min_amount": 25000,
                "max_amount": 500000,
                "tenure_months": [12, 24, 36],
                "interest_rate": 15.5,
                "processing_fee": 2.0,
                "min_credit_score": 600,
                "min_monthly_income": 20000,
                "features": ["Easy eligibility", "Quick disbursal", "Minimal documentation"]
            }
        ]
        self.loan_products.insert_multiple(products)

    def _initialize_mock_customers(self):
        """Initialize mock customer data"""
        customers = [
            {
                "customer_id": "CUST001",
                "name": "Raj Kumar",
                "phone": "9876543210",
                "email": "raj.kumar@email.com",
                "address": "123 MG Road, Bangalore, Karnataka 560001",
                "pan": "ABCDE1234F",
                "aadhaar": "1234-5678-9012",
                "date_of_birth": "1990-05-15",
                "kyc_verified": True,
                "kyc_verified_date": datetime(2024, 1, 10).isoformat(),
                "existing_customer": True,
                "last_loan_date": "2024-03-15",
                "repayment_history": "excellent",
                "created_at": datetime.now().isoformat()
            },
            {
                "customer_id": "CUST002",
                "name": "Priya Sharma",
                "phone": "9123456789",
                "email": "priya.sharma@email.com",
                "address": "456 Park Street, Kolkata, West Bengal 700016",
                "pan": "FGHIJ5678K",
                "aadhaar": "9876-5432-1098",
                "date_of_birth": "1992-08-22",
                "kyc_verified": True,
                "kyc_verified_date": datetime(2024, 2, 20).isoformat(),
                "existing_customer": False,
                "created_at": datetime.now().isoformat()
            },
            {
                "customer_id": "CUST003",
                "name": "Amit Patel",
                "phone": "9988776655",
                "email": "amit.patel@email.com",
                "address": "789 SG Highway, Ahmedabad, Gujarat 380015",
                "pan": "KLMNO9012P",
                "aadhaar": "5555-6666-7777",
                "date_of_birth": "1995-03-10",
                "kyc_verified": False,
                "existing_customer": False,
                "created_at": datetime.now().isoformat()
            }
        ]

        # Encrypt sensitive fields if crypto service available
        if self.crypto:
            for customer in customers:
                customer = self.crypto.encrypt_dict(customer, [
                    'pan', 'aadhaar', 'address', 'email', 'date_of_birth'
                ])

        self.customers.insert_multiple(customers)

    def _initialize_mock_credit_profiles(self):
        """Initialize mock credit profiles"""
        profiles = [
            {
                "pan": "ABCDE1234F",
                "credit_score": 780,
                "credit_history_length": 8,
                "active_loans": 1,
                "total_loan_amount": 250000,
                "default_history": [],
                "credit_utilization": 35,
                "recent_inquiries": 2,
                "last_updated": datetime.now().isoformat()
            },
            {
                "pan": "FGHIJ5678K",
                "credit_score": 720,
                "credit_history_length": 5,
                "active_loans": 2,
                "total_loan_amount": 450000,
                "default_history": [],
                "credit_utilization": 55,
                "recent_inquiries": 4,
                "last_updated": datetime.now().isoformat()
            },
            {
                "pan": "KLMNO9012P",
                "credit_score": 620,
                "credit_history_length": 2,
                "active_loans": 1,
                "total_loan_amount": 180000,
                "default_history": ["30-day-late-2024-01"],
                "credit_utilization": 78,
                "recent_inquiries": 8,
                "last_updated": datetime.now().isoformat()
            }
        ]

        # Encrypt credit scores
        if self.crypto:
            for profile in profiles:
                profile['credit_score_encrypted'] = self.crypto.encrypt(str(profile['credit_score']))

        self.credit_profiles.insert_multiple(profiles)

    def _initialize_mock_employment_records(self):
        """Initialize mock employment records"""
        records = [
            {
                "email": "raj.kumar@email.com",
                "company": "Tech Innovations Pvt Ltd",
                "designation": "Senior Software Engineer",
                "monthly_salary": 85000,
                "annual_income": 1020000,
                "employment_type": "permanent",
                "years_in_company": 4.5,
                "verified": True,
                "verified_date": datetime(2024, 1, 15).isoformat()
            },
            {
                "email": "priya.sharma@email.com",
                "company": "Global Finance Corp",
                "designation": "Financial Analyst",
                "monthly_salary": 65000,
                "annual_income": 780000,
                "employment_type": "permanent",
                "years_in_company": 3.2,
                "verified": True,
                "verified_date": datetime(2024, 2, 25).isoformat()
            },
            {
                "email": "amit.patel@email.com",
                "company": "StartUp Ventures",
                "designation": "Marketing Manager",
                "monthly_salary": 45000,
                "annual_income": 540000,
                "employment_type": "contract",
                "years_in_company": 1.5,
                "verified": True,
                "verified_date": datetime(2024, 3, 10).isoformat()
            }
        ]

        # Encrypt salary information
        if self.crypto:
            for record in records:
                record = self.crypto.encrypt_dict(record, [
                    'monthly_salary', 'annual_income'
                ])

        self.employment_records.insert_multiple(records)

    # ========================================================================
    # GENERIC CRUD OPERATIONS
    # ========================================================================

    def insert(self, table_name: str, data: Dict[str, Any]) -> int:
        """Insert a record"""
        table = getattr(self, table_name)
        return table.insert(data)

    def get_by_id(self, table_name: str, doc_id: int) -> Optional[Dict]:
        """Get record by document ID"""
        table = getattr(self, table_name)
        return table.get(doc_id=doc_id)

    def get_by_field(self, table_name: str, field: str, value: Any) -> Optional[Dict]:
        """Get first record matching field value"""
        table = getattr(self, table_name)
        return table.get(where(field) == value)

    def get_all(self, table_name: str, condition=None) -> List[Dict]:
        """Get all records from table"""
        table = getattr(self, table_name)
        if condition:
            return table.search(condition)
        return table.all()

    def update(self, table_name: str, data: Dict[str, Any], condition) -> List[int]:
        """Update records matching condition"""
        table = getattr(self, table_name)
        data['updated_at'] = datetime.now().isoformat()
        return table.update(data, condition)

    def delete(self, table_name: str, condition) -> List[int]:
        """Delete records matching condition"""
        table = getattr(self, table_name)
        return table.remove(condition)

    # ========================================================================
    # SPECIALIZED QUERY METHODS
    # ========================================================================

    def get_customer_by_phone(self, phone: str) -> Optional[Dict]:
        """Get customer by phone number"""
        return self.customers.get(where('phone') == phone)

    def get_credit_profile_by_pan(self, pan: str) -> Optional[Dict]:
        """Get credit profile by PAN"""
        return self.credit_profiles.get(where('pan') == pan)

    def get_employment_by_email(self, email: str) -> Optional[Dict]:
        """Get employment record by email"""
        return self.employment_records.get(where('email') == email)

    def get_application_by_id(self, app_id: str) -> Optional[Dict]:
        """Get loan application by ID"""
        return self.applications.get(where('application_id') == app_id)

    def get_active_session(self, session_id: str) -> Optional[Dict]:
        """Get active chat session"""
        return self.sessions.get(
            (where('session_id') == session_id) & (where('status') == 'active')
        )

    def get_user_applications(self, customer_id: str) -> List[Dict]:
        """Get all applications for a customer"""
        return self.applications.search(where('customer_id') == customer_id)

    def log_audit(self, audit_data: Dict[str, Any]) -> int:
        """Insert audit log entry"""
        audit_data['timestamp'] = datetime.now().isoformat()
        return self.audit_logs.insert(audit_data)

    # ========================================================================
    # ANALYTICS QUERIES
    # ========================================================================

    def get_application_stats(self) -> Dict[str, Any]:
        """Get application statistics"""
        all_apps = self.applications.all()

        approved = len([a for a in all_apps if a['status'] == 'approved'])
        rejected = len([a for a in all_apps if a['status'] == 'rejected'])
        pending = len([a for a in all_apps if a['status'] not in ['approved', 'rejected', 'closed']])

        total = len(all_apps)
        approval_rate = (approved / total * 100) if total > 0 else 0

        return {
            "total_applications": total,
            "approved": approved,
            "rejected": rejected,
            "pending": pending,
            "approval_rate": approval_rate
        }

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def backup_database(self, backup_path: str):
        """Create database backup"""
        import shutil
        shutil.copy2(self.db.storage._handle.name, backup_path)

    def close(self):
        """Close database connection"""
        self.db.close()
