"""
Configuration Management Module
Handles environment variables and application settings
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application Settings"""

    # Application
    APP_NAME: str = "Loan-ERP-System"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # Security
    SECRET_KEY: str = Field(..., min_length=32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Encryption
    ENCRYPTION_KEY: str = Field(..., min_length=32, max_length=32)
    ENCRYPTION_ALGORITHM: str = "AES-256-CBC"
    SALT: str

    # Database
    DATABASE_TYPE: str = "tinydb"
    DATABASE_PATH: str = "./data/mock/database.json"
    BACKUP_ENABLED: bool = True
    BACKUP_INTERVAL_HOURS: int = 24

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_HEADERS: List[str] = ["*"]

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60

    # Session
    SESSION_TIMEOUT_MINUTES: int = 30
    MAX_CONCURRENT_SESSIONS: int = 3

    # File Storage
    UPLOAD_DIR: str = "./data/storage/uploads"
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "jpg", "jpeg", "png"]

    # Logging
    LOG_DIR: str = "./logs"
    LOG_ROTATION: str = "daily"
    LOG_RETENTION_DAYS: int = 30
    AUDIT_LOG_ENABLED: bool = True

    # Agents
    MASTER_AGENT_TIMEOUT: int = 300
    WORKER_AGENT_TIMEOUT: int = 60
    AGENT_RETRY_ATTEMPTS: int = 3

    # Credit Bureau
    CREDIT_SCORE_MIN: int = 300
    CREDIT_SCORE_MAX: int = 900

    # Loan Configuration
    MIN_LOAN_AMOUNT: int = 25000
    MAX_LOAN_AMOUNT: int = 2000000
    AVAILABLE_TENURES: List[int] = [12, 24, 36, 48, 60]

    # Behavioral Analysis
    BEHAVIORAL_SCORE_WEIGHT: float = 0.10
    BEHAVIORAL_SCORE_THRESHOLD: int = 60

    # Fraud Detection
    FRAUD_CHECK_ENABLED: bool = True
    MAX_RISK_FLAGS: int = 3

    # Feature Flags
    ENABLE_AI_EXPLAINABILITY: bool = True
    ENABLE_PERSONALITY_DETECTION: bool = True
    ENABLE_BEHAVIORAL_SCORING: bool = True
    ENABLE_AUTO_SANCTION: bool = True
    ENABLE_REAL_TIME_VERIFICATION: bool = True

    @validator("ENCRYPTION_KEY")
    def validate_encryption_key(cls, v):
        if len(v) != 32:
            raise ValueError("Encryption key must be exactly 32 characters for AES-256")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
