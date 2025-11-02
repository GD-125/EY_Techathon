"""
Main Application Entry Point
FastAPI application with all routes, middleware, and services
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from src.config.settings import settings
from src.database.db_manager import DatabaseManager
from src.services.encryption.crypto_service import CryptoService
from src.services.auth.jwt_service import JWTService, SessionManager
from src.api.middleware.audit_middleware import AuditLoggingMiddleware
from src.api.routes import chat_routes
from src.models.schemas import APIResponse


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

# Global instances
db_manager = None
crypto_service = None
jwt_service = None
session_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    global db_manager, crypto_service, jwt_service, session_manager

    print(f"\n{'='*70}")
    print(f"  Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"  Environment: {settings.ENVIRONMENT}")
    print(f"{'='*70}\n")

    # Initialize services
    print("Initializing services...")

    # Ensure data directories exist
    os.makedirs("./data/mock", exist_ok=True)
    os.makedirs("./data/storage/uploads", exist_ok=True)
    os.makedirs("./data/storage/sanction_letters", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    # Initialize encryption service
    crypto_service = CryptoService(
        encryption_key=settings.ENCRYPTION_KEY,
        salt=settings.SALT
    )
    print("Encryption service initialized")

    # Initialize database
    db_manager = DatabaseManager(
        db_path=settings.DATABASE_PATH,
        crypto_service=crypto_service
    )
    print("Database initialized")

    # Initialize JWT service
    jwt_service = JWTService(
        secret_key=settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    print("JWT service initialized")

    # Initialize session manager
    session_manager = SessionManager()
    print("Session manager initialized")

    print(f"\nServer running on http://{settings.HOST}:{settings.PORT}")
    print(f"API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"{'='*70}\n")

    yield

    # Cleanup on shutdown
    print("\nShutting down...")
    if db_manager:
        db_manager.close()
    print("Cleanup complete\n")


# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise Loan Processing ERP System with AI Multi-Agent Architecture",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================================
# MIDDLEWARE
# ============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Audit Logging Middleware
@app.middleware("http")
async def audit_middleware(request, call_next):
    """Add audit logging to all requests"""
    if db_manager and settings.AUDIT_LOG_ENABLED:
        middleware = AuditLoggingMiddleware(app, db_manager)
        return await middleware.dispatch(request, call_next)
    return await call_next(request)


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================



# ============================================================================
# ROUTES
# ============================================================================

@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint"""
    return APIResponse(
        success=True,
        message=f"Welcome to {settings.APP_NAME} API",
        data={
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "docs": "/docs",
            "health": "/health"
        }
    )


@app.get("/health", response_model=APIResponse)
async def health_check():
    """Health check endpoint"""
    return APIResponse(
        success=True,
        message="System is healthy",
        data={
            "status": "operational",
            "database": "connected" if db_manager else "disconnected",
            "encryption": "enabled" if crypto_service else "disabled"
        }
    )


@app.get("/api/stats", response_model=APIResponse)
async def get_statistics():
    """Get system statistics"""
    if not db_manager:
        return APIResponse(success=False, message="Database not available")

    stats = db_manager.get_application_stats()

    return APIResponse(
        success=True,
        message="Statistics retrieved",
        data=stats
    )


# Include chat routes
app.include_router(chat_routes.router)

# Include data routes
from src.api.routes import data_routes
app.include_router(data_routes.router)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "errors": [str(exc)] if settings.DEBUG else ["An error occurred"]
        }
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
