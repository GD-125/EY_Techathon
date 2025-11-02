"""
Audit Logging Middleware
Logs all API requests for compliance
"""

import time
import uuid
from datetime import datetime
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for audit trail logging"""

    def __init__(self, app, db_manager):
        super().__init__(app)
        self.db = db_manager

    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Start time
        start_time = time.time()

        # Extract user info if available
        user_id = None
        if hasattr(request.state, 'user'):
            user_id = request.state.user.user_id

        # Process request
        try:
            response = await call_next(request)
            result = "success"
            error_message = None
        except Exception as e:
            result = "failure"
            error_message = str(e)
            raise
        finally:
            # Calculate duration
            duration = time.time() - start_time

            # Log audit entry
            audit_log = {
                "log_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "ip_address": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "duration_ms": round(duration * 1000, 2),
                "result": result,
                "error_message": error_message
            }

            # Save to audit log
            try:
                self.db.log_audit(audit_log)
            except Exception:
                # Don't fail request if audit logging fails
                pass

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response
