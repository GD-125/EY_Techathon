"""
Authentication Middleware
JWT token validation for protected routes
"""

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional


class JWTBearer(HTTPBearer):
    """JWT Bearer token authentication"""

    def __init__(self, jwt_service, session_manager, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)
        self.jwt_service = jwt_service
        self.session_manager = session_manager

    async def __call__(self, request: Request) -> Optional[str]:
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)

        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authentication scheme"
                )

            token = credentials.credentials

            # Check if token is revoked
            if self.session_manager.is_token_revoked(token):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )

            # Verify token
            token_data = self.jwt_service.verify_token(token)
            if not token_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )

            # Update session activity
            self.session_manager.update_session_activity(token_data.session_id)

            # Attach user info to request
            request.state.user = token_data

            return token
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid authorization code"
            )


def get_current_user(request: Request):
    """Extract current user from request"""
    return getattr(request.state, 'user', None)


def require_role(required_role: str):
    """Decorator to require specific role"""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            user = get_current_user(request)
            if not user or user.role != required_role:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. Required role: {required_role}"
                )
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
