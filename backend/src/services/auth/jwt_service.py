"""
JWT Authentication Service
Implements secure token-based authentication with HS256/RS256
Supports access tokens and refresh tokens
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
from jose import JWTError, jwt
from pydantic import BaseModel


class TokenData(BaseModel):
    """Token payload data"""
    user_id: str
    email: str
    role: str
    session_id: str
    token_type: str  # "access" or "refresh"


class TokenPair(BaseModel):
    """Access and refresh token pair"""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int


class JWTService:
    """
    JWT Token Management Service
    Handles token creation, validation, and refresh
    """

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """
        Initialize JWT service

        Args:
            secret_key: Secret key for signing tokens
            algorithm: Signing algorithm (HS256 or RS256)
        """
        self.secret_key = secret_key
        self.algorithm = algorithm

    def create_access_token(
        self,
        user_id: str,
        email: str,
        role: str,
        session_id: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token

        Args:
            user_id: Unique user identifier
            email: User email
            role: User role (customer, agent, admin)
            session_id: Session identifier
            expires_delta: Token expiration time

        Returns:
            Encoded JWT token
        """
        if expires_delta is None:
            expires_delta = timedelta(minutes=30)

        expire = datetime.utcnow() + expires_delta

        to_encode = {
            "sub": user_id,
            "email": email,
            "role": role,
            "session_id": session_id,
            "token_type": "access",
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": "loan-erp-system"
        }

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def create_refresh_token(
        self,
        user_id: str,
        email: str,
        role: str,
        session_id: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT refresh token

        Args:
            user_id: Unique user identifier
            email: User email
            role: User role
            session_id: Session identifier
            expires_delta: Token expiration time

        Returns:
            Encoded refresh token
        """
        if expires_delta is None:
            expires_delta = timedelta(days=7)

        expire = datetime.utcnow() + expires_delta

        to_encode = {
            "sub": user_id,
            "email": email,
            "role": role,
            "session_id": session_id,
            "token_type": "refresh",
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": "loan-erp-system"
        }

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def create_token_pair(
        self,
        user_id: str,
        email: str,
        role: str,
        session_id: str,
        access_expires_minutes: int = 30,
        refresh_expires_days: int = 7
    ) -> TokenPair:
        """
        Create access and refresh token pair

        Args:
            user_id: User identifier
            email: User email
            role: User role
            session_id: Session identifier
            access_expires_minutes: Access token expiration in minutes
            refresh_expires_days: Refresh token expiration in days

        Returns:
            TokenPair object
        """
        access_token = self.create_access_token(
            user_id, email, role, session_id,
            timedelta(minutes=access_expires_minutes)
        )

        refresh_token = self.create_refresh_token(
            user_id, email, role, session_id,
            timedelta(days=refresh_expires_days)
        )

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=access_expires_minutes * 60
        )

    def verify_token(self, token: str) -> Optional[TokenData]:
        """
        Verify and decode JWT token

        Args:
            token: JWT token to verify

        Returns:
            TokenData if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": True, "verify_iss": True},
                issuer="loan-erp-system"
            )

            token_data = TokenData(
                user_id=payload.get("sub"),
                email=payload.get("email"),
                role=payload.get("role"),
                session_id=payload.get("session_id"),
                token_type=payload.get("token_type")
            )

            return token_data

        except JWTError:
            return None

    def decode_token(self, token: str) -> Optional[Dict]:
        """
        Decode token without verification (for debugging)

        Args:
            token: JWT token

        Returns:
            Decoded payload dictionary
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_signature": False}
            )
            return payload
        except JWTError:
            return None

    def is_token_expired(self, token: str) -> bool:
        """
        Check if token is expired

        Args:
            token: JWT token

        Returns:
            True if expired
        """
        payload = self.decode_token(token)
        if not payload:
            return True

        exp = payload.get("exp")
        if not exp:
            return True

        return datetime.fromtimestamp(exp) < datetime.utcnow()

    def get_token_expiry(self, token: str) -> Optional[datetime]:
        """
        Get token expiration datetime

        Args:
            token: JWT token

        Returns:
            Expiration datetime
        """
        payload = self.decode_token(token)
        if not payload:
            return None

        exp = payload.get("exp")
        if not exp:
            return None

        return datetime.fromtimestamp(exp)

    @staticmethod
    def extract_token_from_header(authorization: str) -> Optional[str]:
        """
        Extract token from Authorization header

        Args:
            authorization: Authorization header value (Bearer <token>)

        Returns:
            Token string or None
        """
        if not authorization:
            return None

        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None

        return parts[1]


class SessionManager:
    """
    Manage user sessions and token revocation
    """

    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}
        self.revoked_tokens: set = set()

    def create_session(self, session_id: str, user_id: str, token_pair: TokenPair):
        """Create new session"""
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "access_token": token_pair.access_token,
            "refresh_token": token_pair.refresh_token
        }

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session details"""
        return self.active_sessions.get(session_id)

    def update_session_activity(self, session_id: str):
        """Update last activity timestamp"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["last_activity"] = datetime.utcnow()

    def revoke_session(self, session_id: str):
        """Revoke session and tokens"""
        session = self.active_sessions.pop(session_id, None)
        if session:
            self.revoked_tokens.add(session["access_token"])
            self.revoked_tokens.add(session["refresh_token"])

    def is_token_revoked(self, token: str) -> bool:
        """Check if token is revoked"""
        return token in self.revoked_tokens

    def cleanup_expired_sessions(self, timeout_minutes: int = 30):
        """Remove expired sessions"""
        now = datetime.utcnow()
        expired = [
            sid for sid, session in self.active_sessions.items()
            if (now - session["last_activity"]).total_seconds() > timeout_minutes * 60
        ]

        for sid in expired:
            self.revoke_session(sid)

    def get_user_sessions(self, user_id: str) -> list:
        """Get all active sessions for a user"""
        return [
            {"session_id": sid, **session}
            for sid, session in self.active_sessions.items()
            if session["user_id"] == user_id
        ]
