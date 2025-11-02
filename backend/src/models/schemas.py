"""
Data Models and Schemas
Pydantic models for request/response validation and data structures
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, EmailStr, validator
from enum import Enum


# ============================================================================
# ENUMERATIONS
# ============================================================================

class UserRole(str, Enum):
    CUSTOMER = "customer"
    AGENT = "agent"
    MANAGER = "manager"
    ADMIN = "admin"


class LoanStatus(str, Enum):
    INITIATED = "initiated"
    INFO_GATHERING = "information_gathering"
    KYC_VERIFICATION = "kyc_verification"
    CREDIT_EVALUATION = "credit_evaluation"
    APPROVED = "approved"
    REJECTED = "rejected"
    SANCTION_GENERATED = "sanction_generated"
    DISBURSED = "disbursed"
    CLOSED = "closed"


class PersonalityType(str, Enum):
    ANALYTICAL = "analytical"
    AMIABLE = "amiable"
    EXPRESSIVE = "expressive"
    DRIVER = "driver"


class EmploymentType(str, Enum):
    PERMANENT = "permanent"
    CONTRACT = "contract"
    SELF_EMPLOYED = "self_employed"
    FREELANCE = "freelance"


# ============================================================================
# USER MODELS
# ============================================================================

class UserBase(BaseModel):
    """Base user model"""
    email: EmailStr
    full_name: str
    phone: str
    role: UserRole = UserRole.CUSTOMER


class UserCreate(UserBase):
    """User creation model"""
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    """User login model"""
    email: EmailStr
    password: str


class UserResponse(UserBase):
    """User response model"""
    user_id: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


# ============================================================================
# CUSTOMER MODELS
# ============================================================================

class CustomerKYC(BaseModel):
    """Customer KYC details"""
    customer_id: str
    name: str
    phone: str
    email: EmailStr
    address: str
    pan: str
    aadhaar: str
    date_of_birth: Optional[str] = None
    kyc_verified: bool = False
    kyc_verified_date: Optional[datetime] = None


class EmploymentDetails(BaseModel):
    """Employment information"""
    company: str
    designation: str
    monthly_salary: float
    annual_income: float
    employment_type: EmploymentType
    years_in_company: float
    verified: bool = False


class CreditProfile(BaseModel):
    """Credit bureau data"""
    pan: str
    credit_score: int
    credit_history_length: int  # years
    active_loans: int
    total_loan_amount: float
    credit_utilization: float  # percentage
    default_history: List[str] = []
    recent_inquiries: int


# ============================================================================
# LOAN MODELS
# ============================================================================

class LoanProduct(BaseModel):
    """Loan product details"""
    product_id: str
    name: str
    min_amount: float
    max_amount: float
    tenure_months: List[int]
    interest_rate: float
    processing_fee: float  # percentage
    min_credit_score: int
    min_monthly_income: float


class LoanRequest(BaseModel):
    """Loan application request"""
    phone: str
    loan_amount: float = Field(..., gt=0)
    tenure: int
    purpose: Optional[str] = None

    @validator('loan_amount')
    def validate_amount(cls, v):
        if v < 25000:
            raise ValueError('Minimum loan amount is ₹25,000')
        if v > 2000000:
            raise ValueError('Maximum loan amount is ₹20,00,000')
        return v

    @validator('tenure')
    def validate_tenure(cls, v):
        if v not in [12, 24, 36, 48, 60]:
            raise ValueError('Invalid tenure. Choose from 12, 24, 36, 48, or 60 months')
        return v


class LoanApplication(BaseModel):
    """Complete loan application"""
    application_id: str
    customer_id: str
    customer_name: str
    phone: str
    email: EmailStr
    loan_amount: float
    tenure: int
    product_id: Optional[str] = None
    status: LoanStatus
    created_at: datetime
    updated_at: datetime
    credit_score: Optional[int] = None
    final_score: Optional[float] = None
    interest_rate: Optional[float] = None
    emi: Optional[float] = None
    processing_fee: Optional[float] = None


class LoanDecision(BaseModel):
    """Loan approval/rejection decision"""
    application_id: str
    approved: bool
    final_score: float
    decision_factors: List[Dict[str, Any]]
    explanation: str
    recommended_product: Optional[LoanProduct] = None
    alternatives: Optional[List[Dict[str, Any]]] = None
    decided_at: datetime


class SanctionLetter(BaseModel):
    """Loan sanction letter"""
    sanction_id: str
    application_id: str
    customer_name: str
    loan_amount: float
    interest_rate: float
    tenure: int
    emi: float
    processing_fee: float
    validity_date: datetime
    terms_conditions: List[str]
    generated_at: datetime
    letter_content: str


# ============================================================================
# CONVERSATION MODELS
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message"""
    role: str  # "customer" or "assistant"
    message: str
    timestamp: datetime
    response_time: Optional[float] = None


class ChatSession(BaseModel):
    """Chat session"""
    session_id: str
    customer_id: Optional[str] = None
    phone: Optional[str] = None
    application_id: Optional[str] = None
    personality_type: Optional[PersonalityType] = None
    behavioral_score: Optional[float] = None
    messages: List[ChatMessage] = []
    created_at: datetime
    last_activity: datetime
    status: str = "active"


class ChatRequest(BaseModel):
    """Incoming chat message"""
    session_id: Optional[str] = None
    message: str
    response_time: float = 2.0


class ChatResponse(BaseModel):
    """Outgoing chat response"""
    session_id: str
    message: str
    status: str
    application_id: Optional[str] = None
    personality_detected: Optional[PersonalityType] = None


# ============================================================================
# BEHAVIORAL ANALYSIS MODELS
# ============================================================================

class BehavioralMetrics(BaseModel):
    """Behavioral analysis metrics"""
    total_messages: int
    avg_response_time: float
    avg_message_length: float
    question_ratio: float
    hesitation_markers: int
    confidence_markers: int
    personality_type: PersonalityType
    trust_score: float
    risk_flags: List[str]


# ============================================================================
# EXPLAINABILITY MODELS
# ============================================================================

class DecisionFactor(BaseModel):
    """Individual decision factor"""
    factor: str
    value: Any
    impact: str  # "Positive", "Negative", "Neutral"
    weight: float
    explanation: str


class ExplainableDecision(BaseModel):
    """Explainable loan decision"""
    application_id: str
    approved: bool
    final_score: float
    threshold: float
    factors: List[DecisionFactor]
    summary: str
    recommendations: Optional[List[str]] = None


# ============================================================================
# AUDIT MODELS
# ============================================================================

class AuditLog(BaseModel):
    """Audit trail entry"""
    log_id: str
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    application_id: Optional[str] = None
    action: str
    entity_type: str
    entity_id: Optional[str] = None
    changes: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    result: str  # "success" or "failure"
    error_message: Optional[str] = None


# ============================================================================
# ANALYTICS MODELS
# ============================================================================

class ApplicationStats(BaseModel):
    """Application statistics"""
    total_applications: int
    approved: int
    rejected: int
    pending: int
    approval_rate: float
    avg_processing_time: float  # minutes
    avg_loan_amount: float
    total_disbursed: float


class AgentPerformance(BaseModel):
    """Agent performance metrics"""
    agent_type: str
    total_requests: int
    successful: int
    failed: int
    avg_response_time: float  # seconds
    success_rate: float


# ============================================================================
# API RESPONSE MODELS
# ============================================================================

class APIResponse(BaseModel):
    """Standard API response"""
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginatedResponse(BaseModel):
    """Paginated response"""
    success: bool
    data: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
