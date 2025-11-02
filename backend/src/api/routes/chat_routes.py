"""
Chat API Routes
Handles chatbot conversation endpoints
"""

import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from ...models.schemas import ChatRequest, ChatResponse, APIResponse
from ...services.behavioral_analyzer import BehavioralAnalyzer
from ...services.explainability_engine import ExplainabilityEngine
from ...agents.master.orchestrator import MasterAgent
from ...agents.workers.sales_agent import SalesAgent
from ...agents.workers.verification_agent import VerificationAgent
from ...agents.workers.underwriting_agent import UnderwritingAgent
from ...agents.workers.sanction_agent import SanctionLetterAgent


router = APIRouter(prefix="/api/chat", tags=["Chat"])

# In-memory session storage (replace with Redis in production)
active_sessions = {}


def get_or_create_session(session_id: str):
    """Get existing session or create new one"""
    if session_id and session_id in active_sessions:
        return active_sessions[session_id]

    # Create new session
    new_session_id = session_id or f"SESSION_{uuid.uuid4().hex[:12]}"

    # Initialize components
    behavioral_analyzer = BehavioralAnalyzer()
    explainability_engine = ExplainabilityEngine()

    # Initialize worker agents
    worker_agents = {
        'sales': SalesAgent(None),
        'verification': VerificationAgent(None, None),
        'underwriting': UnderwritingAgent(None, explainability_engine),
        'sanction': SanctionLetterAgent(None)
    }

    # Initialize master agent
    master_agent = MasterAgent(
        None,
        behavioral_analyzer,
        worker_agents
    )

    # Store session
    session = {
        "session_id": new_session_id,
        "master_agent": master_agent,
        "behavioral_analyzer": behavioral_analyzer,
        "created_at": datetime.utcnow().isoformat(),
        "last_activity": datetime.utcnow().isoformat()
    }

    active_sessions[new_session_id] = session
    return session


@router.post("/start", response_model=ChatResponse)
async def start_chat_session():
    """
    Start a new chat session

    Returns:
        ChatResponse with greeting message
    """
    session_id = f"SESSION_{uuid.uuid4().hex[:12]}"
    session = get_or_create_session(session_id)

    greeting = session['master_agent'].start_conversation(session_id)

    return ChatResponse(
        session_id=session_id,
        message=greeting,
        status="active"
    )


@router.post("/message", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,

):
    """
    Send message to chatbot with AI-like natural language processing

    Args:
        request: ChatRequest with session_id and message

    Returns:
        ChatResponse with bot reply
    """
    if not request.session_id or request.session_id not in active_sessions:
        # Start new session if not exists
        session = get_or_create_session(None)
        session_id = session['session_id']
        greeting = session['master_agent'].start_conversation(session_id)
        return ChatResponse(
            session_id=session_id,
            message=greeting + "\n\n" + "Please send your message again.",
            status="active"
        )

    session = active_sessions[request.session_id]

    # AI-like conversational chat with context awareness
    try:
        message = request.message.lower().strip()

        # Initialize conversation context if not exists
        if 'context' not in session:
            session['context'] = {'stage': 'greeting', 'data': {}}

        context = session['context']

        # Detect intent with improved NLP-like pattern matching
        intent = detect_intent(message)

        # Generate contextual response based on intent
        response_message = generate_response(intent, message, context)

        # Update last activity
        session['last_activity'] = datetime.utcnow().isoformat()

        return ChatResponse(
            session_id=request.session_id,
            message=response_message,
            status="active"
        )

    except Exception as e:
        # Better error logging
        import traceback
        error_detail = traceback.format_exc()
        print(f"Chat error: {error_detail}")

        return ChatResponse(
            session_id=request.session_id,
            message="I understand you need assistance. Could you please rephrase your question? I'm here to help with loans, applications, eligibility checks, and more!",
            status="active"
        )


def detect_intent(message: str) -> str:
    """Detect user intent from message using NLP-like pattern matching"""

    # Greeting patterns
    if any(word in message for word in ["hello", "hi", "hey", "welcome", "good morning", "good evening", "good afternoon"]):
        return "greeting"

    # Loan application intent
    if any(phrase in message for phrase in ["apply", "new loan", "want loan", "need loan", "get loan", "loan application"]):
        return "apply_loan"

    # Personal loan specific
    if any(phrase in message for phrase in ["personal loan", "individual loan"]):
        return "personal_loan"

    # Home loan specific
    if any(phrase in message for phrase in ["home loan", "house loan", "mortgage", "property loan"]):
        return "home_loan"

    # Car loan specific
    if any(phrase in message for phrase in ["car loan", "vehicle loan", "auto loan", "bike loan"]):
        return "car_loan"

    # Business loan specific
    if any(phrase in message for phrase in ["business loan", "commercial loan", "msme loan"]):
        return "business_loan"

    # Status check intent
    if any(phrase in message for phrase in ["status", "track", "check application", "my application", "where is my"]):
        return "check_status"

    # Eligibility check intent
    if any(phrase in message for phrase in ["eligible", "eligibility", "qualify", "can i get", "am i eligible"]):
        return "check_eligibility"

    # Interest rate intent
    if any(phrase in message for phrase in ["interest", "rate", "emi", "monthly payment", "installment"]):
        return "interest_rates"

    # Documents intent
    if any(phrase in message for phrase in ["document", "documents", "papers", "kyc", "what do i need", "requirements"]):
        return "documents"

    # Processing time
    if any(phrase in message for phrase in ["how long", "processing time", "how much time", "when will i get", "duration"]):
        return "processing_time"

    # Credit score
    if any(phrase in message for phrase in ["credit score", "cibil", "credit rating"]):
        return "credit_score"

    # Repayment
    if any(phrase in message for phrase in ["repay", "payment", "pay back", "tenure", "duration"]):
        return "repayment"

    # Thank you / Goodbye
    if any(phrase in message for phrase in ["thank", "thanks", "bye", "goodbye", "see you"]):
        return "farewell"

    # Help
    if any(phrase in message for phrase in ["help", "assist", "support", "what can you do"]):
        return "help"

    return "general"


def generate_response(intent: str, message: str, context: dict) -> str:
    """Generate contextual AI-like response based on intent and conversation context"""

    responses = {
        "greeting": """Hello! I'm your AI Loan Assistant. 

I'm here to make your loan journey smooth and hassle-free. I can assist you with:

 Personal Loans - Quick approval in 24 hours
 Home Loans - Attractive rates starting at 8.5%
 Car Loans - Drive your dream car today
 Business Loans - Grow your business

What brings you here today? Feel free to ask me anything!""",

        "apply_loan": """Great! I'd be happy to help you apply for a loan.

We offer several loan products:

1. **Personal Loan** - ₹25,000 to ₹20 Lakhs
   • Quick disbursal in 24-48 hours
   • Minimal documentation
   • Interest rates: 10.5% - 24% p.a.

2. **Home Loan** - Up to ₹5 Crores
   • Tenure up to 30 years
   • Interest rates: 8.5% - 11% p.a.
   • Zero processing fee

3. **Car Loan** - Up to ₹50 Lakhs
   • Funding up to 90% of car value
   • Interest rates: 9% - 15% p.a.
   • Quick approval

4. **Business Loan** - Up to ₹2 Crores
   • Flexible repayment options
   • Interest rates: 12% - 18% p.a.

Which loan are you interested in? Just type the name (e.g., "Personal Loan").""",

        "personal_loan": """**Personal Loan Details:**

 **Loan Amount:** ₹25,000 to ₹20,00,000
 **Tenure:** 12 to 60 months
 **Interest Rate:** 10.5% - 24% p.a. (based on credit score)
 **Disbursal:** 24-48 hours
 **Documentation:** Minimal

**Eligibility Criteria:**
• Age: 21-65 years
• Minimum Income: ₹15,000/month for salaried
• Credit Score: 650+
• Employment: Salaried or Self-employed

**Required Documents:**
 PAN Card & Aadhaar Card
 Last 3 months salary slips
 6 months bank statement

Would you like to check your eligibility or proceed with the application?""",

        "home_loan": """**Home Loan Details:**

 **Loan Amount:** Up to ₹5 Crores
 **Tenure:** Up to 30 years
 **Interest Rate:** 8.5% - 11% p.a.
 **Processing Fee:** Zero (limited period offer)
 **Funding:** Up to 90% of property value

**Eligibility Criteria:**
• Age: 23-65 years
• Minimum Income: ₹25,000/month
• Credit Score: 700+
• Stable employment for 2+ years

**Tax Benefits:**
 Deduction up to ₹1.5 Lakhs on principal (Section 80C)
 Deduction up to ₹2 Lakhs on interest (Section 24)

**Special Features:**
• Part-payment facility
• Top-up loan available
• Insurance coverage

Interested in calculating your EMI or checking eligibility?""",

        "car_loan": """**Car Loan Details:**

 **Loan Amount:** Up to ₹50 Lakhs
 **Tenure:** 12 to 84 months
 **Interest Rate:** 9% - 15% p.a.
 **Funding:** Up to 90% of on-road price
 **Approval:** Within 24 hours

**Eligibility:**
• Age: 21-65 years
• Minimum Income: ₹20,000/month
• Credit Score: 680+

**Documents Required:**
 Identity & Address Proof
 Income Proof (last 3 months)
 Bank Statements
 Car Quotation/Invoice

**Benefits:**
• Zero foreclosure charges
• Flexible EMI options
• Insurance included
• New & Used cars eligible

Ready to apply or need an EMI calculation?""",

        "business_loan": """**Business Loan Details:**

 **Loan Amount:** ₹50,000 to ₹2 Crores
 **Tenure:** 12 to 60 months
 **Interest Rate:** 12% - 18% p.a.
 **Collateral:** Based on loan amount

**Eligibility:**
• Business vintage: 2+ years
• Annual Turnover: ₹10 Lakhs+
• Credit Score: 700+
• Age: 25-65 years

**Documents:**
 Business Registration
 GST Returns (last 2 years)
 ITR (last 2 years)
 Bank Statements (6 months)
 Financial Statements

**Use Cases:**
• Working capital
• Equipment purchase
• Business expansion
• Inventory management

Would you like to know more about eligibility or start your application?""",

        "check_status": """I can help you track your loan application!

To check your application status, I'll need:
1. Your Application ID (format: APP2025XXXXXXXXXX)
2. Registered Mobile Number

Alternatively, you can:
• Visit the **Dashboard** to see all your applications
• Check your email for status updates
• Call our helpline: 1800-XXX-XXXX

Could you please share your Application ID?""",

        "check_eligibility": """Let me help you check your loan eligibility!

To provide an accurate assessment, I need some information:

1. **What type of loan** are you interested in?
   (Personal/Home/Car/Business)

2. **Your monthly income:** (approximate)

3. **Desired loan amount:**

4. **Employment type:**
   • Salaried
   • Self-employed
   • Business owner

5. **Credit score:** (if known)

6. **Existing EMIs:** (if any)

Please share these details and I'll instantly check your eligibility! """,

        "interest_rates": """**Current Interest Rates (October 2025):**

 **Personal Loans:** 10.5% - 24% p.a.
 **Home Loans:** 8.5% - 11% p.a.
 **Car Loans:** 9% - 15% p.a.
 **Business Loans:** 12% - 18% p.a.

**Interest Rate Factors:**
 Credit Score (Most Important)
 Income Level
 Loan Amount & Tenure
 Employment Stability
 Existing Obligations

**EMI Calculation Example:**
For ₹5,00,000 loan at 12% for 3 years:
Monthly EMI: ₹16,607
Total Interest: ₹97,852

Would you like me to calculate EMI for your specific requirements?

**Pro Tip:** A credit score above 750 can get you the lowest rates! """,

        "documents": """**Documents Required by Loan Type:**

 **Personal Loan:**
 PAN Card & Aadhaar Card
 Last 3 months salary slips
 6 months bank statement
 Employment letter

 **Home Loan (Additional):**
 Property documents
 Sale agreement
 Approval plans
 NOC from builder

 **Car Loan (Additional):**
 Car quotation/proforma invoice
 Insurance quote
 Driving license

 **Business Loan (Additional):**
 Business registration certificate
 GST returns (2 years)
 ITR (2 years)
 Financial statements
 Office address proof

**Digital Submission:**
You can upload all documents securely through our **Data Upload** section in the dashboard.

Need help with any specific document?""",

        "processing_time": """**Loan Processing Timeline:**

 **Personal Loan:** 24-48 hours
   • Application to approval: 2-4 hours
   • Document verification: 6-12 hours
   • Disbursal: 24-48 hours

 **Home Loan:** 7-15 days
   • Initial approval: 2-3 days
   • Property valuation: 3-5 days
   • Legal verification: 5-7 days
   • Final disbursal: 7-15 days

 **Car Loan:** 2-3 days
   • Application to approval: 24 hours
   • Documentation: 1-2 days
   • Disbursal: 2-3 days

 **Business Loan:** 5-10 days
   • Initial screening: 1-2 days
   • Financial assessment: 3-5 days
   • Final approval: 5-10 days

**Fast-Track Options Available!**
Submit complete documents for faster processing 

Which loan are you applying for?""",

        "credit_score": """**Understanding Credit Score:**

 **Credit Score Ranges:**
• 750-900: Excellent (Best rates & instant approval)
• 700-749: Good (Competitive rates)
• 650-699: Fair (Moderate rates)
• Below 650: Needs improvement

**How Credit Score Affects Your Loan:**

**Example - ₹10 Lakh Personal Loan:**
• Score 800+: 10.5% interest
• Score 750-799: 12% interest
• Score 700-749: 15% interest
• Score 650-699: 18% interest

**Improve Your Credit Score:**
 Pay EMIs on time
 Keep credit utilization below 30%
 Don't apply for multiple loans
 Maintain old credit accounts
 Check credit report regularly

**Free Credit Score Check:**
We can check your credit score during application at no cost!

Want to know your approximate score?""",

        "repayment": """**Flexible Repayment Options:**

 **Tenure Options:**
• Personal Loan: 12-60 months
• Home Loan: 60-360 months (up to 30 years)
• Car Loan: 12-84 months
• Business Loan: 12-60 months

 **EMI Payment Methods:**
 Auto-debit (ECS/NACH)
 Online payment portal
 Net banking
 Mobile app
 Cheque/DD

**Repayment Features:**

 **Pre-payment:**
• Close loan before tenure ends
• Part-payment allowed
• Minimal or zero charges

 **Moratorium Period:**
• Available on select loans
• Pay interest only initially
• EMI starts after 3-6 months

 **Step-up EMI:**
• Lower EMI initially
• Gradually increases
• Ideal for career starters

 **Step-down EMI:**
• Higher EMI initially
• Reduces over time
• Save on interest

Need an EMI schedule for your loan amount?""",

        "farewell": """Thank you for chatting with me! 

If you need any further assistance, I'm available 24/7. Feel free to return anytime!

**Quick Contact:**
 Helpline: 1800-XXX-XXXX
 Email: support@loan-erp.com
 Chat: Available in Dashboard

Have a wonderful day ahead! """,

        "help": """**I'm Your AI Loan Assistant! Here's how I can help:**

 **What I Can Do:**

1. **Loan Applications**
   • Personal, Home, Car, Business loans
   • Eligibility checking
   • EMI calculations

2. **Information Services**
   • Interest rates & offers
   • Required documents
   • Processing timelines

3. **Application Tracking**
   • Check application status
   • Next steps guidance
   • Document submission

4. **Loan Guidance**
   • Best loan for your needs
   • Comparison between products
   • Tips to improve approval chances

**Sample Questions You Can Ask:**
• "I need a personal loan of 5 lakhs"
• "What documents do I need?"
• "Check my application status"
• "Calculate EMI for home loan"
• "Am I eligible for a car loan?"

Just ask me anything about loans! What would you like to know? """,

        "general": """I'm here to help! I didn't quite understand that, but I can assist you with:

 **Loan Applications** - Personal, Home, Car, Business
 **Eligibility Checks** - See if you qualify
 **Interest Rates** - Get current rates & offers
 **EMI Calculations** - Plan your budget
 **Document Requirements** - Know what you need
 **Application Status** - Track your application

Could you please tell me more specifically what you're looking for?

For example:
• "I want a personal loan"
• "What are the interest rates?"
• "Check my eligibility"
• "What documents do I need?"

I'm here to help! """
    }

    return responses.get(intent, responses["general"])


@router.get("/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """
    Get session summary and statistics

    Args:
        session_id: Session identifier

    Returns:
        Session summary with metrics
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
    master_agent = session['master_agent']

    summary = master_agent.get_session_summary()

    return APIResponse(
        success=True,
        message="Session summary retrieved",
        data=summary
    )


@router.delete("/session/{session_id}")
async def end_session(session_id: str):
    """
    End a chat session

    Args:
        session_id: Session identifier

    Returns:
        Confirmation message
    """
    if session_id in active_sessions:
        del active_sessions[session_id]

    return APIResponse(
        success=True,
        message="Session ended successfully",
        data={"session_id": session_id}
    )


@router.get("/sessions/active")
async def get_active_sessions():
    """
    Get list of all active sessions (admin only)

    Returns:
        List of active sessions
    """
    sessions_list = [
        {
            "session_id": sid,
            "created_at": session['created_at'],
            "last_activity": session['last_activity']
        }
        for sid, session in active_sessions.items()
    ]

    return APIResponse(
        success=True,
        message=f"Found {len(sessions_list)} active sessions",
        data=sessions_list
    )
