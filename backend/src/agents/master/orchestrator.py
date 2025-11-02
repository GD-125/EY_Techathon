"""
Master Agent - Orchestrator
Controls the entire loan workflow and coordinates worker agents
Implements state machine pattern for conversation management
"""

import uuid
from datetime import datetime
from typing import Dict, Optional, Tuple
from enum import Enum

from ...models.schemas import (
    LoanStatus, PersonalityType, ChatMessage,
    LoanApplication
)


class ConversationState(str, Enum):
    """Conversation states"""
    GREETING = "greeting"
    PHONE_COLLECTION = "phone_collection"
    AMOUNT_COLLECTION = "amount_collection"
    TENURE_COLLECTION = "tenure_collection"
    KYC_VERIFICATION = "kyc_verification"
    CREDIT_EVALUATION = "credit_evaluation"
    OFFER_PRESENTATION = "offer_presentation"
    DECISION_ACCEPTANCE = "decision_acceptance"
    SANCTION_GENERATION = "sanction_generation"
    COMPLETED = "completed"
    ERROR = "error"


class MasterAgent:
    """
    Master Agent - Central Orchestrator
    Manages conversation flow and coordinates all worker agents
    """

    def __init__(self, db_manager, behavioral_analyzer, worker_agents: Dict):
        """
        Initialize Master Agent

        Args:
            db_manager: Database manager instance
            behavioral_analyzer: Behavioral analysis service
            worker_agents: Dictionary of worker agents (sales, verification, underwriting, sanction)
        """
        self.db = db_manager
        self.behavioral = behavioral_analyzer
        self.workers = worker_agents

        # State management
        self.state = ConversationState.GREETING
        self.application_data = {}
        self.session_data = {}
        self.personality_type = PersonalityType.AMIABLE

    def start_conversation(self, session_id: str) -> str:
        """Start new conversation"""
        self.session_data['session_id'] = session_id
        self.session_data['started_at'] = datetime.now().isoformat()
        self.state = ConversationState.PHONE_COLLECTION

        return self._get_greeting_message()

    def _get_greeting_message(self) -> str:
        """Generate greeting message"""
        return """Welcome to Tata Capital's AI Loan Assistant!

I'm here to help you get a personal loan quickly and easily - typically in under 5 minutes!

Your information is completely secure with end-to-end encryption.

I can help you with:
- Loan applications
- Status tracking
- Eligibility checks
- Interest rates & EMI calculations
- Required documents

What would you like to know?"""

    def process_message(self, message: str, response_time: float = 2.0) -> Tuple[str, Dict]:
        """
        Process incoming message based on current state

        Args:
            message: Customer message
            response_time: Response time in seconds

        Returns:
            Tuple of (response_message, metadata)
        """
        # Analyze behavioral patterns
        self.behavioral.analyze_message(message, response_time)
        self.personality_type = self.behavioral.get_personality_type()

        # Store message
        self.session_data.setdefault('messages', []).append({
            "role": "customer",
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "response_time": response_time
        })

        # Route to appropriate handler based on state
        handlers = {
            ConversationState.PHONE_COLLECTION: self._handle_phone_collection,
            ConversationState.AMOUNT_COLLECTION: self._handle_amount_collection,
            ConversationState.TENURE_COLLECTION: self._handle_tenure_collection,
            ConversationState.KYC_VERIFICATION: self._handle_kyc_verification,
            ConversationState.CREDIT_EVALUATION: self._handle_credit_evaluation,
            ConversationState.OFFER_PRESENTATION: self._handle_offer_presentation,
            ConversationState.DECISION_ACCEPTANCE: self._handle_decision_acceptance,
        }

        handler = handlers.get(self.state)
        if handler:
            response, metadata = handler(message)
        else:
            response = "I'm processing your request. Please wait..."
            metadata = {}

        # Store response
        self.session_data['messages'].append({
            "role": "assistant",
            "message": response,
            "timestamp": datetime.now().isoformat()
        })

        return response, metadata

    def _handle_phone_collection(self, message: str) -> Tuple[str, Dict]:
        """Handle phone number collection"""
        import re

        # Extract phone number
        phone_match = re.search(r'(\d{10})', message.replace(' ', '').replace('-', ''))

        if not phone_match:
            return (
                "I couldn't find a valid 10-digit mobile number. Please provide your registered mobile number.",
                {"state": self.state}
            )

        phone = phone_match.group(1)

        # Check customer existence
        customer = self.db.get_customer_by_phone(phone)

        if not customer:
            return (
                f"""I couldn't find your profile with number {phone}.

For new customers, we'll need to complete KYC verification. This is a quick process.

Would you like to:
1. Proceed with new customer registration
2. Try a different phone number
3. Call our customer service at 1800-XXX-XXXX

Please type 1, 2, or 3.""",
                {"state": self.state}
            )

        # Initialize application
        self.application_data = {
            "application_id": f"APP{datetime.now().strftime('%Y%m%d%H%M%S')}{uuid.uuid4().hex[:4].upper()}",
            "customer_id": customer['customer_id'],
            "customer_name": customer['name'],
            "phone": phone,
            "email": customer['email'],
            "status": LoanStatus.INITIATED.value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        # Move to next state
        self.state = ConversationState.AMOUNT_COLLECTION

        return (
            f"""Great! I found your profile, {customer['name']}.

{self._get_personality_intro()}

How much loan amount are you looking for? (Minimum ₹25,000, Maximum ₹20,00,000)""",
            {"state": self.state, "customer_found": True}
        )

    def _get_personality_intro(self) -> str:
        """Generate personality-based introduction"""
        intros = {
            PersonalityType.ANALYTICAL: "I'll walk you through each step clearly and answer any questions.",
            PersonalityType.DRIVER: "Let's get straight to it - I'll make this quick and efficient.",
            PersonalityType.EXPRESSIVE: "I'm excited to help you achieve your financial goals!",
            PersonalityType.AMIABLE: "I'm here to make this process smooth and comfortable for you."
        }
        return intros.get(self.personality_type, intros[PersonalityType.AMIABLE])

    def _handle_amount_collection(self, message: str) -> Tuple[str, Dict]:
        """Handle loan amount collection"""
        import re

        # Extract amount
        amount_match = re.search(r'(\d+)', message.replace(',', '').replace('₹', ''))

        if not amount_match:
            return (
                "Please specify the loan amount you need (e.g., 500000 or ₹5,00,000)",
                {"state": self.state}
            )

        amount = int(amount_match.group(1))

        if amount < 25000:
            return (
                "The minimum loan amount is ₹25,000. Please request a higher amount.",
                {"state": self.state}
            )

        if amount > 2000000:
            return (
                "The maximum personal loan amount is ₹20,00,000. For higher amounts, please contact our premium banking services at 1800-XXX-XXXX.",
                {"state": self.state}
            )

        self.application_data['loan_amount'] = amount

        # Move to tenure collection
        self.state = ConversationState.TENURE_COLLECTION

        return (
            f"""Perfect! You're requesting a loan of ₹{amount:,}.

What repayment tenure would you prefer?
• 12 months (1 year)
• 24 months (2 years)
• 36 months (3 years)
• 48 months (4 years)
• 60 months (5 years)

Please type the number of months (e.g., 36).""",
            {"state": self.state, "amount_collected": True}
        )

    def _handle_tenure_collection(self, message: str) -> Tuple[str, Dict]:
        """Handle tenure collection"""
        import re

        # Extract tenure
        tenure_match = re.search(r'(\d+)', message)

        if not tenure_match:
            return (
                "Please specify the tenure in months (12, 24, 36, 48, or 60)",
                {"state": self.state}
            )

        tenure = int(tenure_match.group(1))

        if tenure not in [12, 24, 36, 48, 60]:
            return (
                f"Invalid tenure: {tenure} months. Please choose from 12, 24, 36, 48, or 60 months.",
                {"state": self.state}
            )

        self.application_data['tenure'] = tenure

        # Move to KYC verification
        self.state = ConversationState.KYC_VERIFICATION

        # Trigger KYC verification
        return self._handle_kyc_verification(None)

    def _handle_kyc_verification(self, message: Optional[str]) -> Tuple[str, Dict]:
        """Handle KYC verification stage"""
        # Delegate to Verification Agent
        verification_agent = self.workers.get('verification')
        if not verification_agent:
            return ("System error: Verification service unavailable", {"state": "error"})

        is_verified, kyc_data, verification_message = verification_agent.verify_customer_kyc(
            self.application_data['phone']
        )

        if not is_verified:
            self.state = ConversationState.ERROR
            return (verification_message, {"state": self.state})

        # Verify employment
        is_emp_verified, emp_data, emp_message = verification_agent.verify_employment(
            self.application_data['email']
        )

        if not is_emp_verified:
            self.state = ConversationState.ERROR
            return (emp_message, {"state": self.state})

        self.application_data['employment'] = emp_data
        self.application_data['pan'] = kyc_data.get('pan')

        # Check fraud indicators
        behavioral_flags = self.behavioral.get_risk_flags()
        is_clean, fraud_flags = verification_agent.check_fraud_indicators(
            kyc_data, behavioral_flags
        )

        if not is_clean:
            self.state = ConversationState.ERROR
            return (
                f"""Additional Verification Required

We've detected items that need clarification:
{chr(10).join('* ' + flag for flag in fraud_flags)}

For security, please contact our verification team at 1800-XXX-XXXX with application ID: {self.application_data['application_id']}""",
                {"state": self.state, "fraud_detected": True}
            )

        # Move to credit evaluation
        self.state = ConversationState.CREDIT_EVALUATION

        return (
            f"""Verification Complete!

{verification_message}
{emp_message}

Now evaluating your loan eligibility...
This will take just a moment...""",
            {"state": self.state, "kyc_verified": True}
        )

    def _handle_credit_evaluation(self, message: Optional[str]) -> Tuple[str, Dict]:
        """Handle credit evaluation stage"""
        # Delegate to Underwriting Agent
        underwriting_agent = self.workers.get('underwriting')
        if not underwriting_agent:
            return ("System error: Underwriting service unavailable", {"state": "error"})

        pan = self.application_data['pan']
        loan_amount = self.application_data['loan_amount']
        employment = self.application_data['employment']
        behavioral_score = self.behavioral.get_behavioral_trust_score()
        behavioral_flags = self.behavioral.get_risk_flags()

        # Evaluate creditworthiness
        approved, final_score, explanation, factors = underwriting_agent.evaluate_creditworthiness(
            pan, loan_amount, employment, behavioral_score, behavioral_flags
        )

        self.application_data['credit_evaluation'] = {
            "approved": approved,
            "final_score": final_score,
            "factors": factors,
            "explanation": explanation
        }

        if approved:
            self.state = ConversationState.OFFER_PRESENTATION
            return self._handle_offer_presentation(None)
        else:
            self.state = ConversationState.COMPLETED
            self.application_data['status'] = LoanStatus.REJECTED.value

            # Save application
            self.db.insert('applications', self.application_data)

            return (
                f"""{explanation}

Thank you for your interest in Tata Capital. We recommend improving these areas and reapplying after 3 months.

For personalized advice, please call 1800-XXX-XXXX

Your application ID: {self.application_data['application_id']}""",
                {"state": self.state, "approved": False}
            )

    def _handle_offer_presentation(self, message: Optional[str]) -> Tuple[str, Dict]:
        """Present loan offer to customer"""
        # Delegate to Sales Agent
        sales_agent = self.workers.get('sales')
        if not sales_agent:
            return ("System error: Sales service unavailable", {"state": "error"})

        # Get credit score
        credit_profile = self.db.get_credit_profile_by_pan(self.application_data['pan'])
        employment = self.application_data['employment']

        # Get suitable products
        products = sales_agent.get_suitable_products(
            credit_profile['credit_score'],
            employment['monthly_salary'],
            self.application_data['loan_amount']
        )

        if not products:
            self.state = ConversationState.COMPLETED
            return (
                "Unfortunately, we don't have a suitable product matching your requirements at this time.",
                {"state": self.state}
            )

        best_product = products[0]
        self.application_data['product'] = best_product

        # Create personalized offer
        offer = sales_agent.create_loan_offer(
            best_product,
            self.application_data['loan_amount'],
            self.application_data['tenure'],
            self.personality_type
        )

        explanation = self.application_data['credit_evaluation']['explanation']

        self.state = ConversationState.DECISION_ACCEPTANCE

        return (
            f"""{explanation}

---

{offer}

Would you like to accept this loan offer?
Type 'YES' to proceed or 'NO' to decline.""",
            {"state": self.state, "offer_presented": True}
        )

    def _handle_decision_acceptance(self, message: str) -> Tuple[str, Dict]:
        """Handle customer decision"""
        message_lower = message.lower()

        if any(word in message_lower for word in ['yes', 'accept', 'proceed', 'ok', 'sure']):
            self.state = ConversationState.SANCTION_GENERATION
            return self._generate_sanction_letter()
        else:
            self.state = ConversationState.COMPLETED
            return (
                """I understand. No problem at all!

Your application details will be saved for 30 days. You can resume anytime.

Would you like to:
1. Modify the loan amount or tenure
2. Speak with a loan advisor
3. Exit

Thank you for considering Tata Capital!""",
                {"state": self.state, "declined": True}
            )

    def _generate_sanction_letter(self) -> Tuple[str, Dict]:
        """Generate sanction letter"""
        sanction_agent = self.workers.get('sanction')
        if not sanction_agent:
            return ("System error: Sanction service unavailable", {"state": "error"})

        # Calculate EMI
        sales_agent = self.workers.get('sales')
        emi = sales_agent.calculate_emi(
            self.application_data['loan_amount'],
            self.application_data['product']['interest_rate'],
            self.application_data['tenure']
        )

        # Generate sanction letter
        sanction_letter = sanction_agent.generate_sanction_letter(
            self.application_data,
            self.application_data['product'],
            emi
        )

        filename = sanction_agent.save_sanction_letter(
            self.application_data['application_id'],
            sanction_letter
        )

        self.application_data['status'] = LoanStatus.SANCTION_GENERATED.value
        self.application_data['sanction_letter'] = filename
        self.application_data['emi'] = emi

        # Save application
        self.db.insert('applications', self.application_data)

        self.state = ConversationState.COMPLETED

        return (
            f"""Congratulations! Your loan has been sanctioned!

{sanction_letter}

Your sanction letter: {filename}

NEXT STEPS:
1. Check your email for the loan agreement
2. Sign and upload the agreement
3. Loan disbursed within 24 hours

Thank you for choosing Tata Capital!

Application ID: {self.application_data['application_id']}""",
            {"state": self.state, "sanctioned": True, "application_id": self.application_data['application_id']}
        )

    def get_session_summary(self) -> Dict:
        """Get conversation summary"""
        return {
            "session_id": self.session_data.get('session_id'),
            "application_id": self.application_data.get('application_id'),
            "state": self.state.value,
            "personality_type": self.personality_type.value,
            "behavioral_score": self.behavioral.get_behavioral_trust_score(),
            "application_data": self.application_data,
            "total_messages": len(self.session_data.get('messages', []))
        }
