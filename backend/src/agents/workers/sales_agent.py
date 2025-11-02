"""
Sales Agent
Handles loan product offering and personalized sales pitch
"""

from typing import List, Dict
from ...models.schemas import PersonalityType


class SalesAgent:
    """Sales Agent for loan product matching and offering"""

    def __init__(self, db_manager):
        self.db = db_manager

    def get_suitable_products(
        self,
        credit_score: int,
        monthly_income: float,
        requested_amount: float
    ) -> List[Dict]:
        """
        Find suitable loan products based on customer profile

        Args:
            credit_score: Customer credit score
            monthly_income: Monthly income
            requested_amount: Requested loan amount

        Returns:
            List of suitable products sorted by interest rate
        """
        all_products = self.db.get_all('loan_products')
        suitable = []

        for product in all_products:
            if (credit_score >= product['min_credit_score'] and
                monthly_income >= product['min_monthly_income'] and
                product['min_amount'] <= requested_amount <= product['max_amount']):
                suitable.append(product)

        return sorted(suitable, key=lambda x: x['interest_rate'])

    def create_loan_offer(
        self,
        product: Dict,
        amount: float,
        tenure: int,
        personality: PersonalityType
    ) -> str:
        """
        Create personalized loan offer based on personality type

        Args:
            product: Loan product details
            amount: Loan amount
            tenure: Tenure in months
            personality: Customer personality type

        Returns:
            Personalized offer message
        """
        emi = self.calculate_emi(amount, product['interest_rate'], tenure)
        processing_fee = amount * product['processing_fee'] / 100

        # Personality-based messaging
        if personality == PersonalityType.ANALYTICAL:
            return f"""
📋 DETAILED LOAN OFFER - {product['name']}

Loan Amount: ₹{amount:,.2f}
Interest Rate: {product['interest_rate']}% p.a.
Tenure: {tenure} months
Monthly EMI: ₹{emi:,.2f}
Processing Fee: ₹{processing_fee:,.2f} ({product['processing_fee']}%)
Total Interest: ₹{(emi * tenure - amount):,.2f}
Total Repayment: ₹{(emi * tenure):,.2f}

This offer provides competitive terms based on your profile.
Would you like a detailed breakdown of any component?
"""
        elif personality == PersonalityType.DRIVER:
            return f"""
🎯 LOAN OFFER - {product['name']}

₹{amount:,.0f} @ {product['interest_rate']}% | {tenure} months
EMI: ₹{emi:,.0f}/month

Ready to proceed?
"""
        elif personality == PersonalityType.EXPRESSIVE:
            return f"""
🌟 GREAT NEWS! You qualify for {product['name']}!

Imagine having ₹{amount:,.0f} with just ₹{emi:,.0f} monthly payments!
At {product['interest_rate']}% interest for {tenure} months.

Shall we move forward with this exciting opportunity?
"""
        else:  # AMIABLE
            return f"""
😊 Personalized Offer - {product['name']}

Loan Amount: ₹{amount:,.2f}
Monthly Payment: ₹{emi:,.2f}
Interest Rate: {product['interest_rate']}% p.a.
Tenure: {tenure} months

This plan fits comfortably within your budget. How does this sound?
"""

    @staticmethod
    def calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> float:
        """
        Calculate EMI using reducing balance method

        Args:
            principal: Loan amount
            annual_rate: Annual interest rate
            tenure_months: Tenure in months

        Returns:
            EMI amount
        """
        monthly_rate = annual_rate / (12 * 100)
        if monthly_rate == 0:
            return principal / tenure_months

        emi = (principal * monthly_rate * ((1 + monthly_rate) ** tenure_months) /
               (((1 + monthly_rate) ** tenure_months) - 1))
        return round(emi, 2)
