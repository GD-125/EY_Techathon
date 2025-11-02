"""
Underwriting Agent
Handles credit evaluation with explainable AI
"""

from typing import Tuple, List, Dict, Any


class UnderwritingAgent:
    """Underwriting Agent for credit evaluation"""

    def __init__(self, db_manager, explainability_engine):
        self.db = db_manager
        self.explainer = explainability_engine

    def evaluate_creditworthiness(
        self,
        pan: str,
        loan_amount: float,
        employment: Dict,
        behavioral_score: float,
        behavioral_flags: List[str]
    ) -> Tuple[bool, float, str, List[Dict]]:
        """
        Comprehensive credit evaluation with explainability

        Args:
            pan: PAN number
            loan_amount: Requested loan amount
            employment: Employment details
            behavioral_score: Behavioral trust score
            behavioral_flags: Behavioral risk flags

        Returns:
            Tuple of (approved, final_score, explanation, factors)
        """
        credit_data = self.db.get_credit_profile_by_pan(pan)

        if not credit_data:
            return False, 0, "❌ No credit history found. Unable to process loan.", []

        # Decrypt credit score if encrypted
        if credit_data.get('credit_score_encrypted'):
            from ...services.encryption.crypto_service import CryptoService
            # In real implementation, use injected crypto service
            credit_score = credit_data['credit_score']
        else:
            credit_score = credit_data['credit_score']

        # Collect explanation factors
        factors = []

        # Factor 1: Credit Score (35% weight)
        factors.append(self.explainer.explain_credit_score_impact(credit_score))

        # Factor 2: Income vs Loan Amount (25% weight)
        factors.append(self.explainer.explain_income_impact(
            employment['monthly_salary'], loan_amount
        ))

        # Factor 3: Existing Loans (15% weight)
        factors.append(self.explainer.explain_existing_loans_impact(
            credit_data['active_loans'], credit_data['total_loan_amount']
        ))

        # Factor 4: Employment Stability (15% weight)
        factors.append(self.explainer.explain_employment_stability(
            employment['years_in_company'], employment['employment_type']
        ))

        # Factor 5: Behavioral Score (10% weight)
        factors.append(self.explainer.explain_behavioral_score(
            behavioral_score, behavioral_flags
        ))

        # Calculate final weighted score
        final_score = sum(
            (100 if f['impact'] == 'Positive' else 50 if f['impact'] == 'Neutral' else 0) * f['weight']
            for f in factors
        )

        # Generate explanation
        explanation = self.explainer.generate_full_explanation(factors)

        # Decision threshold
        approved = final_score >= 60

        if approved:
            explanation += "\n\n✅ DECISION: LOAN APPROVED\n"
            explanation += "Your application meets our lending criteria. Congratulations!"
        else:
            explanation += "\n\n❌ DECISION: LOAN DECLINED\n"
            explanation += "Unfortunately, your application doesn't meet current criteria.\n\n"
            explanation += "💡 Recommendations:\n"
            explanation += "  • Improve credit score (current: {0})\n".format(credit_score)
            explanation += "  • Reduce existing debt burden\n"
            explanation += "  • Consider a lower loan amount\n"
            explanation += "  • Build longer employment history\n"

        return approved, final_score, explanation, factors

    def suggest_alternatives(
        self,
        original_amount: float,
        credit_score: int,
        monthly_income: float
    ) -> List[Dict]:
        """
        Suggest alternative loan options

        Args:
            original_amount: Original requested amount
            credit_score: Credit score
            monthly_income: Monthly income

        Returns:
            List of alternative options
        """
        alternatives = []

        # Suggest lower amount
        if original_amount > 100000:
            lower_amount = original_amount * 0.6
            alternatives.append({
                "type": "Lower Amount",
                "amount": lower_amount,
                "reason": "A smaller loan may be approved based on your current profile"
            })

        # Suggest joint application
        alternatives.append({
            "type": "Joint Application",
            "amount": original_amount,
            "reason": "Adding a co-applicant strengthens your application"
        })

        # Suggest secured loan
        alternatives.append({
            "type": "Secured Loan",
            "amount": original_amount,
            "reason": "Providing collateral improves approval chances and reduces rates"
        })

        return alternatives
