"""
Explainability Engine
Provides transparent AI decision explanations (LIME-inspired)
"""

from typing import Dict, List, Any


class ExplainabilityEngine:
    """
    Explainability Engine for transparent loan decisions
    Based on LIME (Local Interpretable Model-agnostic Explanations)
    """

    @staticmethod
    def explain_credit_score_impact(credit_score: int) -> Dict[str, Any]:
        """Explain credit score impact on decision"""
        if credit_score >= 750:
            return {
                "factor": "Credit Score",
                "value": credit_score,
                "impact": "Positive",
                "weight": 0.35,
                "explanation": (
                    f"Your excellent credit score of {credit_score} qualifies you for "
                    f"our best interest rates. This demonstrates strong creditworthiness "
                    f"and payment history."
                )
            }
        elif credit_score >= 650:
            return {
                "factor": "Credit Score",
                "value": credit_score,
                "impact": "Neutral",
                "weight": 0.35,
                "explanation": (
                    f"Your credit score of {credit_score} is good. You qualify for "
                    f"standard loan products. Improving above 750 can get premium rates."
                )
            }
        else:
            return {
                "factor": "Credit Score",
                "value": credit_score,
                "impact": "Negative",
                "weight": 0.35,
                "explanation": (
                    f"Your credit score of {credit_score} is below our preferred threshold. "
                    f"This may result in higher interest rates or require additional verification."
                )
            }

    @staticmethod
    def explain_income_impact(monthly_income: float, loan_amount: float) -> Dict[str, Any]:
        """Explain income-to-loan ratio impact"""
        emi_estimate = loan_amount * 0.02  # Rough 2% EMI
        debt_to_income_ratio = (emi_estimate / monthly_income) * 100

        if debt_to_income_ratio < 30:
            return {
                "factor": "Income vs Loan Amount",
                "value": f"₹{monthly_income:,.0f}/month",
                "impact": "Positive",
                "weight": 0.25,
                "explanation": (
                    f"Your monthly income of ₹{monthly_income:,.0f} comfortably supports "
                    f"the loan. Estimated EMI is {debt_to_income_ratio:.1f}% of income "
                    f"(well within 40% safe limit)."
                )
            }
        elif debt_to_income_ratio < 40:
            return {
                "factor": "Income vs Loan Amount",
                "value": f"₹{monthly_income:,.0f}/month",
                "impact": "Neutral",
                "weight": 0.25,
                "explanation": (
                    f"Your monthly income of ₹{monthly_income:,.0f} adequately supports "
                    f"the loan. Estimated EMI is {debt_to_income_ratio:.1f}% of income, "
                    f"acceptable but near our threshold."
                )
            }
        else:
            return {
                "factor": "Income vs Loan Amount",
                "value": f"₹{monthly_income:,.0f}/month",
                "impact": "Negative",
                "weight": 0.25,
                "explanation": (
                    f"The loan amount may be high relative to income of ₹{monthly_income:,.0f}. "
                    f"Estimated EMI ({debt_to_income_ratio:.1f}%) exceeds our 40% threshold. "
                    f"Consider a lower amount."
                )
            }

    @staticmethod
    def explain_existing_loans_impact(active_loans: int, total_debt: float) -> Dict[str, Any]:
        """Explain existing loans impact"""
        if active_loans == 0:
            return {
                "factor": "Existing Loan Burden",
                "value": "No active loans",
                "impact": "Positive",
                "weight": 0.15,
                "explanation": (
                    "You have no existing loans, indicating strong debt management "
                    "and leaving room for this new loan."
                )
            }
        elif active_loans <= 2 and total_debt < 500000:
            return {
                "factor": "Existing Loan Burden",
                "value": f"{active_loans} loans, ₹{total_debt:,.0f} total",
                "impact": "Neutral",
                "weight": 0.15,
                "explanation": (
                    f"You have {active_loans} existing loan(s) totaling ₹{total_debt:,.0f}. "
                    f"This is manageable but we'll consider total debt obligation."
                )
            }
        else:
            return {
                "factor": "Existing Loan Burden",
                "value": f"{active_loans} loans, ₹{total_debt:,.0f} total",
                "impact": "Negative",
                "weight": 0.15,
                "explanation": (
                    f"You have {active_loans} existing loans totaling ₹{total_debt:,.0f}. "
                    f"This high debt may impact ability to service additional loans."
                )
            }

    @staticmethod
    def explain_employment_stability(years_in_company: float, employment_type: str) -> Dict[str, Any]:
        """Explain employment stability impact"""
        if years_in_company >= 2 and employment_type == "permanent":
            return {
                "factor": "Employment Stability",
                "value": f"{years_in_company} years, {employment_type}",
                "impact": "Positive",
                "weight": 0.15,
                "explanation": (
                    f"Your {years_in_company} years of permanent employment demonstrates "
                    f"excellent job stability, reducing default risk."
                )
            }
        elif years_in_company >= 1:
            return {
                "factor": "Employment Stability",
                "value": f"{years_in_company} years, {employment_type}",
                "impact": "Neutral",
                "weight": 0.15,
                "explanation": (
                    f"Your employment history of {years_in_company} years is acceptable. "
                    f"Longer tenure would strengthen your application."
                )
            }
        else:
            return {
                "factor": "Employment Stability",
                "value": f"{years_in_company} years, {employment_type}",
                "impact": "Negative",
                "weight": 0.15,
                "explanation": (
                    f"Your employment tenure of {years_in_company} years is relatively short, "
                    f"which increases perceived risk."
                )
            }

    @staticmethod
    def explain_behavioral_score(behavioral_score: float, flags: List[str]) -> Dict[str, Any]:
        """Explain behavioral trust score impact"""
        if behavioral_score >= 80 and not flags:
            return {
                "factor": "Behavioral Trust Score",
                "value": f"{behavioral_score:.0f}/100",
                "impact": "Positive",
                "weight": 0.10,
                "explanation": (
                    f"Your interaction patterns show high engagement (score: {behavioral_score:.0f}/100). "
                    f"No behavioral risk flags detected."
                )
            }
        elif behavioral_score >= 60:
            flag_text = f"Minor flags: {', '.join(flags)}" if flags else "No significant concerns."
            return {
                "factor": "Behavioral Trust Score",
                "value": f"{behavioral_score:.0f}/100",
                "impact": "Neutral",
                "weight": 0.10,
                "explanation": (
                    f"Your interaction patterns are normal (score: {behavioral_score:.0f}/100). "
                    f"{flag_text}"
                )
            }
        else:
            return {
                "factor": "Behavioral Trust Score",
                "value": f"{behavioral_score:.0f}/100",
                "impact": "Negative",
                "weight": 0.10,
                "explanation": (
                    f"Some concerns in interaction patterns (score: {behavioral_score:.0f}/100). "
                    f"Flags: {', '.join(flags)}"
                )
            }

    @staticmethod
    def generate_full_explanation(factors: List[Dict[str, Any]]) -> str:
        """Generate comprehensive explanation"""
        positive = [f for f in factors if f['impact'] == 'Positive']
        negative = [f for f in factors if f['impact'] == 'Negative']
        neutral = [f for f in factors if f['impact'] == 'Neutral']

        explanation = "\n📊 LOAN DECISION EXPLANATION\n" + "="*70 + "\n\n"

        if positive:
            explanation += "✅ POSITIVE FACTORS:\n"
            for factor in positive:
                explanation += f"  • {factor['factor']}: {factor['explanation']}\n\n"

        if neutral:
            explanation += "➖ NEUTRAL FACTORS:\n"
            for factor in neutral:
                explanation += f"  • {factor['factor']}: {factor['explanation']}\n\n"

        if negative:
            explanation += "❌ NEGATIVE FACTORS:\n"
            for factor in negative:
                explanation += f"  • {factor['factor']}: {factor['explanation']}\n\n"

        # Calculate weighted score
        total_score = sum(
            (100 if f['impact'] == 'Positive' else 50 if f['impact'] == 'Neutral' else 0) * f['weight']
            for f in factors
        )

        explanation += f"📈 Overall Weighted Score: {total_score:.1f}/100\n"
        explanation += f"🎯 Approval Threshold: 60/100\n"

        return explanation
