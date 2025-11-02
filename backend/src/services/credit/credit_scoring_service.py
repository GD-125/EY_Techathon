"""
Credit Scoring Service with ML-based Risk Assessment
"""
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CreditScore:
    """Credit score result with explainability"""
    score: int
    risk_level: str
    confidence: float
    factors: List[Dict[str, Any]]
    recommendations: List[str]
    shap_values: Optional[Dict[str, float]] = None


class CreditScoringService:
    """
    Advanced credit scoring service with explainability
    Uses multiple factors to calculate credit score and risk level
    """

    def __init__(self):
        self.score_weights = {
            'payment_history': 0.35,
            'credit_utilization': 0.30,
            'credit_history_length': 0.15,
            'credit_mix': 0.10,
            'new_credit': 0.10
        }

        self.risk_thresholds = {
            'excellent': (750, 850),
            'good': (700, 749),
            'fair': (650, 699),
            'poor': (600, 649),
            'very_poor': (300, 599)
        }

    def calculate_credit_score(self, applicant_data: Dict[str, Any]) -> CreditScore:
        """
        Calculate comprehensive credit score with explainability

        Args:
            applicant_data: Dictionary containing applicant information

        Returns:
            CreditScore object with detailed breakdown
        """
        try:
            # Extract features
            income = applicant_data.get('annual_income', 0)
            existing_debt = applicant_data.get('existing_debt', 0)
            payment_history_score = applicant_data.get('payment_history_score', 75)
            credit_age_months = applicant_data.get('credit_age_months', 24)
            num_credit_accounts = applicant_data.get('num_credit_accounts', 3)
            recent_inquiries = applicant_data.get('recent_inquiries', 0)

            # Calculate individual components
            components = self._calculate_components(
                income=income,
                existing_debt=existing_debt,
                payment_history_score=payment_history_score,
                credit_age_months=credit_age_months,
                num_credit_accounts=num_credit_accounts,
                recent_inquiries=recent_inquiries
            )

            # Calculate weighted score
            final_score = self._calculate_weighted_score(components)

            # Determine risk level
            risk_level = self._determine_risk_level(final_score)

            # Calculate confidence
            confidence = self._calculate_confidence(components)

            # Generate factor analysis
            factors = self._generate_factors(components, applicant_data)

            # Generate SHAP-like values for explainability
            shap_values = self._calculate_shap_values(components)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                final_score,
                risk_level,
                components,
                applicant_data
            )

            return CreditScore(
                score=int(final_score),
                risk_level=risk_level,
                confidence=confidence,
                factors=factors,
                recommendations=recommendations,
                shap_values=shap_values
            )

        except Exception as e:
            logger.error(f"Error calculating credit score: {e}")
            # Return default safe score
            return CreditScore(
                score=650,
                risk_level='fair',
                confidence=0.5,
                factors=[],
                recommendations=['Unable to calculate accurate score. Manual review required.']
            )

    def _calculate_components(
        self,
        income: float,
        existing_debt: float,
        payment_history_score: float,
        credit_age_months: int,
        num_credit_accounts: int,
        recent_inquiries: int
    ) -> Dict[str, float]:
        """Calculate individual credit score components"""

        # Payment History (35%)
        payment_component = min(payment_history_score / 100.0, 1.0)

        # Credit Utilization (30%)
        if income > 0:
            debt_to_income = existing_debt / (income / 12)
            utilization_component = max(0, 1 - (debt_to_income / 0.43))  # 43% DTI threshold
        else:
            utilization_component = 0.5

        # Credit History Length (15%)
        age_component = min(credit_age_months / 120.0, 1.0)  # 10 years = perfect

        # Credit Mix (10%)
        mix_component = min(num_credit_accounts / 8.0, 1.0)  # 8+ accounts = perfect

        # New Credit (10%)
        inquiry_component = max(0, 1 - (recent_inquiries / 6.0))  # 6+ inquiries = 0

        return {
            'payment_history': payment_component * 100,
            'credit_utilization': utilization_component * 100,
            'credit_history_length': age_component * 100,
            'credit_mix': mix_component * 100,
            'new_credit': inquiry_component * 100
        }

    def _calculate_weighted_score(self, components: Dict[str, float]) -> float:
        """Calculate final weighted credit score"""
        weighted_sum = sum(
            components[key] * self.score_weights[key]
            for key in self.score_weights
        )

        # Scale to 300-850 range
        final_score = 300 + (weighted_sum / 100.0) * 550
        return min(850, max(300, final_score))

    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level based on score"""
        for level, (min_score, max_score) in self.risk_thresholds.items():
            if min_score <= score <= max_score:
                return level
        return 'unknown'

    def _calculate_confidence(self, components: Dict[str, float]) -> float:
        """Calculate confidence in the credit score"""
        # Higher variance in components = lower confidence
        values = list(components.values())
        mean_val = np.mean(values)
        variance = np.var(values)

        # Normalize to 0-1 range
        confidence = 1.0 - min(variance / 1000.0, 0.5)
        return max(0.5, min(1.0, confidence))

    def _generate_factors(
        self,
        components: Dict[str, float],
        applicant_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate detailed factor analysis"""
        factors = []

        for key, value in components.items():
            impact = 'positive' if value >= 70 else 'negative' if value < 50 else 'neutral'
            importance = self.score_weights[key]

            factors.append({
                'feature': key.replace('_', ' ').title(),
                'value': round(value, 2),
                'impact': impact,
                'importance': importance,
                'description': self._get_factor_description(key, value, applicant_data)
            })

        # Sort by importance
        factors.sort(key=lambda x: x['importance'], reverse=True)
        return factors

    def _calculate_shap_values(self, components: Dict[str, float]) -> Dict[str, float]:
        """Calculate SHAP-like values for explainability"""
        base_score = 650  # Average credit score
        shap_values = {}

        for key, value in components.items():
            # Calculate contribution to final score
            contribution = (value - 70) * self.score_weights[key] * 5.5
            # Ensure float type for JSON serialization
            shap_values[key.replace('_', ' ').title()] = float(round(contribution, 2))

        return shap_values

    def _get_factor_description(
        self,
        factor: str,
        value: float,
        applicant_data: Dict[str, Any]
    ) -> str:
        """Get human-readable description for each factor"""
        descriptions = {
            'payment_history': f"Payment history score of {value:.0f}%. " +
                             ("Excellent payment record." if value >= 80 else
                              "Some payment issues detected." if value >= 60 else
                              "Significant payment problems."),

            'credit_utilization': f"Debt-to-income ratio at {value:.0f}%. " +
                                ("Low utilization is excellent." if value >= 80 else
                                 "Moderate utilization." if value >= 60 else
                                 "High utilization may be concerning."),

            'credit_history_length': f"Credit history length at {value:.0f}%. " +
                                   ("Long credit history is positive." if value >= 70 else
                                    "Building credit history."),

            'credit_mix': f"Credit mix diversity at {value:.0f}%. " +
                        ("Good variety of credit types." if value >= 70 else
                         "Limited credit variety."),

            'new_credit': f"Recent credit inquiry score at {value:.0f}%. " +
                        ("Few recent inquiries." if value >= 80 else
                         "Multiple recent inquiries may indicate risk.")
        }

        return descriptions.get(factor, f"Score: {value:.0f}%")

    def _generate_recommendations(
        self,
        score: float,
        risk_level: str,
        components: Dict[str, float],
        applicant_data: Dict[str, Any]
    ) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []

        # Score-based recommendations
        if score < 700:
            recommendations.append("Consider improving credit score before applying for large loans")

        # Component-specific recommendations
        if components['payment_history'] < 70:
            recommendations.append("Focus on making all payments on time to improve payment history")

        if components['credit_utilization'] < 60:
            recommendations.append("Reduce debt-to-income ratio by paying down existing debts")

        if components['credit_history_length'] < 50:
            recommendations.append("Building longer credit history will improve score over time")

        if components['new_credit'] < 70:
            recommendations.append("Avoid applying for multiple credit accounts in short period")

        # Income-based recommendations
        income = applicant_data.get('annual_income', 0)
        loan_amount = applicant_data.get('loan_amount', 0)

        if loan_amount > income * 0.3:
            recommendations.append("Loan amount is high relative to income. Consider requesting lower amount")

        if not recommendations:
            recommendations.append("Credit profile is strong. Good candidate for loan approval")

        return recommendations

    def assess_loan_risk(
        self,
        applicant_data: Dict[str, Any],
        loan_amount: float,
        loan_term_months: int
    ) -> Dict[str, Any]:
        """
        Comprehensive loan risk assessment

        Returns:
            Dictionary with risk assessment and explainability
        """
        # Calculate credit score
        credit_score = self.calculate_credit_score(applicant_data)

        # Calculate additional risk factors
        monthly_income = applicant_data.get('annual_income', 0) / 12
        monthly_payment = self._calculate_monthly_payment(loan_amount, loan_term_months)
        payment_to_income = (monthly_payment / monthly_income) if monthly_income > 0 else 1.0

        # Determine approval likelihood
        approval_score = self._calculate_approval_score(
            credit_score.score,
            payment_to_income,
            applicant_data
        )

        # Final decision
        decision = 'APPROVED' if approval_score >= 70 else 'REJECTED'

        return {
            'decision': str(decision),
            'approval_score': float(approval_score),
            'credit_score': int(credit_score.score),
            'risk_level': str(credit_score.risk_level),
            'confidence': float(credit_score.confidence),
            'monthly_payment': float(round(monthly_payment, 2)),
            'payment_to_income_ratio': float(round(payment_to_income, 4)),
            'factors': credit_score.factors,
            'shap_values': credit_score.shap_values,
            'recommendations': credit_score.recommendations,
            'reasoning': str(self._generate_reasoning(
                decision,
                credit_score.score,
                payment_to_income,
                approval_score
            ))
        }

    def _calculate_monthly_payment(self, principal: float, months: int, annual_rate: float = 0.08) -> float:
        """Calculate monthly loan payment"""
        if months == 0:
            return 0

        monthly_rate = annual_rate / 12
        if monthly_rate == 0:
            return principal / months

        payment = principal * (monthly_rate * (1 + monthly_rate)**months) / \
                  ((1 + monthly_rate)**months - 1)
        return payment

    def _calculate_approval_score(
        self,
        credit_score: float,
        payment_to_income: float,
        applicant_data: Dict[str, Any]
    ) -> float:
        """Calculate loan approval score"""
        # Credit score component (60%)
        credit_component = (credit_score - 300) / 550 * 60

        # Payment-to-income component (30%)
        pti_component = max(0, (0.35 - payment_to_income) / 0.35) * 30

        # Additional factors (10%)
        employment_months = applicant_data.get('employment_months', 0)
        employment_component = min(employment_months / 24, 1.0) * 10

        approval_score = credit_component + pti_component + employment_component
        return min(100, max(0, approval_score))

    def _generate_reasoning(
        self,
        decision: str,
        credit_score: float,
        payment_to_income: float,
        approval_score: float
    ) -> str:
        """Generate human-readable reasoning for the decision"""
        if decision == 'APPROVED':
            return (f"Application approved based on credit score of {credit_score:.0f} "
                   f"and payment-to-income ratio of {payment_to_income:.1%}. "
                   f"Overall approval score: {approval_score:.1f}/100. "
                   f"Applicant demonstrates good creditworthiness and ability to repay.")
        else:
            reasons = []
            if credit_score < 650:
                reasons.append(f"credit score of {credit_score:.0f} is below threshold")
            if payment_to_income > 0.35:
                reasons.append(f"payment-to-income ratio of {payment_to_income:.1%} exceeds 35% limit")

            reason_text = " and ".join(reasons) if reasons else "risk factors"
            return (f"Application rejected due to {reason_text}. "
                   f"Overall approval score: {approval_score:.1f}/100. "
                   f"Consider addressing these factors before reapplying.")
