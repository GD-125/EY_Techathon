"""
Tests for Credit Scoring Service
"""
import sys
from pathlib import Path

# Add backend src to path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))

from src.services.credit.credit_scoring_service import CreditScoringService


def test_credit_score_calculation():
    """Test basic credit score calculation"""
    service = CreditScoringService()

    applicant_data = {
        'annual_income': 75000,
        'existing_debt': 15000,
        'payment_history_score': 85,
        'credit_age_months': 60,
        'num_credit_accounts': 5,
        'recent_inquiries': 1,
        'loan_amount': 50000
    }

    result = service.calculate_credit_score(applicant_data)

    # Assertions
    assert result.score >= 300 and result.score <= 850, "Credit score out of range"
    assert result.risk_level in ['excellent', 'good', 'fair', 'poor', 'very_poor'], "Invalid risk level"
    assert 0 <= result.confidence <= 1, "Confidence out of range"
    assert len(result.factors) > 0, "No factors provided"
    assert len(result.recommendations) > 0, "No recommendations provided"

    print(f"[PASS] Credit score: {result.score}")
    print(f"[PASS] Risk level: {result.risk_level}")
    print(f"[PASS] Confidence: {result.confidence}")
    print(f"[PASS] Factors count: {len(result.factors)}")


def test_loan_risk_assessment():
    """Test loan risk assessment"""
    service = CreditScoringService()

    applicant_data = {
        'annual_income': 75000,
        'existing_debt': 15000,
        'payment_history_score': 85,
        'credit_age_months': 60,
        'num_credit_accounts': 5,
        'recent_inquiries': 1,
        'employment_months': 48
    }

    result = service.assess_loan_risk(
        applicant_data=applicant_data,
        loan_amount=50000,
        loan_term_months=60
    )

    # Assertions
    assert result['decision'] in ['APPROVED', 'REJECTED'], "Invalid decision"
    assert 'credit_score' in result, "Credit score missing"
    assert 'reasoning' in result, "Reasoning missing"
    assert 'factors' in result, "Factors missing"
    assert 'recommendations' in result, "Recommendations missing"

    print(f"[PASS] Decision: {result['decision']}")
    print(f"[PASS] Approval score: {result['approval_score']}")
    print(f"[PASS] Credit score: {result['credit_score']}")


def test_high_risk_rejection():
    """Test that high-risk applications are rejected"""
    service = CreditScoringService()

    # High-risk applicant
    applicant_data = {
        'annual_income': 30000,
        'existing_debt': 25000,
        'payment_history_score': 55,
        'credit_age_months': 12,
        'num_credit_accounts': 1,
        'recent_inquiries': 5,
        'employment_months': 6
    }

    result = service.assess_loan_risk(
        applicant_data=applicant_data,
        loan_amount=50000,
        loan_term_months=60
    )

    assert result['decision'] == 'REJECTED', "High-risk application should be rejected"
    print(f"[PASS] High-risk correctly rejected")


def test_low_risk_approval():
    """Test that low-risk applications are approved"""
    service = CreditScoringService()

    # Low-risk applicant
    applicant_data = {
        'annual_income': 120000,
        'existing_debt': 10000,
        'payment_history_score': 95,
        'credit_age_months': 120,
        'num_credit_accounts': 8,
        'recent_inquiries': 0,
        'employment_months': 96
    }

    result = service.assess_loan_risk(
        applicant_data=applicant_data,
        loan_amount=50000,
        loan_term_months=60
    )

    assert result['decision'] == 'APPROVED', "Low-risk application should be approved"
    print(f"[PASS] Low-risk correctly approved")


if __name__ == '__main__':
    print("Running Credit Scoring Service Tests...")
    print("=" * 60)

    try:
        test_credit_score_calculation()
        print("\n" + "=" * 60)

        test_loan_risk_assessment()
        print("\n" + "=" * 60)

        test_high_risk_rejection()
        print("\n" + "=" * 60)

        test_low_risk_approval()
        print("\n" + "=" * 60)

        print("\n[SUCCESS] All tests passed!")

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
    except Exception as e:
        print(f"\n[ERROR] Error running tests: {e}")
