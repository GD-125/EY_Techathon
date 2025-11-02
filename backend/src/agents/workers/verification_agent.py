"""
Verification Agent
Handles KYC verification and fraud detection
"""

from typing import Tuple, Dict, List


class VerificationAgent:
    """Verification Agent for KYC and fraud detection"""

    def __init__(self, db_manager, crypto_service):
        self.db = db_manager
        self.crypto = crypto_service

    def verify_customer_kyc(self, phone: str) -> Tuple[bool, Dict, str]:
        """
        Verify customer KYC details

        Args:
            phone: Customer phone number

        Returns:
            Tuple of (is_verified, customer_data, message)
        """
        customer = self.db.get_customer_by_phone(phone)

        if not customer:
            return False, {}, "Customer not found. New KYC verification required."

        # Decrypt sensitive fields if encrypted
        if self.crypto and customer.get('pan_encrypted'):
            customer = self.crypto.decrypt_dict(customer, ['pan', 'aadhaar', 'address', 'email'])

        if customer.get('kyc_verified'):
            return (
                True,
                customer,
                f"✅ KYC verified for {customer['name']}. All documents are in order."
            )
        else:
            return (
                False,
                customer,
                f"⚠️ KYC pending for {customer['name']}. Please upload PAN and Aadhaar."
            )

    def verify_employment(self, email: str) -> Tuple[bool, Dict, str]:
        """
        Verify employment details

        Args:
            email: Customer email

        Returns:
            Tuple of (is_verified, employment_data, message)
        """
        employment = self.db.get_employment_by_email(email)

        if not employment:
            return False, {}, "Employment details not found. Please provide salary slip."

        # Decrypt salary if encrypted
        if self.crypto and employment.get('monthly_salary_encrypted'):
            employment = self.crypto.decrypt_dict(
                employment, ['monthly_salary', 'annual_income']
            )

        if employment.get('verified'):
            return (
                True,
                employment,
                f"✅ Employment verified: {employment['designation']} at {employment['company']}"
            )
        else:
            return (
                False,
                employment,
                "⚠️ Employment verification pending. Additional documents required."
            )

    def check_fraud_indicators(
        self,
        customer_data: Dict,
        behavioral_flags: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Check for potential fraud indicators

        Args:
            customer_data: Customer information
            behavioral_flags: Behavioral risk flags

        Returns:
            Tuple of (is_clean, red_flags)
        """
        red_flags = []

        # Add behavioral flags
        if behavioral_flags:
            red_flags.extend(behavioral_flags)

        # Check phone number pattern
        phone = customer_data.get('phone', '')
        if phone.startswith('0000') or phone.startswith('1111'):
            red_flags.append("Suspicious phone number pattern")

        # Check address completeness
        address = customer_data.get('address', '')
        if len(address) < 20:
            red_flags.append("Incomplete address information")

        # Check email domain
        email = customer_data.get('email', '')
        suspicious_domains = ['tempmail', 'throwaway', 'fakeemail']
        if any(domain in email.lower() for domain in suspicious_domains):
            red_flags.append("Suspicious email domain")

        is_clean = len(red_flags) == 0
        return is_clean, red_flags
