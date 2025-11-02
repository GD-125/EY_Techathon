"""
Sanction Letter Agent
Generates loan sanction letters
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict


class SanctionLetterAgent:
    """Sanction Letter Agent for document generation"""

    def __init__(self, db_manager):
        self.db = db_manager

    def generate_sanction_letter(
        self,
        application: Dict,
        product: Dict,
        emi: float
    ) -> str:
        """
        Generate formal sanction letter

        Args:
            application: Loan application data
            product: Loan product details
            emi: EMI amount

        Returns:
            Sanction letter content
        """
        sanction_id = f"SL{datetime.now().strftime('%Y%m%d')}{uuid.uuid4().hex[:6].upper()}"
        validity_date = (datetime.now() + timedelta(days=30)).strftime("%d-%m-%Y")
        processing_fee = application['loan_amount'] * product['processing_fee'] / 100

        letter = f"""
{'='*70}
                    LOAN SANCTION LETTER
{'='*70}

Sanction ID: {sanction_id}
Date: {datetime.now().strftime("%d-%m-%Y")}
Valid Until: {validity_date}

CUSTOMER DETAILS:
Name: {application['customer_name']}
Customer ID: {application['customer_id']}
Contact: {application['phone']}
Email: {application['email']}

LOAN DETAILS:
Product: {product['name']}
Sanctioned Amount: ₹{application['loan_amount']:,.2f}
Interest Rate: {product['interest_rate']}% per annum
Tenure: {application['tenure']} months
Monthly EMI: ₹{emi:,.2f}
Processing Fee: ₹{processing_fee:,.2f}

REPAYMENT SCHEDULE:
Total Amount Payable: ₹{(emi * application['tenure']):,.2f}
Total Interest: ₹{(emi * application['tenure'] - application['loan_amount']):,.2f}

TERMS & CONDITIONS:
1. This sanction is valid for 30 days from date of issue
2. Disbursement subject to final document verification
3. EMI payments due on 5th of every month
4. Prepayment allowed after 6 months with 2% charges
5. Late payment penalty: 2% per month on overdue amount
6. Insurance coverage optional but recommended
7. Loan agreement must be signed within 15 days

DISBURSEMENT PROCESS:
1. Submit signed loan agreement
2. Provide post-dated cheques/NACH mandate
3. Complete final KYC verification
4. Amount credited to registered bank account
5. First EMI starts 30 days after disbursement

CONTACT INFORMATION:
Customer Care: 1800-XXX-XXXX
Email: support@tatacapital.com
Website: www.tatacapital.com

For queries regarding this sanction letter, quote Sanction ID: {sanction_id}

APPROVED BY: AI Underwriting System
AUTHORIZED SIGNATORY: [Digital Signature]

{'='*70}
              ** System-Generated Document **
{'='*70}

🎉 Congratulations on your loan approval!
We look forward to serving your financial needs.

IMPORTANT: Please read all terms and conditions carefully before accepting.
"""
        return letter

    def save_sanction_letter(self, application_id: str, letter: str) -> str:
        """
        Save sanction letter

        Args:
            application_id: Application ID
            letter: Letter content

        Returns:
            Filename
        """
        import os
        from pathlib import Path

        # Ensure storage directory exists
        storage_dir = Path("./data/storage/sanction_letters")
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"sanction_{application_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = storage_dir / filename

        # Save letter
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(letter)

        return filename
