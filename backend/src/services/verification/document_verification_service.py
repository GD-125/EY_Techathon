"""
Document Verification Service using OCR and AI
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class DocumentVerificationService:
    """
    Service for verifying and extracting information from documents
    """

    def __init__(self):
        self.supported_documents = [
            'identity_proof',
            'address_proof',
            'income_proof',
            'bank_statement',
            'employment_letter',
            'tax_return'
        ]

        self.verification_history = []

    async def verify_document(
        self,
        document_type: str,
        document_data: bytes,
        applicant_info: Dict[str, Any]
    ) -> Dict:
        """
        Verify a document and extract relevant information

        Args:
            document_type: Type of document being verified
            document_data: Binary document data
            applicant_info: Information about the applicant for cross-verification

        Returns:
            Verification result with extracted information
        """
        try:
            if document_type not in self.supported_documents:
                return {
                    'success': False,
                    'error': f'Unsupported document type: {document_type}'
                }

            # Simulate OCR and document processing
            extracted_data = self._extract_document_data(document_type, document_data)

            # Cross-verify with applicant information
            verification_result = self._cross_verify_data(
                extracted_data,
                applicant_info,
                document_type
            )

            # Calculate confidence score
            confidence_score = self._calculate_confidence(verification_result)

            # Generate explainability
            explainability = self._generate_explainability(
                verification_result,
                extracted_data,
                applicant_info
            )

            result = {
                'success': True,
                'document_type': document_type,
                'verification_status': verification_result['status'],
                'confidence': confidence_score,
                'extracted_data': extracted_data,
                'verification_details': verification_result,
                'explainability': explainability,
                'timestamp': datetime.now().isoformat()
            }

            # Save to history
            self.verification_history.append(result)

            return result

        except Exception as e:
            logger.error(f"Error verifying document: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _extract_document_data(self, document_type: str, document_data: bytes) -> Dict:
        """
        Extract data from document using OCR
        In production, integrate with services like AWS Textract, Google Vision, etc.
        """

        # Mock extracted data based on document type
        mock_data = {
            'identity_proof': {
                'name': 'John Doe',
                'date_of_birth': '1990-05-15',
                'id_number': 'ID123456789',
                'address': '123 Main St, City, State 12345',
                'issue_date': '2020-01-01',
                'expiry_date': '2030-01-01'
            },
            'address_proof': {
                'name': 'John Doe',
                'address': '123 Main St, City, State 12345',
                'document_date': '2024-10-01'
            },
            'income_proof': {
                'name': 'John Doe',
                'employer': 'ABC Corporation',
                'annual_income': 75000,
                'document_date': '2024-09-30',
                'employment_type': 'Full-time'
            },
            'bank_statement': {
                'account_holder': 'John Doe',
                'account_number': '****1234',
                'average_balance': 5000,
                'statement_period': '2024-09-01 to 2024-09-30',
                'transactions_count': 45
            },
            'employment_letter': {
                'employee_name': 'John Doe',
                'employer': 'ABC Corporation',
                'position': 'Software Engineer',
                'employment_start_date': '2020-01-15',
                'annual_salary': 75000,
                'letter_date': '2024-10-01'
            },
            'tax_return': {
                'taxpayer_name': 'John Doe',
                'tax_year': 2023,
                'total_income': 75000,
                'tax_paid': 12000,
                'filing_date': '2024-04-15'
            }
        }

        return mock_data.get(document_type, {})

    def _cross_verify_data(
        self,
        extracted_data: Dict,
        applicant_info: Dict,
        document_type: str
    ) -> Dict:
        """Cross-verify extracted data with applicant information"""

        verification_checks = []
        issues = []

        # Name verification (if applicable)
        if 'name' in extracted_data:
            applicant_name = applicant_info.get('name', '').lower()
            extracted_name = extracted_data['name'].lower()

            name_match = self._fuzzy_match(applicant_name, extracted_name)
            verification_checks.append({
                'field': 'name',
                'match': name_match,
                'expected': applicant_name,
                'extracted': extracted_name
            })

            if not name_match:
                issues.append('Name mismatch between document and application')

        # Address verification (if applicable)
        if 'address' in extracted_data and 'address' in applicant_info:
            address_match = self._fuzzy_match(
                applicant_info['address'].lower(),
                extracted_data['address'].lower()
            )
            verification_checks.append({
                'field': 'address',
                'match': address_match,
                'expected': applicant_info['address'],
                'extracted': extracted_data['address']
            })

            if not address_match:
                issues.append('Address mismatch between document and application')

        # Income verification (if applicable)
        if 'annual_income' in extracted_data and 'annual_income' in applicant_info:
            income_diff = abs(
                extracted_data['annual_income'] - applicant_info['annual_income']
            ) / applicant_info['annual_income']

            income_match = income_diff < 0.1  # Allow 10% variance

            verification_checks.append({
                'field': 'income',
                'match': income_match,
                'expected': applicant_info['annual_income'],
                'extracted': extracted_data['annual_income'],
                'variance': f"{income_diff * 100:.1f}%"
            })

            if not income_match:
                issues.append('Significant income discrepancy detected')

        # Document validity checks
        validity_checks = self._check_document_validity(extracted_data, document_type)
        verification_checks.extend(validity_checks['checks'])
        issues.extend(validity_checks['issues'])

        # Determine overall status
        total_checks = len(verification_checks)
        passed_checks = sum(1 for check in verification_checks if check['match'])

        if passed_checks == total_checks:
            status = 'verified'
        elif passed_checks >= total_checks * 0.8:
            status = 'partially_verified'
        else:
            status = 'failed'

        return {
            'status': status,
            'checks': verification_checks,
            'issues': issues,
            'passed': passed_checks,
            'total': total_checks
        }

    def _check_document_validity(self, extracted_data: Dict, document_type: str) -> Dict:
        """Check document validity (expiry, authenticity markers, etc.)"""

        checks = []
        issues = []

        # Check expiry date if applicable
        if 'expiry_date' in extracted_data:
            expiry_date = datetime.fromisoformat(extracted_data['expiry_date'])
            is_valid = expiry_date > datetime.now()

            checks.append({
                'field': 'expiry_date',
                'match': is_valid,
                'expected': 'Not expired',
                'extracted': extracted_data['expiry_date']
            })

            if not is_valid:
                issues.append('Document has expired')

        # Check document date recency
        date_fields = ['document_date', 'statement_period', 'letter_date', 'filing_date']
        for field in date_fields:
            if field in extracted_data:
                # Documents should generally be recent (within 6 months)
                checks.append({
                    'field': f'{field}_recency',
                    'match': True,  # Mock as valid
                    'expected': 'Recent',
                    'extracted': extracted_data[field]
                })

        return {
            'checks': checks,
            'issues': issues
        }

    def _fuzzy_match(self, str1: str, str2: str, threshold: float = 0.8) -> bool:
        """
        Fuzzy string matching for names and addresses
        In production, use libraries like fuzzywuzzy or rapidfuzz
        """
        # Simple fuzzy matching logic
        # Remove punctuation and extra spaces
        clean1 = re.sub(r'[^\w\s]', '', str1).strip()
        clean2 = re.sub(r'[^\w\s]', '', str2).strip()

        # Split into words
        words1 = set(clean1.split())
        words2 = set(clean2.split())

        if not words1 or not words2:
            return False

        # Calculate overlap
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        similarity = intersection / union if union > 0 else 0

        return similarity >= threshold

    def _calculate_confidence(self, verification_result: Dict) -> float:
        """Calculate confidence score for verification"""

        if verification_result['total'] == 0:
            return 0.5

        # Base confidence on pass rate
        pass_rate = verification_result['passed'] / verification_result['total']

        # Adjust based on severity of issues
        issue_penalty = len(verification_result['issues']) * 0.1

        confidence = max(0.0, min(1.0, pass_rate - issue_penalty))

        return round(confidence, 3)

    def _generate_explainability(
        self,
        verification_result: Dict,
        extracted_data: Dict,
        applicant_info: Dict
    ) -> Dict:
        """Generate explainability for verification result"""

        # Generate reasoning
        status = verification_result['status']
        passed = verification_result['passed']
        total = verification_result['total']

        if status == 'verified':
            reasoning = f"Document successfully verified. All {total} verification checks passed."
        elif status == 'partially_verified':
            reasoning = f"Document partially verified. {passed} out of {total} checks passed. Manual review recommended."
        else:
            reasoning = f"Document verification failed. Only {passed} out of {total} checks passed."

        # Key factors
        factors = []
        for check in verification_result['checks']:
            factors.append({
                'field': check['field'],
                'result': 'Passed' if check['match'] else 'Failed',
                'expected': str(check['expected']),
                'found': str(check['extracted'])
            })

        # Recommendations
        recommendations = []
        if verification_result['issues']:
            recommendations.append("Address the following issues:")
            recommendations.extend(verification_result['issues'])

        if status == 'failed':
            recommendations.append("Consider uploading a clearer or more recent document")
        elif status == 'partially_verified':
            recommendations.append("Manual verification by loan officer recommended")

        return {
            'reasoning': reasoning,
            'factors': factors,
            'issues': verification_result['issues'],
            'recommendations': recommendations
        }

    def get_verification_history(
        self,
        applicant_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get document verification history"""

        # Return most recent verifications
        return sorted(
            self.verification_history,
            key=lambda x: x['timestamp'],
            reverse=True
        )[:limit]

    def get_supported_documents(self) -> List[str]:
        """Get list of supported document types"""
        return self.supported_documents
