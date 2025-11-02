"""
Document Analysis Service with ML-based Risk Prediction
Analyzes new loan documents and predicts risk using trained models
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

from .model_trainer import ModelTrainer

logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    """
    Analyzes loan application documents using trained ML models
    Provides risk assessment and explainability
    """

    def __init__(self, model_path: Optional[str] = None):
        self.trainer = ModelTrainer()

        # Load model if path provided
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """Load trained model"""
        try:
            return self.trainer.load_model(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def analyze_document(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single loan application document

        Args:
            document_data: Dictionary containing applicant information

        Returns:
            Analysis results with risk prediction and explainability
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([document_data])

            # Get predictions
            predictions, probabilities = self.trainer.predict(df)

            # Extract result
            prediction = predictions[0]
            risk_probability = probabilities[0][1] if probabilities is not None else 0.5

            # Determine risk level
            risk_level = self._determine_risk_level(risk_probability)

            # Generate explanation
            explanation = self._generate_explanation(document_data, risk_probability)

            # Calculate confidence
            confidence = self._calculate_confidence(probabilities[0] if probabilities is not None else [0.5, 0.5])

            result = {
                'prediction': 'APPROVED' if prediction == 0 else 'REJECTED',
                'risk_probability': float(risk_probability),
                'risk_level': risk_level,
                'confidence': float(confidence),
                'explanation': explanation,
                'timestamp': pd.Timestamp.now().isoformat()
            }

            logger.info(f"Document analyzed: {result['prediction']} (risk: {risk_probability:.3f})")

            return result

        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return {
                'prediction': 'ERROR',
                'error': str(e),
                'risk_probability': 0.5,
                'risk_level': 'unknown',
                'confidence': 0.0
            }

    def analyze_batch(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple documents in batches for efficiency

        Args:
            documents: List of document dictionaries
            batch_size: Number of documents to process at once

        Returns:
            List of analysis results
        """
        results = []

        try:
            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} documents)")

                # Convert to DataFrame
                df = pd.DataFrame(batch)

                # Get predictions
                predictions, probabilities = self.trainer.predict(df)

                # Process results
                for j, (pred, prob) in enumerate(zip(predictions, probabilities if probabilities is not None else [[0.5, 0.5]] * len(predictions))):
                    doc_data = batch[j]
                    risk_prob = prob[1] if len(prob) > 1 else 0.5

                    result = {
                        'document_id': doc_data.get('application_id', f'DOC_{i+j}'),
                        'prediction': 'APPROVED' if pred == 0 else 'REJECTED',
                        'risk_probability': float(risk_prob),
                        'risk_level': self._determine_risk_level(risk_prob),
                        'confidence': float(self._calculate_confidence(prob)),
                        'explanation': self._generate_explanation(doc_data, risk_prob)
                    }

                    results.append(result)

            logger.info(f"Batch analysis complete: {len(results)} documents processed")

            return results

        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            raise

    def analyze_from_file(
        self,
        file_path: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze loan applications from a file

        Args:
            file_path: Path to input file (CSV/Excel)
            output_path: Optional path to save results

        Returns:
            Analysis summary and results
        """
        try:
            logger.info(f"Analyzing documents from {file_path}")

            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            logger.info(f"Loaded {len(df)} documents")

            # Convert to list of dicts
            documents = df.to_dict('records')

            # Analyze
            results = self.analyze_batch(documents)

            # Calculate summary statistics
            summary = self._generate_summary(results)

            # Save results if output path provided
            if output_path:
                self._save_results(results, output_path)

            return {
                'summary': summary,
                'results': results,
                'total_documents': len(results)
            }

        except Exception as e:
            logger.error(f"Error analyzing file: {e}")
            raise

    def _determine_risk_level(self, risk_probability: float) -> str:
        """Determine risk level from probability"""
        if risk_probability < 0.2:
            return 'very_low'
        elif risk_probability < 0.4:
            return 'low'
        elif risk_probability < 0.6:
            return 'medium'
        elif risk_probability < 0.8:
            return 'high'
        else:
            return 'very_high'

    def _calculate_confidence(self, probabilities: np.ndarray) -> float:
        """Calculate prediction confidence"""
        # Confidence based on probability spread
        max_prob = np.max(probabilities)
        # Higher spread = higher confidence
        confidence = (max_prob - 0.5) * 2  # Scale to 0-1
        return max(0.0, min(1.0, confidence))

    def _generate_explanation(
        self,
        document_data: Dict[str, Any],
        risk_probability: float
    ) -> Dict[str, Any]:
        """
        Generate human-readable explanation for the prediction

        Args:
            document_data: Original document data
            risk_probability: Predicted risk probability

        Returns:
            Explanation dictionary
        """
        explanation = {
            'summary': '',
            'key_factors': [],
            'recommendations': []
        }

        # Generate summary
        if risk_probability < 0.3:
            explanation['summary'] = 'Low risk applicant with strong creditworthiness indicators.'
        elif risk_probability < 0.6:
            explanation['summary'] = 'Moderate risk applicant. Additional verification recommended.'
        else:
            explanation['summary'] = 'High risk applicant. Multiple risk factors identified.'

        # Identify key factors
        key_factors = []

        # Income analysis
        income = document_data.get('AMT_INCOME_TOTAL', document_data.get('annual_income', 0))
        credit_amount = document_data.get('AMT_CREDIT', document_data.get('loan_amount', 0))

        if income > 0 and credit_amount > 0:
            loan_to_income = credit_amount / income
            if loan_to_income > 3:
                key_factors.append({
                    'factor': 'Loan-to-Income Ratio',
                    'value': f'{loan_to_income:.2f}',
                    'impact': 'negative',
                    'description': 'Loan amount is high relative to income'
                })
            elif loan_to_income < 1:
                key_factors.append({
                    'factor': 'Loan-to-Income Ratio',
                    'value': f'{loan_to_income:.2f}',
                    'impact': 'positive',
                    'description': 'Loan amount is reasonable for income level'
                })

        # Age analysis
        days_birth = document_data.get('DAYS_BIRTH', 0)
        if days_birth != 0:
            age_years = abs(days_birth) / 365
            if age_years > 25 and age_years < 65:
                key_factors.append({
                    'factor': 'Age',
                    'value': f'{age_years:.0f} years',
                    'impact': 'positive',
                    'description': 'Age within stable employment range'
                })

        # Employment analysis
        days_employed = document_data.get('DAYS_EMPLOYED', 0)
        if days_employed < 0:  # Negative means currently employed
            employment_years = abs(days_employed) / 365
            if employment_years > 2:
                key_factors.append({
                    'factor': 'Employment Stability',
                    'value': f'{employment_years:.1f} years',
                    'impact': 'positive',
                    'description': 'Long-term employment history'
                })
            else:
                key_factors.append({
                    'factor': 'Employment Stability',
                    'value': f'{employment_years:.1f} years',
                    'impact': 'neutral',
                    'description': 'Recent employment history'
                })

        # External sources (credit bureau scores)
        ext_sources = [
            document_data.get('EXT_SOURCE_1'),
            document_data.get('EXT_SOURCE_2'),
            document_data.get('EXT_SOURCE_3')
        ]
        ext_sources = [x for x in ext_sources if x is not None and not pd.isna(x)]

        if ext_sources:
            avg_ext_source = np.mean(ext_sources)
            if avg_ext_source > 0.5:
                key_factors.append({
                    'factor': 'External Credit Score',
                    'value': f'{avg_ext_source:.3f}',
                    'impact': 'positive',
                    'description': 'Good external credit bureau scores'
                })
            elif avg_ext_source < 0.3:
                key_factors.append({
                    'factor': 'External Credit Score',
                    'value': f'{avg_ext_source:.3f}',
                    'impact': 'negative',
                    'description': 'Low external credit bureau scores'
                })

        explanation['key_factors'] = key_factors

        # Generate recommendations
        recommendations = []

        if risk_probability > 0.6:
            recommendations.append('Request additional documentation for verification')
            recommendations.append('Consider requiring a co-signer or guarantor')
            recommendations.append('Evaluate collateral requirements')
        elif risk_probability > 0.4:
            recommendations.append('Perform enhanced due diligence')
            recommendations.append('Verify employment and income documentation')
        else:
            recommendations.append('Standard approval process recommended')
            recommendations.append('Consider for expedited processing')

        explanation['recommendations'] = recommendations

        return explanation

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from results"""
        if not results:
            return {}

        total = len(results)
        approved = sum(1 for r in results if r['prediction'] == 'APPROVED')
        rejected = total - approved

        risk_levels = [r['risk_level'] for r in results]
        avg_risk = np.mean([r['risk_probability'] for r in results])
        avg_confidence = np.mean([r['confidence'] for r in results])

        return {
            'total_documents': total,
            'approved': approved,
            'rejected': rejected,
            'approval_rate': approved / total if total > 0 else 0,
            'average_risk_probability': float(avg_risk),
            'average_confidence': float(avg_confidence),
            'risk_distribution': {
                level: risk_levels.count(level)
                for level in ['very_low', 'low', 'medium', 'high', 'very_high']
            }
        }

    def _save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save analysis results to file"""
        try:
            # Convert to DataFrame
            df_results = pd.DataFrame(results)

            # Expand explanation into separate columns
            if 'explanation' in df_results.columns:
                df_results['explanation_summary'] = df_results['explanation'].apply(
                    lambda x: x.get('summary', '') if isinstance(x, dict) else ''
                )
                df_results['recommendations'] = df_results['explanation'].apply(
                    lambda x: ', '.join(x.get('recommendations', [])) if isinstance(x, dict) else ''
                )
                df_results.drop('explanation', axis=1, inplace=True)

            # Save
            if output_path.endswith('.csv'):
                df_results.to_csv(output_path, index=False)
            elif output_path.endswith(('.xlsx', '.xls')):
                df_results.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Unsupported output format: {output_path}")

            logger.info(f"Results saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
