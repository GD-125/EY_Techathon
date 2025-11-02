"""
Test Script for Credit Risk ML Model
Tests model predictions and incremental learning
"""
import sys
import logging
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.services.ml.model_trainer import ModelTrainer
from src.services.ml.document_analyzer import DocumentAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_prediction():
    """Test single document prediction"""
    logger.info("\n" + "=" * 80)
    logger.info("Test 1: Single Document Prediction")
    logger.info("=" * 80)

    # Load model
    analyzer = DocumentAnalyzer()

    # Find latest model
    model_dir = Path("models")
    if not model_dir.exists():
        logger.error("No models directory found. Train a model first.")
        return

    model_paths = list(model_dir.glob("credit_risk_model_*"))
    if not model_paths:
        logger.error("No saved models found. Train a model first.")
        return

    latest_model = sorted(model_paths)[-1]
    logger.info(f"Loading model from: {latest_model}")

    if not analyzer.load_model(str(latest_model)):
        logger.error("Failed to load model")
        return

    # Test cases
    test_cases = [
        {
            'name': 'Low Risk Applicant',
            'data': {
                'AMT_INCOME_TOTAL': 200000,
                'AMT_CREDIT': 300000,
                'AMT_ANNUITY': 20000,
                'DAYS_BIRTH': -15000,
                'DAYS_EMPLOYED': -4000,
                'EXT_SOURCE_2': 0.75,
                'EXT_SOURCE_3': 0.70
            }
        },
        {
            'name': 'High Risk Applicant',
            'data': {
                'AMT_INCOME_TOTAL': 50000,
                'AMT_CREDIT': 500000,
                'AMT_ANNUITY': 40000,
                'DAYS_BIRTH': -7300,
                'DAYS_EMPLOYED': -365,
                'EXT_SOURCE_2': 0.25,
                'EXT_SOURCE_3': 0.30
            }
        },
        {
            'name': 'Medium Risk Applicant',
            'data': {
                'AMT_INCOME_TOTAL': 120000,
                'AMT_CREDIT': 350000,
                'AMT_ANNUITY': 25000,
                'DAYS_BIRTH': -12000,
                'DAYS_EMPLOYED': -2000,
                'EXT_SOURCE_2': 0.50,
                'EXT_SOURCE_3': 0.45
            }
        }
    ]

    for test_case in test_cases:
        logger.info(f"\n{'-' * 80}")
        logger.info(f"Test Case: {test_case['name']}")
        logger.info(f"{'-' * 80}")

        result = analyzer.analyze_document(test_case['data'])

        logger.info(f"Prediction:       {result['prediction']}")
        logger.info(f"Risk Probability: {result['risk_probability']:.4f}")
        logger.info(f"Risk Level:       {result['risk_level']}")
        logger.info(f"Confidence:       {result['confidence']:.4f}")

        if 'explanation' in result:
            logger.info(f"\nExplanation: {result['explanation']['summary']}")

            if result['explanation']['key_factors']:
                logger.info("\nKey Factors:")
                for factor in result['explanation']['key_factors'][:3]:
                    logger.info(f"  • {factor['factor']}: {factor['description']}")

            if result['explanation']['recommendations']:
                logger.info("\nRecommendations:")
                for rec in result['explanation']['recommendations'][:2]:
                    logger.info(f"  • {rec}")


def test_batch_prediction():
    """Test batch prediction"""
    logger.info("\n" + "=" * 80)
    logger.info("Test 2: Batch Document Prediction")
    logger.info("=" * 80)

    # Load model
    analyzer = DocumentAnalyzer()
    model_dir = Path("models")
    model_paths = list(model_dir.glob("credit_risk_model_*"))
    latest_model = sorted(model_paths)[-1]

    logger.info(f"Loading model from: {latest_model}")
    analyzer.load_model(str(latest_model))

    # Create batch of documents
    documents = [
        {
            'application_id': f'APP{i:04d}',
            'AMT_INCOME_TOTAL': 100000 + i * 10000,
            'AMT_CREDIT': 200000 + i * 20000,
            'DAYS_BIRTH': -10000 - i * 100,
            'DAYS_EMPLOYED': -2000 - i * 50,
            'EXT_SOURCE_2': 0.5 + (i % 5) * 0.05
        }
        for i in range(10)
    ]

    logger.info(f"Analyzing {len(documents)} documents...")

    results = analyzer.analyze_batch(documents)

    # Display summary
    approved = sum(1 for r in results if r['prediction'] == 'APPROVED')
    rejected = len(results) - approved

    logger.info(f"\nBatch Analysis Summary:")
    logger.info(f"  Total Documents:  {len(results)}")
    logger.info(f"  Approved:         {approved} ({approved/len(results)*100:.1f}%)")
    logger.info(f"  Rejected:         {rejected} ({rejected/len(results)*100:.1f}%)")

    avg_risk = sum(r['risk_probability'] for r in results) / len(results)
    logger.info(f"  Average Risk:     {avg_risk:.4f}")

    # Show sample results
    logger.info(f"\nSample Results:")
    for result in results[:5]:
        logger.info(f"  {result['document_id']}: {result['prediction']} "
                   f"(risk: {result['risk_probability']:.3f}, "
                   f"level: {result['risk_level']})")


def test_incremental_learning():
    """Test incremental learning"""
    logger.info("\n" + "=" * 80)
    logger.info("Test 3: Incremental Learning")
    logger.info("=" * 80)

    # Load model
    trainer = ModelTrainer()
    model_dir = Path("models")
    model_paths = list(model_dir.glob("credit_risk_model_*"))
    latest_model = sorted(model_paths)[-1]

    logger.info(f"Loading model from: {latest_model}")
    trainer.load_model(str(latest_model))

    # Create new training data
    new_data = pd.DataFrame([
        {
            'AMT_INCOME_TOTAL': 150000,
            'AMT_CREDIT': 400000,
            'TARGET': 0
        },
        {
            'AMT_INCOME_TOTAL': 80000,
            'AMT_CREDIT': 500000,
            'TARGET': 1
        }
    ] * 50)  # Replicate to have more samples

    logger.info(f"Training incrementally on {len(new_data)} new records...")

    result = trainer.incremental_train(new_data)

    if result['success']:
        logger.info("\nIncremental Training Results:")
        metrics = result.get('metrics', {})

        # Check which metrics are available
        if 'accuracy' in metrics:
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        if 'precision' in metrics:
            logger.info(f"  Precision: {metrics['precision']:.4f}")
        if 'recall' in metrics:
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
        if 'f1_score' in metrics:
            logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")

        if metrics:
            logger.info("\n✓ Incremental learning completed successfully")
        else:
            logger.warning("Metrics not available, but training completed")
    else:
        logger.error(f"✗ Incremental training failed: {result.get('error')}")


def test_file_analysis():
    """Test file-based analysis"""
    logger.info("\n" + "=" * 80)
    logger.info("Test 4: File-Based Analysis")
    logger.info("=" * 80)

    # Check for test dataset (path relative to project root)
    base_path = Path(__file__).parent.parent
    test_file = base_path / "dataset" / "home_credit_default_risk" / "application_test.csv"

    if not test_file.exists():
        logger.warning(f"Test file not found: {test_file}")
        logger.info("Skipping file analysis test")
        return

    # Load model
    analyzer = DocumentAnalyzer()
    model_dir = Path("models")
    model_paths = list(model_dir.glob("credit_risk_model_*"))
    latest_model = sorted(model_paths)[-1]

    logger.info(f"Loading model from: {latest_model}")
    analyzer.load_model(str(latest_model))

    # Load sample from test file
    logger.info(f"Loading sample from {test_file}")
    df_test = pd.read_csv(test_file, nrows=100)  # Load first 100 records

    logger.info(f"Analyzing {len(df_test)} records from file...")

    # Convert to list of dicts
    documents = df_test.to_dict('records')

    # Analyze
    results = analyzer.analyze_batch(documents)

    # Generate summary
    summary = analyzer._generate_summary(results)

    logger.info("\nFile Analysis Summary:")
    logger.info(f"  Total Documents:       {summary['total_documents']}")
    logger.info(f"  Approved:              {summary['approved']}")
    logger.info(f"  Rejected:              {summary['rejected']}")
    logger.info(f"  Approval Rate:         {summary['approval_rate']:.2%}")
    logger.info(f"  Average Risk:          {summary['average_risk_probability']:.4f}")
    logger.info(f"  Average Confidence:    {summary['average_confidence']:.4f}")

    logger.info("\nRisk Distribution:")
    for level, count in summary['risk_distribution'].items():
        if count > 0:
            logger.info(f"  {level:12s}: {count:3d} ({count/summary['total_documents']*100:.1f}%)")


def main():
    """Main test function"""
    logger.info("=" * 80)
    logger.info("Credit Risk Model Testing Suite")
    logger.info("=" * 80)

    try:
        # Run tests
        test_single_prediction()
        test_batch_prediction()
        test_incremental_learning()
        test_file_analysis()

        logger.info("\n" + "=" * 80)
        logger.info("All Tests Completed")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)


if __name__ == "__main__":
    main()
