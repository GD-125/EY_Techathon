"""
Training Script for Credit Risk ML Model
Trains on multiple datasets and saves the model
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


def main():
    """Main training function"""

    logger.info("=" * 80)
    logger.info("Credit Risk Model Training Pipeline")
    logger.info("=" * 80)

    # Initialize trainer
    trainer = ModelTrainer(model_dir="models")

    # Dataset configurations (paths relative to project root)
    base_path = Path(__file__).parent.parent  # Go up to project root

    datasets = [
        {
            'name': 'Home Credit - Application Data',
            'path': str(base_path / 'dataset' / 'home_credit_default_risk' / 'application_train.csv'),
            'type': 'home_credit',
            'sample_size': 50000  # Use sample for faster training
        },
        {
            'name': 'GiveMeSomeCredit',
            'path': str(base_path / 'dataset' / 'GiveMeSomeCredit' / 'cs-training.csv'),
            'type': 'givemesomecredit',
            'sample_size': 30000
        }
    ]

    # Choose which dataset to train on
    print("\nAvailable datasets:")
    for i, ds in enumerate(datasets, 1):
        print(f"{i}. {ds['name']} ({ds['path']})")
    print(f"{len(datasets) + 1}. Train on all datasets (combined)")

    choice = input(f"\nSelect dataset (1-{len(datasets) + 1}): ").strip()

    try:
        choice_idx = int(choice) - 1

        if choice_idx == len(datasets):
            # Train on all datasets
            logger.info("\n" + "=" * 80)
            logger.info("Training on ALL datasets (combined)")
            logger.info("=" * 80)

            dfs_to_combine = []

            for ds in datasets:
                logger.info(f"\nLoading {ds['name']}...")
                df = trainer.load_dataset(
                    dataset_path=ds['path'],
                    dataset_type=ds['type'],
                    sample_size=ds['sample_size']
                )
                dfs_to_combine.append(df)

            # Combine datasets (align columns)
            logger.info("\nCombining datasets...")
            combined_df = pd.concat(dfs_to_combine, ignore_index=True)

            logger.info(f"\nTotal combined records: {len(combined_df)}")

            # Train model
            logger.info("\n" + "-" * 80)
            logger.info("Training Random Forest Model")
            logger.info("-" * 80)

            result = trainer.train_model(
                df=combined_df,
                model_type='random_forest',
                test_size=0.2
            )

        elif 0 <= choice_idx < len(datasets):
            # Train on selected dataset
            ds = datasets[choice_idx]

            logger.info("\n" + "=" * 80)
            logger.info(f"Training on {ds['name']}")
            logger.info("=" * 80)

            # Load dataset
            logger.info(f"\nLoading dataset from {ds['path']}...")
            df = trainer.load_dataset(
                dataset_path=ds['path'],
                dataset_type=ds['type'],
                sample_size=ds['sample_size']
            )

            # Train model
            logger.info("\n" + "-" * 80)
            logger.info("Training Random Forest Model")
            logger.info("-" * 80)

            result = trainer.train_model(
                df=df,
                model_type='random_forest',
                test_size=0.2
            )

        else:
            logger.error("Invalid choice")
            return

        # Display results
        if result['success']:
            logger.info("\n" + "=" * 80)
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)

            metrics = result['metrics']
            logger.info(f"\nModel Performance Metrics:")
            logger.info(f"  Training Accuracy:   {metrics['train_accuracy']:.4f}")
            logger.info(f"  Test Accuracy:       {metrics['test_accuracy']:.4f}")
            logger.info(f"  Precision:           {metrics['precision']:.4f}")
            logger.info(f"  Recall:              {metrics['recall']:.4f}")
            logger.info(f"  F1 Score:            {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC:             {metrics['roc_auc']:.4f}")

            if 'top_features' in metrics:
                logger.info(f"\nTop 10 Important Features:")
                for i, (feature, importance) in enumerate(metrics['top_features'][:10], 1):
                    logger.info(f"  {i:2d}. {feature:40s} {importance:.6f}")

            # Save model
            logger.info("\n" + "-" * 80)
            logger.info("Saving Model...")
            logger.info("-" * 80)

            success = trainer.save_model('credit_risk_model')

            if success:
                logger.info("✓ Model saved successfully")
            else:
                logger.error("✗ Failed to save model")

            # Test prediction
            logger.info("\n" + "-" * 80)
            logger.info("Testing Prediction on Sample Data")
            logger.info("-" * 80)

            # Create test sample
            test_sample = {
                'AMT_INCOME_TOTAL': 180000,
                'AMT_CREDIT': 450000,
                'AMT_ANNUITY': 25000,
                'DAYS_BIRTH': -15000,  # ~41 years old
                'DAYS_EMPLOYED': -3000,  # ~8 years employed
                'EXT_SOURCE_2': 0.65,
                'EXT_SOURCE_3': 0.55
            }

            analyzer = DocumentAnalyzer()
            analyzer.trainer = trainer

            result = analyzer.analyze_document(test_sample)

            logger.info(f"\nTest Sample Analysis:")
            logger.info(f"  Prediction:         {result['prediction']}")
            logger.info(f"  Risk Probability:   {result['risk_probability']:.4f}")
            logger.info(f"  Risk Level:         {result['risk_level']}")
            logger.info(f"  Confidence:         {result['confidence']:.4f}")

            if 'explanation' in result:
                logger.info(f"\nExplanation:")
                logger.info(f"  {result['explanation']['summary']}")

        else:
            logger.error("\n✗ Training failed")
            logger.error(f"Error: {result.get('error', 'Unknown error')}")

    except ValueError:
        logger.error("Invalid input. Please enter a number.")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)

    logger.info("\n" + "=" * 80)
    logger.info("Training Pipeline Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
