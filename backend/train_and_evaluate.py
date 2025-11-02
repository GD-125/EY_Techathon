"""
Enhanced Training Script with Professional Evaluation and Visualizations
Trains models, generates comprehensive reports, and selects the best model
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.services.ml.model_trainer import ModelTrainer
from src.services.ml.model_evaluator import ModelEvaluator
from src.services.ml.document_analyzer import DocumentAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_and_evaluate_model(
    trainer: ModelTrainer,
    evaluator: ModelEvaluator,
    df: pd.DataFrame,
    model_name: str,
    model_type: str = 'random_forest'
):
    """Train a model and generate comprehensive evaluation"""

    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*80}")

    # Prepare features
    X, y = trainer.prepare_features(df, is_training=True)

    # Remove invalid targets
    valid_mask = y.notna() & (y != -1)
    X = X[valid_mask]
    y = y[valid_mask]

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    X_train_scaled = trainer.scaler.fit_transform(X_train)
    X_test_scaled = trainer.scaler.transform(X_test)

    # Initialize and train model
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    elif model_type == 'gradient_boosting':
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=1
        )
    elif model_type == 'sgd':
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(
            loss='log_loss',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info("Training model...")
    model.fit(X_train_scaled, y_train)

    # Make predictions
    logger.info("Generating predictions...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Get probabilities
    if hasattr(model, 'predict_proba'):
        y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba_test = model.decision_function(X_test_scaled)

    # Evaluate model
    logger.info("Evaluating model and generating visualizations...")

    feature_importances = None
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_

    evaluation_report = evaluator.evaluate_model(
        y_true=y_test.values,
        y_pred=y_pred_test,
        y_pred_proba=y_pred_proba_test,
        model_name=model_name,
        feature_names=trainer.feature_names,
        feature_importances=feature_importances
    )

    # Store model in trainer
    trainer.base_model = model

    # Add training accuracy to report
    from sklearn.metrics import accuracy_score
    evaluation_report['metrics']['train_accuracy'] = float(
        accuracy_score(y_train, y_pred_train)
    )

    return evaluation_report, model


def main():
    """Main training and evaluation pipeline"""

    logger.info("=" * 80)
    logger.info("Enhanced Model Training & Evaluation Pipeline")
    logger.info("=" * 80)

    # Initialize
    trainer = ModelTrainer(model_dir="models")
    evaluator = ModelEvaluator(output_dir="evaluation_reports")

    # Dataset configurations
    base_path = Path(__file__).parent.parent

    datasets = [
        {
            'name': 'Home Credit',
            'path': str(base_path / 'dataset' / 'home_credit_default_risk' / 'application_train.csv'),
            'type': 'home_credit',
            'sample_size': 50000
        },
        {
            'name': 'GiveMeSomeCredit',
            'path': str(base_path / 'dataset' / 'GiveMeSomeCredit' / 'cs-training.csv'),
            'type': 'givemesomecredit',
            'sample_size': 30000
        }
    ]

    print("\n" + "="*80)
    print("TRAINING OPTIONS")
    print("="*80)
    print("\nDatasets:")
    for i, ds in enumerate(datasets, 1):
        print(f"  {i}. {ds['name']}")

    print("\nModels to Train:")
    print("  1. Random Forest only")
    print("  2. Gradient Boosting only")
    print("  3. Both models (comparison)")

    # Get user choices
    dataset_choice = input(f"\nSelect dataset (1-{len(datasets)}): ").strip()
    model_choice = input("Select model option (1-3): ").strip()

    try:
        dataset_idx = int(dataset_choice) - 1
        model_option = int(model_choice)

        if not (0 <= dataset_idx < len(datasets)):
            logger.error("Invalid dataset choice")
            return

        if not (1 <= model_option <= 3):
            logger.error("Invalid model choice")
            return

        # Load dataset
        ds = datasets[dataset_idx]
        logger.info(f"\n{'='*80}")
        logger.info(f"Loading dataset: {ds['name']}")
        logger.info(f"{'='*80}")

        df = trainer.load_dataset(
            dataset_path=ds['path'],
            dataset_type=ds['type'],
            sample_size=ds['sample_size']
        )

        # Train models
        reports = []

        if model_option == 1 or model_option == 3:
            # Train Random Forest
            rf_report, rf_model = train_and_evaluate_model(
                trainer, evaluator, df,
                model_name=f"RandomForest_{ds['name']}",
                model_type='random_forest'
            )
            reports.append(rf_report)

            # Save Random Forest model
            logger.info("\nSaving Random Forest model...")
            trainer.base_model = rf_model
            trainer.save_model(f"random_forest_{ds['name'].lower().replace(' ', '_')}")

        if model_option == 2 or model_option == 3:
            # Train Gradient Boosting
            gb_report, gb_model = train_and_evaluate_model(
                trainer, evaluator, df,
                model_name=f"GradientBoosting_{ds['name']}",
                model_type='gradient_boosting'
            )
            reports.append(gb_report)

            # Save Gradient Boosting model
            logger.info("\nSaving Gradient Boosting model...")
            trainer.base_model = gb_model
            trainer.save_model(f"gradient_boosting_{ds['name'].lower().replace(' ', '_')}")

        # Display results
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)

        for report in reports:
            metrics = report['metrics']
            print(f"\n{report['model_name']}:")
            print(f"  {'Metric':<25} {'Score':<10}")
            print(f"  {'-'*35}")
            print(f"  {'Training Accuracy':<25} {metrics['train_accuracy']:>8.4f}")
            print(f"  {'Test Accuracy':<25} {metrics['accuracy']:>8.4f}")
            print(f"  {'Precision':<25} {metrics['precision']:>8.4f}")
            print(f"  {'Recall':<25} {metrics['recall']:>8.4f}")
            print(f"  {'F1 Score':<25} {metrics['f1_score']:>8.4f}")
            print(f"  {'ROC-AUC':<25} {metrics['roc_auc']:>8.4f}")
            print(f"\n  Visualizations saved:")
            for plot_name, plot_path in report['plots'].items():
                print(f"    • {plot_name}: {Path(plot_path).name}")

        # Compare models if multiple
        if len(reports) > 1:
            logger.info("\n" + "="*80)
            logger.info("Model Comparison")
            logger.info("="*80)

            comparison = evaluator.compare_models(reports)

            print(f"\n{'='*80}")
            print("BEST MODEL SELECTION")
            print(f"{'='*80}")
            print(f"\nBest Model: {comparison['best_model']['model_name']}")
            print(f"Composite Score: {comparison['summary']['best_composite_score']:.4f}")
            print(f"\nRanking:")
            for i, rank in enumerate(comparison['ranking'], 1):
                print(f"  {i}. {rank['model']:<30} Score: {rank['composite_score']:.4f}")

            if comparison['comparison_plot']:
                print(f"\nComparison plots saved: {Path(comparison['comparison_plot']).name}")

            # Save best model for dashboard
            best_model_path = Path("models") / "best_model_info.json"
            import json
            with open(best_model_path, 'w') as f:
                json.dump({
                    'best_model_id': comparison['best_model']['report_id'],
                    'model_name': comparison['best_model']['model_name'],
                    'composite_score': comparison['summary']['best_composite_score'],
                    'metrics': comparison['best_model']['metrics'],
                    'plots': comparison['best_model']['plots'],
                    'timestamp': comparison['best_model']['timestamp']
                }, f, indent=2)

            logger.info(f"✓ Best model info saved: {best_model_path}")

        # Test sample prediction
        logger.info("\n" + "="*80)
        logger.info("Testing Sample Prediction")
        logger.info("="*80)

        test_sample = {
            'AMT_INCOME_TOTAL': 180000,
            'AMT_CREDIT': 450000,
            'AMT_ANNUITY': 25000,
            'DAYS_BIRTH': -15000,
            'DAYS_EMPLOYED': -3000,
            'EXT_SOURCE_2': 0.65,
            'EXT_SOURCE_3': 0.55
        }

        analyzer = DocumentAnalyzer()
        analyzer.trainer = trainer

        result = analyzer.analyze_document(test_sample)

        print(f"\nTest Sample Analysis:")
        print(f"  Prediction:         {result['prediction']}")
        print(f"  Risk Probability:   {result['risk_probability']:.4f}")
        print(f"  Risk Level:         {result['risk_level']}")
        print(f"  Confidence:         {result['confidence']:.4f}")

        if 'explanation' in result:
            print(f"\nExplanation: {result['explanation']['summary']}")

        # Summary
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"\n✓ Models trained: {len(reports)}")
        print(f"✓ Evaluation reports: evaluation_reports/reports/")
        print(f"✓ Visualizations: evaluation_reports/plots/")
        if len(reports) > 1:
            print(f"✓ Model comparison: evaluation_reports/comparisons/")
        print(f"✓ Saved models: models/")
        print("\nNext steps:")
        print("  1. Review evaluation plots in 'evaluation_reports/plots/'")
        print("  2. Check best model selection in 'models/best_model_info.json'")
        print("  3. Use the best model in your dashboard")
        print("  4. Run 'python test_model.py' for additional testing")

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)


if __name__ == "__main__":
    main()
