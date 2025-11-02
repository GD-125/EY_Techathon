"""
Complete ML Model Training Pipeline with Visualizations
Trains multiple models: XGBoost, LightGBM, Random Forest, Neural Network
Generates comprehensive visualizations and evaluation metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import neural network libraries
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("TensorFlow not available. Neural Network training will be skipped.")


class LoanModelTrainer:
    """
    Complete ML training pipeline for loan approval prediction
    """

    def __init__(self, data_path, output_dir='./ml_models'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)

        self.models = {}
        self.results = {}

    def load_and_preprocess_data(self):
        """Load and preprocess the loan data"""
        print("\n" + "="*80)
        print("📊 LOADING AND PREPROCESSING DATA")
        print("="*80)

        # Load data
        print(f"\n Loading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} records with {len(self.df.columns)} columns")

        # Display basic info
        print(f"\nDataset shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nMissing values:\n{self.df.isnull().sum()[self.df.isnull().sum() > 0]}")

        # Handle missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.df[col].fillna(self.df[col].median(), inplace=True)

        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'unknown', inplace=True)

        print("✓ Missing values handled")

        # Encode categorical variables
        self.label_encoders = {}
        encode_cols = ['loan_purpose', 'education_level', 'marital_status', 'home_ownership']

        for col in encode_cols:
            if col in self.df.columns:
                self.label_encoders[col] = LabelEncoder()
                self.df[col] = self.label_encoders[col].fit_transform(self.df[col].astype(str))

        print(f"✓ Encoded {len(self.label_encoders)} categorical columns")

        # Create derived features
        if 'annual_income' in self.df.columns and 'loan_amount' in self.df.columns:
            self.df['loan_to_income_ratio'] = self.df['loan_amount'] / (self.df['annual_income'] + 1)

        if 'existing_debt' in self.df.columns and 'annual_income' in self.df.columns:
            self.df['debt_to_income_ratio'] = self.df['existing_debt'] / ((self.df['annual_income'] / 12) + 1)

        print("✓ Created derived features")

        # Prepare features and target
        target_col = 'loan_status'
        if target_col in self.df.columns:
            self.y = (self.df[target_col] == 'approved').astype(int)
        else:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        feature_cols = [col for col in self.df.columns
                       if col not in ['application_id', 'name', target_col]
                       and self.df[col].dtype in [np.int64, np.float64]]

        self.X = self.df[feature_cols]
        self.feature_names = feature_cols

        print(f"\n✓ Features selected: {len(self.feature_names)}")
        print(f"  Features: {self.feature_names}")
        print(f"\n✓ Target distribution:")
        print(f"  Approved: {self.y.sum()} ({self.y.mean()*100:.1f}%)")
        print(f"  Rejected: {len(self.y) - self.y.sum()} ({(1-self.y.mean())*100:.1f}%)")

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        print(f"\n✓ Data split:")
        print(f"  Training set: {len(self.X_train)} samples")
        print(f"  Test set: {len(self.X_test)} samples")

        # Scale features for neural network
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("✓ Features scaled for neural network")

    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "="*80)
        print("🌲 TRAINING RANDOM FOREST")
        print("="*80)

        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)

        print("\nPerforming grid search...")
        grid_search.fit(self.X_train, self.y_train)

        self.models['random_forest'] = grid_search.best_estimator_
        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV F1-score: {grid_search.best_score_:.4f}")

        # Evaluate
        self._evaluate_model('random_forest', self.models['random_forest'])

    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n" + "="*80)
        print("🚀 TRAINING XGBOOST")
        print("="*80)

        # Hyperparameter tuning
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)

        print("\nPerforming grid search...")
        grid_search.fit(self.X_train, self.y_train)

        self.models['xgboost'] = grid_search.best_estimator_
        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV F1-score: {grid_search.best_score_:.4f}")

        # Evaluate
        self._evaluate_model('xgboost', self.models['xgboost'])

    def train_lightgbm(self):
        """Train LightGBM model"""
        print("\n" + "="*80)
        print("⚡ TRAINING LIGHTGBM")
        print("="*80)

        # Hyperparameter tuning
        param_grid = {
            'num_leaves': [31, 50],
            'max_depth': [5, 10],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'min_child_samples': [20, 30]
        }

        lgb_model = lgb.LGBMClassifier(random_state=42)
        grid_search = GridSearchCV(lgb_model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)

        print("\nPerforming grid search...")
        grid_search.fit(self.X_train, self.y_train)

        self.models['lightgbm'] = grid_search.best_estimator_
        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV F1-score: {grid_search.best_score_:.4f}")

        # Evaluate
        self._evaluate_model('lightgbm', self.models['lightgbm'])

    def train_neural_network(self):
        """Train Neural Network model"""
        if not KERAS_AVAILABLE:
            print("\n⚠️  Skipping Neural Network (TensorFlow not available)")
            return

        print("\n" + "="*80)
        print("🧠 TRAINING NEURAL NETWORK")
        print("="*80)

        # Build model
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )

        print("\nModel architecture:")
        model.summary()

        # Train
        history = model.fit(
            self.X_train_scaled, self.y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        self.models['neural_network'] = model
        self.nn_history = history

        # Evaluate
        self._evaluate_model('neural_network', model, use_scaled=True)

    def _evaluate_model(self, model_name, model, use_scaled=False):
        """Evaluate a trained model"""
        print(f"\n📊 Evaluating {model_name}...")

        X_test = self.X_test_scaled if use_scaled else self.X_test

        # Predictions
        if model_name == 'neural_network':
            y_pred_proba = model.predict(X_test).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }

        self.results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }

        print(f"\n✓ {model_name.upper()} Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    def visualize_results(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*80)
        print("📈 GENERATING VISUALIZATIONS")
        print("="*80)

        # 1. Model Comparison
        self._plot_model_comparison()

        # 2. Confusion Matrices
        self._plot_confusion_matrices()

        # 3. ROC Curves
        self._plot_roc_curves()

        # 4. Feature Importance (for tree models)
        self._plot_feature_importance()

        # 5. Neural Network Training History
        if 'neural_network' in self.models and KERAS_AVAILABLE:
            self._plot_nn_history()

        print("\n✓ All visualizations saved to:", f"{self.output_dir}/plots/")

    def _plot_model_comparison(self):
        """Plot comparison of all models"""
        fig, ax = plt.subplots(figsize=(12, 6))

        metrics_df = pd.DataFrame({
            model: results['metrics']
            for model, results in self.results.items()
        }).T

        metrics_df.plot(kind='bar', ax=ax)
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(f"{self.output_dir}/plots/model_comparison_{self.timestamp}.png", dpi=300)
        print("✓ Model comparison plot saved")
        plt.close()

    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name.upper()}\nConfusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/confusion_matrices_{self.timestamp}.png", dpi=300)
        print("✓ Confusion matrices plot saved")
        plt.close()

    def _plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))

        for model_name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            auc = results['metrics']['roc_auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(f"{self.output_dir}/plots/roc_curves_{self.timestamp}.png", dpi=300)
        print("✓ ROC curves plot saved")
        plt.close()

    def _plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        tree_models = {k: v for k, v in self.models.items()
                      if k in ['random_forest', 'xgboost', 'lightgbm']}

        if not tree_models:
            return

        fig, axes = plt.subplots(1, len(tree_models), figsize=(8*len(tree_models), 6))

        if len(tree_models) == 1:
            axes = [axes]

        for idx, (model_name, model) in enumerate(tree_models.items()):
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:15]  # Top 15 features

                axes[idx].barh(range(len(indices)), importances[indices])
                axes[idx].set_yticks(range(len(indices)))
                axes[idx].set_yticklabels([self.feature_names[i] for i in indices])
                axes[idx].set_xlabel('Importance')
                axes[idx].set_title(f'{model_name.upper()}\nTop 15 Features')
                axes[idx].invert_yaxis()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/feature_importance_{self.timestamp}.png", dpi=300)
        print("✓ Feature importance plot saved")
        plt.close()

    def _plot_nn_history(self):
        """Plot neural network training history"""
        history = self.nn_history.history

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(history['loss'], label='Training Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Accuracy
        axes[0, 1].plot(history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Precision
        axes[1, 0].plot(history['precision'], label='Training Precision')
        axes[1, 0].plot(history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Recall
        axes[1, 1].plot(history['recall'], label='Training Recall')
        axes[1, 1].plot(history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/nn_training_history_{self.timestamp}.png", dpi=300)
        print("✓ Neural network training history plot saved")
        plt.close()

    def save_models(self):
        """Save all trained models"""
        print("\n" + "="*80)
        print("💾 SAVING MODELS")
        print("="*80)

        for model_name, model in self.models.items():
            if model_name == 'neural_network' and KERAS_AVAILABLE:
                model.save(f"{self.output_dir}/models/{model_name}_{self.timestamp}.h5")
            else:
                joblib.dump(model, f"{self.output_dir}/models/{model_name}_{self.timestamp}.pkl")
            print(f"✓ Saved {model_name}")

        # Save preprocessors
        joblib.dump(self.scaler, f"{self.output_dir}/models/scaler_{self.timestamp}.pkl")
        joblib.dump(self.label_encoders, f"{self.output_dir}/models/label_encoders_{self.timestamp}.pkl")
        print("✓ Saved preprocessors")

        # Save results summary
        summary = pd.DataFrame({
            model: results['metrics']
            for model, results in self.results.items()
        }).T
        summary.to_csv(f"{self.output_dir}/results_summary_{self.timestamp}.csv")
        print("✓ Saved results summary")

    def generate_report(self):
        """Generate comprehensive training report"""
        print("\n" + "="*80)
        print("📄 GENERATING TRAINING REPORT")
        print("="*80)

        report_path = f"{self.output_dir}/training_report_{self.timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("LOAN APPROVAL MODEL TRAINING REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Total Samples: {len(self.df)}\n")
            f.write(f"Training Samples: {len(self.X_train)}\n")
            f.write(f"Test Samples: {len(self.X_test)}\n\n")

            f.write("="*80 + "\n")
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("="*80 + "\n\n")

            for model_name, results in self.results.items():
                f.write(f"\n{model_name.upper()}:\n")
                f.write("-" * 40 + "\n")
                for metric, value in results['metrics'].items():
                    f.write(f"  {metric:15s}: {value:.4f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("BEST MODEL\n")
            f.write("="*80 + "\n\n")

            best_model = max(self.results.items(),
                           key=lambda x: x[1]['metrics']['f1_score'])
            f.write(f"Model: {best_model[0]}\n")
            f.write(f"F1-Score: {best_model[1]['metrics']['f1_score']:.4f}\n")

        print(f"✓ Training report saved to: {report_path}")

    def run_complete_training(self):
        """Run complete training pipeline"""
        print("\n" + "="*80)
        print("🚀 STARTING COMPLETE ML TRAINING PIPELINE")
        print("="*80)

        start_time = datetime.now()

        # Load data
        self.load_and_preprocess_data()

        # Train all models
        self.train_random_forest()
        self.train_xgboost()
        self.train_lightgbm()
        self.train_neural_network()

        # Visualize
        self.visualize_results()

        # Save
        self.save_models()
        self.generate_report()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"\nTotal training time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"\nAll outputs saved to: {self.output_dir}/")
        print(f"  - Models: {self.output_dir}/models/")
        print(f"  - Plots: {self.output_dir}/plots/")
        print(f"  - Report: training_report_{self.timestamp}.txt")

        # Print best model
        best_model = max(self.results.items(),
                        key=lambda x: x[1]['metrics']['f1_score'])
        print(f"\n🏆 Best Model: {best_model[0].upper()}")
        print(f"   F1-Score: {best_model[1]['metrics']['f1_score']:.4f}")
        print(f"   Accuracy: {best_model[1]['metrics']['accuracy']:.4f}")


def main():
    """Main execution"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python train_model.py <data_file.csv>")
        print("\nExample:")
        print("  python train_model.py ../data/mock/sample_loan_data.csv")
        sys.exit(1)

    data_path = sys.argv[1]

    if not os.path.exists(data_path):
        print(f"Error: File not found: {data_path}")
        sys.exit(1)

    # Create trainer
    trainer = LoanModelTrainer(data_path)

    # Run complete training
    trainer.run_complete_training()


if __name__ == "__main__":
    main()
