"""
ML Model Training Service with Incremental Learning
Handles large datasets efficiently with batch processing and model persistence
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import joblib
import json
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Advanced ML training pipeline with incremental learning support
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self.base_model = None
        self.incremental_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

        # Training metadata
        self.feature_names = []
        self.training_history = []
        self.model_version = "1.0.0"

        # Configuration
        self.chunk_size = 10000  # Process data in chunks for memory efficiency
        self.target_column = 'TARGET'

    def load_dataset(
        self,
        dataset_path: str,
        dataset_type: str = 'home_credit',
        sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load dataset with efficient chunked reading for large files

        Args:
            dataset_path: Path to dataset file
            dataset_type: Type of dataset (home_credit, givemesomecredit, lending_club)
            sample_size: Optional sample size for faster testing

        Returns:
            Processed DataFrame
        """
        logger.info(f"Loading dataset from {dataset_path}")

        try:
            file_path = Path(dataset_path)

            if not file_path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")

            # Load with chunking for large files
            if file_path.suffix == '.csv':
                if sample_size:
                    df = pd.read_csv(dataset_path, nrows=sample_size)
                else:
                    # Load in chunks and concatenate
                    chunks = []
                    for chunk in pd.read_csv(dataset_path, chunksize=self.chunk_size):
                        chunks.append(chunk)
                    df = pd.concat(chunks, ignore_index=True)
            elif file_path.suffix in ['.xls', '.xlsx']:
                df = pd.read_excel(dataset_path)
                if sample_size:
                    df = df.sample(n=min(sample_size, len(df)))
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

            # Preprocess based on dataset type
            df = self._preprocess_dataset(df, dataset_type)

            return df

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def _preprocess_dataset(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Preprocess dataset based on type"""

        if dataset_type == 'home_credit':
            # Home Credit dataset preprocessing
            if 'TARGET' not in df.columns:
                # For test data without target
                df['TARGET'] = -1

            # Handle specific column mappings
            self.target_column = 'TARGET'

        elif dataset_type == 'givemesomecredit':
            # GiveMeSomeCredit dataset
            if 'SeriousDlqin2yrs' in df.columns:
                df['TARGET'] = df['SeriousDlqin2yrs']
                self.target_column = 'TARGET'

        elif dataset_type == 'lending_club':
            # Lending Club dataset
            if 'loan_status' in df.columns:
                # Map loan status to binary target
                df['TARGET'] = df['loan_status'].apply(
                    lambda x: 1 if x in ['Charged Off', 'Default'] else 0
                )
                self.target_column = 'TARGET'

        return df

    def prepare_features(
        self,
        df: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training/prediction with efficient processing

        Args:
            df: Input DataFrame
            is_training: Whether this is for training (True) or prediction (False)

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df_prep = df.copy()

        # Separate features and target
        if self.target_column in df_prep.columns:
            y = df_prep[self.target_column]
            X = df_prep.drop(columns=[self.target_column])
        else:
            y = None
            X = df_prep

        # Identify column types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove ID columns and high-cardinality columns
        id_patterns = ['id', 'ID', 'key', 'KEY', 'SK_']
        cols_to_drop = [col for col in X.columns if any(pattern in col for pattern in id_patterns)]

        # Also drop high cardinality categorical columns (>100 unique values)
        for col in categorical_cols[:]:
            if X[col].nunique() > 100:
                cols_to_drop.append(col)
                categorical_cols.remove(col)

        X = X.drop(columns=cols_to_drop, errors='ignore')

        # Handle numeric columns - fill missing with median
        for col in numeric_cols:
            if col in X.columns:
                if is_training:
                    median_val = X[col].median()
                    X[col].fillna(median_val, inplace=True)
                else:
                    X[col].fillna(0, inplace=True)

                # Handle infinite values
                X[col].replace([np.inf, -np.inf], 0, inplace=True)

        # Handle categorical columns - encode
        for col in categorical_cols:
            if col in X.columns:
                if is_training:
                    # Fit label encoder
                    le = LabelEncoder()
                    X[col] = X[col].astype(str).fillna('missing')
                    X[col] = le.fit_transform(X[col])
                    self.label_encoders[col] = le
                else:
                    # Transform using existing encoder
                    if col in self.label_encoders:
                        X[col] = X[col].astype(str).fillna('missing')
                        # Handle unseen labels
                        le = self.label_encoders[col]
                        X[col] = X[col].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        X[col] = 0

        # Store feature names
        if is_training:
            self.feature_names = X.columns.tolist()
        else:
            # Ensure same features as training
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]

        # Convert to numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        return X, y

    def train_model(
        self,
        df: pd.DataFrame,
        model_type: str = 'random_forest',
        test_size: float = 0.2,
        incremental: bool = False
    ) -> Dict[str, Any]:
        """
        Train ML model with the dataset

        Args:
            df: Training DataFrame
            model_type: Type of model (random_forest, gradient_boosting, sgd)
            test_size: Test set proportion
            incremental: Whether to use incremental learning

        Returns:
            Training results dictionary
        """
        logger.info(f"Training {model_type} model on {len(df)} records")

        try:
            # Prepare features
            X, y = self.prepare_features(df, is_training=True)

            # Remove records with missing target
            valid_mask = y.notna() & (y != -1)
            X = X[valid_mask]
            y = y[valid_mask]

            logger.info(f"Training with {len(X)} samples and {len(X.columns)} features")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Initialize model
            if incremental:
                # Use SGD for incremental learning
                model = SGDClassifier(
                    loss='log_loss',
                    max_iter=1000,
                    random_state=42,
                    n_jobs=-1
                )
                self.incremental_model = model
            else:
                if model_type == 'random_forest':
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=15,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=-1,
                        verbose=1
                    )
                elif model_type == 'gradient_boosting':
                    model = GradientBoostingClassifier(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=42,
                        verbose=1
                    )
                elif model_type == 'sgd':
                    model = SGDClassifier(
                        loss='log_loss',
                        max_iter=1000,
                        random_state=42,
                        n_jobs=-1
                    )
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                self.base_model = model

            # Train model
            logger.info("Training model...")
            model.fit(X_train_scaled, y_train)

            # Evaluate
            logger.info("Evaluating model...")
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            # Get probabilities for AUC
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = model.decision_function(X_test_scaled)

            # Calculate metrics
            metrics = {
                'train_accuracy': float(accuracy_score(y_train, y_pred_train)),
                'test_accuracy': float(accuracy_score(y_test, y_pred_test)),
                'precision': float(precision_score(y_test, y_pred_test, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred_test, zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred_test, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
                'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
            }

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = dict(zip(self.feature_names, importances))
                # Get top 20 features
                top_features = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]
                metrics['top_features'] = top_features

            # Store training history
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'model_type': model_type,
                'n_samples': len(X),
                'n_features': len(X.columns),
                'metrics': metrics,
                'incremental': incremental
            }
            self.training_history.append(training_record)

            logger.info(f"Training complete. Test Accuracy: {metrics['test_accuracy']:.4f}, "
                       f"AUC-ROC: {metrics['roc_auc']:.4f}")

            return {
                'success': True,
                'metrics': metrics,
                'n_samples': len(X),
                'n_features': len(X.columns),
                'model_type': model_type
            }

        except Exception as e:
            logger.error(f"Error training model: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def incremental_train(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Incrementally train on new data patterns

        Args:
            new_data: New DataFrame with recent data

        Returns:
            Training results
        """
        logger.info(f"Incremental training on {len(new_data)} new records")

        try:
            # Check if incremental model exists
            if self.incremental_model is None:
                logger.info("No incremental model found. Training new SGD model...")
                return self.train_model(new_data, model_type='sgd', incremental=True)

            # Prepare features
            X_new, y_new = self.prepare_features(new_data, is_training=False)

            # Remove invalid targets
            valid_mask = y_new.notna() & (y_new != -1)
            X_new = X_new[valid_mask]
            y_new = y_new[valid_mask]

            # Scale features
            X_new_scaled = self.scaler.transform(X_new)

            # Partial fit
            self.incremental_model.partial_fit(X_new_scaled, y_new)

            # Evaluate on new data
            y_pred = self.incremental_model.predict(X_new_scaled)

            metrics = {
                'accuracy': float(accuracy_score(y_new, y_pred)),
                'precision': float(precision_score(y_new, y_pred, zero_division=0)),
                'recall': float(recall_score(y_new, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y_new, y_pred, zero_division=0))
            }

            logger.info(f"Incremental training complete. Accuracy: {metrics['accuracy']:.4f}")

            return {
                'success': True,
                'metrics': metrics,
                'n_new_samples': len(X_new)
            }

        except Exception as e:
            logger.error(f"Error in incremental training: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def predict(self, data: pd.DataFrame, use_incremental: bool = False) -> np.ndarray:
        """
        Make predictions on new data

        Args:
            data: DataFrame with features
            use_incremental: Use incremental model if available

        Returns:
            Array of predictions
        """
        try:
            # Choose model
            model = self.incremental_model if use_incremental and self.incremental_model else self.base_model

            if model is None:
                raise ValueError("No trained model available")

            # Prepare features
            X, _ = self.prepare_features(data, is_training=False)

            # Scale
            X_scaled = self.scaler.transform(X)

            # Predict
            predictions = model.predict(X_scaled)

            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)
            else:
                probabilities = None

            return predictions, probabilities

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def save_model(self, model_name: str = 'credit_risk_model') -> bool:
        """
        Save trained model and metadata

        Args:
            model_name: Name for the saved model

        Returns:
            True if successful
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = self.model_dir / f"{model_name}_{timestamp}"
            model_path.mkdir(parents=True, exist_ok=True)

            # Save models
            if self.base_model:
                joblib.dump(self.base_model, model_path / 'base_model.pkl')

            if self.incremental_model:
                joblib.dump(self.incremental_model, model_path / 'incremental_model.pkl')

            # Save scaler
            joblib.dump(self.scaler, model_path / 'scaler.pkl')

            # Save label encoders
            joblib.dump(self.label_encoders, model_path / 'label_encoders.pkl')

            # Save metadata
            metadata = {
                'model_version': self.model_version,
                'feature_names': self.feature_names,
                'training_history': self.training_history,
                'saved_at': timestamp,
                'target_column': self.target_column
            }

            with open(model_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model saved to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, model_path: str) -> bool:
        """
        Load trained model and metadata

        Args:
            model_path: Path to saved model directory

        Returns:
            True if successful
        """
        try:
            model_path = Path(model_path)

            # Load models
            if (model_path / 'base_model.pkl').exists():
                self.base_model = joblib.load(model_path / 'base_model.pkl')

            if (model_path / 'incremental_model.pkl').exists():
                self.incremental_model = joblib.load(model_path / 'incremental_model.pkl')

            # Load scaler
            self.scaler = joblib.load(model_path / 'scaler.pkl')

            # Load label encoders
            self.label_encoders = joblib.load(model_path / 'label_encoders.pkl')

            # Load metadata
            with open(model_path / 'metadata.json', 'r') as f:
                metadata = json.load(f)

            self.model_version = metadata['model_version']
            self.feature_names = metadata['feature_names']
            self.training_history = metadata['training_history']
            self.target_column = metadata.get('target_column', 'TARGET')

            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Model version: {self.model_version}, Features: {len(self.feature_names)}")

            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
