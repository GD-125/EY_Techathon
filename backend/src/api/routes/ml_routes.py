"""
API Routes for ML Model Training and Prediction
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import pandas as pd
import logging
from pathlib import Path
import tempfile
import os

from ...services.ml.model_trainer import ModelTrainer
from ...services.ml.document_analyzer import DocumentAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])

# Global instances
model_trainer = ModelTrainer(model_dir="models")
document_analyzer = DocumentAnalyzer()


class TrainingRequest(BaseModel):
    """Request model for training"""
    dataset_path: str
    dataset_type: str = 'home_credit'
    model_type: str = 'random_forest'
    sample_size: Optional[int] = None
    test_size: float = 0.2


class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    data: Dict[str, Any]


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    documents: List[Dict[str, Any]]


class IncrementalTrainingRequest(BaseModel):
    """Request model for incremental training"""
    new_data: List[Dict[str, Any]]


@router.post("/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train ML model on specified dataset

    Example:
    ```json
    {
        "dataset_path": "dataset/home_credit_default_risk/application_train.csv",
        "dataset_type": "home_credit",
        "model_type": "random_forest",
        "sample_size": 10000
    }
    ```
    """
    try:
        logger.info(f"Starting training with dataset: {request.dataset_path}")

        # Load dataset
        df = model_trainer.load_dataset(
            dataset_path=request.dataset_path,
            dataset_type=request.dataset_type,
            sample_size=request.sample_size
        )

        # Train model
        result = model_trainer.train_model(
            df=df,
            model_type=request.model_type,
            test_size=request.test_size
        )

        if result['success']:
            # Save model in background
            background_tasks.add_task(model_trainer.save_model)

            return JSONResponse(
                status_code=200,
                content={
                    'status': 'success',
                    'message': 'Model trained successfully',
                    'metrics': result['metrics'],
                    'n_samples': result['n_samples'],
                    'n_features': result['n_features']
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Training failed'))

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {str(e)}")
    except Exception as e:
        logger.error(f"Error in training endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/incremental")
async def incremental_train(request: IncrementalTrainingRequest):
    """
    Incrementally train model on new data

    Example:
    ```json
    {
        "new_data": [
            {
                "AMT_INCOME_TOTAL": 180000,
                "AMT_CREDIT": 450000,
                "TARGET": 0
            }
        ]
    }
    ```
    """
    try:
        logger.info(f"Incremental training with {len(request.new_data)} records")

        # Convert to DataFrame
        df_new = pd.DataFrame(request.new_data)

        # Incremental train
        result = model_trainer.incremental_train(df_new)

        if result['success']:
            return JSONResponse(
                status_code=200,
                content={
                    'status': 'success',
                    'message': 'Incremental training completed',
                    'metrics': result['metrics'],
                    'n_new_samples': result['n_new_samples']
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Training failed'))

    except Exception as e:
        logger.error(f"Error in incremental training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict_single(request: PredictionRequest):
    """
    Analyze a single loan application document

    Example:
    ```json
    {
        "data": {
            "AMT_INCOME_TOTAL": 180000,
            "AMT_CREDIT": 450000,
            "DAYS_BIRTH": -15000,
            "DAYS_EMPLOYED": -3000,
            "EXT_SOURCE_2": 0.65
        }
    }
    ```
    """
    try:
        # Analyze document
        result = document_analyzer.analyze_document(request.data)

        if result.get('prediction') == 'ERROR':
            raise HTTPException(status_code=500, detail=result.get('error', 'Analysis failed'))

        return JSONResponse(
            status_code=200,
            content={
                'status': 'success',
                'analysis': result
            }
        )

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Analyze multiple loan applications in batch

    Example:
    ```json
    {
        "documents": [
            {
                "application_id": "APP001",
                "AMT_INCOME_TOTAL": 180000,
                "AMT_CREDIT": 450000
            },
            {
                "application_id": "APP002",
                "AMT_INCOME_TOTAL": 120000,
                "AMT_CREDIT": 300000
            }
        ]
    }
    ```
    """
    try:
        logger.info(f"Batch prediction for {len(request.documents)} documents")

        # Analyze batch
        results = document_analyzer.analyze_batch(request.documents)

        # Generate summary
        summary = document_analyzer._generate_summary(results)

        return JSONResponse(
            status_code=200,
            content={
                'status': 'success',
                'summary': summary,
                'results': results,
                'total_processed': len(results)
            }
        )

    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/file")
async def predict_from_file(file: UploadFile = File(...)):
    """
    Analyze loan applications from uploaded CSV/Excel file

    Upload a file with loan application data and get predictions for all records
    """
    try:
        logger.info(f"Processing uploaded file: {file.filename}")

        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload CSV or Excel file"
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Analyze file
            result = document_analyzer.analyze_from_file(tmp_path)

            return JSONResponse(
                status_code=200,
                content={
                    'status': 'success',
                    'summary': result['summary'],
                    'total_documents': result['total_documents'],
                    'results': result['results'][:100]  # Return first 100 results
                }
            )

        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model/save")
async def save_model(model_name: str = "credit_risk_model"):
    """
    Save the currently trained model

    Args:
        model_name: Name for the saved model
    """
    try:
        success = model_trainer.save_model(model_name)

        if success:
            return JSONResponse(
                status_code=200,
                content={
                    'status': 'success',
                    'message': f'Model saved as {model_name}'
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save model")

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model/load")
async def load_model(model_path: str):
    """
    Load a saved model

    Args:
        model_path: Path to the saved model directory
    """
    try:
        success = model_trainer.load_model(model_path)

        if success:
            # Update analyzer with loaded model
            document_analyzer.trainer = model_trainer

            return JSONResponse(
                status_code=200,
                content={
                    'status': 'success',
                    'message': f'Model loaded from {model_path}',
                    'model_version': model_trainer.model_version,
                    'n_features': len(model_trainer.feature_names)
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info")
async def get_model_info():
    """
    Get information about the currently loaded model
    """
    try:
        if model_trainer.base_model is None and model_trainer.incremental_model is None:
            return JSONResponse(
                status_code=200,
                content={
                    'status': 'no_model',
                    'message': 'No model currently loaded'
                }
            )

        info = {
            'status': 'loaded',
            'model_version': model_trainer.model_version,
            'n_features': len(model_trainer.feature_names),
            'feature_names': model_trainer.feature_names[:20],  # First 20 features
            'has_base_model': model_trainer.base_model is not None,
            'has_incremental_model': model_trainer.incremental_model is not None,
            'training_history': model_trainer.training_history[-5:] if model_trainer.training_history else []
        }

        return JSONResponse(status_code=200, content=info)

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            'status': 'healthy',
            'service': 'ML Service',
            'model_loaded': model_trainer.base_model is not None or model_trainer.incremental_model is not None
        }
    )
