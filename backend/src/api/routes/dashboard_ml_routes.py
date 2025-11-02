"""
Dashboard ML API Routes with File Upload and Visualization Support
Professional endpoints for dashboard integration
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import pandas as pd
import logging
from pathlib import Path
import tempfile
import os
import json

from ...services.ml.model_trainer import ModelTrainer
from ...services.ml.document_analyzer import DocumentAnalyzer
from ...services.ml.model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard/ml", tags=["Dashboard ML"])

# Global instances
model_trainer = ModelTrainer(model_dir="models")
document_analyzer = DocumentAnalyzer()
model_evaluator = ModelEvaluator(output_dir="evaluation_reports")

# Load best model on startup
def load_best_model():
    """Load the best model if available"""
    try:
        best_model_info_path = Path("models/best_model_info.json")
        if best_model_info_path.exists():
            with open(best_model_info_path, 'r') as f:
                info = json.load(f)

            # Find the model directory
            model_dir = Path("models")
            model_paths = list(model_dir.glob("*"))

            # Try to load the most recent model
            if model_paths:
                latest_model = sorted(model_paths, key=lambda x: x.stat().st_mtime)[-1]
                if latest_model.is_dir():
                    model_trainer.load_model(str(latest_model))
                    document_analyzer.trainer = model_trainer
                    logger.info(f"Loaded model: {latest_model.name}")
    except Exception as e:
        logger.warning(f"Could not load best model: {e}")


# Load on startup
load_best_model()


class FileAnalysisRequest(BaseModel):
    """Request for file analysis with options"""
    generate_report: bool = True
    include_visualizations: bool = True


@router.get("/best-model")
async def get_best_model_info():
    """
    Get information about the currently loaded best model
    """
    try:
        best_model_info_path = Path("models/best_model_info.json")

        if not best_model_info_path.exists():
            return JSONResponse(
                status_code=200,
                content={
                    'status': 'no_model',
                    'message': 'No best model selected yet. Train models first.',
                    'has_model': False
                }
            )

        with open(best_model_info_path, 'r') as f:
            info = json.load(f)

        # Add current model status
        info['has_model'] = (
            model_trainer.base_model is not None or
            model_trainer.incremental_model is not None
        )

        return JSONResponse(status_code=200, content=info)

    except Exception as e:
        logger.error(f"Error getting best model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-file")
async def analyze_uploaded_file(file: UploadFile = File(...)):
    """
    Analyze loan applications from uploaded file
    Returns comprehensive analysis with visualizations

    Accepts: CSV, Excel files
    Returns: Analysis results with statistics and risk distribution
    """
    try:
        logger.info(f"Analyzing uploaded file: {file.filename}")

        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Upload CSV or Excel file."
            )

        # Check if model is loaded
        if model_trainer.base_model is None and model_trainer.incremental_model is None:
            raise HTTPException(
                status_code=400,
                detail="No model loaded. Train a model first."
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Load data
            logger.info("Loading data from file...")
            if tmp_path.endswith('.csv'):
                df = pd.read_csv(tmp_path)
            else:
                df = pd.read_excel(tmp_path)

            logger.info(f"Loaded {len(df)} records")

            # Analyze
            documents = df.to_dict('records')
            results = document_analyzer.analyze_batch(documents, batch_size=1000)

            # Generate comprehensive summary
            summary = document_analyzer._generate_summary(results)

            # Calculate additional insights
            risk_levels = [r['risk_level'] for r in results]
            risk_probabilities = [r['risk_probability'] for r in results]

            insights = {
                'total_analyzed': len(results),
                'approval_summary': {
                    'total_approved': summary['approved'],
                    'total_rejected': summary['rejected'],
                    'approval_rate': summary['approval_rate'],
                    'rejection_rate': 1 - summary['approval_rate']
                },
                'risk_analysis': {
                    'average_risk': summary['average_risk_probability'],
                    'median_risk': float(np.median(risk_probabilities)),
                    'std_risk': float(np.std(risk_probabilities)),
                    'min_risk': float(np.min(risk_probabilities)),
                    'max_risk': float(np.max(risk_probabilities)),
                    'risk_distribution': summary['risk_distribution']
                },
                'confidence_metrics': {
                    'average_confidence': summary['average_confidence'],
                    'high_confidence_count': sum(1 for r in results if r['confidence'] > 0.8),
                    'low_confidence_count': sum(1 for r in results if r['confidence'] < 0.5)
                },
                'recommendations': {
                    'high_risk_count': risk_levels.count('high') + risk_levels.count('very_high'),
                    'manual_review_recommended': sum(
                        1 for r in results
                        if r['risk_level'] in ['high', 'very_high'] or r['confidence'] < 0.6
                    ),
                    'fast_track_eligible': sum(
                        1 for r in results
                        if r['risk_level'] in ['low', 'very_low'] and r['confidence'] > 0.8
                    )
                }
            }

            # Sample results (first 100 for response)
            sample_results = results[:100]

            return JSONResponse(
                status_code=200,
                content={
                    'status': 'success',
                    'file_name': file.filename,
                    'insights': insights,
                    'sample_results': sample_results,
                    'full_results_count': len(results),
                    'message': f'Successfully analyzed {len(results)} loan applications'
                }
            )

        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-file-detailed")
async def analyze_file_with_export(
    file: UploadFile = File(...),
    export_format: str = "csv"
):
    """
    Analyze file and export detailed results
    Returns download link for full results
    """
    try:
        logger.info(f"Detailed analysis of: {file.filename}")

        # Validate
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if model_trainer.base_model is None and model_trainer.incremental_model is None:
            raise HTTPException(status_code=400, detail="No model loaded")

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Analyze
            result = document_analyzer.analyze_from_file(tmp_path)

            # Create export file
            export_dir = Path("exports")
            export_dir.mkdir(exist_ok=True)

            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            export_filename = f"analysis_results_{timestamp}.{export_format}"
            export_path = export_dir / export_filename

            # Export results
            document_analyzer._save_results(result['results'], str(export_path))

            return JSONResponse(
                status_code=200,
                content={
                    'status': 'success',
                    'summary': result['summary'],
                    'total_documents': result['total_documents'],
                    'export_file': export_filename,
                    'download_url': f'/api/dashboard/ml/download/{export_filename}'
                }
            )

        finally:
            os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detailed analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{filename}")
async def download_export(filename: str):
    """Download exported analysis results"""
    try:
        file_path = Path("exports") / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluation-plots")
async def get_evaluation_plots():
    """
    Get list of available evaluation plots for the best model
    """
    try:
        best_model_info_path = Path("models/best_model_info.json")

        if not best_model_info_path.exists():
            return JSONResponse(
                status_code=200,
                content={
                    'status': 'no_plots',
                    'plots': [],
                    'message': 'No evaluation plots available yet'
                }
            )

        with open(best_model_info_path, 'r') as f:
            info = json.load(f)

        plots = info.get('plots', {})

        # Check which plots exist
        available_plots = {}
        for plot_name, plot_path in plots.items():
            if Path(plot_path).exists():
                available_plots[plot_name] = {
                    'path': plot_path,
                    'filename': Path(plot_path).name,
                    'url': f'/api/dashboard/ml/plot/{Path(plot_path).name}'
                }

        return JSONResponse(
            status_code=200,
            content={
                'status': 'success',
                'plots': available_plots,
                'model_name': info.get('model_name', 'Unknown'),
                'timestamp': info.get('timestamp', '')
            }
        )

    except Exception as e:
        logger.error(f"Error getting evaluation plots: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plot/{filename}")
async def get_plot_image(filename: str):
    """
    Serve evaluation plot images
    """
    try:
        # Look in plots directory
        plot_path = Path("evaluation_reports/plots") / filename

        if not plot_path.exists():
            raise HTTPException(status_code=404, detail="Plot not found")

        return FileResponse(
            path=str(plot_path),
            media_type='image/png'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving plot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-metrics")
async def get_model_metrics():
    """
    Get detailed metrics for dashboard display
    """
    try:
        best_model_info_path = Path("models/best_model_info.json")

        if not best_model_info_path.exists():
            return JSONResponse(
                status_code=200,
                content={
                    'status': 'no_metrics',
                    'message': 'No model metrics available'
                }
            )

        with open(best_model_info_path, 'r') as f:
            info = json.load(f)

        metrics = info.get('metrics', {})

        # Format for dashboard
        dashboard_metrics = {
            'primary_metrics': {
                'accuracy': {
                    'value': metrics.get('accuracy', 0),
                    'label': 'Accuracy',
                    'format': 'percentage'
                },
                'precision': {
                    'value': metrics.get('precision', 0),
                    'label': 'Precision',
                    'format': 'percentage'
                },
                'recall': {
                    'value': metrics.get('recall', 0),
                    'label': 'Recall',
                    'format': 'percentage'
                },
                'f1_score': {
                    'value': metrics.get('f1_score', 0),
                    'label': 'F1 Score',
                    'format': 'percentage'
                }
            },
            'advanced_metrics': {
                'roc_auc': {
                    'value': metrics.get('roc_auc', 0),
                    'label': 'ROC-AUC',
                    'format': 'decimal'
                },
                'average_precision': {
                    'value': metrics.get('average_precision', 0),
                    'label': 'Avg Precision',
                    'format': 'decimal'
                }
            },
            'confusion_matrix': {
                'true_positives': metrics.get('true_positives', 0),
                'true_negatives': metrics.get('true_negatives', 0),
                'false_positives': metrics.get('false_positives', 0),
                'false_negatives': metrics.get('false_negatives', 0)
            },
            'model_info': {
                'name': info.get('model_name', 'Unknown'),
                'timestamp': info.get('timestamp', ''),
                'composite_score': info.get('composite_score', 0)
            }
        }

        return JSONResponse(status_code=200, content=dashboard_metrics)

    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_system_statistics():
    """
    Get overall system statistics for dashboard
    """
    try:
        stats = {
            'models': {
                'total_models': len(list(Path("models").glob("*"))) if Path("models").exists() else 0,
                'has_active_model': (
                    model_trainer.base_model is not None or
                    model_trainer.incremental_model is not None
                )
            },
            'evaluations': {
                'total_reports': len(list(Path("evaluation_reports/reports").glob("*.json")))
                    if Path("evaluation_reports/reports").exists() else 0,
                'total_plots': len(list(Path("evaluation_reports/plots").glob("*.png")))
                    if Path("evaluation_reports/plots").exists() else 0
            },
            'exports': {
                'total_exports': len(list(Path("exports").glob("*")))
                    if Path("exports").exists() else 0
            }
        }

        # Get training history if available
        if model_trainer.training_history:
            latest_training = model_trainer.training_history[-1]
            stats['latest_training'] = {
                'timestamp': latest_training.get('timestamp', ''),
                'n_samples': latest_training.get('n_samples', 0),
                'metrics': latest_training.get('metrics', {})
            }

        return JSONResponse(status_code=200, content=stats)

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for dashboard"""
    return JSONResponse(
        status_code=200,
        content={
            'status': 'healthy',
            'service': 'Dashboard ML Service',
            'model_loaded': (
                model_trainer.base_model is not None or
                model_trainer.incremental_model is not None
            )
        }
    )
