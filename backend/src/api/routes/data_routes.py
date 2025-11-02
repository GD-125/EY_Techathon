"""
API Routes for Data Upload and Analysis
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import tempfile
import os
from pathlib import Path

# Import services
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from services.credit.credit_scoring_service import CreditScoringService
from utils.data_processor import DataProcessor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data", tags=["data"])

# Initialize services
credit_service = CreditScoringService()
data_processor = DataProcessor()

# Store uploaded files temporarily
uploaded_files = {}


class AnalyzeRequest(BaseModel):
    """Request model for data analysis"""
    file_id: str


@router.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """
    Upload loan data file for analysis

    Accepts CSV and all Excel formats (.xls, .xlsx, .xlsm, .xlsb)
    """
    try:
        # Validate file type - accept CSV and all Excel formats
        filename_lower = file.filename.lower()
        if not (filename_lower.endswith('.csv') or
                filename_lower.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb'))):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Accepted formats: .csv, .xls, .xlsx, .xlsm, .xlsb"
            )

        # Create temporary file
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            # Write uploaded file to temp file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Load and validate data
        df, metadata = data_processor.load_data(temp_file_path)

        # Generate file ID
        file_id = f"file_{len(uploaded_files) + 1}_{file.filename}"

        # Store file info
        uploaded_files[file_id] = {
            'path': temp_file_path,
            'filename': file.filename,
            'dataframe': df,
            'metadata': metadata
        }

        # Convert to JSON-serializable format
        response_data = {
            'success': True,
            'file_id': file_id,
            'filename': file.filename,
            'rows': int(metadata['rows']),
            'columns': int(metadata['columns']),
            'validation': metadata['validation']
        }

        response_data = data_processor.convert_to_json_serializable(response_data)

        return JSONResponse(response_data)

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze_data(request: AnalyzeRequest):
    """
    Analyze uploaded loan data with ML predictions and explainability

    Returns:
        Comprehensive analysis including:
        - Data quality metrics
        - ML predictions for each application
        - Feature importance and SHAP values
        - Model performance metrics
        - Recommendations
    """
    try:
        # Get uploaded file
        if request.file_id not in uploaded_files:
            raise HTTPException(
                status_code=404,
                detail="File not found. Please upload the file first."
            )

        file_info = uploaded_files[request.file_id]
        df = file_info['dataframe']

        # Perform data analysis
        analysis = data_processor.analyze_data(df)

        # Generate predictions with explainability
        predictions = data_processor.generate_predictions_with_explainability(
            df=df,
            credit_scoring_service=credit_service
        )

        # Generate model metrics
        model_metrics = data_processor.generate_model_metrics(df, predictions)

        # Convert all data to JSON-serializable format
        result = {
            'success': True,
            'file_id': request.file_id,
            'total_records': analysis['total_records'],
            'quality_score': analysis['quality_score'],
            'missing_values': analysis['missing_values'],
            'duplicates': analysis['duplicates'],
            'approval_rate': analysis.get('approval_rate', 0),
            'predictions': predictions,
            'model_metrics': model_metrics,
            'data_statistics': {
                'numeric_stats': analysis.get('numeric_stats', {}),
                'categorical_stats': analysis.get('categorical_stats', {}),
                'loan_status_distribution': analysis.get('loan_status_distribution', {})
            }
        }

        # Convert to JSON-serializable format
        result = data_processor.convert_to_json_serializable(result)

        return JSONResponse(result)

    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files")
async def list_uploaded_files():
    """
    List all uploaded files
    """
    files_list = []

    for file_id, info in uploaded_files.items():
        files_list.append({
            'file_id': file_id,
            'filename': info['filename'],
            'rows': info['metadata']['rows'],
            'columns': info['metadata']['columns'],
            'upload_time': info.get('upload_time', 'N/A')
        })

    return JSONResponse({
        'success': True,
        'files': files_list,
        'total': len(files_list)
    })


@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """
    Delete an uploaded file
    """
    try:
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")

        # Delete temporary file
        file_path = uploaded_files[file_id]['path']
        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove from storage
        del uploaded_files[file_id]

        return JSONResponse({
            'success': True,
            'message': f'File {file_id} deleted successfully'
        })

    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample")
async def get_sample_data():
    """
    Get sample loan data for testing
    """
    try:
        # Load sample data
        sample_path = Path(__file__).parent.parent.parent.parent / "data" / "mock" / "sample_loan_data.csv"

        if not sample_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Sample data file not found"
            )

        df, metadata = data_processor.load_data(str(sample_path))

        # Convert first 10 rows to dict
        sample_records = df.head(10).to_dict('records')

        return JSONResponse({
            'success': True,
            'sample_data': sample_records,
            'total_records': len(df),
            'columns': list(df.columns)
        })

    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
