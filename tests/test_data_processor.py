"""
Tests for Data Processor
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add backend src to path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))

from src.utils.data_processor import DataProcessor


def test_load_sample_data():
    """Test loading sample CSV data"""
    processor = DataProcessor()

    # Load sample data
    sample_path = Path(__file__).parent.parent / "data" / "mock" / "sample_loan_data.csv"

    if not sample_path.exists():
        print("[WARN] Sample data file not found, skipping test")
        return

    df, metadata = processor.load_data(str(sample_path))

    # Assertions
    assert df is not None, "DataFrame is None"
    assert len(df) > 0, "DataFrame is empty"
    assert metadata['rows'] == len(df), "Row count mismatch"
    assert metadata['columns'] == len(df.columns), "Column count mismatch"

    print(f"[PASS] Loaded {metadata['rows']} rows and {metadata['columns']} columns")
    print(f"[PASS] Validation: {metadata['validation']['valid']}")


def test_data_validation():
    """Test data validation"""
    processor = DataProcessor()

    # Create test DataFrame
    test_data = {
        'application_id': ['LA001', 'LA002', 'LA003'],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'annual_income': [75000, 62000, 95000],
        'loan_amount': [50000, 30000, 75000],
        'credit_score': [720, 680, 750],
        'loan_status': ['approved', 'approved', 'approved']
    }

    df = pd.DataFrame(test_data)

    validation = processor.validate_data(df)

    # Assertions
    assert validation['valid'] == True, "Valid data marked as invalid"
    assert len(validation['issues']) == 0, "Issues found in valid data"

    print(f"[PASS] Data validation passed")
    print(f"[PASS] Issues: {len(validation['issues'])}")


def test_data_quality_score():
    """Test data quality score calculation"""
    processor = DataProcessor()

    # Create perfect data
    test_data = {
        'application_id': ['LA001', 'LA002', 'LA003'],
        'annual_income': [75000, 62000, 95000],
        'credit_score': [720, 680, 750]
    }

    df = pd.DataFrame(test_data)

    quality_score = processor._calculate_quality_score(df)

    # Perfect data should have high quality score
    assert quality_score >= 0.9, f"Quality score too low: {quality_score}"

    print(f"[PASS] Quality score: {quality_score:.3f}")


def test_feature_preparation():
    """Test feature preparation"""
    processor = DataProcessor()

    # Create test data
    test_data = {
        'application_id': ['LA001', 'LA002'],
        'annual_income': [75000, 62000],
        'loan_amount': [50000, 30000],
        'credit_score': [720, 680],
        'existing_debt': [15000, 8000],
        'loan_purpose': ['home_improvement', 'debt_consolidation']
    }

    df = pd.DataFrame(test_data)

    df_prep, features = processor.prepare_features(df)

    # Assertions
    assert len(df_prep) == len(df), "Row count changed"
    assert len(features) > 0, "No features prepared"
    assert 'loan_to_income_ratio' in df_prep.columns, "Derived feature not created"

    print(f"[PASS] Features prepared: {len(features)}")
    print(f"[PASS] Derived features created")


def test_data_analysis():
    """Test data analysis"""
    processor = DataProcessor()

    # Create test data
    test_data = {
        'application_id': ['LA001', 'LA002', 'LA003'],
        'annual_income': [75000, 62000, 95000],
        'loan_amount': [50000, 30000, 75000],
        'credit_score': [720, 680, 750],
        'loan_status': ['approved', 'approved', 'rejected']
    }

    df = pd.DataFrame(test_data)

    analysis = processor.analyze_data(df)

    # Assertions
    assert 'total_records' in analysis, "Total records missing"
    assert analysis['total_records'] == 3, "Wrong record count"
    assert 'quality_score' in analysis, "Quality score missing"
    assert 'numeric_stats' in analysis, "Numeric stats missing"
    assert 'approval_rate' in analysis, "Approval rate missing"

    print(f"[PASS] Total records: {analysis['total_records']}")
    print(f"[PASS] Quality score: {analysis['quality_score']:.3f}")
    print(f"[PASS] Approval rate: {analysis['approval_rate']:.2%}")


if __name__ == '__main__':
    print("Running Data Processor Tests...")
    print("=" * 60)

    try:
        test_load_sample_data()
        print("\n" + "=" * 60)

        test_data_validation()
        print("\n" + "=" * 60)

        test_data_quality_score()
        print("\n" + "=" * 60)

        test_feature_preparation()
        print("\n" + "=" * 60)

        test_data_analysis()
        print("\n" + "=" * 60)

        print("\n[SUCCESS] All tests passed!")

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
    except Exception as e:
        print(f"\n[FAIL] Error running tests: {e}")
        import traceback
        traceback.print_exc()
