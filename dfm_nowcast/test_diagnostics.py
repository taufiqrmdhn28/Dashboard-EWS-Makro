import pytest
import numpy as np
import pandas as pd
import warnings
from dfm_nowcast import (
    check_collinearity,
    compute_residual_normality
)
from dfm_nowcast.data import verify_data_sheets, verify_dates
from dfm_nowcast.estimation import safe_inv

def test_verify_data_sheets_malformed():
    # Missing required key: InfoQ is missing. Other keys are mock dataframes.
    bad_dict = {
        'InfoM': pd.DataFrame({'INCLUDED': [1]}),
        'MonthlyData': pd.DataFrame({'col': [1]}),
        'QuarterlyData': pd.DataFrame({'col': [1]}),
        'Calendar': pd.DataFrame({'col': [1]})
    }
    with pytest.raises(ValueError, match="Malformed input data: key 'InfoQ' is missing"):
        verify_data_sheets(bad_dict)

    # Missing column in InfoM
    bad_dict2 = {
        'InfoM': pd.DataFrame({'INCLUDED': [1]}),
        'MonthlyData': pd.DataFrame({'col': [1]}),
        'InfoQ': pd.DataFrame({'INCLUDED': [1], 'Indicator Code': ['col'], 'log': [0], 'QoQ': [0], 'YoY': [0], 'Months lag': [0], 'Days lag': [0]}),
        'QuarterlyData': pd.DataFrame({'col': [1]}),
        'Calendar': pd.DataFrame({'col': [1]})
    }
    with pytest.raises(ValueError, match="Malformed input data: 'InfoM' sheet is missing required column"):
        verify_data_sheets(bad_dict2)

def test_verify_dates_warnings():
    # Duplicate dates
    df_dup = pd.DataFrame({
        'Date': ['2020-01-01', '2020-01-01', '2020-02-01'],
        'Value': [1, 2, 3]
    })
    with pytest.warns(RuntimeWarning, match="Duplicate date entries found"):
        verify_dates(df_dup, 'MonthlyData')

    # Non-standard dates
    df_bad_format = pd.DataFrame({
        'Date': ['abc', 'def', 'ghi'],
        'Value': [1, 2, 3]
    })
    with pytest.warns(RuntimeWarning, match="Non-standard date formats found"):
        verify_dates(df_bad_format, 'MonthlyData')

def test_safe_inv_singular():
    # Singular matrix (all zeros)
    singular_matrix = np.zeros((3, 3))
    with pytest.warns(RuntimeWarning, match="Covariance matrix is singular or near-singular"):
        inv_m = safe_inv(singular_matrix)
    # Check that it produces the pseudo-inverse (which is also zero for all-zero matrix)
    np.testing.assert_array_equal(inv_m, np.zeros((3, 3)))

def test_collinearity_check():
    # Create collinear dataset
    x1 = np.random.randn(100)
    x2 = x1 * 1.0001 + np.random.randn(100) * 1e-8
    df_collinear = pd.DataFrame({
        'x1': x1,
        'x2': x2
    })
    
    data_dict = {
        'MonthlyData': df_collinear,
        'InfoM': pd.DataFrame({
            'INCLUDED': [1, 1],
            'Indicator Code': ['x1', 'x2']
        })
    }
    
    with pytest.warns(RuntimeWarning, match="Extremely Ill-conditioned/Multicollinear matrix"):
        check_collinearity(data_dict)

def test_residual_normality_metrics():
    # Generate random actuals and predictions
    np.random.seed(42)
    actual = np.random.randn(100)
    pred = actual + np.random.normal(0, 0.5, 100)
    
    metrics = compute_residual_normality(actual, pred)
    assert metrics['count'] == 100
    assert 'skewness' in metrics
    assert 'kurtosis' in metrics
    assert 'jb_stat' in metrics
    assert 'jb_pval' in metrics
    assert 'shapiro_stat' in metrics
    assert 'shapiro_pval' in metrics
    assert metrics['jb_pval'] > 0
    assert metrics['shapiro_pval'] > 0
