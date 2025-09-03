"""
Common utility functions for structural breakpoint detection methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional


def validate_input_data(values: np.ndarray, periods: np.ndarray) -> None:
    """
    Validate input data for structural breakpoint detection.
    
    Args:
        values: Time series values
        periods: Period indicators
        
    Raises:
        ValueError: If data is invalid
    """
    if len(values) != len(periods):
        raise ValueError("Values and periods must have the same length")
    
    if len(values) < 10:
        raise ValueError("Time series must have at least 10 observations")
    
    if not np.all(np.isfinite(values)):
        raise ValueError("Values must be finite")
    
    unique_periods = np.unique(periods)
    if len(unique_periods) < 2:
        raise ValueError("Must have at least 2 different periods")
    
    if not np.all(np.isin(unique_periods, [0, 1])):
        raise ValueError("Periods must be 0 or 1")


def standardize_output(predictors: Dict[str, float], metadata: Dict[str, Any], 
                      series_id: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Standardize output format across methods.
    
    Args:
        predictors: Raw predictor dictionary
        metadata: Raw metadata dictionary
        series_id: Optional series identifier
        
    Returns:
        Tuple of (standardized_predictors, standardized_metadata)
    """
    # Ensure predictors are floats
    std_predictors = {}
    for key, value in predictors.items():
        if isinstance(value, (int, float)):
            std_predictors[key] = float(value)
        else:
            std_predictors[key] = float('nan')
    
    # Add series ID if provided
    if series_id is not None:
        std_predictors['id'] = series_id
        metadata['id'] = series_id
    
    # Ensure metadata has standard fields
    if 'method' not in metadata:
        metadata['method'] = 'unknown'
    
    if 'n_observations' not in metadata:
        metadata['n_observations'] = len(predictors.get('values', []))
    
    return std_predictors, metadata


def create_empty_predictors(series_id: Optional[str] = None) -> Dict[str, float]:
    """
    Create empty predictor dictionary with default values.
    
    Args:
        series_id: Optional series identifier
        
    Returns:
        Dictionary with default predictor values
    """
    predictors = {
        'p_mean': 0.5,
        'p_var': 0.5,
        'p_structural_break': 0.5,
        'break_strength': 0.0,
        'confidence': 0.0
    }
    
    if series_id is not None:
        predictors['id'] = series_id
    
    return predictors


def create_empty_metadata(series_id: Optional[str] = None, method: str = 'unknown') -> Dict[str, Any]:
    """
    Create empty metadata dictionary with default values.
    
    Args:
        series_id: Optional series identifier
        method: Method name
        
    Returns:
        Dictionary with default metadata values
    """
    metadata = {
        'method': method,
        'n_observations': 0,
        'processing_time': 0.0,
        'status': 'failed'
    }
    
    if series_id is not None:
        metadata['id'] = series_id
    
    return metadata
