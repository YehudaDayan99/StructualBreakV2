"""
Batch processing for Wavelet21 method.

This module handles batch processing of multiple time series using wavelet analysis.
"""

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from typing import Tuple, Dict, Any
import logging

from .wavelet21_method import Wavelet21Method
from .config import N_JOBS, SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_wavelet_series(id_value, g: pd.DataFrame, config: Dict[str, Any] = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Process a single time series using Wavelet21 method.
    
    Args:
        id_value: Series identifier
        g: DataFrame with time series data
        config: Wavelet21 configuration
        
    Returns:
        Tuple of (predictors_dict, metadata_dict)
    """
    try:
        # Sort by time
        g_sorted = g.sort_index(level='time')
        values = g_sorted['value'].to_numpy(float)
        periods = g_sorted['period'].to_numpy()
        
        # Initialize Wavelet21 method
        method = Wavelet21Method(config)
        
        # Compute predictors
        predictors, metadata = method.compute_predictors(values, periods)
        
        # Add series ID
        predictors['id'] = id_value
        metadata['id'] = id_value
        
        return predictors, metadata
        
    except Exception as e:
        logger.error(f"Error processing series {id_value}: {str(e)}")
        
        # Return default values on error
        default_predictors = {
            'id': id_value,
            'p_wavelet_break': 0.5,
            'break_strength': 0.0,
            'frequency_energy_ratio': 0.5,
            'wavelet_variance_ratio': 0.5,
            'dominant_frequency_shift': 0.0,
            'confidence': 0.0,
            'n_breakpoints': 0.0
        }
        default_metadata = {
            'id': id_value,
            'method': 'wavelet21',
            'status': 'failed',
            'error': str(e),
            'n_observations': len(g) if g is not None else 0
        }
        return default_predictors, default_metadata


def validate_input_dataframe(df: pd.DataFrame) -> None:
    """
    Validate input DataFrame for Wavelet21 processing.
    
    Args:
        df: Input DataFrame
        
    Raises:
        ValueError: If DataFrame is invalid
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must have MultiIndex with 'id' and 'time' levels")
    
    required_levels = ['id', 'time']
    if not all(level in df.index.names for level in required_levels):
        raise ValueError(f"DataFrame index must have levels: {required_levels}")
    
    required_columns = ['value', 'period']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must have columns: {required_columns}")
    
    if df.empty:
        raise ValueError("DataFrame cannot be empty")


def run_wavelet_batch(input_parquet: str, out_pred_parquet: str, out_meta_parquet: str,
                     config: Dict[str, Any] = None, n_jobs: int = None, 
                     verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Wavelet21 batch processing on multiple time series.
    
    Args:
        input_parquet: Path to input parquet file
        out_pred_parquet: Path to output predictors file
        out_meta_parquet: Path to output metadata file
        config: Wavelet21 configuration
        n_jobs: Number of parallel jobs
        verbose: Whether to show progress
        
    Returns:
        Tuple of (predictors_df, metadata_df)
    """
    if n_jobs is None:
        n_jobs = N_JOBS
    
    if config is None:
        config = {}
    
    # Load data
    logger.info(f"Loading data from {input_parquet}")
    df = pd.read_parquet(input_parquet)
    validate_input_dataframe(df)
    
    # Group by series ID
    grouped = df.groupby(level='id')
    series_ids = list(grouped.groups.keys())
    
    logger.info(f"Processing {len(series_ids)} series with {n_jobs} parallel jobs")
    
    # Process series in parallel
    if n_jobs == 1:
        results = []
        for series_id in tqdm(series_ids, disable=not verbose):
            g = grouped.get_group(series_id)
            result = process_wavelet_series(series_id, g, config)
            results.append(result)
    else:
        results = Parallel(n_jobs=n_jobs, verbose=1 if verbose else 0)(
            delayed(process_wavelet_series)(series_id, grouped.get_group(series_id), config)
            for series_id in series_ids
        )
    
    # Combine results
    predictors_list = []
    metadata_list = []
    
    for preds, meta in results:
        predictors_list.append(preds)
        metadata_list.append(meta)
    
    # Create output DataFrames
    predictors_df = pd.DataFrame(predictors_list)
    metadata_df = pd.DataFrame(metadata_list)
    
    # Save results
    logger.info(f"Saving predictors to {out_pred_parquet}")
    predictors_df.to_parquet(out_pred_parquet)
    
    logger.info(f"Saving metadata to {out_meta_parquet}")
    metadata_df.to_parquet(out_meta_parquet)
    
    return predictors_df, metadata_df


def get_wavelet_summary(predictors_df: pd.DataFrame, metadata_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for Wavelet21 batch processing.
    
    Args:
        predictors_df: Predictors DataFrame
        metadata_df: Metadata DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'n_series': len(predictors_df),
        'n_successful': len(metadata_df[metadata_df['status'] == 'success']),
        'n_failed': len(metadata_df[metadata_df['status'] == 'failed']),
        'avg_processing_time': metadata_df['processing_time'].mean(),
        'avg_break_strength': predictors_df['break_strength'].mean(),
        'avg_confidence': predictors_df['confidence'].mean(),
        'avg_n_breakpoints': predictors_df['n_breakpoints'].mean()
    }
    
    return summary
