"""
Structural Breakpoint Detection Package

A modular package for detecting structural breakpoints in time series data
using multiple methods including Roy24 and Wavelet21 approaches.
"""

from .methods import Roy24Method, Wavelet21Method, BaseMethod
from .methods.base import CommonConfig, validate_input_data, standardize_output

__version__ = "2.0.0"
__author__ = "Structural Breakpoint Detection Team"
__description__ = "Multi-Method Structural Breakpoint Detection Package"

# Convenience functions for backward compatibility
def quick_setup(method: str = 'roy24', **kwargs) -> dict:
    """
    Quick setup for structural breakpoint detection.
    
    Args:
        method: Method to use ('roy24' or 'wavelet21')
        **kwargs: Method-specific parameters
        
    Returns:
        Configuration dictionary
    """
    config = CommonConfig.get_default_config()
    config.update(kwargs)
    config['method'] = method
    return config


def run_batch(input_parquet: str, out_pred_parquet: str, out_meta_parquet: str,
              method: str = 'roy24', **kwargs) -> tuple:
    """
    Run batch processing with specified method.
    
    Args:
        input_parquet: Input parquet file path
        out_pred_parquet: Output predictors file path
        out_meta_parquet: Output metadata file path
        method: Method to use ('roy24' or 'wavelet21')
        **kwargs: Method-specific parameters
        
    Returns:
        Tuple of (predictors_df, metadata_df)
    """
    if method == 'roy24':
        from .methods.roy24.batch_processor import run_batch as roy24_run_batch
        return roy24_run_batch(input_parquet, out_pred_parquet, out_meta_parquet, **kwargs)
    elif method == 'wavelet21':
        from .methods.wavelet21.batch_processor import run_wavelet_batch
        return run_wavelet_batch(input_parquet, out_pred_parquet, out_meta_parquet, **kwargs)
    elif method == 'tsfm':  # additive adapter for TSFM
        # TSFM currently produces only predictors; create a minimal metadata file.
        from .methods.TSFM.batch_processor import run_tsfm_batch
        import pandas as _pd
        preds = run_tsfm_batch(
            input_parquet=input_parquet,
            out_pred_parquet=out_pred_parquet,
            n_ids=kwargs.get('n_ids'),
            random_sample=kwargs.get('random_sample', False),
            seed=kwargs.get('seed', 42),
            cfg=kwargs.get('cfg'),
        )
        # Minimal metadata: just id index (preserves API contract without impacting Roy/Wavelet)
        meta = _pd.DataFrame(index=preds.index).reset_index()[['id']]
        from pathlib import Path as _Path
        out_meta_p = _Path(out_meta_parquet)
        out_meta_p.parent.mkdir(parents=True, exist_ok=True)
        if out_meta_p.suffix.lower() == '.csv':
            meta.to_csv(out_meta_p, index=False)
        else:
            meta.to_parquet(out_meta_p, index=False)
        return preds, meta
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'roy24', 'wavelet21', or 'tsfm'")


def compute_predictors_for_values(values, periods, method: str = 'roy24', **kwargs) -> tuple:
    """
    Compute predictors for a single time series.
    
    Args:
        values: Time series values
        periods: Period indicators
        method: Method to use ('roy24' or 'wavelet21')
        **kwargs: Method-specific parameters
        
    Returns:
        Tuple of (predictors_dict, metadata_dict)
    """
    if method == 'roy24':
        from .methods.roy24.predictor_extractor import compute_predictors_for_values as roy24_compute
        return roy24_compute(values, periods, **kwargs)
    elif method == 'wavelet21':
        method_instance = Wavelet21Method(kwargs)
        return method_instance.compute_predictors(values, periods, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'roy24' or 'wavelet21'")


def validate_config(method: str = 'roy24') -> None:
    """
    Validate configuration for specified method.
    
    Args:
        method: Method to validate ('roy24' or 'wavelet21')
    """
    if method == 'roy24':
        from .methods.roy24.config import validate_config as roy24_validate
        roy24_validate()
    elif method == 'wavelet21':
        from .methods.wavelet21.config import validate_config as wavelet21_validate
        wavelet21_validate()
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'roy24' or 'wavelet21'")


__all__ = [
    'Roy24Method', 'Wavelet21Method', 'BaseMethod',
    'CommonConfig', 'validate_input_data', 'standardize_output',
    'quick_setup', 'run_batch', 'compute_predictors_for_values', 'validate_config'
]