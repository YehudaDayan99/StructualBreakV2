"""
Wavelet21 method for structural breakpoint detection.

Wavelet-based method for detecting structural breakpoints in time series data.
This is a placeholder implementation that can be extended with actual wavelet analysis.
"""

from .wavelet21_method import Wavelet21Method
from .config import (
    WAVELET_TYPE, DECOMPOSITION_LEVELS, THRESHOLD_METHOD,
    SEED, N_JOBS, validate_config
)
from .wavelet_analysis import (
    wavelet_decomposition, detect_breakpoints, compute_wavelet_features
)
from .feature_extractor import (
    extract_wavelet_predictors, compute_break_strength, analyze_frequency_bands
)
from .batch_processor import (
    process_wavelet_series, run_wavelet_batch, get_wavelet_summary
)

__all__ = [
    'Wavelet21Method',
    'WAVELET_TYPE', 'DECOMPOSITION_LEVELS', 'THRESHOLD_METHOD',
    'SEED', 'N_JOBS', 'validate_config',
    'wavelet_decomposition', 'detect_breakpoints', 'compute_wavelet_features',
    'extract_wavelet_predictors', 'compute_break_strength', 'analyze_frequency_bands',
    'process_wavelet_series', 'run_wavelet_batch', 'get_wavelet_summary'
]
