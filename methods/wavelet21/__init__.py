"""
Wavelet21 method for structural breakpoint detection.

MODW (Maximal Overlap Discrete Wavelet Transform) methodology
for robust multi-resolution structural break detection.
"""

from .wavelet21_method import Wavelet21Method
from .config import (
    WAVELET_TYPE, DECOMPOSITION_LEVELS, ALPHA, BOUNDARY_WIN_FRAC, 
    BOUNDARY_WIN_MIN, LENGTH_BUCKET, MC_REPS, THRESHOLD_MODE, SEED, CACHE_FILE,
    validate_config
)
from .wavelet_analysis import (
    build_features_for_id, ThresholdCache, modwt_coeffs, simulate_thresholds
)
from .feature_extractor import (
    extract_wavelet_predictors, compute_break_strength_from_modw, 
    compute_wavelet_confidence_from_modw
)

__all__ = [
    'Wavelet21Method',
    'WAVELET_TYPE', 'DECOMPOSITION_LEVELS', 'ALPHA', 'BOUNDARY_WIN_FRAC', 
    'BOUNDARY_WIN_MIN', 'LENGTH_BUCKET', 'MC_REPS', 'THRESHOLD_MODE', 'SEED', 'CACHE_FILE',
    'validate_config',
    'build_features_for_id', 'ThresholdCache', 'modwt_coeffs', 'simulate_thresholds',
    'extract_wavelet_predictors', 'compute_break_strength_from_modw', 
    'compute_wavelet_confidence_from_modw'
]
