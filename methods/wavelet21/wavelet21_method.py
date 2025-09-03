"""
Wavelet21 method implementation for structural breakpoint detection.

This implements the MODW (Maximal Overlap Discrete Wavelet Transform) methodology
for robust multi-resolution structural break detection.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import time

from ..base import BaseMethod
from .wavelet_analysis import build_features_for_id, ThresholdCache
from .feature_extractor import extract_wavelet_predictors
from .config import (
    WAVELET_TYPE, DECOMPOSITION_LEVELS, ALPHA, BOUNDARY_WIN_FRAC, 
    BOUNDARY_WIN_MIN, LENGTH_BUCKET, MC_REPS, THRESHOLD_MODE, SEED, CACHE_FILE
)


class Wavelet21Method(BaseMethod):
    """
    Wavelet21 method for structural breakpoint detection.
    
    Implements MODW (Maximal Overlap Discrete Wavelet Transform) methodology
    for robust multi-resolution structural break detection.
    """
    
    version = "1.0.0"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Wavelet21 method.
        
        Args:
            config: Method-specific configuration dictionary
        """
        super().__init__(config)
        
        # Set default Wavelet21-specific parameters
        self.wavelet_type = self.config.get('wavelet_type', WAVELET_TYPE)
        self.decomposition_levels = self.config.get('decomposition_levels', DECOMPOSITION_LEVELS)
        self.alpha = self.config.get('alpha', ALPHA)
        self.boundary_win_frac = self.config.get('boundary_win_frac', BOUNDARY_WIN_FRAC)
        self.boundary_win_min = self.config.get('boundary_win_min', BOUNDARY_WIN_MIN)
        self.length_bucket = self.config.get('length_bucket', LENGTH_BUCKET)
        self.mc_reps = self.config.get('mc_reps', MC_REPS)
        self.threshold_mode = self.config.get('threshold_mode', THRESHOLD_MODE)
        self.seed = self.config.get('seed', SEED)
        self.cache_file = self.config.get('cache_file', CACHE_FILE)
        
        # Initialize threshold cache
        self.threshold_cache = ThresholdCache(self.cache_file)
    
    def compute_predictors(self, values: np.ndarray, periods: np.ndarray, 
                          **kwargs) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Compute structural break predictors using Wavelet21 MODW method.
        
        Args:
            values: Time series values
            periods: Period indicators (0 for pre-break, 1 for post-break)
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (predictors_dict, metadata_dict)
        """
        start_time = time.time()
        
        try:
            # Set random seed for reproducibility
            np.random.seed(self.seed)
            
            # Use MODW feature extraction
            predictors = extract_wavelet_predictors(
                values, periods, self.threshold_cache
            )
            
            # Create metadata
            processing_time = time.time() - start_time
            metadata = {
                'method': 'wavelet21',
                'processing_time': processing_time,
                'status': 'success',
                'n_observations': len(values),
                'wavelet_type': self.wavelet_type,
                'decomposition_levels': self.decomposition_levels,
                'alpha': self.alpha,
                'threshold_mode': self.threshold_mode,
                'boundary_win_frac': self.boundary_win_frac,
                'boundary_win_min': self.boundary_win_min,
                'mc_reps': self.mc_reps,
                'cache_file': self.cache_file
            }
            
            return predictors, metadata
            
        except Exception as e:
            # Return default values on error
            processing_time = time.time() - start_time
            default_predictors = {
                'p_wavelet_break': 0.5,
                'break_strength': 0.0,
                'confidence': 0.0,
                'resid_kurtosis': 0.0,
                'resid_skewness': 0.0,
                'mean_diff': 0.0,
                'log_var_ratio': 0.0,
                'S_local_max_over_j': 0.0,
                'cnt_local_sum_over_j': 0,
                'log_energy_ratio_l2norm_over_j': 0.0
            }
            default_metadata = {
                'method': 'wavelet21',
                'processing_time': processing_time,
                'status': 'failed',
                'error': str(e),
                'n_observations': len(values)
            }
            return default_predictors, default_metadata
    
    def validate_config(self) -> None:
        """
        Validate Wavelet21-specific configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.decomposition_levels <= 0:
            raise ValueError("decomposition_levels must be positive")
        
        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError("alpha must be between 0 and 1")
        
        if self.boundary_win_frac <= 0:
            raise ValueError("boundary_win_frac must be positive")
        
        if self.boundary_win_min <= 0:
            raise ValueError("boundary_win_min must be positive")
        
        if self.mc_reps <= 0:
            raise ValueError("mc_reps must be positive")
        
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
    
    def get_method_info(self) -> Dict[str, str]:
        """
        Get information about the Wavelet21 method.
        
        Returns:
            Dictionary with method information
        """
        info = super().get_method_info()
        info.update({
            'approach': 'MODW (Maximal Overlap Discrete Wavelet Transform)',
            'features': 'Multi-resolution analysis, MODWT coefficients, Monte Carlo thresholds, residual diagnostics',
            'wavelet_type': self.wavelet_type,
            'decomposition_levels': str(self.decomposition_levels),
            'alpha': str(self.alpha),
            'threshold_mode': self.threshold_mode
        })
        return info
