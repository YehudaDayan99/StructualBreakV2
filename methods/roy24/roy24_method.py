"""
Roy24 method implementation for structural breakpoint detection.

This class wraps the existing Roy24 implementation to conform to the BaseMethod interface.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import time

from ..base import BaseMethod
from .predictor_extractor import compute_predictors_for_values
from .config import B_BOOT, ENERGY_ENABLE, ENERGY_B, ENERGY_MAX_N_PER_PERIOD, SEED


class Roy24Method(BaseMethod):
    """
    Roy24 method for structural breakpoint detection.
    
    Nonparametric method based on Roy et al. 2024 for detecting structural
    breakpoints in time series data using kernel smoothing and conditional tests.
    """
    
    version = "1.0.0"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Roy24 method.
        
        Args:
            config: Method-specific configuration dictionary
        """
        super().__init__(config)
        
        # Set default Roy24-specific parameters
        self.b_boot = self.config.get('b_boot', B_BOOT)
        self.energy_enable = self.config.get('energy_enable', ENERGY_ENABLE)
        self.energy_b = self.config.get('energy_b', ENERGY_B)
        self.energy_max_n = self.config.get('energy_max_n', ENERGY_MAX_N_PER_PERIOD)
        self.seed = self.config.get('seed', SEED)
    
    def compute_predictors(self, values: np.ndarray, periods: np.ndarray, 
                          **kwargs) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Compute structural break predictors using Roy24 method.
        
        Args:
            values: Time series values
            periods: Period indicators (0 for pre-break, 1 for post-break)
            **kwargs: Additional parameters (B_boot, energy_enable, etc.)
            
        Returns:
            Tuple of (predictors_dict, metadata_dict)
        """
        start_time = time.time()
        
        # Extract parameters from kwargs or use instance defaults
        B_boot = kwargs.get('B_boot', self.b_boot)
        energy_enable = kwargs.get('energy_enable', self.energy_enable)
        energy_B = kwargs.get('energy_B', self.energy_b)
        energy_max_n = kwargs.get('energy_max_n', self.energy_max_n)
        seed = kwargs.get('seed', self.seed)
        
        try:
            # Use existing Roy24 implementation
            predictors, metadata = compute_predictors_for_values(
                values, periods,
                B_boot=B_boot,
                energy_enable=energy_enable,
                energy_B=energy_B,
                energy_max_n=energy_max_n,
                seed=seed
            )
            
            # Add processing metadata
            processing_time = time.time() - start_time
            metadata.update({
                'method': 'roy24',
                'processing_time': processing_time,
                'status': 'success',
                'n_observations': len(values),
                'b_boot': B_boot,
                'energy_enable': energy_enable
            })
            
            return predictors, metadata
            
        except Exception as e:
            # Return default values on error
            processing_time = time.time() - start_time
            default_predictors = {
                'p_mu_lag1': 0.5, 'p_sigma_lag1': 0.5, 'overlap_frac_lag1': 0.0,
                'p_mu_vol': 0.5, 'p_sigma_vol': 0.5, 'overlap_frac_vol': 0.0,
                'p_mu_resid_lag1': 0.5, 'p_sigma_resid_lag1': 0.5, 'overlap_frac_resid_lag1': 0.0,
                'p_mean': 0.5, 'p_var': 0.5, 'p_MWU': 0.5, 'p_energy': np.nan, 'acf_absdiff_l1': 0.0
            }
            default_metadata = {
                'method': 'roy24',
                'processing_time': processing_time,
                'status': 'failed',
                'error': str(e),
                'n_observations': len(values)
            }
            return default_predictors, default_metadata
    
    def validate_config(self) -> None:
        """
        Validate Roy24-specific configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.b_boot <= 0:
            raise ValueError("B_boot must be positive")
        
        if self.energy_b <= 0:
            raise ValueError("energy_b must be positive")
        
        if self.energy_max_n <= 0:
            raise ValueError("energy_max_n must be positive")
        
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
    
    def get_method_info(self) -> Dict[str, str]:
        """
        Get information about the Roy24 method.
        
        Returns:
            Dictionary with method information
        """
        info = super().get_method_info()
        info.update({
            'paper': 'Roy et al. 2024 - Nonparametric method of structural break detection',
            'approach': 'Kernel smoothing with conditional tests',
            'features': 'Mean/variance tests, energy distance, autocorrelation analysis'
        })
        return info
