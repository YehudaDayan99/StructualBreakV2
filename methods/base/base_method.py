"""
Abstract base class for structural breakpoint detection methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np


class BaseMethod(ABC):
    """
    Abstract base class for structural breakpoint detection methods.
    
    All methods must implement the core interface methods to ensure
    compatibility with the batch processing pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the method with optional configuration.
        
        Args:
            config: Method-specific configuration dictionary
        """
        self.config = config or {}
        self.method_name = self.__class__.__name__
    
    @abstractmethod
    def compute_predictors(self, values: np.ndarray, periods: np.ndarray, 
                          **kwargs) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Compute structural break predictors for a single time series.
        
        Args:
            values: Time series values
            periods: Period indicators (0 for pre-break, 1 for post-break)
            **kwargs: Additional method-specific parameters
            
        Returns:
            Tuple of (predictors_dict, metadata_dict)
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate method-specific configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    def get_method_info(self) -> Dict[str, str]:
        """
        Get information about the method.
        
        Returns:
            Dictionary with method information
        """
        return {
            'name': self.method_name,
            'description': self.__doc__ or 'No description available',
            'version': getattr(self, 'version', '1.0.0')
        }
    
    def preprocess_data(self, values: np.ndarray, periods: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess input data (can be overridden by subclasses).
        
        Args:
            values: Raw time series values
            periods: Raw period indicators
            
        Returns:
            Tuple of (processed_values, processed_periods)
        """
        # Default: no preprocessing
        return values, periods
    
    def postprocess_results(self, predictors: Dict[str, float], 
                          metadata: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Postprocess results (can be overridden by subclasses).
        
        Args:
            predictors: Raw predictor dictionary
            metadata: Raw metadata dictionary
            
        Returns:
            Tuple of (processed_predictors, processed_metadata)
        """
        # Default: no postprocessing
        return predictors, metadata
