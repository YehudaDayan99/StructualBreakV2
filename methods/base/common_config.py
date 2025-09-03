"""
Common configuration parameters shared across all methods.
"""

import os
from typing import Final, Dict, Any


class CommonConfig:
    """
    Common configuration parameters for all structural breakpoint detection methods.
    """
    
    # Random seed for reproducibility
    SEED: Final[int] = 123
    
    # Parallel processing
    N_JOBS: Final[int] = max(1, os.cpu_count() - 1)
    
    # File paths (can be overridden)
    DEFAULT_INPUT_PATH: Final[str] = 'input.parquet'
    DEFAULT_OUTPUT_PRED_PATH: Final[str] = 'predictors.parquet'
    DEFAULT_OUTPUT_META_PATH: Final[str] = 'metadata.parquet'
    
    # Common statistical parameters
    EPSILON: Final[float] = 1e-12
    MIN_VARIANCE: Final[float] = 1e-12
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get default configuration dictionary.
        
        Returns:
            Dictionary with default configuration values
        """
        return {
            'seed': cls.SEED,
            'n_jobs': cls.N_JOBS,
            'epsilon': cls.EPSILON,
            'min_variance': cls.MIN_VARIANCE,
            'input_path': cls.DEFAULT_INPUT_PATH,
            'output_pred_path': cls.DEFAULT_OUTPUT_PRED_PATH,
            'output_meta_path': cls.DEFAULT_OUTPUT_META_PATH
        }
    
    @classmethod
    def validate_common_config(cls, config: Dict[str, Any]) -> None:
        """
        Validate common configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        if 'seed' in config and config['seed'] < 0:
            raise ValueError("Seed must be non-negative")
        
        if 'n_jobs' in config and config['n_jobs'] <= 0:
            raise ValueError("Number of jobs must be positive")
        
        if 'epsilon' in config and config['epsilon'] <= 0:
            raise ValueError("Epsilon must be positive")
        
        if 'min_variance' in config and config['min_variance'] <= 0:
            raise ValueError("Minimum variance must be positive")
