"""
Configuration module for ADIA structural breakpoint detection.

This module contains all the runtime parameters and constants used throughout
the ADIA pipeline for detecting structural breakpoints in time series data.
"""

import os
from typing import Final

# ==================== Runtime Configuration ====================

# Bootstrap and statistical testing parameters
B_BOOT: Final[int] = 80                 # circular-shift bootstrap reps per conditional test
ENERGY_ENABLE: Final[bool] = False       # energy-distance p-value is expensive at scale
ENERGY_B: Final[int] = 40                # permutations for energy p-value (if enabled)
ENERGY_MAX_N_PER_PERIOD: Final[int] = 400  # subsample cap per period for energy (if enabled)

# Random seed for reproducibility
SEED: Final[int] = 123

# Parallel processing
N_JOBS: Final[int] = max(1, os.cpu_count() - 1)  # parallel workers

# ==================== File Paths ====================
# These can be overridden via environment variables or command line arguments
DEFAULT_INPUT_PATH: Final[str] = '/content/drive/MyDrive/ADIA/X_train.parquet'
DEFAULT_OUTPUT_PRED_PATH: Final[str] = '/content/drive/MyDrive/ADIA/X_train_predictors.parquet'
DEFAULT_OUTPUT_META_PATH: Final[str] = '/content/drive/MyDrive/ADIA/X_train_metadata.parquet'

# ==================== Statistical Parameters ====================
# Grid generation parameters
GRID_MIN_POINTS: Final[int] = 40
GRID_MAX_POINTS: Final[int] = 120

# Bandwidth parameters
BANDWIDTH_POWER: Final[float] = -0.2  # h = n^BANDWIDTH_POWER

# Overlap threshold for conditional tests
OVERLAP_THRESHOLD: Final[float] = 5.0

# Numerical stability constants
EPSILON: Final[float] = 1e-12
MIN_VARIANCE: Final[float] = 1e-12

# ACF lag values for analysis
ACF_LAGS: Final[list] = [1, 5, 10]

# ==================== Validation Functions ====================

def validate_config() -> None:
    """Validate configuration parameters."""
    if B_BOOT <= 0:
        raise ValueError("B_BOOT must be positive")
    if ENERGY_B <= 0:
        raise ValueError("ENERGY_B must be positive")
    if ENERGY_MAX_N_PER_PERIOD <= 0:
        raise ValueError("ENERGY_MAX_N_PER_PERIOD must be positive")
    if SEED < 0:
        raise ValueError("SEED must be non-negative")
    if N_JOBS <= 0:
        raise ValueError("N_JOBS must be positive")
    if BANDWIDTH_POWER >= 0:
        raise ValueError("BANDWIDTH_POWER must be negative for proper bandwidth scaling")

def get_bandwidth(n: int) -> float:
    """Calculate bandwidth based on sample size and configuration."""
    return n ** BANDWIDTH_POWER

# Validate configuration on import
validate_config()
