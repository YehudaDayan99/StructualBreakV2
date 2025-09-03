"""
Configuration for Wavelet21 structural breakpoint detection method.
"""

import os
from typing import Final

# Wavelet-specific parameters
WAVELET_TYPE: Final[str] = 'db4'  # Daubechies 4 wavelet
DECOMPOSITION_LEVELS: Final[int] = 4  # Number of decomposition levels
THRESHOLD_METHOD: Final[str] = 'soft'  # Thresholding method: 'soft' or 'hard'

# Bootstrap and statistical testing parameters
B_BOOT: Final[int] = 80  # Bootstrap replicates
ENERGY_ENABLE: Final[bool] = False  # Energy distance tests (expensive)
ENERGY_B: Final[int] = 40  # Permutations for energy tests
ENERGY_MAX_N_PER_PERIOD: Final[int] = 400  # Max sample size for energy tests

# Random seed for reproducibility
SEED: Final[int] = 123

# Parallel processing
N_JOBS: Final[int] = max(1, os.cpu_count() - 1)

# Wavelet threshold parameters
THRESHOLD_FACTOR: Final[float] = 0.1  # Threshold factor for noise reduction
MIN_BREAK_STRENGTH: Final[float] = 0.3  # Minimum break strength to consider significant

# Frequency band analysis
FREQUENCY_BANDS: Final[list] = ['low', 'medium', 'high']  # Frequency bands to analyze

def validate_config() -> None:
    """Validate Wavelet21 configuration parameters."""
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
    if DECOMPOSITION_LEVELS <= 0:
        raise ValueError("DECOMPOSITION_LEVELS must be positive")
    if THRESHOLD_FACTOR <= 0:
        raise ValueError("THRESHOLD_FACTOR must be positive")
    if MIN_BREAK_STRENGTH < 0 or MIN_BREAK_STRENGTH > 1:
        raise ValueError("MIN_BREAK_STRENGTH must be between 0 and 1")

# Validate configuration on import
validate_config()
