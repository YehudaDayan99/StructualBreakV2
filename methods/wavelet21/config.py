"""
Configuration for Wavelet21 structural breakpoint detection method.

Based on MODW (Maximal Overlap Discrete Wavelet Transform) methodology
for robust multi-resolution structural break detection.
"""

import os
import math
from typing import Final

# MODW-specific parameters
WAVELET_TYPE: Final[str] = "la8"  # LA(8) wavelet per paper
DECOMPOSITION_LEVELS: Final[int] = 3  # Number of decomposition levels (J)
ALPHA: Final[float] = 0.05  # Significance level for thresholds
BOUNDARY_WIN_FRAC: Final[float] = 0.01  # Boundary window fraction
BOUNDARY_WIN_MIN: Final[int] = 10  # Minimum boundary window size
LENGTH_BUCKET: Final[int] = 100  # MC cache bucket size for n
MC_REPS: Final[int] = 400  # Monte Carlo repetitions for thresholds
THRESHOLD_MODE: Final[str] = "mc"  # "mc" or "universal"

# Bootstrap and statistical testing parameters
B_BOOT: Final[int] = 80  # Bootstrap replicates
ENERGY_ENABLE: Final[bool] = False  # Energy distance tests (expensive)
ENERGY_B: Final[int] = 40  # Permutations for energy tests
ENERGY_MAX_N_PER_PERIOD: Final[int] = 400  # Max sample size for energy tests

# Random seed for reproducibility
SEED: Final[int] = 42

# Parallel processing
N_JOBS: Final[int] = max(1, os.cpu_count() - 1)

# Cache file for thresholds
CACHE_FILE: Final[str] = "threshold_cache.json"

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
    if ALPHA <= 0 or ALPHA >= 1:
        raise ValueError("ALPHA must be between 0 and 1")
    if BOUNDARY_WIN_FRAC <= 0:
        raise ValueError("BOUNDARY_WIN_FRAC must be positive")
    if BOUNDARY_WIN_MIN <= 0:
        raise ValueError("BOUNDARY_WIN_MIN must be positive")
    if MC_REPS <= 0:
        raise ValueError("MC_REPS must be positive")

# Validate configuration on import
validate_config()
