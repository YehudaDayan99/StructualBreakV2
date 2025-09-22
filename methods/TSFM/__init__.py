"""TSFM method package (additive, does not affect Roy24/Wavelet21)."""

from .config import TSFMConfig  # re-export
from .feature_extractor import extract_tsfm_predictors  # re-export
from .batch_processor import run_tsfm_batch  # re-export

__all__ = [
    "TSFMConfig",
    "extract_tsfm_predictors",
    "run_tsfm_batch",
]


