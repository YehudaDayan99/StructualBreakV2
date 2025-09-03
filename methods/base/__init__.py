"""
Base classes and common utilities for structural breakpoint detection methods.
"""

from .base_method import BaseMethod
from .common_config import CommonConfig
from .utils import validate_input_data, standardize_output

__all__ = ['BaseMethod', 'CommonConfig', 'validate_input_data', 'standardize_output']
