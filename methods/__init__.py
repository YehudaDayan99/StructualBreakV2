"""
Methods package for structural breakpoint detection.

This package contains different modeling techniques for structural breakpoint detection:
- roy24: Nonparametric method based on Roy et al. 2024
- wavelet21: MODW (Maximal Overlap Discrete Wavelet Transform) method
"""

from .base import BaseMethod
from .roy24 import Roy24Method
from .wavelet21 import Wavelet21Method

__all__ = ['BaseMethod', 'Roy24Method', 'Wavelet21Method']
