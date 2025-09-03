"""
Feature extraction for Wavelet21 method.

This module extracts structural break predictors from wavelet analysis.
"""

import numpy as np
from typing import Dict, Any, Tuple


def extract_wavelet_predictors(values: np.ndarray, periods: np.ndarray,
                              wavelet_coeffs: Dict[str, np.ndarray],
                              breakpoints: np.ndarray,
                              min_break_strength: float = 0.3) -> Dict[str, float]:
    """
    Extract structural break predictors from wavelet analysis.
    
    Args:
        values: Time series values
        periods: Period indicators
        wavelet_coeffs: Wavelet coefficients
        breakpoints: Detected breakpoint indices
        min_break_strength: Minimum break strength threshold
        
    Returns:
        Dictionary of structural break predictors
    """
    predictors = {}
    
    # Basic breakpoint analysis
    n_breaks = len(breakpoints)
    predictors['n_breakpoints'] = float(n_breaks)
    
    # Break strength analysis
    break_strength = compute_break_strength(values, periods, breakpoints)
    predictors['break_strength'] = break_strength
    predictors['p_wavelet_break'] = 1.0 if break_strength > min_break_strength else 0.5
    
    # Frequency band analysis
    freq_features = analyze_frequency_bands(wavelet_coeffs, periods)
    predictors.update(freq_features)
    
    # Wavelet variance analysis
    var_ratio = compute_wavelet_variance_ratio(wavelet_coeffs, periods)
    predictors['wavelet_variance_ratio'] = var_ratio
    
    # Confidence based on multiple indicators
    confidence = compute_wavelet_confidence(predictors)
    predictors['confidence'] = confidence
    
    return predictors


def compute_break_strength(values: np.ndarray, periods: np.ndarray,
                          breakpoints: np.ndarray) -> float:
    """
    Compute the strength of detected breakpoints.
    
    Args:
        values: Time series values
        periods: Period indicators
        breakpoints: Detected breakpoint indices
        
    Returns:
        Break strength (0-1 scale)
    """
    if len(breakpoints) == 0:
        return 0.0
    
    # Analyze variance change around breakpoints
    strength_scores = []
    
    for bp in breakpoints:
        if 0 < bp < len(values) - 1:
            # Compare variance before and after breakpoint
            before_var = np.var(values[max(0, bp-10):bp])
            after_var = np.var(values[bp:min(len(values), bp+10)])
            
            # Normalize variance change
            if before_var > 0:
                var_change = abs(after_var - before_var) / before_var
                strength_scores.append(min(var_change, 1.0))
    
    return np.mean(strength_scores) if strength_scores else 0.0


def analyze_frequency_bands(wavelet_coeffs: Dict[str, np.ndarray],
                           periods: np.ndarray) -> Dict[str, float]:
    """
    Analyze frequency band characteristics.
    
    Args:
        wavelet_coeffs: Wavelet coefficients
        periods: Period indicators
        
    Returns:
        Dictionary of frequency band features
    """
    features = {}
    
    # Energy in different frequency bands
    low_energy = 0
    high_energy = 0
    
    for key, coeffs in wavelet_coeffs.items():
        energy = np.sum(coeffs ** 2)
        
        if 'detail_1' in key or 'detail_2' in key:
            high_energy += energy
        elif 'approx' in key or 'detail_3' in key or 'detail_4' in key:
            low_energy += energy
    
    total_energy = low_energy + high_energy
    if total_energy > 0:
        features['frequency_energy_ratio'] = high_energy / total_energy
        features['low_freq_energy'] = low_energy / total_energy
        features['high_freq_energy'] = high_energy / total_energy
    else:
        features['frequency_energy_ratio'] = 0.5
        features['low_freq_energy'] = 0.5
        features['high_freq_energy'] = 0.5
    
    return features


def compute_wavelet_variance_ratio(wavelet_coeffs: Dict[str, np.ndarray],
                                  periods: np.ndarray) -> float:
    """
    Compute variance ratio using wavelet coefficients.
    
    Args:
        wavelet_coeffs: Wavelet coefficients
        periods: Period indicators
        
    Returns:
        Variance ratio (0-1 scale)
    """
    # Separate coefficients by period
    period_0_coeffs = []
    period_1_coeffs = []
    
    for key, coeffs in wavelet_coeffs.items():
        # This is a simplified approach - in practice, you'd need to
        # map coefficients back to original time indices
        mid_point = len(coeffs) // 2
        period_0_coeffs.extend(coeffs[:mid_point])
        period_1_coeffs.extend(coeffs[mid_point:])
    
    if len(period_0_coeffs) > 0 and len(period_1_coeffs) > 0:
        var_0 = np.var(period_0_coeffs)
        var_1 = np.var(period_1_coeffs)
        
        if var_0 > 0:
            return min(var_1 / var_0, 2.0) / 2.0  # Normalize to 0-1
    
    return 0.5


def compute_wavelet_confidence(predictors: Dict[str, float]) -> float:
    """
    Compute confidence score based on multiple wavelet indicators.
    
    Args:
        predictors: Dictionary of predictor values
        
    Returns:
        Confidence score (0-1 scale)
    """
    # Combine multiple indicators
    indicators = []
    
    # Break strength indicator
    if 'break_strength' in predictors:
        indicators.append(predictors['break_strength'])
    
    # Frequency energy ratio indicator
    if 'frequency_energy_ratio' in predictors:
        # Values far from 0.5 indicate structural change
        freq_indicator = abs(predictors['frequency_energy_ratio'] - 0.5) * 2
        indicators.append(freq_indicator)
    
    # Variance ratio indicator
    if 'wavelet_variance_ratio' in predictors:
        # Values far from 0.5 indicate structural change
        var_indicator = abs(predictors['wavelet_variance_ratio'] - 0.5) * 2
        indicators.append(var_indicator)
    
    # Number of breakpoints indicator
    if 'n_breakpoints' in predictors:
        n_breaks = predictors['n_breakpoints']
        break_indicator = min(n_breaks / 5.0, 1.0)  # Normalize
        indicators.append(break_indicator)
    
    return np.mean(indicators) if indicators else 0.5
