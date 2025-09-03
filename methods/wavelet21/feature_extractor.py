"""
Feature extraction for Wavelet21 method.

This module extracts structural break predictors from MODW analysis.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from .wavelet_analysis import build_features_for_id, ThresholdCache


def extract_wavelet_predictors(values: np.ndarray, periods: np.ndarray,
                              thr_cache: Optional[ThresholdCache] = None) -> Dict[str, float]:
    """
    Extract structural break predictors using MODW analysis.
    
    Args:
        values: Time series values
        periods: Period indicators
        thr_cache: Optional threshold cache for efficiency
        
    Returns:
        Dictionary of structural break predictors
    """
    # Use the comprehensive MODW feature extraction
    features = build_features_for_id(values, periods, thr_cache)
    
    # Add compatibility layer for existing interface
    predictors = {}
    predictors.update(features)
    
    # Compute derived predictors
    predictors['p_wavelet_break'] = compute_break_strength_from_modw(features)
    predictors['confidence'] = compute_wavelet_confidence_from_modw(features)
    
    return predictors


def compute_break_strength_from_modw(features: Dict[str, float]) -> float:
    """
    Compute break strength from MODW features.
    
    Args:
        features: Dictionary of MODW features
        
    Returns:
        Break strength (0-1 scale)
    """
    # Use MODW-specific features to determine break strength
    indicators = []
    
    # Local max over scales
    if 'S_local_max_over_j' in features:
        s_max = features['S_local_max_over_j']
        # Normalize based on typical threshold values
        indicators.append(min(s_max / 3.0, 1.0))
    
    # Exceedance counts
    if 'cnt_local_sum_over_j' in features:
        cnt = features['cnt_local_sum_over_j']
        indicators.append(min(cnt / 10.0, 1.0))
    
    # Energy ratio L2 norm
    if 'log_energy_ratio_l2norm_over_j' in features:
        energy_norm = features['log_energy_ratio_l2norm_over_j']
        indicators.append(min(energy_norm / 2.0, 1.0))
    
    # KS test p-values (lower p-values indicate stronger breaks)
    ks_indicators = []
    for key, value in features.items():
        if key.startswith('j') and key.endswith('_ks_p'):
            ks_indicators.append(1.0 - value)  # Convert p-value to strength
    
    if ks_indicators:
        indicators.append(np.mean(ks_indicators))
    
    return np.mean(indicators) if indicators else 0.5


def compute_wavelet_confidence_from_modw(features: Dict[str, float]) -> float:
    """
    Compute confidence score from MODW features.
    
    Args:
        features: Dictionary of MODW features
        
    Returns:
        Confidence score (0-1 scale)
    """
    # Combine multiple MODW indicators
    indicators = []
    
    # Break strength
    break_strength = compute_break_strength_from_modw(features)
    indicators.append(break_strength)
    
    # Residual diagnostics
    if 'arch_lm_p' in features:
        # Lower p-value indicates stronger evidence of heteroscedasticity
        arch_indicator = 1.0 - features['arch_lm_p']
        indicators.append(arch_indicator)
    
    # Ljung-Box tests
    lb_indicators = []
    for key, value in features.items():
        if 'ljungbox_p' in key:
            lb_indicators.append(1.0 - value)  # Convert p-value to confidence
    
    if lb_indicators:
        indicators.append(np.mean(lb_indicators))
    
    # Segment shift indicators
    if 'ks_p_raw' in features:
        indicators.append(1.0 - features['ks_p_raw'])
    
    if 'ks_p_abs' in features:
        indicators.append(1.0 - features['ks_p_abs'])
    
    return np.mean(indicators) if indicators else 0.5


# Compatibility functions for existing interface
def analyze_frequency_bands(wavelet_coeffs: Dict[str, np.ndarray],
                           periods: np.ndarray) -> Dict[str, float]:
    """
    Analyze frequency band characteristics (compatibility function).
    
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
    Compute variance ratio using wavelet coefficients (compatibility function).
    
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
