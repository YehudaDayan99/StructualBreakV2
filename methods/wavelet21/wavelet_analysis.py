"""
MODW (Maximal Overlap Discrete Wavelet Transform) analysis for Wavelet21 method.

This module implements the robust multi-resolution procedure for detecting
structural breakpoints using MODWT on standardized residuals.
"""

import os
import math
import json
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Stats & models
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import kurtosis, skew, ks_2samp

# Wavelets
import pywt

# Optional ARCH
try:
    from arch import arch_model
    HAVE_ARCH = True
except ImportError:
    HAVE_ARCH = False

from .config import (
    WAVELET_TYPE, DECOMPOSITION_LEVELS, ALPHA, BOUNDARY_WIN_FRAC, 
    BOUNDARY_WIN_MIN, LENGTH_BUCKET, MC_REPS, THRESHOLD_MODE, SEED, CACHE_FILE
)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Global random number generator
rng_global = np.random.default_rng(SEED)


class ThresholdCache:
    """Cache for Monte Carlo thresholds to avoid recomputation."""
    
    def __init__(self, path: str = CACHE_FILE):
        self.path = path
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.store = json.load(f)
            except Exception:
                self.store = {}
        else:
            self.store = {}

    def key(self, n_bucket: int, J: int, law: str, wavelet: str, alpha: float):
        return f"n{n_bucket}_J{J}_law{law}_w{wavelet}_a{alpha}"

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.store, f)
        os.replace(tmp, self.path)


def first_boundary_index(periods: np.ndarray) -> int:
    """Return first index of period 1 (boundary position in original x)."""
    d = np.diff(periods.astype(int))
    idx = np.flatnonzero(d != 0)
    if len(idx) == 0:
        # No boundary found; default to middle
        return len(periods) // 2
    return int(idx[0] + 1)


def make_stationary(x: np.ndarray) -> np.ndarray:
    """Make series stationary via first difference or log-difference."""
    x = np.asarray(x, dtype=float)
    if np.all(np.isfinite(x)) and np.all(x > 0):
        return np.diff(np.log(x))
    return np.diff(x)


def standardize(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Standardize array to zero mean and unit variance."""
    m, s = np.mean(z), np.std(z)
    s = s if s > eps else eps
    return (z - m) / s


def fit_arma_resid(x: np.ndarray) -> np.ndarray:
    """Fit ARMA(1,1) to x and return standardized residuals."""
    try:
        model = ARIMA(x, order=(1, 0, 1))
        res = model.fit(method="statespace", disp=0)
        resid = res.resid
        return standardize(resid)
    except Exception:
        # Fallback to simple standardization
        return standardize(x)


def arch_lm_pvalue(resid: np.ndarray, lags: int = 10) -> float:
    """Compute ARCH LM test p-value."""
    try:
        test = het_arch(resid, nlags=lags)
        return float(test[1])  # returns (lm_stat, lm_pvalue, f_stat, f_pvalue)
    except Exception:
        return 1.0


def fit_garch_t_resid(x: np.ndarray) -> np.ndarray:
    """Fit GARCH(1,1)-t if arch is available; else return ARMA residuals."""
    if not HAVE_ARCH:
        return fit_arma_resid(x)
    try:
        am = arch_model(x, mean="ARX", lags=1, vol="GARCH", p=1, q=1, dist="StudentsT")
        r = am.fit(disp="off")
        resid = r.std_resid  # standardized residuals
        return standardize(resid)
    except Exception:
        return fit_arma_resid(x)


def modwt_coeffs(resid: np.ndarray, wavelet: str = WAVELET_TYPE, J: int = DECOMPOSITION_LEVELS) -> Dict[int, np.ndarray]:
    """Return dict {level: coeff_array} using MODWT if available, else SWT."""
    # Try MODWT (PyWavelets >= 1.5 has modwt)
    if hasattr(pywt, "modwt"):
        try:
            Ws = pywt.modwt(resid, wavelet=wavelet, level=J, mode="periodization")
            # pywt.modwt returns list [W1,..,WJ]
            return {j+1: np.asarray(Ws[j]) for j in range(len(Ws))}
        except Exception:
            pass
    
    # Fallback to SWT (stationary wavelet transform)
    try:
        swt_coeffs = pywt.swt(resid, wavelet=wavelet, level=J, start_level=0, trim_approx=True, norm=True)
        # swt returns list of tuples (cA_j, cD_j) from level 1..J
        return {j+1: np.asarray(swt_coeffs[j][1]) for j in range(len(swt_coeffs))}
    except Exception:
        # Final fallback: return empty dict
        return {}


def simulate_thresholds(n: int, J: int, wavelet: str, alpha: float, law: str, 
                       reps: int, rng: np.random.Generator) -> Dict[int, float]:
    """Return dict {j: q_{j,alpha}(n)} for max |W_{j,t}| under i.i.d. noise."""
    maxes = {j: [] for j in range(1, J+1)}
    
    for _ in range(reps):
        if law == "t":
            # df=8 as a robust default
            z = rng.standard_t(df=8, size=n)
            z = z / np.std(z)
        else:
            z = rng.standard_normal(size=n)
        
        W = modwt_coeffs(z, wavelet, J)
        for j in range(1, J+1):
            if j in W:
                maxes[j].append(np.max(np.abs(W[j])))
    
    return {j: float(np.quantile(maxes[j], 1.0 - alpha)) for j in range(1, J+1) if maxes[j]}


def universal_thresholds(n: int, J: int, wavelet: str) -> Dict[int, float]:
    """Donoho-Johnstone universal threshold as fallback."""
    thr = math.sqrt(2.0 * math.log(max(n, 2)))
    return {j: thr for j in range(1, J+1)}


def residual_diagnostics(resid: np.ndarray) -> Dict[str, float]:
    """Compute residual diagnostic features."""
    feats = {}
    feats["resid_kurtosis"] = float(kurtosis(resid, fisher=True, bias=False))
    feats["resid_skewness"] = float(skew(resid, bias=False))
    
    for lag in (10, 20):
        try:
            lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
            feats[f"ljungbox_p_ac_{lag}"] = float(lb["lb_pvalue"].iloc[0])
            lb2 = acorr_ljungbox(resid**2, lags=[lag], return_df=True)
            feats[f"ljungbox_p_ac2_{lag}"] = float(lb2["lb_pvalue"].iloc[0])
        except Exception:
            feats[f"ljungbox_p_ac_{lag}"] = 1.0
            feats[f"ljungbox_p_ac2_{lag}"] = 1.0
    
    feats["arch_lm_p"] = arch_lm_pvalue(resid, lags=10)
    return feats


def ewma_vol(x: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Compute EWMA volatility."""
    s2 = 0.0
    out = np.empty_like(x)
    for i, xi in enumerate(x):
        s2 = lam * s2 + (1 - lam) * xi * xi
        out[i] = math.sqrt(max(s2, 1e-12))
    return out


def simple_segment_shifts(x_pre: np.ndarray, per0: np.ndarray, per1: np.ndarray) -> Dict[str, float]:
    """Compute simple segment shift features."""
    feats = {}
    x0, x1 = x_pre[per0], x_pre[per1]
    
    if len(x0) > 0 and len(x1) > 0:
        feats["mean_diff"] = float(np.mean(x1) - np.mean(x0))
        v0, v1 = np.var(x0), np.var(x1)
        feats["log_var_ratio"] = float(np.log((v1 + 1e-12) / (v0 + 1e-12)))
        
        def acf1(z):
            if len(z) < 2:
                return 0.0
            return float(np.corrcoef(z[:-1], z[1:])[0,1])
        
        feats["acf1_diff"] = acf1(x1) - acf1(x0)
        
        # Robust statistics
        feats["median_diff"] = float(np.median(x1) - np.median(x0))
        mad = lambda z: np.median(np.abs(z - np.median(z))) * 1.4826
        feats["log_mad_ratio"] = float(np.log((mad(x1) + 1e-12) / (mad(x0) + 1e-12)))
        
        # KS tests
        try:
            feats["ks_p_raw"] = float(ks_2samp(x0, x1, alternative="two-sided").pvalue)
            feats["ks_p_abs"] = float(ks_2samp(np.abs(x0), np.abs(x1)).pvalue)
        except Exception:
            feats["ks_p_raw"] = 1.0
            feats["ks_p_abs"] = 1.0
        
        # EWMA vol ratio
        v0_ew = np.median(ewma_vol(x_pre[per0]))
        v1_ew = np.median(ewma_vol(x_pre[per1]))
        feats["ewma_vol_log_ratio"] = float(np.log((v1_ew + 1e-12) / (v0_ew + 1e-12)))
    else:
        # Default values when segments are empty
        feats.update({
            "mean_diff": 0.0, "log_var_ratio": 0.0, "acf1_diff": 0.0,
            "median_diff": 0.0, "log_mad_ratio": 0.0, "ks_p_raw": 1.0,
            "ks_p_abs": 1.0, "ewma_vol_log_ratio": 0.0
        })
    
    return feats


def modwt_feature_block(W: Dict[int, np.ndarray], B_resid: int, w: int, 
                       per0_idx: np.ndarray, per1_idx: np.ndarray,
                       thresholds: Dict[int, float], common_thr: float) -> Dict[str, float]:
    """Compute MODWT feature block."""
    feats = {}
    
    # Handle empty wavelet coefficients
    if not W:
        return {
            "S_local_max_over_j": 0.0, "cnt_local_sum_over_j": 0,
            "cnt_common_sum_over_j": 0, "log_energy_ratio_l2norm_over_j": 0.0
        }
    
    # Window slice around boundary in residual index space
    lo = max(B_resid - 1 - w, 0)
    hi = min(B_resid - 1 + w + 1, len(next(iter(W.values()))))
    sl = slice(lo, hi)

    energy_ratios = []
    for j, wj in W.items():
        wj = np.asarray(wj)
        
        # Localized max and exceedance counts
        s_local = float(np.max(np.abs(wj[sl]))) if hi > lo else float(np.max(np.abs(wj)))
        cnt = int(np.sum(np.abs(wj[sl]) > thresholds.get(j, 1.0)))
        cnt_common = int(np.sum(np.abs(wj[sl]) > common_thr))
        
        feats[f"j{j}_S_local"] = s_local
        feats[f"j{j}_cnt_local"] = cnt
        feats[f"j{j}_cnt_common"] = cnt_common
        
        # Energy ratios
        e0 = float(np.mean(wj[per0_idx]**2)) if np.any(per0_idx) else 0.0
        e1 = float(np.mean(wj[per1_idx]**2)) if np.any(per1_idx) else 0.0
        ler = float(np.log((e1 + 1e-12) / (e0 + 1e-12)))
        feats[f"j{j}_log_energy_ratio"] = ler
        energy_ratios.append(ler)
        
        # Distribution shifts
        mu0 = float(np.mean(wj[per0_idx])) if np.any(per0_idx) else 0.0
        mu1 = float(np.mean(wj[per1_idx])) if np.any(per1_idx) else 0.0
        sd0 = float(np.std(wj[per0_idx])) if np.any(per0_idx) else 1e-12
        sd1 = float(np.std(wj[per1_idx])) if np.any(per1_idx) else 1e-12
        
        feats[f"j{j}_d_mean"] = mu1 - mu0
        feats[f"j{j}_log_sd_ratio"] = float(np.log((sd1 + 1e-12) / (sd0 + 1e-12)))
        
        try:
            feats[f"j{j}_ks_p"] = float(ks_2samp(wj[per0_idx], wj[per1_idx]).pvalue)
        except Exception:
            feats[f"j{j}_ks_p"] = 1.0
        
        # Local energy fraction
        total_e = float(np.sum(wj**2)) + 1e-12
        local_e = float(np.sum(wj[sl]**2)) if hi > lo else float(np.sum(np.abs(wj)**2))
        feats[f"j{j}_local_energy_frac"] = local_e / total_e

    # Cross-scale summaries
    if W:
        feats["S_local_max_over_j"] = float(np.max([feats[f"j{j}_S_local"] for j in W.keys()]))
        feats["cnt_local_sum_over_j"] = int(np.sum([feats[f"j{j}_cnt_local"] for j in W.keys()]))
        feats["cnt_common_sum_over_j"] = int(np.sum([feats[f"j{j}_cnt_common"] for j in W.keys()]))
        feats["log_energy_ratio_l2norm_over_j"] = float(np.sqrt(np.sum(np.square(energy_ratios))))
    else:
        feats.update({
            "S_local_max_over_j": 0.0, "cnt_local_sum_over_j": 0,
            "cnt_common_sum_over_j": 0, "log_energy_ratio_l2norm_over_j": 0.0
        })

    return feats


def build_features_for_id(values: np.ndarray, periods: np.ndarray, 
                         thr_cache: Optional[ThresholdCache] = None) -> Dict[str, float]:
    """Build comprehensive MODW features for a single time series."""
    x = np.asarray(values, dtype=float)
    per = np.asarray(periods, dtype=int)
    n = len(x)
    B = first_boundary_index(per)  # boundary in original index

    # Preprocess to (log-)diffed series
    x_pre = make_stationary(x)

    # Align period masks to residual length (diff reduces length by 1)
    per0 = per[:-1] == 0
    per1 = per[:-1] == 1

    # Residual model under H0
    arma_resid = fit_arma_resid(x_pre)
    p_arch = arch_lm_pvalue(arma_resid, lags=10)
    
    if p_arch < 0.05:
        resid = fit_garch_t_resid(x_pre)
        law = "t"
    else:
        resid = arma_resid
        law = "normal"

    resid = standardize(resid)

    # MODWT/SWT coefficients
    W = modwt_coeffs(resid, wavelet=WAVELET_TYPE, J=DECOMPOSITION_LEVELS)

    # Thresholds
    n_resid = len(resid)
    n_bucket = int(round(n_resid / LENGTH_BUCKET) * LENGTH_BUCKET)
    
    if thr_cache is None:
        thr_cache = ThresholdCache()
    
    key = thr_cache.key(n_bucket, DECOMPOSITION_LEVELS, law, WAVELET_TYPE, ALPHA)
    q_dict = thr_cache.get(key)
    
    if q_dict is None:
        if THRESHOLD_MODE == "mc":
            q_dict = simulate_thresholds(n_bucket, DECOMPOSITION_LEVELS, WAVELET_TYPE, 
                                       ALPHA, law, MC_REPS, rng_global)
        else:
            q_dict = universal_thresholds(n_bucket, DECOMPOSITION_LEVELS, WAVELET_TYPE)
        thr_cache.set(key, q_dict)

    q_common = q_dict.get(max(W.keys(), default=1), 1.0)

    # Boundary window in residual space (B-1 due to diff)
    w = max(BOUNDARY_WIN_MIN, int(BOUNDARY_WIN_FRAC * n_resid))
    feats = {}

    # Residual diagnostics & base shifts
    feats.update(residual_diagnostics(resid))
    feats.update(simple_segment_shifts(x_pre, per0, per1))

    # MODWT features
    feats.update(modwt_feature_block(W, B_resid=B, w=w, per0_idx=per0, per1_idx=per1,
                                   thresholds={int(k): float(v) for k, v in q_dict.items()},
                                   common_thr=float(q_common)))

    return feats
