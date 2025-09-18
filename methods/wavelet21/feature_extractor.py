"""
Feature extraction for Wavelet21 method (replacement).

This module implements a faithful version of the wavelet-based methodology:
  - Fit a null (no-break) model to obtain standardized residuals (ARMA → optional GARCH-t).
  - Apply MODWT (or SWT fallback) using LA(8), J≤3.
  - Calibrate scale-specific thresholds via Monte Carlo (cached) for max |W_{j,t}|.
  - Build boundary-local features (localized maxima, exceedance counts, local energy fraction),
    segment contrasts (energy/mean/sd ratios, KS), and cross-scale summaries.

Public entry point:
    extract_wavelet_predictors(values: np.ndarray, periods: np.ndarray,
                               thr_cache: Optional[ThresholdCache]) -> Dict[str, float]

This is a drop-in replacement for the previous file, but self-contained (no imports
from .wavelet_analysis). It preserves compatibility helpers: compute_break_strength_from_modw,
compute_wavelet_confidence_from_modw, analyze_frequency_bands, compute_wavelet_variance_ratio.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Union
import warnings
import logging

# Core dependencies
from scipy.stats import kurtosis, skew, ks_2samp
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# Wavelets
import pywt

# Optional ARCH
try:
    from arch import arch_model  # type: ignore
    HAVE_ARCH = True
except Exception:  # pragma: no cover
    HAVE_ARCH = False


# =============================
# Configuration & RNG
# =============================
@dataclass
class WaveletConfig:
    wavelet: str = "db8"           # Daubechies 8 (substitute for LA(8))
    J: int = 3                     # number of levels
    alpha: float = 0.05            # exceedance level for thresholds
    boundary_win_frac: float = 0.01
    boundary_win_min: int = 10
    length_bucket: int = 100       # bucket size for MC cache by n
    mc_reps: int = 400             # increase for higher-fidelity thresholds
    use_mc_thresholds: bool = True # if False, fallback to universal thresholds
    random_state: int = 42
    # Optional residual-first configuration (unused until wired)
    null_model: Optional["NullModelCfg"] = None
    # Whether to use residual-first pipeline (off by default for backward compatibility)
    use_residuals: bool = False
    # Contrast engine: "recon" (DWT level reconstruction) or "swt" (legacy)
    contrast_engine: str = "recon"


DEFAULT_CFG = WaveletConfig()


# =============================
# Threshold Cache
# =============================
class ThresholdCache:
    """Cache for Monte Carlo thresholds by (n, J, wavelet, alpha)."""
    
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._cache: Dict[Tuple[int, int, str, float], Dict[int, float]] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load existing cache from disk."""
        try:
            import json
            import ast
            import re
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
                # Convert string keys back to tuples
                for key_str, thresholds in data.items():
                    n: int
                    J: int
                    wavelet: str
                    alpha: float
                    parsed_ok = False
                    # First try safe literal eval for tuple-like strings
                    try:
                        tpl = ast.literal_eval(key_str)
                        if (
                            isinstance(tpl, tuple)
                            and len(tpl) == 4
                            and isinstance(tpl[0], int)
                            and isinstance(tpl[1], int)
                            and isinstance(tpl[2], str)
                            and isinstance(tpl[3], (int, float))
                        ):
                            n, J, wavelet, alpha = tpl
                            parsed_ok = True
                    except Exception:
                        pass

                    # Fallback: support legacy keys like "n100_J3_lawnormal_wla8_a0.05"
                    if not parsed_ok:
                        m = re.match(r"^n(\d+)_J(\d+)_[A-Za-z0-9]+_w([A-Za-z0-9]+)_a([0-9.]+)$", key_str)
                        if m:
                            n = int(m.group(1))
                            J = int(m.group(2))
                            wtoken = m.group(3)
                            # Legacy had a leading 'w' in wavelet token; strip if present
                            wavelet = wtoken[1:] if wtoken.startswith('w') else wtoken
                            alpha = float(m.group(4))
                            parsed_ok = True

                    if parsed_ok:
                        self._cache[(n, J, wavelet, float(alpha))] = thresholds
                    else:
                        # Skip unknown key formats silently
                        continue
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass  # Start with empty cache
    
    def _save_cache(self):
        """Save cache to disk."""
        import json
        # Convert tuple keys to strings for JSON serialization
        data = {str(k): v for k, v in self._cache.items()}
        with open(self.cache_path, 'w') as f:
            json.dump(data, f)
    
    def get(self, n: int, J: int, wavelet: str, alpha: float) -> Optional[Dict[int, float]]:
        """Get cached thresholds."""
        return self._cache.get((n, J, wavelet, alpha))
    
    def set(self, n: int, J: int, wavelet: str, alpha: float, thresholds: Dict[int, float]):
        """Set cached thresholds."""
        self._cache[(n, J, wavelet, alpha)] = thresholds
        self._save_cache()


# =============================
# Utility Functions
# =============================
def _swt_details(x: np.ndarray, wavelet: str = 'sym4', J: int = 3) -> list:
    """Sanity harness for SWT stability (as per PDF debugging guide).
    
    Args:
        x: Input time series
        wavelet: Wavelet name
        J: Number of decomposition levels
        
    Returns:
        List of detail coefficients, each as 1D array
    """
    coeffs = pywt.swt(x, wavelet, level=J, trim_approx=True, norm=True)
    out = []
    for c in coeffs:
        if isinstance(c, tuple):
            out.append(np.asarray(c[1]).reshape(-1))
        else:
            out.append(np.asarray(c).reshape(-1))
    return out


def _test_swt_stability(x: np.ndarray, wavelet: str = 'sym4', J: int = 3) -> bool:
    """Test SWT stability with smoke test (as per PDF debugging guide).
    
    Args:
        x: Input time series
        wavelet: Wavelet name  
        J: Number of decomposition levels
        
    Returns:
        True if stable, False otherwise
    """
    try:
        det = _swt_details(x, wavelet, J)
        for j, Wj in enumerate(det, 1):
            if not (Wj.ndim == 1 and Wj.size == x.size):
                return False
        return True
    except Exception:
        return False


def _pad_to_pow2_periodic(x: np.ndarray) -> Tuple[np.ndarray, int]:
    """Pad to next power of two using periodic wrap; return padded array and original length.
    """
    n = int(x.size)
    if n <= 0:
        return x.copy(), n
    target = 1
    while target < n:
        target <<= 1
    if target == n:
        return x.copy(), n
    reps = int(np.ceil(target / n))
    x_rep = np.tile(x, reps)[:target]
    return x_rep, n


def _dwt_level_reconstruction(series: np.ndarray, wavelet: str, level: int, J: int) -> np.ndarray:
    """Reconstruct N-length level-j detail component via DWT.

    Uses periodic padding to next power-of-two for stable decomposition, then trims back.
    """
    x = np.asarray(series, dtype=float)
    x_pad, n_orig = _pad_to_pow2_periodic(x)
    wobj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(data_len=x_pad.size, filter_len=wobj.dec_len)
    actual_J = min(J, max_level)
    lvl = min(level, actual_J)
    coeffs = pywt.wavedec(x_pad, wavelet=wobj, level=actual_J, mode='periodization')
    # coeffs = [cA_J, cD_J, cD_{J-1}, ..., cD_1]
    # Zero all but cD_lvl
    for j in range(1, actual_J + 1):
        if j == lvl:
            continue
        coeffs[-j] = np.zeros_like(coeffs[-j])
    # Optionally keep approximation zero to isolate detail energy
    coeffs[0] = np.zeros_like(coeffs[0])
    x_rec = pywt.waverec(coeffs, wavelet=wobj, mode='periodization')
    # Ensure 1D and trim to original length
    x_rec = np.asarray(x_rec, dtype=float).reshape(-1)
    if x_rec.size > n_orig:
        x_rec = x_rec[:n_orig]
    return x_rec
def _standardize(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Standardize array to zero mean, unit variance."""
    m = float(np.mean(z))
    s = float(np.std(z))
    s = s if s > eps else eps
    return (z - m) / s


def _arma_resid(x: np.ndarray) -> np.ndarray:
    res = ARIMA(x, order=(1, 0, 1)).fit(method="statespace")
    return _standardize(res.resid)


def _arch_lm_pvalue(resid: np.ndarray, lags: int = 10) -> float:
    # returns (lm_stat, lm_pvalue, f_stat, f_pvalue)
    return float(het_arch(resid, nlags=lags)[1])


def _ljungbox_pvalue(resid: np.ndarray, lags: int = 10) -> float:
    # returns (lb_stat, lb_pvalue)
    try:
        # Ensure we have enough data and valid lag
        if len(resid) <= lags:
            return 1.0
        result = acorr_ljungbox(resid, lags=lags, return_df=False)
        if isinstance(result, tuple):
            return float(result[1])
        else:
            return float(result.iloc[0, 1])  # p-value is in second column
    except Exception:
        return 1.0  # Return non-significant p-value on error


def _modwt_coeffs(resid: np.ndarray, wavelet: str, J: int) -> Dict[int, np.ndarray]:
    """Extract MODWT or SWT coefficients with proper error handling."""
    # Handle odd-length data by padding to even length
    data = resid.copy()
    if len(data) % 2 == 1:
        data = np.append(data, data[-1])  # Pad with last value (forward fill)
    
    # Calculate maximum decomposition level based on data length
    max_level = pywt.swt_max_level(len(data))
    actual_J = min(J, max_level)
    
    if actual_J < J:
        warnings.warn(f"Requested J={J} but data length {len(data)} only supports J={actual_J}")
    
    try:
        # Try MODWT first; normalize output to 1D arrays of length len(data)
        if hasattr(pywt, "modwt"):
            Ws = pywt.modwt(data, wavelet=wavelet, level=actual_J, mode="periodization")
            # Normalize MODWT result
            modwt_ok = False
            if isinstance(Ws, np.ndarray) and Ws.ndim == 2 and Ws.shape[1] == len(data):
                out = {j + 1: np.asarray(Ws[j, :], dtype=float).reshape(-1) for j in range(Ws.shape[0])}
                modwt_ok = True
            elif isinstance(Ws, (list, tuple)):
                out = {}
                modwt_ok = True
                for j in range(len(Ws)):
                    wj = Ws[j]
                    if isinstance(wj, tuple):
                        wj = wj[1] if len(wj) > 1 else wj[0]
                    arr = np.asarray(wj, dtype=float).reshape(-1)
                    if arr.size != len(data):
                        modwt_ok = False
                        break
                    out[j + 1] = arr
            else:
                out = {}

            if modwt_ok and out:
                return out

        # Fallback to SWT with full-length detail coefficients
        swt = pywt.swt(data, wavelet=wavelet, level=actual_J, trim_approx=True, norm=True)
        return {j + 1: np.asarray(swt[j][1], dtype=float).reshape(-1) for j in range(len(swt))}
    except Exception as e:
        warnings.warn(f"Wavelet transform failed: {e}. Using identity transform.")
        # Fallback: return the original data as a single "level"
        return {1: np.asarray(data, dtype=float)}


def _simulate_thresholds(n: int, J: int, wavelet: str, alpha: float, law: str,
                         reps: int, rng: np.random.Generator) -> Dict[int, float]:
    """Monte-Carlo q_{j,alpha}(n) for max |W_{j,t}| under i.i.d. noise."""
    thresholds = {}
    
    # Calculate actual J based on data length
    max_level = pywt.swt_max_level(n)
    actual_J = min(J, max_level)
    
    for j in range(1, actual_J + 1):
        max_coeffs = []
        for _ in range(reps):
            if law == "gaussian":
                noise = rng.standard_normal(n)
            else:  # t-distribution
                noise = rng.standard_t(df=4, size=n)
            
            # Apply wavelet transform
            W = _modwt_coeffs(noise, wavelet=wavelet, J=j)
            if j in W:
                max_coeffs.append(np.max(np.abs(W[j])))
        
        if max_coeffs:
            thresholds[j] = float(np.quantile(max_coeffs, 1 - alpha))
    
    return thresholds


def _get_thresholds(n: int, J: int, wavelet: str, alpha: float, 
                   thr_cache: Optional[ThresholdCache], cfg: WaveletConfig) -> Dict[int, float]:
    """Get thresholds from cache or compute via Monte Carlo."""
    
    # Calculate actual J based on data length
    max_level = pywt.swt_max_level(n)
    actual_J = min(J, max_level)
    
    # Check cache first
    if thr_cache is not None:
        cached = thr_cache.get(n, actual_J, wavelet, alpha)
        if cached is not None:
            return cached
    
    # Compute thresholds
    rng = np.random.default_rng(cfg.random_state)
    thresholds = _simulate_thresholds(n, actual_J, wavelet, alpha, "gaussian", 
                                    cfg.mc_reps, rng)
    
    # Cache the results
    if thr_cache is not None:
        thr_cache.set(n, actual_J, wavelet, alpha, thresholds)
    
    return thresholds


# =============================
# Residual-first (isolated additions; unused until wired)
# =============================
@dataclass
class NullModelCfg:
    model: str = "arima"             # placeholder for future "garch" option
    arima_order: tuple = (1, 0, 1)
    resid_law: str = "t"             # "normal" or "t"
    min_len: int = 300               # fallback to z-score if shorter


def fit_null_model(full_values: np.ndarray, cfg: NullModelCfg):
    """
    Fit a simple H0 on the full series and return standardized residuals and diagnostics.
    Isolated helper; not used by the main pipeline until explicitly wired.
    """
    x = np.asarray(full_values, dtype=float)

    # Fallback path for short or non-finite series
    if len(x) < cfg.min_len or not np.isfinite(x).all():
        m = float(np.nanmean(x))
        s = float(np.nanstd(x)) if np.isfinite(np.nanstd(x)) else 1.0
        s = s if s > 1e-12 else 1e-12
        eps = (x - m) / s
        meta = dict(law="normal", nu=np.nan, ljungbox_p=np.nan, archlm_p=np.nan)
        return eps, meta

    # ARIMA residuals as baseline H0
    res = ARIMA(x, order=cfg.arima_order).fit(method="statespace")
    resid = np.asarray(res.resid, dtype=float)
    s = float(np.nanstd(resid))
    s = s if s > 1e-12 else 1e-12
    eps = resid / s

    # Diagnostics: Ljung–Box on residuals and squared residuals
    try:
        lb_stat, lb_p = acorr_ljungbox(eps, lags=20, return_df=False)
        if isinstance(lb_p, (list, tuple, np.ndarray)):
            lb_p_val = float(lb_p[-1])
        else:
            lb_p_val = float(lb_p)
    except Exception:
        lb_p_val = np.nan

    try:
        lb2_stat, lb2_p = acorr_ljungbox(eps ** 2, lags=20, return_df=False)
        if isinstance(lb2_p, (list, tuple, np.ndarray)):
            lb2_p_val = float(lb2_p[-1])
        else:
            lb2_p_val = float(lb2_p)
    except Exception:
        lb2_p_val = np.nan

    law = "t" if str(cfg.resid_law).lower().startswith("t") else "normal"
    nu = 8.0 if law == "t" else np.nan

    meta = dict(law=law, nu=nu, ljungbox_p=lb_p_val, archlm_p=lb2_p_val)
    return eps, meta


def h0_meta_to_feats(meta: Dict[str, Any]) -> Dict[str, float]:
    """
    Map H0 diagnostics to flat feature dict. Not used until residual pipeline is enabled.
    """
    return {
        "h0_ljungbox_p": float(meta.get("ljungbox_p", np.nan)) if meta.get("ljungbox_p") is not None else np.nan,
        "h0_archlm_p": float(meta.get("archlm_p", np.nan)) if meta.get("archlm_p") is not None else np.nan,
        "h0_err_is_t": 1.0 if meta.get("law") == "t" else 0.0,
        "h0_t_nu": float(meta.get("nu", np.nan)) if meta.get("nu") is not None else np.nan,
    }


# =============================
# Period-aware contrasts on residual MODWT (helpers)
# =============================
def _safe_var(x: np.ndarray) -> float:
    v = float(np.nanvar(x, ddof=1)) if x.size > 1 else 0.0
    return v + 1e-12


def _mad_normalized(x: np.ndarray) -> float:
    # Normalized MAD (approx. std under normality): 1.4826 * median(|x - median(x)|)
    if x.size == 0:
        return 1e-12
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    return 1.4826 * mad + 1e-12


# =============================
# Feature Extraction
# =============================
def _wavelet_features_for_series(values: np.ndarray, periods: np.ndarray, 
                                cfg: WaveletConfig, 
                                thr_cache: Optional[ThresholdCache]) -> Dict[str, float]:
    """Extract comprehensive wavelet features for a single series."""
    
    # Ensure we have enough data
    if len(values) < 20:
        return _default_features()
    
    # Split by period
    per = periods.astype(int)
    x_pre = values[per == 0]
    x_post = values[per == 1]
    
    if len(x_pre) < 10 or len(x_post) < 10:
        return _default_features()
    
    # Use pre-period for residual modeling
    per1 = per[:-1] == 1

    # Residual model under H0 (flag-controlled)
    if cfg.use_residuals and cfg.null_model is not None:
        try:
            resid, h0_meta = fit_null_model(values, cfg.null_model)
        except Exception:
            # Fallback to original ARMA residuals if null-model fitting fails
            arma_resid = _arma_resid(x_pre)
            p_arch = _arch_lm_pvalue(arma_resid, lags=10)
            if p_arch < 0.05 and HAVE_ARCH:
                try:
                    am = arch_model(arma_resid, vol='Garch', p=1, q=1, dist='t')
                    r = am.fit()
                    resid = r.std_resid
                except Exception:
                    resid = arma_resid
            else:
                resid = arma_resid
            h0_meta = {"ljungbox_p": np.nan, "archlm_p": np.nan, "law": "normal", "nu": np.nan}
    else:
        arma_resid = _arma_resid(x_pre)
        p_arch = _arch_lm_pvalue(arma_resid, lags=10)
        if p_arch < 0.05 and HAVE_ARCH:
            try:
                am = arch_model(arma_resid, vol='Garch', p=1, q=1, dist='t')
                r = am.fit()
                resid = r.std_resid
            except Exception:
                resid = arma_resid
        else:
            resid = arma_resid
        h0_meta = {"ljungbox_p": np.nan, "archlm_p": np.nan, "law": "normal", "nu": np.nan}
    
    resid = _standardize(resid)
    
    # MODWT/SWT coefficients
    W = _modwt_coeffs(resid, wavelet=cfg.wavelet, J=cfg.J)
    
    # Get actual J used (may be less than requested)
    actual_J = len(W)
    
    # Thresholds (MC or universal) with bucketing by residual length
    n_resid = len(resid)
    bucket_n = (n_resid // cfg.length_bucket) * cfg.length_bucket
    bucket_n = max(bucket_n, cfg.length_bucket)
    
    if cfg.use_mc_thresholds:
        thresholds = _get_thresholds(bucket_n, cfg.J, cfg.wavelet, cfg.alpha, thr_cache, cfg)
    else:
        # Universal thresholds (simplified)
        thresholds = {j: 2.0 * np.sqrt(2 * np.log(n_resid)) for j in range(1, actual_J + 1)}
    
    features = {}
    
    # 1. Scale-specific local maxima and exceedances
    for j in range(1, actual_J + 1):
        if j not in W:
            continue  # Skip if this level doesn't exist
            
        # Get detail coefficients robustly (as per PDF instructions)
        Wj_raw = W[j]
        if isinstance(Wj_raw, tuple):
            Wj_raw = Wj_raw[1]  # detail (cD)
        Wj = np.asarray(Wj_raw).reshape(-1)
        nj = Wj.size
        
        # Optional debug
        # logging.debug(f"Level {j}: Wj_raw type={type(Wj_raw)}, Wj shape={Wj.shape}, nj={nj}")
        
        if nj == 0:
            logging.warning(f"Level {j}: Empty coefficients, skipping")
            continue
            
        thresh_j = thresholds.get(j, 2.0 * np.sqrt(2 * np.log(n_resid)))
        
        # Exceedance fraction
        exceed_count = int(np.sum(np.abs(Wj) > thresh_j))
        features[f"j{j}_exceed_count"] = float(exceed_count)
        features[f"j{j}_exceed_frac"] = exceed_count / float(nj)
        
        # Localized stats (avoid reusing Wj - use Wj_win for windowed analysis)
        # For boundary analysis, use a window around the boundary
        win_lo = max(0, nj // 2 - 10)
        win_hi = min(nj, nj // 2 + 10)
        Wj_win = Wj[win_lo:win_hi]
        S_j = float(np.max(np.abs(Wj_win))) if Wj_win.size else 0.0
        features[f"j{j}_S_local_max"] = S_j
        
        # Local maxima over threshold (using full array)
        local_max = np.max(Wj)
        features[f"j{j}_local_max"] = float(local_max)
        features[f"j{j}_exceeds_thresh"] = float(local_max > thresh_j)
        
        # Energy (using full array)
        energy_j = float(np.sum(Wj * Wj))
        features[f"j{j}_energy"] = energy_j
        features[f"j{j}_energy_norm"] = float(energy_j / nj)
    
    # 2. Period-aware contrasts on residuals (only when residuals path is active)
    if cfg.use_residuals and cfg.null_model is not None:
        pre_idx = np.flatnonzero(per == 0)
        post_idx = np.flatnonzero(per == 1)
        fam = cfg.wavelet
        engine = (cfg.contrast_engine or "recon").lower()
        if engine == "recon":
            # DWT reconstruction-based per-level detail at full length
            for j in range(1, actual_J + 1):
                Wj_full = _dwt_level_reconstruction(resid, wavelet=fam, level=j, J=cfg.J)
                pre_idx_clip = pre_idx[pre_idx < Wj_full.size]
                post_idx_clip = post_idx[post_idx < Wj_full.size]
                if pre_idx_clip.size > 1 and post_idx_clip.size > 1:
                    v0 = _safe_var(Wj_full[pre_idx_clip])
                    v1 = _safe_var(Wj_full[post_idx_clip])
                    m0 = _mad_normalized(Wj_full[pre_idx_clip])
                    m1 = _mad_normalized(Wj_full[post_idx_clip])
                    features[f"wav_{fam}_L{j}_var_logratio"] = float(np.log(v1 / v0))
                    features[f"wav_{fam}_L{j}_mad_logratio"] = float(np.log(m1 / m0))
        else:
            # Legacy SWT/MODWT path (best-effort)
            # Ensure we use full-length coefficients for contrasts
            use_W = True
            for j in range(1, actual_J + 1):
                if j not in W or np.asarray(W[j]).reshape(-1).size != len(resid):
                    use_W = False
                    break
            if not use_W:
                resid_even = resid if len(resid) % 2 == 0 else np.append(resid, resid[-1])
                swt = pywt.swt(resid_even, wavelet=fam, level=min(cfg.J, pywt.swt_max_level(len(resid_even))), trim_approx=True, norm=True)
                Wc = {j + 1: np.asarray(swt[j][1], dtype=float).reshape(-1) for j in range(len(swt))}
            else:
                Wc = {j: np.asarray(W[j]).reshape(-1) for j in W}

            for j in range(1, len(Wc) + 1):
                Wj = Wc[j]
                pre_idx_clip = pre_idx[pre_idx < Wj.size]
                post_idx_clip = post_idx[post_idx < Wj.size]
                if pre_idx_clip.size > 1 and post_idx_clip.size > 1:
                    v0 = _safe_var(Wj[pre_idx_clip])
                    v1 = _safe_var(Wj[post_idx_clip])
                    m0 = _mad_normalized(Wj[pre_idx_clip])
                    m1 = _mad_normalized(Wj[post_idx_clip])
                    features[f"wav_{fam}_L{j}_var_logratio"] = float(np.log(v1 / v0))
                    features[f"wav_{fam}_L{j}_mad_logratio"] = float(np.log(m1 / m0))

    # 3. Cross-scale summaries
    all_max = [features[f"j{j}_local_max"] for j in range(1, actual_J + 1) if f"j{j}_local_max" in features]
    all_exceed = [features[f"j{j}_exceed_count"] for j in range(1, actual_J + 1) if f"j{j}_exceed_count" in features]
    all_energy = [features[f"j{j}_energy"] for j in range(1, actual_J + 1) if f"j{j}_energy" in features]
    
    features["S_local_max_over_j"] = float(np.max(all_max)) if all_max else 0.0
    features["cnt_local_sum_over_j"] = float(np.sum(all_exceed)) if all_exceed else 0.0
    features["energy_sum_over_j"] = float(np.sum(all_energy)) if all_energy else 0.0
    
    # 4. Segment contrasts (pre vs post)
    if len(x_pre) > 0 and len(x_post) > 0:
        # Energy ratios
        energy_pre = np.sum(x_pre ** 2)
        energy_post = np.sum(x_post ** 2)
        if energy_pre > 0:
            features["energy_ratio_post_pre"] = float(energy_post / energy_pre)
        
        # Mean and variance ratios
        mean_pre, mean_post = np.mean(x_pre), np.mean(x_post)
        var_pre, var_post = np.var(x_pre), np.var(x_post)
        
        if mean_pre != 0:
            features["mean_ratio_post_pre"] = float(mean_post / mean_pre)
        if var_pre > 0:
            features["var_ratio_post_pre"] = float(var_post / var_pre)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = ks_2samp(x_pre, x_post)
        features["ks_stat_raw"] = float(ks_stat)
        features["ks_p_raw"] = float(ks_p)
        
        # KS test on absolute values
        ks_stat_abs, ks_p_abs = ks_2samp(np.abs(x_pre), np.abs(x_post))
        features["ks_stat_abs"] = float(ks_stat_abs)
        features["ks_p_abs"] = float(ks_p_abs)
    
    # 5. Residual diagnostics
    # Maintain existing arch_lm_p when available (for backward compatibility)
    try:
        arch_p = _arch_lm_pvalue(np.asarray(resid, dtype=float), lags=10)
    except Exception:
        arch_p = np.nan
    features["arch_lm_p"] = float(arch_p) if arch_p == arch_p else np.nan

    # If residual-first path used, also expose h0_* diagnostics
    if cfg.use_residuals and cfg.null_model is not None and isinstance(h0_meta, dict):
        features.update(h0_meta_to_feats(h0_meta))
    
    # Ljung-Box tests at different lags (ensure lag < len(resid))
    n_resid = len(resid)
    for lag in [5, 10, 15]:
        actual_lag = min(lag, n_resid - 1)
        if actual_lag > 0:
            lb_p = _ljungbox_pvalue(resid, lags=actual_lag)
            features[f"ljungbox_p_ac_{lag}"] = float(lb_p)
        else:
            features[f"ljungbox_p_ac_{lag}"] = 1.0
    
    # Ljung-Box on squared residuals
    for lag in [5, 10, 15]:
        actual_lag = min(lag, n_resid - 1)
        if actual_lag > 0:
            lb_p_sq = _ljungbox_pvalue(resid ** 2, lags=actual_lag)
            features[f"ljungbox_p_ac2_{lag}"] = float(lb_p_sq)
        else:
            features[f"ljungbox_p_ac2_{lag}"] = 1.0
    
    # 6. Energy ratio L2 norm (cross-scale)
    if len(all_energy) > 1:
        energy_norm = np.linalg.norm(all_energy)
        features["log_energy_ratio_l2norm_over_j"] = float(np.log(energy_norm + 1e-12))
    
    return features


def _default_features() -> Dict[str, float]:
    """Return default features when data is insufficient."""
    features = {}
    
    # Default values for all expected features
    for j in range(1, 4):  # J=3
        features[f"j{j}_local_max"] = 0.0
        features[f"j{j}_exceeds_thresh"] = 0.0
        features[f"j{j}_exceed_count"] = 0.0
        features[f"j{j}_exceed_frac"] = 0.0
        features[f"j{j}_energy"] = 0.0
        features[f"j{j}_energy_norm"] = 0.0
    
    features.update({
        "S_local_max_over_j": 0.0,
        "cnt_local_sum_over_j": 0.0,
        "energy_sum_over_j": 0.0,
        "energy_ratio_post_pre": 1.0,
        "mean_ratio_post_pre": 1.0,
        "var_ratio_post_pre": 1.0,
        "ks_stat_raw": 0.0,
        "ks_p_raw": 1.0,
        "ks_stat_abs": 0.0,
        "ks_p_abs": 1.0,
        "arch_lm_p": 1.0,
        "log_energy_ratio_l2norm_over_j": 0.0
    })
    
    for lag in [5, 10, 15]:
        features[f"ljungbox_p_ac_{lag}"] = 1.0
        features[f"ljungbox_p_ac2_{lag}"] = 1.0
    
    return features


# =============================
# Main Entry Point
# =============================
def extract_wavelet_predictors(values: np.ndarray, periods: np.ndarray,
                               thr_cache: Optional[ThresholdCache] = None,
                               cfg: WaveletConfig = DEFAULT_CFG) -> Dict[str, float]:
    """Extract structural break predictors using MODWT analysis.

    Args:
        values: Time series values
        periods: Period indicators (0=pre, 1=post)
        thr_cache: Optional threshold cache for efficiency
        cfg: Wavelet configuration
        
    Returns:
        Dict[str, float]: rich set of boundary-aware, calibrated features per series.
    """
    features = _wavelet_features_for_series(values, periods, cfg, thr_cache)
    
    # Compatibility layer: add derived summary indicators if caller expects them
    features["p_wavelet_break"] = compute_break_strength_from_modw(features)
    features["confidence"] = compute_wavelet_confidence_from_modw(features)
    
    return features


# =============================
# Compatibility Functions
# =============================
def compute_break_strength_from_modw(features: Dict[str, float]) -> float:
    """Compute break strength from MODW features.
    
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
    """Compute confidence score from MODW features.
    
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