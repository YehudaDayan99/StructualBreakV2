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

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import json
import math
import os

import numpy as np

# Stats & diagnostics
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
    wavelet: str = "la8"           # LA(8) as in paper
    J: int = 3                     # number of levels
    alpha: float = 0.05            # exceedance level for thresholds
    boundary_win_frac: float = 0.01
    boundary_win_min: int = 10
    length_bucket: int = 100       # bucket size for MC cache by n
    mc_reps: int = 400             # increase for higher-fidelity thresholds
    use_mc_thresholds: bool = True # if False, fallback to universal thresholds
    random_state: int = 42


DEFAULT_CFG = WaveletConfig()
_rng = np.random.default_rng(DEFAULT_CFG.random_state)


# =============================
# Threshold cache (JSON-backed)
# =============================
class ThresholdCache:
    """Simple JSON-backed cache for MC thresholds.

    Keys are built from (n_bucket, J, law, wavelet, alpha). Values are {j: q_{j,alpha}}.
    If path is None, store in-memory only.
    """
    def __init__(self, path: Optional[str] = "threshold_cache.json") -> None:
        self.path = path
        self.store: Dict[str, Dict[str, float]] = {}
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.store = json.load(f)
            except Exception:
                self.store = {}

    def key(self, n_bucket: int, J: int, law: str, wavelet: str, alpha: float) -> str:
        return f"n{n_bucket}_J{J}_law{law}_w{wavelet}_a{alpha}"

    def get(self, key: str) -> Optional[Dict[int, float]]:
        raw = self.store.get(key)
        if raw is None:
            return None
        # keys may be str in JSON, cast to int
        return {int(k): float(v) for k, v in raw.items()}

    def set(self, key: str, value: Dict[int, float]) -> None:
        self.store[key] = {str(k): float(v) for k, v in value.items()}
        if self.path:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.store, f)
            os.replace(tmp, self.path)


# =============================
# Helpers
# =============================

def _first_boundary_index(periods: np.ndarray) -> int:
    d = np.diff(periods.astype(int))
    idx = np.flatnonzero(d != 0)
    return int(idx[0] + 1) if len(idx) else int(len(periods) // 2)


def _make_stationary(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.diff(np.log(x)) if np.all(x > 0) else np.diff(x)


def _standardize(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    m = float(np.mean(z))
    s = float(np.std(z))
    s = s if s > eps else eps
    return (z - m) / s


def _arma_resid(x: np.ndarray) -> np.ndarray:
    res = ARIMA(x, order=(1, 0, 1)).fit(method="statespace", disp=0)
    return _standardize(res.resid)


def _arch_lm_pvalue(resid: np.ndarray, lags: int = 10) -> float:
    # returns (lm_stat, lm_pvalue, f_stat, f_pvalue)
    return float(het_arch(resid, nlags=lags)[1])


def _garch_t_resid(x: np.ndarray) -> np.ndarray:
    if not HAVE_ARCH:
        return _arma_resid(x)
    am = arch_model(x, mean="ARX", lags=1, vol="GARCH", p=1, q=1, dist="StudentsT")
    r = am.fit(disp="off")
    resid = r.std_resid  # standardized residuals
    return _standardize(np.asarray(resid, dtype=float))


def _modwt_coeffs(resid: np.ndarray, wavelet: str, J: int) -> Dict[int, np.ndarray]:
    # Use MODWT if available, else SWT (stationary) as a fallback
    if hasattr(pywt, "modwt"):
        Ws = pywt.modwt(resid, wavelet=wavelet, level=J, mode="periodization")
        return {j + 1: np.asarray(Ws[j], dtype=float) for j in range(len(Ws))}
    swt = pywt.swt(resid, wavelet=wavelet, level=J, trim_approx=True, norm=True)
    return {j + 1: np.asarray(swt[j][1], dtype=float) for j in range(len(swt))}


def _simulate_thresholds(n: int, J: int, wavelet: str, alpha: float, law: str,
                         reps: int, rng: np.random.Generator) -> Dict[int, float]:
    """Monte-Carlo q_{j,alpha}(n) for max |W_{j,t}| under i.i.d. noise."""
    maxes = {j: [] for j in range(1, J + 1)}
    for _ in range(int(reps)):
        if law == "t":
            z = rng.standard_t(df=8, size=n).astype(float)
            z /= (np.std(z) + 1e-12)
        else:
            z = rng.standard_normal(size=n).astype(float)
        W = _modwt_coeffs(z, wavelet, J)
        for j in range(1, J + 1):
            maxes[j].append(float(np.max(np.abs(W[j]))))
    return {j: float(np.quantile(maxes[j], 1.0 - alpha)) for j in range(1, J + 1)}


def _universal_thresholds(n: int, J: int) -> Dict[int, float]:
    # Conservative fallback (same threshold across levels)
    thr = math.sqrt(2.0 * math.log(max(n, 2)))
    return {j: thr for j in range(1, J + 1)}


def _ewma_vol(x: np.ndarray, lam: float = 0.94) -> np.ndarray:
    s2 = 0.0
    out = np.empty_like(x, dtype=float)
    for i, xi in enumerate(x):
        s2 = lam * s2 + (1.0 - lam) * float(xi) * float(xi)
        out[i] = math.sqrt(max(s2, 1e-12))
    return out


def _residual_diagnostics(resid: np.ndarray) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    feats["resid_kurtosis"] = float(kurtosis(resid, fisher=True, bias=False))
    feats["resid_skewness"] = float(skew(resid, bias=False))
    for lag in (10, 20):
        lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
        feats[f"ljungbox_p_ac_{lag}"] = float(lb["lb_pvalue"].iloc[0])
        lb2 = acorr_ljungbox(resid ** 2, lags=[lag], return_df=True)
        feats[f"ljungbox_p_ac2_{lag}"] = float(lb2["lb_pvalue"].iloc[0])
    feats["arch_lm_p"] = _arch_lm_pvalue(resid, lags=10)
    return feats


def _simple_segment_shifts(x_pre: np.ndarray, per0: np.ndarray, per1: np.ndarray) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    x0, x1 = x_pre[per0], x_pre[per1]
    feats["mean_diff"] = float(np.mean(x1) - np.mean(x0)) if (len(x0) and len(x1)) else 0.0
    v0, v1 = float(np.var(x0)) if len(x0) else 0.0, float(np.var(x1)) if len(x1) else 0.0
    feats["log_var_ratio"] = float(math.log((v1 + 1e-12) / (v0 + 1e-12)))

    def _acf1(z: np.ndarray) -> float:
        return 0.0 if len(z) < 2 else float(np.corrcoef(z[:-1], z[1:])[0, 1])

    feats["acf1_diff"] = _acf1(x1) - _acf1(x0)
    feats["median_diff"] = float(np.median(x1) - np.median(x0)) if (len(x0) and len(x1)) else 0.0
    mad = lambda z: 1.4826 * np.median(np.abs(z - np.median(z))) if len(z) else 0.0
    feats["log_mad_ratio"] = float(math.log((mad(x1) + 1e-12) / (mad(x0) + 1e-12)))

    # KS p-values
    try:
        feats["ks_p_raw"] = float(ks_2samp(x0, x1).pvalue) if (len(x0) > 1 and len(x1) > 1) else 1.0
        feats["ks_p_abs"] = float(ks_2samp(np.abs(x0), np.abs(x1)).pvalue) if (len(x0) > 1 and len(x1) > 1) else 1.0
    except Exception:
        feats["ks_p_raw"], feats["ks_p_abs"] = 1.0, 1.0

    # EWMA volatility ratio (log)
    if len(x0) and len(x1):
        v0_ew = float(np.median(_ewma_vol(x0)))
        v1_ew = float(np.median(_ewma_vol(x1)))
    else:
        v0_ew, v1_ew = 0.0, 0.0
    feats["ewma_vol_log_ratio"] = float(math.log((v1_ew + 1e-12) / (v0_ew + 1e-12)))

    return feats


def _modwt_feature_block(W: Dict[int, np.ndarray], B_resid: int, w: int,
                         per0_idx: np.ndarray, per1_idx: np.ndarray,
                         thresholds: Dict[int, float], common_thr: float) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    # Boundary window within residual index
    lo = max(B_resid - 1 - w, 0)
    length = len(next(iter(W.values())))
    hi = min(B_resid - 1 + w + 1, length)
    sl = slice(lo, hi)

    energy_log_ratios = []
    for j, wj in W.items():
        wj = np.asarray(wj, dtype=float)
        s_local = float(np.max(np.abs(wj[sl]))) if hi > lo else float(np.max(np.abs(wj)))
        cnt = int(np.sum(np.abs(wj[sl]) > thresholds[j]))
        cnt_common = int(np.sum(np.abs(wj[sl]) > common_thr))

        e0 = float(np.mean(wj[per0_idx] ** 2)) if np.any(per0_idx) else 0.0
        e1 = float(np.mean(wj[per1_idx] ** 2)) if np.any(per1_idx) else 0.0
        ler = float(math.log((e1 + 1e-12) / (e0 + 1e-12)))

        mu0 = float(np.mean(wj[per0_idx])) if np.any(per0_idx) else 0.0
        mu1 = float(np.mean(wj[per1_idx])) if np.any(per1_idx) else 0.0
        sd0 = float(np.std(wj[per0_idx])) if np.any(per0_idx) else 1e-12
        sd1 = float(np.std(wj[per1_idx])) if np.any(per1_idx) else 1e-12

        ks_p = float(ks_2samp(wj[per0_idx], wj[per1_idx]).pvalue) if (np.any(per0_idx) and np.any(per1_idx)) else 1.0
        local_e_frac = float(np.sum(wj[sl] ** 2) / (np.sum(wj ** 2) + 1e-12))

        feats[f"j{j}_S_local"] = s_local
        feats[f"j{j}_cnt_local"] = cnt
        feats[f"j{j}_cnt_common"] = cnt_common
        feats[f"j{j}_log_energy_ratio"] = ler
        feats[f"j{j}_d_mean"] = mu1 - mu0
        feats[f"j{j}_log_sd_ratio"] = float(math.log((sd1 + 1e-12) / (sd0 + 1e-12)))
        feats[f"j{j}_ks_p"] = ks_p
        feats[f"j{j}_local_energy_frac"] = local_e_frac

        energy_log_ratios.append(ler)

    # Cross-scale summaries
    feats["S_local_max_over_j"] = float(np.max([feats[f"j{j}_S_local"] for j in W.keys()]))
    feats["cnt_local_sum_over_j"] = int(np.sum([feats[f"j{j}_cnt_local"] for j in W.keys()]))
    feats["cnt_common_sum_over_j"] = int(np.sum([feats[f"j{j}_cnt_common"] for j in W.keys()]))
    feats["log_energy_ratio_l2norm_over_j"] = float(np.sqrt(np.sum(np.square(energy_log_ratios))))

    return feats


# =============================
# Main per-series builder
# =============================

def _wavelet_features_for_series(values: np.ndarray, periods: np.ndarray,
                                 cfg: WaveletConfig, thr_cache: Optional[ThresholdCache]) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    per = np.asarray(periods, dtype=int)
    n = len(x)
    B = _first_boundary_index(per)

    # Preprocess to (log-)diffed series; align period masks due to diff length -= 1
    x_pre = _make_stationary(x)
    per0 = per[:-1] == 0
    per1 = per[:-1] == 1

    # Residual model under H0
    arma_resid = _arma_resid(x_pre)
    p_arch = _arch_lm_pvalue(arma_resid, lags=10)
    if p_arch < 0.05:
        resid = _garch_t_resid(x_pre)
        law = "t"
    else:
        resid = arma_resid
        law = "normal"
    resid = _standardize(resid)

    # MODWT/SWT coefficients
    W = _modwt_coeffs(resid, wavelet=cfg.wavelet, J=cfg.J)

    # Thresholds (MC or universal) with bucketing by residual length
    n_resid = len(resid)
    n_bucket = int(round(n_resid / cfg.length_bucket) * cfg.length_bucket)
    cache = thr_cache if thr_cache is not None else ThresholdCache()
    key = cache.key(n_bucket, cfg.J, law, cfg.wavelet, cfg.alpha)
    q_dict = cache.get(key)
    if q_dict is None:
        if cfg.use_mc_thresholds:
            q_dict = _simulate_thresholds(n_bucket, cfg.J, cfg.wavelet, cfg.alpha, law, cfg.mc_reps, _rng)
        else:
            q_dict = _universal_thresholds(n_bucket, cfg.J)
        cache.set(key, q_dict)

    q_common = q_dict[max(W.keys())]

    # Boundary window half-width
    w = max(cfg.boundary_win_min, int(cfg.boundary_win_frac * n_resid))

    feats: Dict[str, float] = {}
    # Global residual diagnostics & simple shifts on preprocessed series
    feats.update(_residual_diagnostics(resid))
    feats.update(_simple_segment_shifts(x_pre, per0, per1))

    # MODWT feature block
    feats.update(_modwt_feature_block(W, B_resid=B, w=w, per0_idx=per0, per1_idx=per1,
                                     thresholds=q_dict, common_thr=float(q_common)))

    # Meta
    feats["n_obs"] = int(n)
    feats["len_period0"] = int(B)
    feats["len_period1"] = int(n - B)

    return feats


# =============================
# Public API (drop-in replacement)
# =============================

def extract_wavelet_predictors(values: np.ndarray, periods: np.ndarray,
                               thr_cache: Optional[ThresholdCache] = None,
                               cfg: WaveletConfig = DEFAULT_CFG) -> Dict[str, float]:
    """Extract structural break predictors using MODWT analysis.

    Args:
        values: 1D array of series values.
        periods: 1D array of 0/1 period labels with one 0→1 boundary.
        thr_cache: Optional ThresholdCache (JSON-backed by default if None).
        cfg: WaveletConfig with wavelet/levels/alpha/MC settings.

    Returns:
        Dict[str, float]: rich set of boundary-aware, calibrated features per series.
    """
    features = _wavelet_features_for_series(values, periods, cfg, thr_cache)

    # Compatibility layer: add derived summary indicators if caller expects them
    features["p_wavelet_break"] = compute_break_strength_from_modw(features)
    features["confidence"] = compute_wavelet_confidence_from_modw(features)

    return features


# =============================
# Derived indicators (compatibility)
# =============================

def compute_break_strength_from_modw(features: Dict[str, float]) -> float:
    """Heuristic 0–1 strength combining localized spikes, counts, and energy shifts."""
    parts = []
    s_max = features.get("S_local_max_over_j")
    if s_max is not None:
        parts.append(min(float(s_max) / 3.0, 1.0))
    cnt = features.get("cnt_local_sum_over_j")
    if cnt is not None:
        parts.append(min(float(cnt) / 10.0, 1.0))
    enr = features.get("log_energy_ratio_l2norm_over_j")
    if enr is not None:
        parts.append(min(float(enr) / 2.0, 1.0))
    ks_vals = [1.0 - float(v) for k, v in features.items() if k.startswith("j") and k.endswith("_ks_p")]
    if ks_vals:
        parts.append(float(np.nanmean(ks_vals)))
    return float(np.nanmean(parts)) if parts else 0.5


def compute_wavelet_confidence_from_modw(features: Dict[str, float]) -> float:
    """Heuristic 0–1 confidence combining strength + residual tests.
    (These are *not* used in the paper; keep for backward compatibility.)
    """
    parts = [compute_break_strength_from_modw(features)]
    arch_p = features.get("arch_lm_p")
    if arch_p is not None:
        parts.append(1.0 - float(arch_p))
    lb = [1.0 - float(v) for k, v in features.items() if "ljungbox_p" in k]
    if lb:
        parts.append(float(np.nanmean(lb)))
    for k in ("ks_p_raw", "ks_p_abs"):
        if k in features:
            parts.append(1.0 - float(features[k]))
    return float(np.nanmean(parts)) if parts else 0.5


# =============================
# Compatibility helpers (legacy)
# =============================

def analyze_frequency_bands(wavelet_coeffs: Dict[str, np.ndarray], periods: np.ndarray) -> Dict[str, float]:
    """Legacy placeholder retained for API compatibility. Prefer MODWT features above.
    Computes crude band energy shares if given a dict of coeff arrays with keys including
    'detail_1', 'detail_2', 'detail_3', 'detail_4', 'approx', etc.
    """
    features: Dict[str, float] = {}
    low_energy = 0.0
    high_energy = 0.0
    for key, coeffs in wavelet_coeffs.items():
        energy = float(np.sum(np.asarray(coeffs, dtype=float) ** 2))
        if ("detail_1" in key) or ("detail_2" in key):
            high_energy += energy
        elif ("approx" in key) or ("detail_3" in key) or ("detail_4" in key):
            low_energy += energy
    total = low_energy + high_energy
    if total <= 0:
        features.update({"frequency_energy_ratio": 0.5, "low_freq_energy": 0.5, "high_freq_energy": 0.5})
    else:
        features.update({
            "frequency_energy_ratio": high_energy / total,
            "low_freq_energy": low_energy / total,
            "high_freq_energy": high_energy / total,
        })
    return features


def compute_wavelet_variance_ratio(wavelet_coeffs: Dict[str, np.ndarray], periods: np.ndarray) -> float:
    """Legacy placeholder retained for API compatibility.
    Attempts a crude variance ratio from concatenated coeff halves.
    """
    period_0_coeffs = []
    period_1_coeffs = []
    for _, coeffs in wavelet_coeffs.items():
        coeffs = np.asarray(coeffs, dtype=float)
        mid = len(coeffs) // 2
        period_0_coeffs.extend(coeffs[:mid])
        period_1_coeffs.extend(coeffs[mid:])
    if len(period_0_coeffs) and len(period_1_coeffs):
        v0 = float(np.var(period_0_coeffs))
        v1 = float(np.var(period_1_coeffs))
        if v0 > 0:
            return float(min(v1 / v0, 2.0) / 2.0)
    return 0.5
