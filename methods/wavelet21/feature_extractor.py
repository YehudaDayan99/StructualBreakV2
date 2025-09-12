from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import json, hashlib, warnings, logging

import numpy as np
import pywt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# Optional: Student-t GARCH standardization (fallbacks if missing)
try:
    from arch.univariate import arch_model
    _HAS_ARCH = True
except Exception:
    _HAS_ARCH = False


# =============================
# Config
# =============================
@dataclass
class WaveletConfig:
    # Wavelet families & levels
    wavelets: Tuple[str, ...] = ("sym4", "db8")
    J: int = 3                 # number of SWT/MODWT detail levels (1..J)
    alpha: float = 0.05        # exceedance alpha for local thresholds
    windows: Tuple[int, ...] = (16, 32, 64)  # half-window around boundary

    # Residual (H0) modeling
    use_garch: bool = False
    error_dist: str = "t"      # "t" or "normal"
    nu_default: int = 8        # fallback df

    # Threshold cache (re-uses existing interface)
    cache_path: str = "threshold_cache.json"

    # MC parameters (fast & cached)
    mc_reps: int = 2000
    random_state: int = 12345


DEFAULT_CFG = WaveletConfig()


# =============================
# Threshold cache (compatible)
# Keys stay (n, J, wavelet, alpha) to avoid breaking old files.
# We encode "window length" as n for the local-window thresholds.
# =============================
class ThresholdCache:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._cache: Dict[Tuple[int,int,str,float], Dict[str, float]] = {}
        self._load()

    def _load(self):
        try:
            with open(self.cache_path, "r") as f:
                raw = json.load(f)
            for k_str, v in raw.items():
                # parse tuple-like string keys
                try:
                    # eval safely
                    n,J,w,alpha = eval(k_str, {"__builtins__":{}})
                    if isinstance(n,int) and isinstance(J,int) and isinstance(w,str):
                        self._cache[(n,J,w,float(alpha))] = {kk: float(vv) for kk,vv in v.items()}
                except Exception:
                    continue
        except Exception:
            pass

    def _save(self):
        data = {str(k): v for k, v in self._cache.items()}
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(data, f, indent=2)

    def get(self, n: int, J: int, wavelet: str, alpha: float) -> Optional[Dict[str, float]]:
        return self._cache.get((n, J, wavelet, float(alpha)))

    def set(self, n: int, J: int, wavelet: str, alpha: float, thresholds: Dict[str, float]):
        self._cache[(n, J, wavelet, float(alpha))] = thresholds
        self._save()


# =============================
# Residual (H0) pipeline
# =============================
def _std_residuals_full_series(x: np.ndarray, use_garch: bool, error_dist: str, nu_default: int) -> Tuple[np.ndarray, Dict[str, float]]:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    diag: Dict[str, float] = {}

    # Simple ARMA(1,0,1) on full series; robust standardization
    try:
        res = ARIMA(x, order=(1,0,1)).fit(method="statespace")
        resid = res.resid
    except Exception:
        # fallback AR(1)
        xlag = np.roll(x,1); xlag[0]=0.0
        phi = np.corrcoef(x[1:], xlag[1:])[0,1] if np.std(xlag[1:])>0 else 0.0
        resid = x - phi*xlag
        diag["arma_phi"] = float(phi)

    if _HAS_ARCH and use_garch:
        try:
            dist = "StudentsT" if error_dist.lower().startswith("t") else "normal"
            am = arch_model(resid, vol="GARCH", p=1, q=1, dist=dist, rescale=True)
            r = am.fit(disp="off")
            eps = r.resid / np.where(r.conditional_volatility==0, 1e-8, r.conditional_volatility)
            if hasattr(r, "distribution") and hasattr(r.distribution, "nu"):
                diag["resid_nu"] = float(r.distribution.nu)
            else:
                diag["resid_nu"] = float(nu_default)
            diag["garch_aic"] = float(r.aic); diag["garch_bic"] = float(r.bic)
        except Exception:
            eps = resid / (np.std(resid) or 1.0)
            diag["resid_nu"] = float(nu_default)
    else:
        eps = resid / (np.std(resid) or 1.0)
        diag["resid_nu"] = float(nu_default)

    # simple diagnostics
    try:
        diag["lb_p_resid"] = float(_lb_p(eps, 10))
        diag["lb_p_resid_sq"] = float(_lb_p(eps**2, 10))
        diag["arch_lm_p"] = float(_arch_lm_p(eps, 10))
        diag["kurtosis_resid"] = float(stats.kurtosis(eps, fisher=True, bias=False))
    except Exception:
        pass

    return np.asarray(eps, dtype=float), diag


def _lb_p(arr, lags=10) -> float:
    if len(arr) <= lags:
        return 1.0
    try:
        out = acorr_ljungbox(arr, lags=[lags], return_df=True)
        return float(out["lb_pvalue"].iloc[0])
    except Exception:
        return 1.0

def _arch_lm_p(arr, lags=10) -> float:
    try:
        stat, p, _, _ = het_arch(arr, nlags=lags)
        return float(p)
    except Exception:
        return 1.0


# =============================
# Wavelet helpers
# =============================
def _swt_detail_coeffs(eps: np.ndarray, wavelet: str, J: int) -> Dict[int, np.ndarray]:
    # Stationary Wavelet Transform (MODWT-like, no decimation)
    J_eff = min(J, pywt.swt_max_level(len(eps)))
    coeffs = pywt.swt(eps, wavelet, level=J_eff, trim_approx=True, norm=True)
    return {j: coeffs[j-1][1].astype(float) for j in range(1, J_eff+1)}  # detail coeffs


def _mad(x: np.ndarray) -> float:
    if x.size == 0: return np.nan
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def _find_boundary(period: np.ndarray) -> int:
    # single 0..0,1..1 regime
    d = np.diff(period.astype(int))
    idx = np.where(d != 0)[0]
    return int(idx[0] + 1) if idx.size else int(np.argmax(period))


# =============================
# Local thresholds via MC (cached)
# We store per-level quantiles for max|Wj| using "n = window length"
# so we don't break prior cache schema.
# =============================
def _simulate_local_max_quantiles(m: int, wavelet: str, J: int, alpha: float, dist: str, nu: float, n_sims: int, seed: int) -> Dict[int, float]:
    rng = np.random.default_rng(seed)
    out: Dict[int, float] = {}
    J_eff = min(J, pywt.swt_max_level(m))
    for j in range(1, J_eff+1):
        draws = []
        for _ in range(n_sims):
            x = rng.standard_t(df=nu, size=m) if dist.lower().startswith("t") else rng.standard_normal(size=m)
            dj = _swt_detail_coeffs(x, wavelet, j)
            d = dj[j]
            draws.append(np.max(np.abs(d)))
        out[j] = float(np.quantile(draws, 1.0 - alpha))
    return out


def _get_window_thresholds(cache: ThresholdCache, m: int, wavelet: str, J: int, alpha: float, dist: str, nu: float, n_sims: int, seed: int) -> Dict[int, float]:
    # Encode window length m into 'n' position of the legacy key
    cached = cache.get(m, J, wavelet, alpha)
    if cached is None:
        q = _simulate_local_max_quantiles(m, wavelet, J, alpha, dist, nu, n_sims, seed)
        cache.set(m, J, wavelet, alpha, {f"L{j}": qj for j,qj in q.items()})
        cached = cache.get(m, J, wavelet, alpha)
    # normalize to dict[int,float]
    out = {}
    for j in range(1, min(J, pywt.swt_max_level(m)) + 1):
        out[j] = float(cached.get(f"L{j}", 2.0*np.sqrt(2*np.log(max(m,2)))))
    return out


# =============================
# Main per-id feature computation
# =============================
def extract_wavelet_predictors(values: np.ndarray,
                               periods: np.ndarray,
                               thr_cache: Optional[ThresholdCache] = None,
                               cfg: WaveletConfig = DEFAULT_CFG) -> Dict[str, float]:
    """
    Public entry (kept stable).
    Returns a feature dict. It includes:
      - NEW: period-aware contrasts on SWT detail coeffs per level (energy/var/MAD logratios)
      - NEW: boundary-local maxima & exceedances across windows (w∈{16,32,64})
      - NEW: H0 diagnostics (lb_p, arch_lm_p, kurtosis)
      - NEW: classical pre/post t & F tests on raw series
      - LEGACY: j*-style features (local_max/exceed_count/energy) for backward-compat
    """
    # Ensure arrays
    x = np.asarray(values, dtype=float)
    p = np.asarray(periods, dtype=int)
    feats: Dict[str, float] = {}

    # Guardrails
    if x.size < 40 or (p==0).sum() < 10 or (p==1).sum() < 10:
        return _defaults_legacy()

    # 1) Residual-first standardization under H0
    eps, diag = _std_residuals_full_series(x, cfg.use_garch, cfg.error_dist, cfg.nu_default)
    nu_eff = float(diag.get("resid_nu", cfg.nu_default))
    for k,v in diag.items():
        feats[f"h0_{k}"] = float(v)

    # Pre/Post masks on original periods
    pre_mask = (p==0); post_mask = (p==1)

    # 2) SWT coeffs per family & period-aware contrasts
    B = _find_boundary(p)

    for w in cfg.wavelets:
        dj_all = _swt_detail_coeffs(eps, w, cfg.J)
        for j, d in dj_all.items():
            d_pre, d_post = d[pre_mask], d[post_mask]

            # energy/var/robust scale per segment
            e0 = float(np.mean(d_pre**2)) if d_pre.size else np.nan
            e1 = float(np.mean(d_post**2)) if d_post.size else np.nan
            v0 = float(np.var(d_pre, ddof=1)) if d_pre.size>1 else np.nan
            v1 = float(np.var(d_post, ddof=1)) if d_post.size>1 else np.nan
            s0 = float(_mad(d_pre)) if d_pre.size else np.nan
            s1 = float(_mad(d_post)) if d_post.size else np.nan

            # log-ratios (post vs pre)
            def _lr(a,b):
                a = 1e-12 if (a is None or not np.isfinite(a) or a<=0) else a
                b = 1e-12 if (b is None or not np.isfinite(b) or b<=0) else b
                return float(np.log(b/a))
            feats[f"wav_{w}_L{j}_energy_logratio"] = _lr(e0,e1)
            feats[f"wav_{w}_L{j}_var_logratio"]    = _lr(v0,v1)
            feats[f"wav_{w}_L{j}_mad_logratio"]    = _lr(s0,s1)

            # 3) Boundary-local features across windows
            n = len(d)
            for w_half in cfg.windows:
                lo = max(0, B - w_half)
                hi = min(n, B + w_half + 1)
                win = d[lo:hi]
                feats[f"wav_{w}_L{j}_localmax_w{w_half}"] = float(np.max(np.abs(win))) if win.size else 0.0

                # local exceedances vs calibrated q_{j,α}(m)
                if thr_cache is None:
                    thr_cache = ThresholdCache(cfg.cache_path)
                q = _get_window_thresholds(thr_cache, m=w_half, wavelet=w, J=cfg.J,
                                           alpha=cfg.alpha, dist=cfg.error_dist, nu=nu_eff,
                                           n_sims=cfg.mc_reps, seed=cfg.random_state)[j]
                feats[f"wav_{w}_L{j}_exceed_w{w_half}"] = float(np.sum(np.abs(win) > q))

            # aggregates across windows
            vals = [feats.get(f"wav_{w}_L{j}_localmax_w{wh}", np.nan) for wh in cfg.windows]
            feats[f"wav_{w}_L{j}_localmax_w_mean"] = float(np.nanmean(vals))
            feats[f"wav_{w}_L{j}_localmax_w_max"]  = float(np.nanmax(vals))

    # 4) Classical raw-series tests (cheap & useful)
    x0, x1 = x[pre_mask], x[post_mask]
    if x0.size > 3 and x1.size > 3:
        t_stat, t_p = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit")
        feats["ttest_stat_signed"] = float(t_stat)
        feats["ttest_neglog10p"]   = float(-np.log10(max(t_p, 1e-300)))
        v0, v1 = np.var(x0, ddof=1), np.var(x1, ddof=1)
        if v0 > 0 and v1 > 0:
            F = v1 / v0
            df1, df0 = len(x1)-1, len(x0)-1
            p_f = 2*min(stats.f.cdf(F, df1, df0), 1 - stats.f.cdf(F, df1, df0))
            feats["ftest_ratio"]     = float(F)
            feats["ftest_neglog10p"] = float(-np.log10(max(p_f, 1e-300)))

    # 5) Add a legacy-compatible block (j*-style) so nothing downstream breaks
    #    We compute on the last wavelet in list for determinism.
    w_legacy = cfg.wavelets[-1]
    dj_legacy = _swt_detail_coeffs(eps, w_legacy, cfg.J)
    all_max, all_exceed, all_energy = [], [], []
    for j, d in dj_legacy.items():
        n = len(d)
        # universal-ish threshold for legacy (not used for decisions)
        thr = 2.0 * np.sqrt(2*np.log(max(n,2)))
        exceed = int(np.sum(np.abs(d) > thr))
        e = float(np.sum(d*d))
        feats[f"j{j}_exceed_count"] = float(exceed)
        feats[f"j{j}_exceed_frac"]  = float(exceed / max(n,1))
        feats[f"j{j}_local_max"]    = float(np.max(d)) if n else 0.0
        feats[f"j{j}_energy"]       = e
        feats[f"j{j}_energy_norm"]  = float(e / max(n,1))
        all_max.append(feats[f"j{j}_local_max"]); all_exceed.append(exceed); all_energy.append(e)

    if all_max:
        feats["S_local_max_over_j"]    = float(np.max(all_max))
        feats["cnt_local_sum_over_j"]  = float(np.sum(all_exceed))
        feats["energy_sum_over_j"]     = float(np.sum(all_energy))
        feats["log_energy_ratio_l2norm_over_j"] = float(np.log(np.linalg.norm(all_energy)+1e-12))

    # Derived “compatibility” indicators often expected by downstream code
    feats["p_wavelet_break"] = _compat_break_strength(feats)
    feats["confidence"]      = _compat_confidence(feats)

    return feats

# --- Back-compat exports expected by methods/wavelet21/__init__.py ---

def compute_break_strength_from_modw(features: Dict[str, float]) -> float:
    """Legacy API: alias to our internal break-strength heuristic."""
    return _compat_break_strength(features)

def compute_wavelet_confidence_from_modw(features: Dict[str, float]) -> float:
    """Legacy API: alias to our internal confidence heuristic."""
    return _compat_confidence(features)

# =============================
# Defaults (legacy-safe)
# =============================
def _defaults_legacy() -> Dict[str, float]:
    f = {}
    for j in range(1, 4):
        f[f"j{j}_local_max"] = 0.0
        f[f"j{j}_exceeds_thresh"] = 0.0
        f[f"j{j}_exceed_count"] = 0.0
        f[f"j{j}_exceed_frac"] = 0.0
        f[f"j{j}_energy"] = 0.0
        f[f"j{j}_energy_norm"] = 0.0
    f.update({
        "S_local_max_over_j": 0.0,
        "cnt_local_sum_over_j": 0.0,
        "energy_sum_over_j": 0.0,
        "log_energy_ratio_l2norm_over_j": 0.0,
        "p_wavelet_break": 0.0,
        "confidence": 0.0
    })
    return f


# =============================
# Compatibility “scores”
# =============================
def _compat_break_strength(features: Dict[str, float]) -> float:
    parts = []
    smax = features.get("S_local_max_over_j", 0.0)
    parts.append(min(smax/3.0, 1.0))
    cnt = features.get("cnt_local_sum_over_j", 0.0)
    parts.append(min(cnt/10.0, 1.0))
    enr = features.get("log_energy_ratio_l2norm_over_j", 0.0)
    parts.append(min(enr/2.0, 1.0))
    return float(np.mean(parts)) if parts else 0.0

def _compat_confidence(features: Dict[str, float]) -> float:
    parts = []
    parts.append(_compat_break_strength(features))
    parts.append(1.0 - features.get("h0_arch_lm_p", 1.0))
    for k,v in features.items():
        if "h0_lb_p" in k or "ljungbox_p" in k:
            parts.append(1.0 - float(v))
    return float(np.mean(parts)) if parts else 0.0
