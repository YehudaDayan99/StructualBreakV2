"""
TSFM Feature Extraction (Project integration)
 - Placeholder forecaster by default (no external deps)
 - Optional TimesFM engine (graceful fallback)
Implements Blocks A–D, with optional E when quantiles available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable, List

import numpy as np
import pandas as pd

# Allow standalone loading (e.g., Colab with sys.path set to .../methods)
try:  # preferred relative import when part of package
    from .config import TSFMConfig
except Exception:  # fallback absolute import when loaded standalone
    from TSFM.config import TSFMConfig  # type: ignore


# -----------------------------
# Optional dependencies
# -----------------------------
try:
    from scipy import stats  # for Block D
except Exception:  # pragma: no cover
    stats = None


# -----------------------------
# Forecast engines
# -----------------------------

class _PlaceholderForecaster:
    """Simple baseline forecaster used when TimesFM is not installed.
    - pick_best_ctx: chooses from a small grid by backtesting MAE on x0 tail
    - forecast: last-value or seasonal-naive depending on detected season
    """

    def __init__(self, max_context: int, max_horizon: int):
        self.max_context = max_context
        self.max_horizon = max_horizon

    def _robust_mad(self, x: np.ndarray) -> float:
        return float(np.median(np.abs(x - np.median(x)))) + 1e-8

    def _seasonal_naive(self, x: np.ndarray, H: int) -> np.ndarray:
        for s in (7, 12, 24, 30, 52):
            if 2 * s <= len(x):
                return np.tile(x[-s:], int(np.ceil(H / s)))[:H]
        return np.repeat(x[-1], H)

    def pick_best_ctx(self, x0: np.ndarray, H: int, grid: Tuple[int, ...] = (128, 256, 512)) -> int:
        Hs = max(8, min(H, int(0.25 * len(x0))))
        mad = self._robust_mad(x0)
        bestL, best_err = None, np.inf
        for L in grid:
            if L > len(x0):
                continue
            ctx = x0[-L:]
            yhat = self._seasonal_naive(ctx, Hs)
            ytrue = x0[-Hs:]
            err = float(np.mean(np.abs(yhat - ytrue)) / mad)
            if err < best_err:
                best_err, bestL = err, L
        return bestL or min(len(x0), max(grid))

    def forecast(self, x0: np.ndarray, H: int, L_star: Optional[int] = None,
                 return_quantiles: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        L = L_star or self.pick_best_ctx(x0, H)
        ctx = x0[-L:]
        point = self._seasonal_naive(ctx, H).astype(float)
        return point, None

    def forecast_batch(
        self,
        contexts: List[np.ndarray],
        H: int,
        return_quantiles: bool = False,
        use_mixed_precision: bool = True,
        amp_dtype: str = "bfloat16",
    ) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
        points: List[np.ndarray] = []
        for ctx in contexts:
            points.append(self._seasonal_naive(np.asarray(ctx).reshape(-1), H).astype(float))
        return points, None


class _TimesFMForecaster:
    """Thin wrapper for TimesFM (optional)."""

    def __init__(self, max_context: int, max_horizon: int, use_quantiles: bool):
        try:
            import timesfm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("TimesFM not available. Install per repo and pip install -e .") from e
        Model = getattr(timesfm, "TimesFM_2p5_200M_torch", None)
        ForecastConfig = getattr(timesfm, "ForecastConfig", None)
        if Model is None or ForecastConfig is None:
            raise RuntimeError("TimesFM API not found. Please pin a compatible commit.")
        self.max_horizon = int(max_horizon)
        self.model = Model()
        self.model.load_checkpoint()
        self.model.compile(
            ForecastConfig(
                max_context=max_context,
                max_horizon=max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=use_quantiles,
                fix_quantile_crossing=True,
            )
        )

    def pick_best_ctx(self, x0: np.ndarray, H: int, grid: Tuple[int, ...] = (128, 256, 512, 1024)) -> int:
        Hs = max(8, min(int(H), int(self.max_horizon), int(0.25 * len(x0))))
        mad = float(np.median(np.abs(x0 - np.median(x0)))) + 1e-8
        bestL, best_err = None, np.inf
        for L in grid:
            if L > len(x0):
                continue
            ctx = x0[-L:].astype(np.float32)
            pf, _ = self.model.forecast(horizon=Hs, inputs=[ctx])
            yhat = np.asarray(pf[0], dtype=np.float32)
            ytrue = x0[-Hs:]
            err = float(np.mean(np.abs(yhat - ytrue)) / mad)
            if err < best_err:
                best_err, bestL = err, L
        return bestL or min(len(x0), max(grid))

    def forecast(self, x0: np.ndarray, H: int, L_star: Optional[int] = None,
                 return_quantiles: bool = True,
                 use_mixed_precision: bool = True,
                 amp_dtype: str = "bfloat16") -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Cap horizon to avoid ValueError when H > compiled max_horizon
        H_use = int(min(int(H), self.max_horizon))
        L = L_star or self.pick_best_ctx(x0, H_use)
        ctx = x0[-L:].astype(np.float32)
        # Optional mixed precision + inference mode (CUDA only)
        pf = qf = None
        try:
            import torch  # type: ignore
            if use_mixed_precision and torch.cuda.is_available():
                dtype = torch.bfloat16 if amp_dtype.lower() == "bfloat16" else torch.float16
                with torch.inference_mode():
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        pf, qf = self.model.forecast(horizon=H_use, inputs=[ctx])
            else:
                pf, qf = self.model.forecast(horizon=H_use, inputs=[ctx])
        except Exception:
            # Fallback to standard path if torch not available or autocast fails
            pf, qf = self.model.forecast(horizon=H_use, inputs=[ctx])
        point = np.asarray(pf[0], dtype=np.float32)
        quants = np.asarray(qf[0]) if (return_quantiles and qf is not None) else None
        return point, quants

    def forecast_batch(
        self,
        contexts: List[np.ndarray],
        H: int,
        return_quantiles: bool = True,
        use_mixed_precision: bool = True,
        amp_dtype: str = "bfloat16",
    ) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
        """Batch forecast for a list of contexts with the same horizon H.
        Returns lists aligned to inputs. Quantiles optional.
        """
        H_use = int(min(int(H), self.max_horizon))
        inputs = [np.asarray(c, dtype=np.float32).reshape(-1) for c in contexts]
        pf = qf = None
        try:
            import torch  # type: ignore
            if use_mixed_precision and torch.cuda.is_available():
                dtype = torch.bfloat16 if amp_dtype.lower() == "bfloat16" else torch.float16
                with torch.inference_mode():
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        pf, qf = self.model.forecast(horizon=H_use, inputs=inputs)
            else:
                pf, qf = self.model.forecast(horizon=H_use, inputs=inputs)
        except Exception:
            pf, qf = self.model.forecast(horizon=H_use, inputs=inputs)
        points = [np.asarray(p, dtype=np.float32) for p in pf]
        quants = [np.asarray(q) for q in qf] if (return_quantiles and qf is not None) else None
        return points, quants


def _robust_scales(x0: np.ndarray) -> Tuple[float, float]:
    mad = float(np.median(np.abs(x0 - np.median(x0)))) + 1e-8
    rms = float(np.sqrt(np.mean((x0 - np.mean(x0)) ** 2))) + 1e-8
    return mad, rms


def _pct_index(H: int, p: int) -> int:
    return max(1, int(np.ceil(H * p / 100)))


def _best_xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 8) -> Tuple[int, float]:
    best_lag, best_corr = 0, -1.0
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            c = np.corrcoef(a[L:], b[: len(b) - L])[0, 1]
        else:
            c = np.corrcoef(a[: len(a) + L], b[-L:])[0, 1]
        c = float(0.0 if np.isnan(c) else c)
        if c > best_corr:
            best_lag, best_corr = L, c
    return best_lag, best_corr


def _neglogp(p: float, cap: float = 12.0) -> float:
    return float(np.clip(-np.log10(max(p, 1e-300)), 0, cap))


def _block_A(yhat: np.ndarray, x1: np.ndarray, x0: np.ndarray) -> Dict[str, float]:
    H = len(x1)
    mad, rms = _robust_scales(x0)
    abs_err = np.abs(yhat - x1)
    feats: Dict[str, float] = {
        "mae_full": float(abs_err.mean() / mad),
        "rmse_full": float(np.sqrt(np.mean((yhat - x1) ** 2)) / rms),
        "mae_early": float(abs_err[: _pct_index(H, 10)].mean() / mad),
        "mae_late": float(abs_err[-_pct_index(H, 10) :].mean() / mad),
    }
    feats["err_ratio_early_late"] = float(feats["mae_early"] / (feats["mae_late"] + 1e-12))
    feats["err_slope"] = float(feats["mae_early"] - feats["mae_late"])
    for p in (10, 25, 50, 75, 90):
        feats[f"mae_p{p}"] = float(abs_err[: _pct_index(H, p)].mean() / mad)
    return feats


def _block_B(yhat: np.ndarray, x1: np.ndarray, x0: np.ndarray) -> Dict[str, float]:
    mad, _ = _robust_scales(x0)
    H = len(x1)
    lv = np.repeat(x0[-1], H)
    # seasonal naive baseline
    sn = None
    for s in (7, 12, 24, 30, 52):
        if 2 * s <= len(x0):
            sn = np.tile(x0[-s:], int(np.ceil(H / s)))[:H]
            break
    if sn is None:
        sn = np.repeat(x0[-1], H)
    corr = float(np.nan_to_num(np.corrcoef(yhat, x1)[0, 1]))
    lag, corr_lag = _best_xcorr_lag(yhat, x1, 8)
    amp_ratio = float((np.std(x1) + 1e-8) / (np.std(yhat) + 1e-8))
    mae_t = float(np.mean(np.abs(yhat - x1)) / mad)
    mae_lv = float(np.mean(np.abs(lv - x1)) / mad)
    mae_sn = float(np.mean(np.abs(sn - x1)) / mad)
    return {
        "corr_full": corr,
        "xcorr_best_lag": float(lag),
        "xcorr_best_corr": float(corr_lag),
        "amp_ratio": amp_ratio,
        "mae_ratio_vs_lastvalue": float(mae_t / (mae_lv + 1e-12)),
        "mae_ratio_vs_seasonal": float(mae_t / (mae_sn + 1e-12)),
    }


def _block_C(yhat_p0: np.ndarray, x0_tail: np.ndarray, yhat_p1: np.ndarray, x1: np.ndarray) -> Dict[str, float]:
    try:
        from sklearn.linear_model import Ridge
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Install scikit-learn for TSFM Block C: pip install scikit-learn") from e
    reg = Ridge(alpha=1e-2)
    reg.fit(yhat_p0.reshape(-1, 1), x0_tail)
    ycal = reg.predict(yhat_p1.reshape(-1, 1))
    mad, _ = _robust_scales(x0_tail)
    mae_raw = float(np.mean(np.abs(yhat_p1 - x1)) / mad)
    mae_cal = float(np.mean(np.abs(ycal - x1)) / mad)
    return {
        "calib_coef": float(np.clip(reg.coef_[0], -10, 10)),
        "calib_intercept": float(np.clip(reg.intercept_, -1e3, 1e3)),
        "calib_delta_err": float((mae_raw - mae_cal) / (mae_raw + 1e-12)),
        "mean_diff_norm": float((x1.mean() - yhat_p1.mean()) / (mad + 1e-12)),
        "var_ratio_post_pre": float(np.var(x1) / (np.var(x0_tail) + 1e-12)),
    }


def _block_D(res_p0: np.ndarray, res_p1: np.ndarray) -> Dict[str, float]:
    if stats is None:  # pragma: no cover
        raise RuntimeError("Install scipy for TSFM Block D: pip install scipy")
    feats: Dict[str, float] = {}
    _, p = stats.ttest_ind(res_p0, res_p1, equal_var=False, nan_policy="omit")
    feats["nlp_t_welch"] = _neglogp(float(p))
    try:
        _, p = stats.mannwhitneyu(res_p0, res_p1, alternative="two-sided")
        feats["nlp_mwu"] = _neglogp(float(p))
    except Exception:
        feats["nlp_mwu"] = 0.0
    _, p = stats.levene(res_p0, res_p1, center="median")
    feats["nlp_levene"] = _neglogp(float(p))
    _, p = stats.ks_2samp(res_p0, res_p1, alternative="two-sided", mode="asymp")
    feats["nlp_ks"] = _neglogp(float(p))
    return feats


def _extract_quantiles(qf: Optional[np.ndarray]) -> Dict[float, np.ndarray]:
    if qf is None:
        return {}
    # Heuristic mapping; adjust if your TimesFM build differs
    if qf.ndim == 2 and qf.shape[1] >= 10:
        return {0.10: qf[:, 1], 0.25: qf[:, 3], 0.50: qf[:, 5], 0.75: qf[:, 7], 0.90: qf[:, 9]}
    return {}


def _block_E(qf: Optional[np.ndarray], x1: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    qm = _extract_quantiles(qf)
    if not qm:
        out["quantiles_present"] = 0.0
        return out
    out["quantiles_present"] = 1.0
    iw80 = float(np.mean(qm[0.90] - qm[0.10]))
    iw50 = float(np.mean(qm[0.75] - qm[0.25]))
    out["iw_80"] = iw80
    out["iw_50"] = iw50
    covered80 = float(((x1 >= qm[0.10]) & (x1 <= qm[0.90])).mean())
    covered50 = float(((x1 >= qm[0.25]) & (x1 <= qm[0.75])).mean())
    out["cov_err_80"] = float(0.80 - covered80)
    out["cov_err_50"] = float(0.50 - covered50)
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        e = x1 - qm[q]
        out[f"pinball_q{int(q*100)}"] = float(np.mean(np.maximum(q * e, (q - 1) * e)))
    return out


_GLOBAL_FC_CACHE: dict = {}


def _make_forecaster(cfg: TSFMConfig):
    """Create or reuse a process-global forecaster to avoid reloading models per id."""
    # Cache key excludes mixed precision flags to avoid unnecessary recompiles
    key = (
        cfg.engine,
        int(cfg.max_context),
        int(cfg.max_horizon),
        bool(cfg.use_quantiles),
    )
    fc = _GLOBAL_FC_CACHE.get(key)
    if fc is not None:
        return fc
    if cfg.engine == "timesfm":
        fc = _TimesFMForecaster(cfg.max_context, cfg.max_horizon, cfg.use_quantiles)
    else:
        fc = _PlaceholderForecaster(cfg.max_context, cfg.max_horizon)
    _GLOBAL_FC_CACHE[key] = fc
    return fc


def extract_tsfm_predictors(values: np.ndarray, periods: np.ndarray, cfg: TSFMConfig) -> Dict[str, float]:
    """Extract TSFM-based structural-break features for a single series.

    values: 1D floats; periods: 0/1 array indicating pre/post boundary.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    periods = np.asarray(periods, dtype=int).reshape(-1)
    if values.size != periods.size:
        raise ValueError("values and periods must be the same length")

    x0 = values[periods == 0]
    x1 = values[periods == 1]
    if (x0.size == 0) or (x1.size == 0):
        return {"mae_full": np.nan}  # degenerate case

    H = int(x1.size)
    forecaster = _make_forecaster(cfg)
    # Pick context length on x0 tail, forecast x1
    Lstar = forecaster.pick_best_ctx(x0, H)
    pf, qf = forecaster.forecast(
        x0,
        H,
        L_star=Lstar,
        return_quantiles=cfg.use_quantiles,
        use_mixed_precision=getattr(cfg, "use_mixed_precision", True),
        amp_dtype=getattr(cfg, "amp_dtype", "bfloat16"),
    )

    # Pseudo-horizon on Period-0 tail for calibration/tests
    Hs = max(8, min(H, int(getattr(forecaster, "max_horizon", H)), int(0.25 * len(x0))))
    L2 = min(Lstar, max(1, len(x0) - Hs))
    pf0, _ = forecaster.forecast(
        x0[:-Hs],
        Hs,
        L_star=L2,
        return_quantiles=False,
        use_mixed_precision=getattr(cfg, "use_mixed_precision", True),
        amp_dtype=getattr(cfg, "amp_dtype", "bfloat16"),
    )
    x0_tail = x0[-Hs:]

    # Blocks
    A = _block_A(pf, x1, x0)
    B = _block_B(pf, x1, x0)
    C = _block_C(pf0, x0_tail, pf, x1)
    res_p0 = x0_tail - pf0
    res_p1 = x1 - pf
    D = _block_D(res_p0, res_p1)
    E = _block_E(qf, x1) if cfg.use_quantiles else {}

    out: Dict[str, float] = {"ctx_len": float(Lstar)}
    out.update(A); out.update(B); out.update(C); out.update(D); out.update(E)
    return out


def extract_tsfm_predictors_batch(
    series_list: List[Tuple[np.ndarray, np.ndarray]],
    cfg: TSFMConfig,
) -> List[Dict[str, float]]:
    """Batched variant: compute features for multiple series.
    Each item is (values, periods) arrays.
    Uses batch forecasts per equal horizon to utilize GPU better.
    """
    forecaster = _make_forecaster(cfg)
    # Precompute splits and L* per id
    entries = []  # holds dict per series with computed items
    for idx, (values, periods) in enumerate(series_list):
        v = np.asarray(values, dtype=float).reshape(-1)
        p = np.asarray(periods, dtype=int).reshape(-1)
        x0 = v[p == 0]
        x1 = v[p == 1]
        if (x0.size == 0) or (x1.size == 0):
            entries.append({
                "idx": idx,
                "degenerate": True,
                "x0": x0,
                "x1": x1,
            })
            continue
        H = int(x1.size)
        Lstar = forecaster.pick_best_ctx(x0, H)
        entries.append({
            "idx": idx,
            "degenerate": False,
            "x0": x0,
            "x1": x1,
            "H": H,
            "Lstar": Lstar,
        })

    # Batch forecast for Period-1 by horizon H
    from collections import defaultdict
    by_H: Dict[int, List[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        if not e.get("degenerate", False):
            by_H[int(e["H"])].append(i)
    # For each H, build contexts list
    pf_dict: Dict[int, np.ndarray] = {}
    qf_dict: Dict[int, Optional[np.ndarray]] = {}
    for H, idxs in by_H.items():
        contexts = [entries[i]["x0"][-entries[i]["Lstar"]:] for i in idxs]
        points, quants = None, None
        if hasattr(forecaster, "forecast_batch"):
            points, quants = forecaster.forecast_batch(
                contexts,
                H,
                return_quantiles=cfg.use_quantiles,
                use_mixed_precision=getattr(cfg, "use_mixed_precision", True),
                amp_dtype=getattr(cfg, "amp_dtype", "bfloat16"),
            )
        else:
            points = []
            quants = [] if cfg.use_quantiles else None
            for ctx in contexts:
                p1, q1 = forecaster.forecast(
                    ctx, H, L_star=len(ctx), return_quantiles=cfg.use_quantiles,
                    use_mixed_precision=getattr(cfg, "use_mixed_precision", True),
                    amp_dtype=getattr(cfg, "amp_dtype", "bfloat16"),
                )
                points.append(p1)
                if cfg.use_quantiles and q1 is not None:
                    quants.append(q1)
        for j, i in enumerate(idxs):
            pf_dict[i] = points[j]
            qf_dict[i] = (quants[j] if (cfg.use_quantiles and quants is not None) else None)

    # Pseudo-horizon on Period-0 tail, batched by Hs
    by_Hs: Dict[int, List[int]] = defaultdict(list)
    Hs_map: Dict[int, int] = {}
    L2_map: Dict[int, int] = {}
    for i, e in enumerate(entries):
        if e.get("degenerate", False):
            continue
        x0 = e["x0"]
        H = e["H"]
        Hs = max(8, min(H, int(getattr(forecaster, "max_horizon", H)), int(0.25 * len(x0))))
        L2 = min(e["Lstar"], max(1, len(x0) - Hs))
        Hs_map[i] = Hs
        L2_map[i] = L2
        by_Hs[int(Hs)].append(i)

    pf0_dict: Dict[int, np.ndarray] = {}
    for Hs, idxs in by_Hs.items():
        contexts0 = [entries[i]["x0"][: -Hs][-L2_map[i]:] for i in idxs]
        points0, _ = None, None
        if hasattr(forecaster, "forecast_batch"):
            points0, _ = forecaster.forecast_batch(
                contexts0,
                Hs,
                return_quantiles=False,
                use_mixed_precision=getattr(cfg, "use_mixed_precision", True),
                amp_dtype=getattr(cfg, "amp_dtype", "bfloat16"),
            )
        else:
            points0 = []
            for ctx in contexts0:
                p0, _ = forecaster.forecast(
                    ctx, Hs, L_star=len(ctx), return_quantiles=False,
                    use_mixed_precision=getattr(cfg, "use_mixed_precision", True),
                    amp_dtype=getattr(cfg, "amp_dtype", "bfloat16"),
                )
                points0.append(p0)
        for j, i in enumerate(idxs):
            pf0_dict[i] = points0[j]

    # Assemble features
    out_list: List[Dict[str, float]] = []
    for i, e in enumerate(entries):
        if e.get("degenerate", False):
            out_list.append({"mae_full": np.nan})
            continue
        x0 = e["x0"]; x1 = e["x1"]
        pf = pf_dict[i]
        pf0 = pf0_dict[i]
        # Blocks
        A = _block_A(pf, x1, x0)
        B = _block_B(pf, x1, x0)
        Hs = Hs_map[i]
        x0_tail = x0[-Hs:]
        C = _block_C(pf0, x0_tail, pf, x1)
        res_p0 = x0_tail - pf0
        res_p1 = x1 - pf
        D = _block_D(res_p0, res_p1)
        qf = qf_dict.get(i)
        E = _block_E(qf, x1) if cfg.use_quantiles else {}
        out: Dict[str, float] = {"ctx_len": float(e["Lstar"]) }
        out.update(A); out.update(B); out.update(C); out.update(D); out.update(E)
        out_list.append(out)
    return out_list