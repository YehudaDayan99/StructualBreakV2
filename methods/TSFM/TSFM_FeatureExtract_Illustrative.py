"""
TSFM Feature Extraction (Illustrative)
--------------------------------------
Goal: show HOW to build TimesFM-based features for a structural-break
classifier (Yes/No) over ~10,000 series (≈30% positives).

This is *illustrative*, not drop-in executable. Adapt names to the
exact TimesFM repo API revision you pin (commit hash + checkpoint tag).

Design choices (why):
- Treat TimesFM as a counterfactual forecaster under H0 (no break).
- Compute *dimensionless* features (normalized by Period-0 scale)
  so they compare across varying units/lengths/distributions.
- Capture mean/level shifts (calib_intercept), variance/scale shifts
  (amp_ratio, calib_coef), and distributional change (tests on residuals).
- Add uncertainty features using the quantile head (if enabled).

Files produced:
- TSFM.csv (one row per id)

Author: (your name)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
import pandas as pd

# Optional deps
try:
    from scipy import stats
except Exception:
    stats = None  # illustrate; ensure scipy installed in your env

# ---------------------------------------------------------------------
# 0) TimesFM wrapper (adapt to the exact repo API you pin)
# ---------------------------------------------------------------------
try:
    import timesfm  # pip install -e . after git clone google-research/timesfm
    MODEL_CLS = getattr(timesfm, "TimesFM_2p5_200M_torch", None)
    ForecastConfig = getattr(timesfm, "ForecastConfig", None)
except Exception:
    timesfm = None
    MODEL_CLS = None
    ForecastConfig = None


@dataclass
class TfmCfg:
    max_context: int = 2048         # raise to 4096/8192/16384 if GPU allows
    max_horizon: int = 1024         # cap by data horizon
    use_quantiles: bool = True      # enable quantile head if available
    enable_bfloat16: bool = False   # try on GPU to save memory


class TSFMForecaster:
    """Thin, defensive wrapper around TimesFM.

    Assumes repo API like:
        model = timesfm.TimesFM_2p5_200M_torch()
        model.load_checkpoint()
        model.compile(timesfm.ForecastConfig(...))
        point_fcst, quant_fcst = model.forecast(horizon=H, inputs=[ctx, ...])
    """
    def __init__(self, cfg: TfmCfg):
        if MODEL_CLS is None or ForecastConfig is None:
            raise RuntimeError(
                "TimesFM not available. Did you `git clone` and `pip install -e .`?"
            )
        self.cfg = cfg
        self.model = MODEL_CLS()
        # Uses the default checkpoint baked into the repo helper.
        self.model.load_checkpoint()
        self.model.compile(
            ForecastConfig(
                max_context=cfg.max_context,
                max_horizon=cfg.max_horizon,
                normalize_inputs=True,  # repo-side normalization
                use_continuous_quantile_head=cfg.use_quantiles,
                fix_quantile_crossing=True,
            )
        )
        # Optional: set dtype/bfloat16 here depending on backend

    def _robust_mad(self, x: np.ndarray) -> float:
        return float(np.median(np.abs(x - np.median(x)))) + 1e-8

    def pick_best_ctx(self, x0: np.ndarray, H: int,
                      grid: Tuple[int, ...] = (128, 256, 512, 1024)) -> int:
        """Mini backtest on the tail of x0 to pick L*.
        Uses MAE normalized by MAD(x0) for comparability.
        """
        Hs = max(8, min(H, int(0.25 * len(x0))))
        mad = self._robust_mad(x0)
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
                 return_quantiles: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        L = L_star or self.pick_best_ctx(x0, H)
        ctx = x0[-L:].astype(np.float32)
        pf, qf = self.model.forecast(horizon=int(H), inputs=[ctx])
        point = np.asarray(pf[0], dtype=np.float32)
        if return_quantiles and qf is not None:
            quants = np.asarray(qf[0])  # shape depends on repo; adapt mapping below
        else:
            quants = None
        return point, quants


# ---------------------------------------------------------------------
# 1) Data access (illustrative). Ensure your parquet has columns: id, period, value
# ---------------------------------------------------------------------

def load_series(input_parquet: str) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    df = pd.read_parquet(input_parquet)
    for gid, g in df.groupby('id'):
        x0 = g[g['period'] == 0]['value'].to_numpy(np.float32)
        x1 = g[g['period'] == 1]['value'].to_numpy(np.float32)
        # Basic hygiene: interpolate interior NaNs; drop leading/trailing NaNs
        x0 = pd.Series(x0).interpolate(limit_direction='both').to_numpy(np.float32)
        x1 = pd.Series(x1).interpolate(limit_direction='both').to_numpy(np.float32)
        yield int(gid), x0, x1


# ---------------------------------------------------------------------
# 2) Feature helpers — invariance, baselines, alignment
# ---------------------------------------------------------------------

def robust_scales(x0: np.ndarray) -> Tuple[float, float]:
    mad = float(np.median(np.abs(x0 - np.median(x0)))) + 1e-8
    rms = float(np.sqrt(np.mean((x0 - np.mean(x0))**2))) + 1e-8
    return mad, rms


def pct_index(H: int, p: int) -> int:
    return max(1, int(np.ceil(H * p / 100)))


def seasonal_naive(x0: np.ndarray, H: int) -> np.ndarray:
    for s in (7, 12, 24, 30, 52):
        if 2 * s <= len(x0):
            return np.tile(x0[-s:], int(np.ceil(H / s)))[:H]
    return np.repeat(x0[-1], H)


def best_xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 8) -> Tuple[int, float]:
    best_lag, best_corr = 0, -1.0
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            c = np.corrcoef(a[L:], b[:len(b) - L])[0, 1]
        else:
            c = np.corrcoef(a[:len(a) + L], b[-L:])[0, 1]
        c = float(0.0 if np.isnan(c) else c)
        if c > best_corr:
            best_lag, best_corr = L, c
    return best_lag, best_corr


# ---------------------------------------------------------------------
# 3) Feature Blocks A–E
# ---------------------------------------------------------------------

PCTS = [10, 25, 50, 75, 90]


def block_A(yhat: np.ndarray, x1: np.ndarray, x0: np.ndarray) -> Dict[str, float]:
    """Error magnitude/dynamics — length invariant via percentile horizons.
    Captures early shock (mae_early), persistence (mae_p90), etc.
    """
    H = len(x1)
    mad, rms = robust_scales(x0)
    abs_err = np.abs(yhat - x1)
    feats = {
        'mae_full': float(abs_err.mean() / mad),
        'rmse_full': float(np.sqrt(np.mean((yhat - x1) ** 2)) / rms),
        'mae_early': float(abs_err[:pct_index(H, 10)].mean() / mad),
        'mae_late': float(abs_err[-pct_index(H, 10):].mean() / mad),
    }
    feats['err_ratio_early_late'] = float(
        feats['mae_early'] / (feats['mae_late'] + 1e-12)
    )
    feats['err_slope'] = float(feats['mae_early'] - feats['mae_late'])
    for p in PCTS:
        feats[f'mae_p{p}'] = float(abs_err[:pct_index(H, p)].mean() / mad)
    return feats


def block_B(yhat: np.ndarray, x1: np.ndarray, x0: np.ndarray) -> Dict[str, float]:
    """Shape agreement & baselines — correlation/lag + naive contrasts.
    Amplitude ratio ~ scale change; low corr/shifted lag ~ phase change.
    """
    mad, _ = robust_scales(x0)
    H = len(x1)
    lv = np.repeat(x0[-1], H)
    sn = seasonal_naive(x0, H)
    corr = float(np.nan_to_num(np.corrcoef(yhat, x1)[0, 1]))
    lag, corr_lag = best_xcorr_lag(yhat, x1, 8)
    amp_ratio = float((np.std(x1) + 1e-8) / (np.std(yhat) + 1e-8))
    mae_t = float(np.mean(np.abs(yhat - x1)) / mad)
    mae_lv = float(np.mean(np.abs(lv - x1)) / mad)
    mae_sn = float(np.mean(np.abs(sn - x1)) / mad)
    return {
        'corr_full': corr,
        'xcorr_best_lag': float(lag),
        'xcorr_best_corr': float(corr_lag),
        'amp_ratio': amp_ratio,
        'mae_ratio_vs_lastvalue': float(mae_t / (mae_lv + 1e-12)),
        'mae_ratio_vs_seasonal': float(mae_t / (mae_sn + 1e-12)),
    }


def block_C(yhat_p0: np.ndarray, x0_tail: np.ndarray,
            yhat_p1: np.ndarray, x1: np.ndarray) -> Dict[str, float]:
    """Calibration stress-test — tiny ridge mapping yhat->x on Period-0 tail.
    Intercept ~= level shift (mean), slope ~= scale change (variance).
    """
    try:
        from sklearn.linear_model import Ridge
    except Exception as e:
        raise RuntimeError("Install scikit-learn for block_C: pip install scikit-learn") from e

    reg = Ridge(alpha=1e-2)
    reg.fit(yhat_p0.reshape(-1, 1), x0_tail)
    ycal = reg.predict(yhat_p1.reshape(-1, 1))
    mad, _ = robust_scales(x0_tail)
    mae_raw = float(np.mean(np.abs(yhat_p1 - x1)) / mad)
    mae_cal = float(np.mean(np.abs(ycal - x1)) / mad)
    return {
        'calib_coef': float(np.clip(reg.coef_[0], -10, 10)),
        'calib_intercept': float(np.clip(reg.intercept_, -1e3, 1e3)),
        'calib_delta_err': float((mae_raw - mae_cal) / (mae_raw + 1e-12)),
        # Optional: explicit contrasts
        'mean_diff_norm': float((x1.mean() - yhat_p1.mean()) / (mad + 1e-12)),
        'var_ratio_post_pre': float(np.var(x1) / (np.var(x0_tail) + 1e-12)),
    }


def _neglogp(p: float, cap: float = 12.0) -> float:
    return float(np.clip(-np.log10(max(p, 1e-300)), 0, cap))


def block_D(res_p0: np.ndarray, res_p1: np.ndarray) -> Dict[str, float]:
    """Statistical tests on residuals — report as -log10 p (bounded).
    Requires scipy. Under H0, distributions should match.
    """
    if stats is None:
        raise RuntimeError("Install scipy for block_D: pip install scipy")
    feats = {}
    # Welch t-test (mean shift)
    _, p = stats.ttest_ind(res_p0, res_p1, equal_var=False, nan_policy='omit')
    feats['nlp_t_welch'] = _neglogp(float(p))
    # Mann–Whitney U (median/robust)
    try:
        _, p = stats.mannwhitneyu(res_p0, res_p1, alternative='two-sided')
        feats['nlp_mwu'] = _neglogp(float(p))
    except Exception:
        feats['nlp_mwu'] = 0.0
    # Brown–Forsythe via Levene(center='median') (variance shift)
    _, p = stats.levene(res_p0, res_p1, center='median')
    feats['nlp_levene'] = _neglogp(float(p))
    # KS two-sample (distributional)
    _, p = stats.ks_2samp(res_p0, res_p1, alternative='two-sided', mode='asymp')
    feats['nlp_ks'] = _neglogp(float(p))
    return feats


# Quantile mapping helper — *adapt indices to the repo output you use*
Q_LIST = [0.10, 0.25, 0.50, 0.75, 0.90]


def _extract_quantiles(qf: np.ndarray) -> Dict[float, np.ndarray]:
    """Map repo quantile output to deciles/quarters; adjust as needed.
    Common pattern in 2.5 demos: qf shape ~ (H, 11) with mean + 0.1..0.9.
    We'll assume qf[:,1]→q10, qf[:,3]→q25, qf[:,5]→q50, qf[:,7]→q75, qf[:,9]→q90.
    """
    if qf is None:
        return {}
    H, C = qf.shape[:2]
    if C < 10:
        # Quantiles not returned
        return {}
    return {0.10: qf[:, 1], 0.25: qf[:, 3], 0.50: qf[:, 5], 0.75: qf[:, 7], 0.90: qf[:, 9]}


def _pinball(y: np.ndarray, qhat: np.ndarray, q: float) -> float:
    e = y - qhat
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def block_E(qf: Optional[np.ndarray], x1: np.ndarray) -> Dict[str, float]:
    """Uncertainty features from quantile head: interval widths, coverage errors,
    pinball losses. These are scale/length invariant and react to shifts.
    """
    out: Dict[str, float] = {}
    qm = _extract_quantiles(qf)
    if not qm:
        out['quantiles_present'] = 0.0
        return out
    out['quantiles_present'] = 1.0
    # Interval widths (sharpness)
    iw80 = float(np.mean(qm[0.90] - qm[0.10]))
    iw50 = float(np.mean(qm[0.75] - qm[0.25]))
    out['iw_80'] = iw80
    out['iw_50'] = iw50
    # Coverage error (empirical - nominal)
    for (a, b, name) in [(0.10, 0.90, '80'), (0.25, 0.75, '50')]:
        covered = float(((x1 >= qm[a]) & (x1 <= qm[b])).mean())
        out[f'cov_err_{name}'] = float((b - a) - covered)
    # Pinball losses
    for q in Q_LIST:
        out[f'pinball_q{int(q*100)}'] = _pinball(x1, qm[q], q)
    return out


# ---------------------------------------------------------------------
# 4) Extraction pipeline — illustrates the end-to-end loop
# ---------------------------------------------------------------------

def extract_tsfm_features(input_parquet: str,
                          checkpoint_note: str,
                          out_csv: str,
                          tfm_cfg: Optional[TfmCfg] = None) -> None:
    """Illustrative end-to-end extractor (no batching/persistence shown).

    checkpoint_note: free-text to log the repo commit/checkpoint you used.
    """
    tfm_cfg = tfm_cfg or TfmCfg()
    tfm = TSFMForecaster(tfm_cfg)
    rows: List[Dict[str, float]] = []

    for gid, x0, x1 in load_series(input_parquet):
        # 1) Choose context and forecast Period-1
        H = len(x1)
        Lstar = tfm.pick_best_ctx(x0, H)
        pf, qf = tfm.forecast(x0, H, L_star=Lstar, return_quantiles=True)

        # 2) Pseudo-horizon on Period-0 tail for calibration/tests
        Hs = max(8, min(H, int(0.25 * len(x0))))
        pf0, _ = tfm.forecast(x0[:-Hs], Hs, L_star=min(Lstar, len(x0) - Hs),
                               return_quantiles=False)
        x0_tail = x0[-Hs:]

        # 3) Blocks A–E
        A = block_A(pf, x1, x0)
        B = block_B(pf, x1, x0)
        C = block_C(pf0, x0_tail, pf, x1)
        res_p0 = x0_tail - pf0
        res_p1 = x1 - pf
        D = block_D(res_p0, res_p1)
        E = block_E(qf, x1)

        row = {'id': gid, 'ctx_len': float(Lstar), 'checkpoint_note': checkpoint_note}
        row.update(A); row.update(B); row.update(C); row.update(D); row.update(E)
        rows.append(row)

    # 4) Write CSV (simple; you may prefer Parquet)
    cols = sorted(set().union(*[r.keys() for r in rows]))
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_csv, index=False)


# ---------------------------------------------------------------------
# 5) Classifier training (illustrative)
# ---------------------------------------------------------------------

def train_classifier(X: pd.DataFrame, y: pd.Series):
    """Train a supervised classifier with 30/70 imbalance.
    Illustrative hyperparameters; prefer a proper CV grid in practice.
    """
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise RuntimeError("Install xgboost for training: pip install xgboost") from e

    model = XGBClassifier(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric='logloss',
        scale_pos_weight=(0.70 / 0.30),  # ≈2.33 for 30% positives
        n_jobs=4,
        random_state=42,
    )
    # Example: 5-fold stratified CV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, brier_score_loss

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, briers = [], []
    for tr, va in skf.split(X, y):
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y.iloc[va], p))
        briers.append(brier_score_loss(y.iloc[va], p))
    print(f"CV AUC: {np.mean(aucs):.3f}±{np.std(aucs):.3f} | Brier: {np.mean(briers):.3f}")
    return model


# ---------------------------------------------------------------------
# 6) Usage sketch (non-executable; adapt paths & schema)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Extract features
    # extract_tsfm_features(
    #     input_parquet="/path/to/train.parquet",
    #     checkpoint_note="timesfm@<commit>/<checkpoint_tag>",
    #     out_csv="TSFM.csv",
    #     tfm_cfg=TfmCfg(max_context=1024, max_horizon=512, use_quantiles=True),
    # )

    # 2) Merge with labels & existing features (Wavelet/Roy)
    # feats = pd.read_csv("TSFM.csv")
    # labels = pd.read_csv("labels.csv")  # columns: id, y
    # join = labels.merge(feats, on="id", how="left").fillna(0)

    # 3) Train classifier
    # y = join["y"].astype(int)
    # X = join.drop(columns=["id", "y", "checkpoint_note"])  # pick columns as needed
    # model = train_classifier(X, y)

    # Notes:
    # - Record repo commit hash + checkpoint tag with your artifacts.
    # - Cache (id→L*, pf, qf) externally to reuse in ablations.
    # - Monitor NaNs; clip -log10 p to ≤12 for stability.
    pass
