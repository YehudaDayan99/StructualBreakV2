# Wavelet21 Optimization Plan — Steps (0) Residual-First and (A) Period-Aware Contrasts

**Project:** ADIA Structural Break Classification  
**Repo:** `https://github.com/YehudaDayan99/StructualBreakV2`  
**Scope of this doc:** Ship a stable residual-first pipeline and period-aware wavelet contrasts on MODWT coefficients, wired into `GeneratingFeatures2.ipynb` and consumed by `Model_X.ipynb`.

---

## 1) Rationale (Short)
- **Residual-first (Step 0):** Fit a single null model H₀ on the **full** series (Period0+1). Transform to standardized residuals \(\hat\varepsilon_t\). MODWT on residuals reduces confounding from trend/volatility so boundary energy reflects *true* departures.
- **Period-aware contrasts (Step A):** On residual MODWT coefficients \(W_{j,t}\), compute *continuous* pre/post dispersion contrasts per level \(j\): **log variance ratio** and **log MAD ratio**, plus optional directional location (Hedges’ g). Continuous contrasts give graded evidence that tree models exploit well.

---

## 2) Files to Edit / Touch
- `methods/wavelet21/feature_extractor.py` — add residual-first H₀ modeling; compute per-level pre/post contrasts on MODWT residuals.
- `methods/wavelet21/config.py` — add `NullModelCfg` knobs (model, law, etc.) to `WaveletConfig`.
- `methods/wavelet21/batch_processor.py` — (light) ensure H₀ diagnostics are passed to output.
- (Optional) `methods/base/utils.py` — helpers for diagnostics.

---

## 3) Step (0): Residual-First Pipeline

### 3.1 Logic
Fit **one** H₀ on full series. Produce standardized residuals and basic diagnostics (Ljung–Box p on \(\hat\varepsilon_t\), ARCH-LM proxy on \(\hat\varepsilon_t^2\), assumed residual law and \(\nu\) if Student-t). Run MODWT on residuals.

### 3.2 Minimal Implementation

**Add a null-model config**
```python
# methods/wavelet21/feature_extractor.py
from dataclasses import dataclass

@dataclass
class NullModelCfg:
    model: str = "arima"          # "arima" or "garch"
    arima_order: tuple = (1,0,1)
    resid_law: str = "t"          # "normal" or "t"
    min_len: int = 300
```

**Fit H₀ and standardize residuals**
```python
# methods/wavelet21/feature_extractor.py
import numpy as np
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox as lb

def fit_null_model(full_vals: np.ndarray, cfg: NullModelCfg):
    x = np.asarray(full_vals, float)
    if len(x) < cfg.min_len:
        eps = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
        meta = dict(law="normal", nu=np.nan, ljungbox_p=np.nan, archlm_p=np.nan)
        return eps, meta

    res = ARIMA(x, order=cfg.arima_order).fit(method="statespace")
    resid_raw = res.resid
    eps = resid_raw / (np.nanstd(resid_raw) + 1e-12)

    lb_p    = float(lb(eps,   lags=[20], return_df=True)["lb_pvalue"].iloc[-1])
    lb_p_sq = float(lb(eps**2,lags=[20], return_df=True)["lb_pvalue"].iloc[-1])

    law = "t" if cfg.resid_law.lower().startswith("t") else "normal"
    nu  = 8.0 if law == "t" else np.nan
    meta = dict(law=law, nu=nu, ljungbox_p=lb_p, archlm_p=lb_p_sq)
    return eps, meta

def h0_meta_to_feats(meta):
    return {
        "h0_ljungbox_p": meta["ljungbox_p"],
        "h0_archlm_p": meta["archlm_p"],
        "h0_err_is_t": 1 if meta["law"] == "t" else 0,
        "h0_t_nu": meta.get("nu", np.nan),
    }
```

**Use residuals → MODWT → features**
```python
def extract_wavelet_features(values: np.ndarray, period: np.ndarray, wcfg, h0cfg):
    eps, meta = fit_null_model(values, h0cfg)        # Step (0)
    W = modwt(eps, wavelet=wcfg.wavelet, J=wcfg.J)   # MODWT on residuals
    feats = {}
    feats.update(h0_meta_to_feats(meta))             # diagnostics
    feats.update(pre_post_contrasts(W, period, wcfg))# Step (A)
    # keep your boundary-local features if present
    return feats, meta
```

**Config plumbing**
```python
# methods/wavelet21/config.py
class WaveletConfig:
    wavelet: str = "sym4"  # LA(8)
    J: int = 3
    alpha: float = 0.05
    null_model = NullModelCfg()
```

---

## 4) Step (A): Period-Aware Contrasts on MODWT Residuals

### 4.1 Logic
For each level \(j\), compute pre vs post dispersion contrasts:
- `var_logratio_j = log(Var(Wj_post)/Var(Wj_pre))`
- `mad_logratio_j = log(MAD(Wj_post)/MAD(Wj_pre))`
- (optional) `hedges_g_j` for directional location change.

**Naming:** `wav_{family}_L{j}_{stat}` e.g., `wav_sym4_L2_var_logratio`

### 4.2 Minimal Implementation
```python
import numpy as np
from scipy import stats

def _split_masks(period: np.ndarray):
    pre_idx  = np.flatnonzero(period == 0)
    post_idx = np.flatnonzero(period == 1)
    return pre_idx, post_idx

def _safe_var(x):  return float(np.nanvar(x, ddof=1)) + 1e-12
def _safe_mad(x):  return float(stats.median_abs_deviation(x, scale='normal')) + 1e-12

def pre_post_contrasts(W: dict[int, np.ndarray], period: np.ndarray, wcfg):
    pre_idx, post_idx = _split_masks(period)
    feats = {}
    fam = wcfg.wavelet
    for j, Wj in W.items():
        v0, v1 = _safe_var(Wj[pre_idx]),  _safe_var(Wj[post_idx])
        m0, m1 = _safe_mad(Wj[pre_idx]), _safe_mad(Wj[post_idx])

        feats[f"wav_{fam}_L{j}_var_logratio"] = np.log(v1 / v0)
        feats[f"wav_{fam}_L{j}_mad_logratio"] = np.log(m1 / m0)

        # optional: directional location effect
        mu0, mu1 = float(np.nanmean(Wj[pre_idx])), float(np.nanmean(Wj[post_idx]))
        n0, n1 = len(pre_idx), len(post_idx)
        s_pooled = np.sqrt(((n0-1)*v0 + (n1-1)*v1) / max(n0+n1-2, 1))
        feats[f"wav_{fam}_L{j}_hedges_g"] = (mu1 - mu0) / (s_pooled + 1e-12)
    return feats
```

---

## 5) Notebook Hooks & QA

### 5.1 GeneratingFeatures2.ipynb (dry run on ~200 ids)
```python
from StructualBreak.methods.wavelet21.feature_extractor import NullModelCfg
from StructualBreak.methods.wavelet21.config import WaveletConfig
from StructualBreak import run_batch

wcfg = WaveletConfig(); wcfg.wavelet="sym4"; wcfg.J=3; wcfg.alpha=0.05
wcfg.null_model = NullModelCfg(model="arima", resid_law="t")

pred_df, meta_df = run_batch(
    input_parquet="/content/drive/MyDrive/ADIA/X_train.parquet",
    out_pred_parquet="/content/drive/MyDrive/ADIA/Wavelet.parquet",
    out_meta_parquet="/content/drive/MyDrive/ADIA/Wavelet_meta.parquet",
    method="wavelet21",
    n_jobs=2, verbose=True
)
```

**Sanity check cell**
```python
import numpy as np, pandas as pd
df = pd.read_parquet("/content/drive/MyDrive/ADIA/Wavelet.parquet")  # or CSV
wcols = [c for c in df.columns if c.startswith("wav_")]
assert df[wcols].std(numeric_only=True).replace(0, np.nan).notna().all(), "Constant wavelet columns"
display(df.filter(regex=r"wav_.*_(var|mad)_logratio$").head(5))
display(df[["h0_ljungbox_p","h0_archlm_p","h0_err_is_t","h0_t_nu"]].head(5))
```

### 5.2 Model_X.ipynb (quick A/B)
- Compare **Wavelet-only** vs **Roy-only** vs **Both**.
- Keep isotonic calibration for Brier.

---

## 6) Acceptance Checks

- **Data product:** `Wavelet.parquet/CSV` has:
  - `h0_*` diagnostics columns.
  - `wav_*_L{j}_(var|mad)_logratio` (and optional `_hedges_g`) for all levels.
- **QA:** No all-NaN or constant columns; sane distributions.
- **Model_X:** Wavelet-only block yields **AUC/Brier lift** vs prior wavelet block; several `wav_*_logratio` enter Top-30 importances.

---

## 7) Next Steps After This Doc (not implemented here)
- Calibrated boundary-local exceedances keyed by `(wavelet, J, alpha, window_m, resid_law, nu)`.
- Add `db8` family and cap at `J<=3` with correlation pruning.
- Robust variance tests (Brown–Forsythe) as cheap classical features.