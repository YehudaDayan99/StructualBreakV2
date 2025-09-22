# TSFM Feature Plan for Structural Break Challenge (Revised)

## 0) What this document is

A step‑by‑step implementation plan for adding a **TimesFM (TSFM)** feature block to classify **structural break = Yes/No** on a training set of **10,000 series** with **\~30% True (break) / 70% False (no break)**. Written for a junior data scientist; includes invariance choices, mean/variance shift capture, and statistical‑test features.

---

## 1) Methodological overview of TSFM (very short)

* **What**: Decoder‑only Transformer for univariate forecasting using **patching** (input/output patches as tokens), trained on broad real+synthetic corpora; strong **zero‑shot** performance.
* **Inference**: Provide a context window → get multi‑step forecast; masking during pretraining enables variable context lengths.
* **Why for us**: Treat TSFM as a **counterfactual forecaster under H₀ (no break)**. Deviations post‑boundary become features.

---

## 2) Intuition: TSFM for structural‑break detection

Use **pre‑boundary (Period‑0)** as context, forecast **post‑boundary (Period‑1)** without peeking. If a true break occurs, we expect:

* **Early‑horizon error spike** (shock), possibly stabilizing later.
* **Shape/phase mismatch** (low corr., non‑zero best lag).
* **Level/scale drift** (mean/variance change).
  We encode these as **dimensionless, length‑robust** features + **statistical tests** on residuals.

---

## 3) High‑level algorithm

For each series id with arrays `x0` (Period‑0) and `x1` (Period‑1):

1. **Select context length** `L*` via quick CV on the tail of `x0`.
2. **Zero‑shot forecast** `ŷ1 = TSFM(x0[-L*:], horizon=len(x1))`.
3. Build **Feature Block A** (normalized error magnitude/dynamics; percentile horizons for length robustness).
4. Build **Feature Block B** (shape agreement, phase, baseline contrasts).
5. Build **Feature Block C** (calibration stress‑test → level/scale drift sensitivity).
6. Build **Feature Block D** (**statistical tests** on residuals: Welch t, MWU, Levene/BF, KS; report as `−log10(p)`).
7. **Persist features** → join with existing feature sets → **train classifier** with class‑imbalance handling and stratified CV.

---

## 4) Step‑by‑step plan (each step: title, logic, pseudocode, tech notes)

### Step 1 — Environment & data wiring

**Logic**: Wrap TimesFM; load `(id, x0, x1)` from parquet; ensure determinism.

```python
# pip install numpy pandas pyarrow scipy scikit-learn xgboost
# plus TimesFM deps per repo (torch or jax, depending on checkpoint)
from pathlib import Path
import pandas as pd

def load_series(input_parquet: str):
    df = pd.read_parquet(input_parquet)
    for gid, g in df.groupby('id'):
        x0 = g[g['period']==0]['value'].to_numpy()
        x1 = g[g['period']==1]['value'].to_numpy()
        yield int(gid), x0, x1
```

**Notes**: Fix random seeds; use float32. Keep wrapper framework‑agnostic (PyTorch/JAX).

---

### Step 2 — TSFM forecaster with context selection

**Logic**: Expose `.forecast(context, horizon)` and pick `L*` via pseudo‑horizon backtest on `x0`.

```python
class TSFMForecaster:
    def __init__(self, ckpt, in_patch=32, out_patch=128, max_ctx=512):
        self.model = load_tsfm_from_checkpoint(ckpt, in_patch, out_patch)
        self.max_ctx = max_ctx

    def _scale(self, x):
        m = x[:min(32, len(x))].mean(); s = x[:min(32, len(x))].std() + 1e-8
        return (x-m)/s, (m, s)

    def pick_best_ctx(self, x0, H, grid=(128,256,512)):
        Hs = max(8, min(H, int(0.25*len(x0))))
        bestL, best = None, 1e9
        for L in grid:
            if L>len(x0):
                continue
            ctx = x0[-L:]
            z,(m,s) = self._scale(ctx)
            yhat = self.model.forecast(z, horizon=Hs)*s + m
            ytrue = x0[-Hs:]
            mad = np.median(np.abs(x0-np.median(x0)))+1e-8
            err = np.mean(np.abs(yhat-ytrue))/mad
            if err < best:
                best, bestL = err, L
        return bestL or min(len(x0), max(grid))

    def forecast(self, x0, H, L_star=None):
        L = L_star or self.pick_best_ctx(x0,H)
        z,(m,s) = self._scale(x0[-L:])
        return self.model.forecast(z, horizon=H)*s + m
```

**Notes**: Batch ids for speed; cache forecasts for reuse.

---

### Step 3 — Feature Block A: error magnitude & dynamics (length‑robust)

**Logic**: Normalize errors by **Period‑0 MAD/RMS** and compute early/median/late slices using **percentiles** (length‑invariant).

```python
def robust_scales(x0):
    mad = np.median(np.abs(x0 - np.median(x0))) + 1e-8
    rms = np.sqrt(np.mean((x0 - np.mean(x0))**2)) + 1e-8
    return mad, rms

PCTS = [10, 25, 50, 75, 90]

def pct_index(H, p):
    return max(1, int(np.ceil(H*p/100)))

def block_A(yhat, x1, x0):
    H = len(x1); mad, rms = robust_scales(x0)
    abs_err = np.abs(yhat - x1)
    feats = {
      'mae_full': abs_err.mean()/mad,
      'rmse_full': np.sqrt(np.mean((yhat-x1)**2))/rms,
      'mae_early': abs_err[:pct_index(H,10)].mean()/mad,
      'mae_late':  abs_err[-pct_index(H,10):].mean()/mad,
    }
    feats['err_ratio_early_late'] = feats['mae_early']/(feats['mae_late']+1e-12)
    feats['err_slope'] = feats['mae_early'] - feats['mae_late']
    for p in PCTS:
        feats[f'mae_p{p}'] = abs_err[:pct_index(H,p)].mean()/mad
    return feats
```

**Notes**: Percentile horizons make features comparable across varying `H`.

---

### Step 4 — Feature Block B: shape agreement & baseline contrasts

**Logic**: Capture correlation, best lag, amplitude ratio; benchmark TSFM vs last‑value & seasonal‑naive.

```python
def seasonal_naive(x0, H):
    for s in (7,12,24,30,52):
        if 2*s <= len(x0):
            return np.tile(x0[-s:], int(np.ceil(H/s)))[:H]
    return np.repeat(x0[-1], H)

def best_xcorr_lag(a,b,max_lag=8):
    lags = range(-max_lag,max_lag+1)
    best=(0,-1.0)
    for L in lags:
        if L>=0:
            c=np.corrcoef(a[L:], b[:len(b)-L])[0,1]
        else:
            c=np.corrcoef(a[:len(a)+L], b[-L:])[0,1]
        if np.isnan(c): c=0.0
        if c>best[1]: best=(L,c)
    return best

def block_B(yhat,x1,x0):
    mad,_= robust_scales(x0)
    H=len(x1)
    lv=np.repeat(x0[-1],H)
    sn=seasonal_naive(x0,H)
    corr=np.corrcoef(yhat,x1)[0,1]
    lag,corr_lag=best_xcorr_lag(yhat,x1,8)
    amp_ratio=(np.std(x1)+1e-8)/(np.std(yhat)+1e-8)
    mae_t=np.mean(np.abs(yhat-x1))/mad
    mae_lv=np.mean(np.abs(lv-x1))/mad
    mae_sn=np.mean(np.abs(sn-x1))/mad
    return {
      'corr_full': np.nan_to_num(corr),
      'xcorr_best_lag': lag,
      'xcorr_best_corr': np.nan_to_num(corr_lag),
      'amp_ratio': amp_ratio,
      'mae_ratio_vs_lastvalue': mae_t/(mae_lv+1e-12),
      'mae_ratio_vs_seasonal': mae_t/(mae_sn+1e-12),
    }
```

**Notes**: These features are dimensionless and robust to length.

---

### Step 5 — Feature Block C: calibration stress‑test (mean/variance drift)

**Logic**: Fit a tiny ridge on Period‑0 tail mapping `ŷ→x`; apply to Period‑1 forecast. Large error drop ⇒ distribution shift near the boundary. `calib_intercept` ≈ **level shift**; `calib_coef` & `amp_ratio` ≈ **scale change**.

```python
from sklearn.linear_model import Ridge

def block_C(yhat_p0, x0_tail, yhat_p1, x1):
    reg = Ridge(alpha=1e-2)
    reg.fit(yhat_p0.reshape(-1,1), x0_tail)
    ycal = reg.predict(yhat_p1.reshape(-1,1))
    mad,_ = robust_scales(x0_tail)
    mae_raw = np.mean(np.abs(yhat_p1-x1))/mad
    mae_cal = np.mean(np.abs(ycal-x1))/mad
    return {
      'calib_coef': float(np.clip(reg.coef_[0], -10, 10)),
      'calib_intercept': float(np.clip(reg.intercept_, -1e3, 1e3)),
      'calib_delta_err': (mae_raw-mae_cal)/(mae_raw+1e-12)
    }
```

**Notes**: Also add simple contrasts: `mean_diff_norm=(x1.mean()-yhat_p1.mean())/MAD(x0)`, `var_ratio_post_pre=np.var(x1)/(np.var(x0)+1e-12)` if desired.

---

### Step 6 — Feature Block D: statistical tests on residuals

**Logic**: Under H₀, post‑residuals ≈ pre‑tail residuals. Significant changes (small p) → evidence of break. Report **`-log10(p)`** (bounded) for numeric stability.

```python
from scipy import stats

def neglogp(p, cap=12.0):
    return float(np.clip(-np.log10(max(p, 1e-300)), 0, cap))

def block_D(res_p0, res_p1):
    feats={}
    # Mean shift (Welch t)
    t, p = stats.ttest_ind(res_p0, res_p1, equal_var=False, nan_policy='omit')
    feats['nlp_t_welch'] = neglogp(p)
    # Median shift (MWU)
    u, p = stats.mannwhitneyu(res_p0, res_p1, alternative='two-sided')
    feats['nlp_mwu'] = neglogp(p)
    # Variance shift (Brown–Forsythe via Levene with center='median')
    w, p = stats.levene(res_p0, res_p1, center='median')
    feats['nlp_levene'] = neglogp(p)
    # Distributional shift (KS)
    ks, p = stats.ks_2samp(res_p0, res_p1, alternative='two-sided', mode='asymp')
    feats['nlp_ks'] = neglogp(p)
    return feats
```

**Notes**: Define residuals `res_p0 = x0_tail - yhat_p0`, `res_p1 = x1 - yhat_p1`. Clip/replace NaNs.

---

### Step 7 — Extraction pipeline, merge, and classifier training

**Logic**: Produce one feature row per id, merge with existing features, train a classifier with imbalance handling and robust CV.

```python
import csv
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

def extract_tsfm_features(input_parquet, checkpoint, out_csv):
    tsfm = TSFMForecaster(checkpoint)
    rows=[]
    for gid,x0,x1 in load_series(input_parquet):
        H=len(x1); L*= tsfm.pick_best_ctx(x0,H)
        yhat1 = tsfm.forecast(x0,H,L*)
        # pseudo-horizon on Period-0 tail for calibration/tests
        Hs = max(8, min(H, int(0.25*len(x0))))
        yhat0 = tsfm.forecast(x0[:-Hs], Hs, L_star=min(L*, len(x0)-Hs))
        x0_tail = x0[-Hs:]
        A = block_A(yhat1, x1, x0)
        B = block_B(yhat1, x1, x0)
        C = block_C(yhat0, x0_tail, yhat1, x1)
        res_p0 = x0_tail - yhat0
        res_p1 = x1 - yhat1
        D = block_D(res_p0, res_p1)
        row={'id':gid,'ctx_len':L*, **A, **B, **C, **D}
        rows.append(row)
    cols=sorted(set().union(*[r.keys() for r in rows]))
    with open(out_csv,'w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

# After extraction: merge TSFM.csv with labels and existing features, then train

def train_classifier(X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    model = XGBClassifier(
        n_estimators=800, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, eval_metric='logloss',
        scale_pos_weight= (0.70/0.30)  # ~2.33 for 30% positives
    )
    # Fit/validate across folds; report AUC, Brier, PR-AUC
    return model
```

**Notes**:

* **Imbalance**: use `scale_pos_weight≈2.33`, or focal loss (optional). Also compute class‑weighted Brier.
* **CV**: 5‑fold **StratifiedKFold** (shuffled); ensure no leakage.
* **Metrics**: AUC (ROC & PR), **Brier score**, calibration curve; plot feature importances & SHAP for diagnostics.

---

## 5) Invariance, mean/variance capture, and tests — quick checklist

* **Length invariance**: percentile‑based horizons (`mae_p10/p25/p50/p75/p90`) and dimensionless ratios.
* **Scale invariance**: normalize by Period‑0 MAD/RMS; use correlations/ratios.
* **Pattern sensitivity (by design)**: correlation/lag features respond to phase/shape changes; baseline contrasts reduce false alarms.
* **Mean shift**: `calib_intercept`, `mean_diff_norm` (optional), rising `mae_*`.
* **Variance/scale shift**: `amp_ratio`, `calib_coef`, `var_ratio_post_pre` (optional).
* **Statistical tests**: `nlp_t_welch`, `nlp_mwu`, `nlp_levene`, `nlp_ks` computed on residuals.

---

## 6) Deliverables & filenames

* `tsfm_features.py` — `TSFMForecaster`, Blocks A–D, extraction function.
* `TSFM.csv` — one row per id with all features.
* Updated training notebook/script — merges features, trains classifier, logs metrics.

---

## 7) Acceptance criteria (10k series; \~30/70 imbalance)

* End‑to‑end extraction runs deterministically; `TSFM.csv` has ≥20 columns.
* TSFM‑only block yields **AUC > 0.60** on validation; stacked with existing features improves **AUC and Brier** vs baseline.
* Statistical test features are bounded (`-log10 p ≤ 12`), non‑NaN.
* Classifier calibrated (optional isotonic/Platt) and passes hold‑out sanity checks.

---

## Repo-driven practical additions (TimesFM 2.5)

**Objective**: add concrete implementation details from the official repository so a junior DS can implement quickly.

### A) Model/version & install

* Use **TimesFM 2.5 (200M, PyTorch)** with optional **continuous quantile head**; context up to **16k**.
* Install from repo (editable install). Pin the **commit hash** and **HF checkpoint tag** in your run logs.

```python
!git clone https://github.com/google-research/timesfm.git
%cd timesfm
!pip install -e .

import timesfm
model = timesfm.TimesFM_2p5_200M_torch()
model.load_checkpoint()
model.compile(timesfm.ForecastConfig(
    max_context=1024,      # raise if GPU allows
    max_horizon=256,       # cap by data horizon
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    fix_quantile_crossing=True,
))
point_fcst, quant_fcst = model.forecast(horizon=H, inputs=[ctx])
```

**Practical notes**: batch by **bucketing** lengths; provide `inputs` as a **list of 1D arrays**; test GPU **bfloat16/float16** inference for speed/memory.

### B) Data hygiene & length handling

* **NaNs**: interpolate interior, drop leading/trailing.
* **Very long contexts**: truncate to last **≤16k** points or mean‑pool (factor 2) before feeding TimesFM.
* **Scale**: repo does `normalize_inputs`; we still compute **MAD/RMS** on Period‑0 for feature normalization.

### C) Context selection & caching

* Keep our `pick_best_ctx` backtest. Cache `(id → L*, pf, qf)` on disk to reuse across runs.

### D) Uncertainty features (enable quantile head)

* Add **Feature Block E** using repo quantiles: **interval widths** (80%/50%), **coverage error** (empirical − nominal), and **pinball losses** at {0.1, 0.25, 0.5, 0.75, 0.9}. See pseudocode in plan.

### E) Training at 30/70 class split

* Use **StratifiedKFold(5)**; `scale_pos_weight≈2.33` in XGBoost/LightGBM.
* Track **AUC, PR‑AUC, Brier**, and calibration. Run ablations: existing features vs TSFM A–E vs stacked.

### F) Acceptance (additions)

* `TSFM.csv` now includes **quantile‑based features**; no NaNs; p‑values reported as **−log10 p ≤ 12**.
* Reproducible runs with **commit hash + checkpoint tag** recorded.


Done—added a new canvas section: “Repo-driven practical additions (TimesFM 2.5)”. Highlights a junior can follow:

* **Version & install**: Use **TimesFM 2.5, 200M (PyTorch)**; install via `git clone` + `pip install -e .`; compile with `ForecastConfig` and enable the **continuous quantile head**. Up to **16k context**, longer horizons supported. ([GitHub][1])
* **API usage**: `timesfm.TimesFM_2p5_200M_torch().load_checkpoint().compile(...); model.forecast(horizon=H, inputs=[array,...])` returns **point** and **quantile** forecasts. ([GitHub][1])
* **Batching & speed**: Batch by bucketing similar lengths; try **bfloat16/float16** on GPU; cache `(id → L*, point, quantiles)` to disk. (Repo shows list-of-arrays batching pattern.) ([Hugging Face][2])
* **Data hygiene**: Interpolate interior NaNs, drop leading/trailing; truncate or mean-pool if context >16k. (2.5 removes frequency flag—less preprocessing.) ([GitHub][1])
* **New features (Block E)**: From quantiles—**interval widths (80/50%)**, **coverage error**, **pinball losses** at {0.1, 0.25, 0.5, 0.75, 0.9}. These are length/scale-invariant and sensitive to distribution shift. ([GitHub][1])
* **Training**: 10k series, 30/70 class split—use **StratifiedKFold(5)** and `scale_pos_weight≈2.33`; track AUC, PR-AUC, **Brier**, and calibration. (Plan notes invariance, mean/variance shift capture, and residual-based tests.)

If you want, I can now generate a `tsfm_features.py` that matches the repo API and the A–E feature blocks.

[1]: https://github.com/google-research/timesfm "GitHub - google-research/timesfm: TimesFM (Time Series Foundation Model) is a pretrained time-series foundation model developed by Google Research for time-series forecasting."
[2]: https://huggingface.co/google/timesfm-2.5-200m-pytorch "google/timesfm-2.5-200m-pytorch · Hugging Face"
