#!/usr/bin/env python3
"""
Wavelet21 — example script (synthetic demo + simple training run)
================================================================

This script demonstrates the Wavelet21 feature extractor in two steps:
  1) **Synthetic demo** — generate a toy series with a structural break and print features.
  2) **Simple training run** — read `X_train.parquet` (MultiIndex [id,time], columns ['value','period']),
     compute features per id, and write a CSV. You can optionally select a subset of ids.

Usage
-----
# Run synthetic demo + full training on all ids
python wavelet21_example_full.py --x X_train.parquet --out Wavelet.csv

# Same, but only process a subset of 500 randomly sampled ids
python wavelet21_example_full.py --x X_train.parquet --out Wavelet_subset.csv \
  --n_ids 500 --random_sample --seed 42

# Skip the synthetic demo
python wavelet21_example_full.py --x X_train.parquet --out Wavelet.csv --skip_demo

# Wavelet config tweaks (paper-like defaults)
python wavelet21_example_full.py --x X_train.parquet --out Wavelet.csv \
  --alpha 0.05 --J 3 --wavelet la8 --cache threshold_cache.json

Dependencies
------------
- numpy, pandas, scipy, statsmodels, pywavelets, (optional) arch
- Make sure you replaced your repo's feature extractor with the new one we wrote.

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence
import io
import contextlib

import numpy as np
import pandas as pd
from tqdm import tqdm

# Preferred import from repo layout; fallback to local file
try:
    from methods.wavelet21.feature_extractor import (
        extract_wavelet_predictors,
        ThresholdCache,
        WaveletConfig,
    )
except Exception:  # pragma: no cover
    from feature_extractor import (
        extract_wavelet_predictors,
        ThresholdCache,
        WaveletConfig,
    )


# -----------------------------
# Synthetic demo utilities
# -----------------------------

def make_synthetic(
    n0: int = 500,
    n1: int = 500,
    kind: str = "var_shift",  # 'var_shift' | 'mean_shift' | 'both'
    mu0: float = 0.0,
    mu1: float = 0.0,
    sigma0: float = 1.0,
    sigma1: float = 2.0,
    phi: float = 0.3,  # AR(1) coeff both segments
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a toy series with a single 0→1 boundary.

    Returns
    -------
    values: np.ndarray shape (n0+n1,)
    periods: np.ndarray of 0/1 labels
    """
    rng = np.random.default_rng(seed)
    e0 = rng.standard_normal(n0)
    e1 = rng.standard_normal(n1)
    x0 = np.zeros(n0, dtype=float)
    x1 = np.zeros(n1, dtype=float)

    if kind in ("mean_shift", "both"):
        mu1_use = mu1
    else:
        mu1_use = mu0

    if kind in ("var_shift", "both"):
        s0, s1 = sigma0, sigma1
    else:
        s0, s1 = sigma0, sigma0

    # AR(1) in each segment
    for t in range(n0):
        x0[t] = mu0 + phi * (x0[t-1] if t > 0 else 0.0) + s0 * e0[t]
    for t in range(n1):
        x1[t] = mu1_use + phi * (x1[t-1] if t > 0 else 0.0) + s1 * e1[t]

    values = np.concatenate([x0, x1], axis=0)
    periods = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)], axis=0)
    return values, periods


def run_synthetic_demo(cfg: WaveletConfig, cache_path: str) -> None:
    print("\n[1/2] Synthetic demo: variance shift (sigma 1 → 2) with AR(1) phi=0.3 ...")
    vals, per = make_synthetic(kind="var_shift")
    feats = extract_wavelet_predictors(vals, per, thr_cache=ThresholdCache(cache_path), cfg=cfg)
    # Show a compact selection of the most indicative features
    keys = [
        "S_local_max_over_j",
        "cnt_local_sum_over_j",
        "cnt_common_sum_over_j",
        "log_energy_ratio_l2norm_over_j",
        "j1_S_local", "j2_S_local", "j3_S_local",
        "j1_log_energy_ratio", "j2_log_energy_ratio", "j3_log_energy_ratio",
        "arch_lm_p", "ljungbox_p_ac_10", "ljungbox_p_ac2_10",
    ]
    print("Top features (variance shift):")
    for k in keys:
        if k in feats:
            print(f"  {k:30s} = {feats[k]:.6f}")

    print("\nSynthetic demo: mean shift (+1.0) with same variance ...")
    vals2, per2 = make_synthetic(kind="mean_shift", mu0=0.0, mu1=1.0, sigma0=1.0, sigma1=1.0)
    feats2 = extract_wavelet_predictors(vals2, per2, thr_cache=ThresholdCache(cache_path), cfg=cfg)
    for k in keys:
        if k in feats2:
            print(f"  {k:30s} = {feats2[k]:.6f}")


# -----------------------------
# Training run
# -----------------------------

def pick_subset(ids: Sequence, n_ids: Optional[int], random_sample: bool, seed: int) -> list:
    ids_unique = list(dict.fromkeys(ids))  # preserve order
    if (n_ids is None) or (n_ids <= 0) or (n_ids >= len(ids_unique)):
        return ids_unique
    if random_sample:
        rng = np.random.default_rng(seed)
        return list(rng.choice(ids_unique, size=n_ids, replace=False))
    return ids_unique[:n_ids]


def run_training(
    x_path: str,
    out_path: str,
    cfg: WaveletConfig,
    cache_path: str,
    n_ids: Optional[int] = None,
    random_sample: bool = False,
    seed: int = 42,
) -> None:
    print("\n[2/2] Training run: loading X ...")
    X = pd.read_parquet(x_path)

    # Ensure MultiIndex [id,time]
    if not isinstance(X.index, pd.MultiIndex):
        if {"id", "time"}.issubset(set(X.columns)):
            X = X.set_index(["id", "time"]).sort_index()
        else:
            X.index = X.index.set_names(["id", "time"]) if X.index.nlevels == 2 else X.index
    X = X.sort_index()

    # Sanity check
    required_cols = {"value", "period"}
    if not required_cols.issubset(set(X.columns)):
        raise ValueError("X must contain columns: ['value','period']")

    all_ids = X.index.get_level_values(0).unique().tolist()
    sel_ids = pick_subset(all_ids, n_ids=n_ids, random_sample=random_sample, seed=seed)
    print(f"Processing {len(sel_ids)} ids (out of {len(all_ids)}) ...")

    cache = ThresholdCache(cache_path)
    rows = []
    for id_ in tqdm(sel_ids, desc="Wavelet21 (ids)", total=len(sel_ids)):
        df = X.xs(id_, level=0)
        vals = df["value"].to_numpy(float)
        per = df["period"].to_numpy(int)
        # Silence optimizer/diagnostic chatter from downstream libs per-id
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            feats = extract_wavelet_predictors(vals, per, thr_cache=cache, cfg=cfg)
        feats["id"] = id_
        rows.append(feats)

    F = pd.DataFrame(rows).set_index("id").sort_index()
    F.to_csv(out_path)
    print(f"Saved {out_path} with shape {F.shape}")
    # Final preview for quick inspection
    try:
        preview = pd.read_csv(out_path).head()
        print("\nPreview (head):")
        print(preview)
    except Exception:
        pass


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Wavelet21 example: synthetic demo + simple training run")
    ap.add_argument("--x", required=True, help="Path to X_train.parquet")
    ap.add_argument("--out", default="Wavelet.csv", help="Output CSV path (features per id)")
    ap.add_argument("--alpha", type=float, default=0.05, help="Exceedance level for thresholds")
    ap.add_argument("--J", type=int, default=3, help="MODWT levels (paper: ≤3)")
    ap.add_argument("--wavelet", type=str, default="la8", help="Wavelet family (default la8)")
    ap.add_argument("--cache", type=str, default="threshold_cache.json", help="Threshold cache file")
    ap.add_argument("--n_ids", type=int, default=None, help="If set, process only this many ids")
    ap.add_argument("--random_sample", action="store_true", help="If set, sample ids at random")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling and synthetic demo")
    ap.add_argument("--skip_demo", action="store_true", help="Skip the synthetic demo step")
    args = ap.parse_args()

    cfg = WaveletConfig(wavelet=args.wavelet, J=args.J, alpha=args.alpha)
    # Enable residual-first + period-aware contrasts (DWT reconstruction) via config if desired
    # cfg.use_residuals = True
    # cfg.contrast_engine = "recon"  # or "swt" for legacy

    if not args.skip_demo:
        run_synthetic_demo(cfg, cache_path=args.cache)

    run_training(
        x_path=args.x,
        out_path=args.out,
        cfg=cfg,
        cache_path=args.cache,
        n_ids=args.n_ids,
        random_sample=args.random_sample,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
