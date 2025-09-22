from __future__ import annotations

"""TSFM example runner — mirrors wavelet_21_example_full.run_training.
Uses placeholder engine by default; switch to TimesFM via config.
"""

import io
import contextlib
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Prefer absolute import via package; fallback to local repo-relative path
try:
    from StructualBreakV2.methods.TSFM.feature_extractor import TSFMConfig, extract_tsfm_predictors
except Exception:  # pragma: no cover
    from methods.TSFM.feature_extractor import TSFMConfig, extract_tsfm_predictors


def run_training(
    x_path: str,
    out_path: str,
    cfg: TSFMConfig,
    n_ids: Optional[int] = None,
    random_sample: bool = False,
    seed: int = 42,
) -> None:
    print("\n[TSFM] Training run: loading X ...")
    X = pd.read_parquet(x_path)
    if not isinstance(X.index, pd.MultiIndex):
        if {"id", "time"}.issubset(set(X.columns)):
            X = X.set_index(["id", "time"]).sort_index()
        else:
            X.index = X.index.set_names(["id", "time"]) if X.index.nlevels == 2 else X.index
    X = X.sort_index()

    required_cols = {"value", "period"}
    if not required_cols.issubset(set(X.columns)):
        raise ValueError("X must contain columns: ['value','period']")

    all_ids = X.index.get_level_values(0).unique().tolist()
    if (n_ids is None) or (n_ids <= 0) or (n_ids >= len(all_ids)):
        sel_ids = all_ids
    else:
        if random_sample:
            import numpy as np
            rng = np.random.default_rng(seed)
            sel_ids = list(rng.choice(all_ids, size=n_ids, replace=False))
        else:
            sel_ids = all_ids[:n_ids]
    print(f"Processing {len(sel_ids)} ids (out of {len(all_ids)}) ...")

    rows = []
    for id_ in tqdm(sel_ids, desc="TSFM (ids)", total=len(sel_ids)):
        df = X.xs(id_, level=0)
        vals = df["value"].to_numpy(float)
        per = df["period"].to_numpy(int)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            feats = extract_tsfm_predictors(vals, per, cfg=cfg)
        feats["id"] = id_
        rows.append(feats)

    F = pd.DataFrame(rows).set_index("id").sort_index()
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    F.to_csv(out_path_p)
    print(f"Saved {out_path} with shape {F.shape}")


