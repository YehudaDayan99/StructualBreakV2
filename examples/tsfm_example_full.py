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

# Prefer absolute import via package; fallback to methods/ path only (avoids package root)
try:
    from StructualBreakV2.methods.TSFM.feature_extractor import TSFMConfig, extract_tsfm_predictors, extract_tsfm_predictors_batch
except Exception:  # pragma: no cover
    import sys, os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    methods_path = os.path.join(repo_root, "methods")
    if methods_path not in sys.path:
        sys.path.insert(0, methods_path)
    from TSFM.feature_extractor import TSFMConfig, extract_tsfm_predictors


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

    # Bucket ids by similar horizon H to batch forecasts
    # Build (values, periods) per id
    series_map = {}
    for id_ in sel_ids:
        df = X.xs(id_, level=0)
        series_map[id_] = (df["value"].to_numpy(float), df["period"].to_numpy(int))

    # Simple horizon buckets
    def horizon_for(id_):
        per = series_map[id_][1]
        return int((per == 1).sum())

    buckets = {128: [], 256: [], 512: [], 1024: [], 10_000: []}
    for id_ in sel_ids:
        H = horizon_for(id_)
        if H <= 128:
            buckets[128].append(id_)
        elif H <= 256:
            buckets[256].append(id_)
        elif H <= 512:
            buckets[512].append(id_)
        elif H <= 1024:
            buckets[1024].append(id_)
        else:
            buckets[10_000].append(id_)

    rows = []
    for cap, ids in buckets.items():
        if not ids:
            continue
        batch = [series_map[i] for i in ids]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            feats_list = extract_tsfm_predictors_batch(batch, cfg=cfg)
        for i, id_ in enumerate(ids):
            feats = feats_list[i]
            feats["id"] = id_
            rows.append(feats)

    F = pd.DataFrame(rows).set_index("id").sort_index()
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    F.to_csv(out_path_p)
    print(f"Saved {out_path} with shape {F.shape}")


