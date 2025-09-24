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

    # Build (values, periods) per id using a single groupby (faster than many xs)
    xr = X.reset_index()
    g = xr.groupby('id', sort=False)
    sel_set = set(sel_ids)
    series_map = {}
    for gid, gdf in g:
        if gid in sel_set:
            series_map[gid] = (
                gdf["value"].to_numpy(dtype=float),
                gdf["period"].to_numpy(dtype=int),
            )

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

    # Parallel input prep: prebuild batches in threads to overlap with GPU compute
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def make_batch(ids):
        return [series_map[i] for i in ids]

    # Prepare batches concurrently
    prep_futures = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for cap, ids in buckets.items():
            if ids:
                prep_futures[cap] = ex.submit(make_batch, ids)

        rows = []
        # Consume in deterministic cap order
        for cap in [128, 256, 512, 1024, 10_000]:
            ids = buckets.get(cap, [])
            if not ids:
                continue
            batch = prep_futures[cap].result()
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


