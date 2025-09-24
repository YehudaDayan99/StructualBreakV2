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
    shard_offset: int = 0,
    shard_size: Optional[int] = None,
    resume_append: bool = True,
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
        base_ids = all_ids
    else:
        if random_sample:
            import numpy as np
            rng = np.random.default_rng(seed)
            base_ids = list(rng.choice(all_ids, size=n_ids, replace=False))
        else:
            base_ids = all_ids[:n_ids]
    # Apply sharding window if requested
    if shard_size is not None and shard_size > 0:
        sel_ids = base_ids[shard_offset: shard_offset + shard_size]
    else:
        sel_ids = base_ids
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

    # Coarser buckets to increase batch sizes
    buckets = {512: [], 1024: [], 10_000: []}
    for id_ in sel_ids:
        H = horizon_for(id_)
        if H <= 512:
            buckets[512].append(id_)
        elif H <= 1024:
            buckets[1024].append(id_)
        else:
            buckets[10_000].append(id_)

    # Parallel input prep: prebuild batches in threads to overlap with GPU compute
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def make_batch(ids):
        # include id as 3rd element so L* cache can persist per-id
        return [(series_map[i][0], series_map[i][1], str(i)) for i in ids]

    # Prepare batches concurrently
    prep_futures = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for cap, ids in buckets.items():
            if ids:
                prep_futures[cap] = ex.submit(make_batch, ids)

        rows = []
        # Consume in deterministic cap order with progress per batch
        from tqdm import tqdm as _tqdm
        progress = _tqdm(total=len(sel_ids), desc="TSFM (ids)")
        try:
            for cap in [512, 1024, 10_000]:
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
                progress.update(len(ids))
        finally:
            progress.close()

    F = pd.DataFrame(rows).set_index("id").sort_index()
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    if resume_append and out_path_p.exists():
        # Append new rows without duplicating existing ids
        try:
            F_prev = pd.read_csv(out_path_p).set_index("id")
            F_combined = pd.concat([F_prev[~F_prev.index.isin(F.index)], F]).sort_index()
            F_combined.to_csv(out_path_p)
            print(f"Appended to {out_path} (now {F_combined.shape})")
        except Exception:
            F.to_csv(out_path_p)
            print(f"Saved {out_path} with shape {F.shape}")
    else:
        F.to_csv(out_path_p)
        print(f"Saved {out_path} with shape {F.shape}")


