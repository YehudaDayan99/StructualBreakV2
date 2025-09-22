from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import pandas as pd
from tqdm import tqdm

from .config import TSFMConfig
from .feature_extractor import extract_tsfm_predictors


def run_tsfm_batch(
    input_parquet: str,
    out_pred_parquet: str,
    n_ids: Optional[int] = None,
    random_sample: bool = False,
    seed: int = 42,
    cfg: Optional[TSFMConfig] = None,
) -> pd.DataFrame:
    """Run TSFM feature extraction over a dataset and save CSV.

    Mirrors shape of other batch processors; returns the DataFrame of predictors.
    """
    cfg = cfg or TSFMConfig()
    X = pd.read_parquet(input_parquet)
    if not isinstance(X.index, pd.MultiIndex):
        if {"id", "time"}.issubset(X.columns):
            X = X.set_index(["id", "time"]).sort_index()
        else:
            raise ValueError("Input must have MultiIndex [id,time] or columns ['id','time'].")
    required = {"value", "period"}
    if not required.issubset(X.columns):
        raise ValueError("Input must contain columns ['value','period'].")

    ids_all: List[int] = X.index.get_level_values(0).unique().tolist()
    if (n_ids is None) or (n_ids <= 0) or (n_ids >= len(ids_all)):
        sel_ids = ids_all
    else:
        if random_sample:
            import numpy as np
            rng = np.random.default_rng(seed)
            sel_ids = list(rng.choice(ids_all, size=n_ids, replace=False))
        else:
            sel_ids = ids_all[:n_ids]

    rows = []
    for id_ in tqdm(sel_ids, desc="TSFM (ids)", total=len(sel_ids)):
        df = X.xs(id_, level=0)
        vals = df["value"].to_numpy(float)
        per = df["period"].to_numpy(int)
        feats = extract_tsfm_predictors(vals, per, cfg=cfg)
        feats["id"] = id_
        rows.append(feats)

    F = pd.DataFrame(rows).set_index("id").sort_index()
    out_path = Path(out_pred_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        F.to_csv(out_path)
    else:
        F.to_parquet(out_path)
    return F


