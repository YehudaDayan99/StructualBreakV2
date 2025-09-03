"""
Batch processing for ADIA structural breakpoint detection.
"""

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from typing import Tuple, Dict, Any
import logging

from .predictor_extractor import compute_predictors_for_values
from .config import N_JOBS, SEED, B_BOOT, ENERGY_ENABLE, ENERGY_B, ENERGY_MAX_N_PER_PERIOD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_one_series(id_value, g: pd.DataFrame, B_boot: int = None, seed: int = None,
                      energy_enable: bool = None, energy_B: int = None, energy_max_n: int = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
    try:
        g_sorted = g.sort_index(level='time')
        vals = g_sorted['value'].to_numpy(float)
        per = g_sorted['period'].to_numpy()
        if B_boot is None:
            B_boot = B_BOOT
        if seed is None:
            seed = SEED
        if energy_enable is None:
            energy_enable = ENERGY_ENABLE
        if energy_B is None:
            energy_B = ENERGY_B
        if energy_max_n is None:
            energy_max_n = ENERGY_MAX_N_PER_PERIOD
        series_seed = seed + int(hash(str(id_value)) % 10_000)
        preds, meta = compute_predictors_for_values(vals, per, B_boot=B_boot, seed=series_seed,
                                                    energy_enable=energy_enable, energy_B=energy_B,
                                                    energy_max_n=energy_max_n)
        preds['id'] = id_value
        meta['id'] = id_value
        return preds, meta
    except Exception as e:
        logger.error(f"Error processing series {id_value}: {str(e)}")
        default_preds = {
            'id': id_value,
            'p_mu_lag1': 0.5, 'p_sigma_lag1': 0.5, 'overlap_frac_lag1': 0.0,
            'p_mu_vol': 0.5, 'p_sigma_vol': 0.5, 'overlap_frac_vol': 0.0,
            'p_mu_resid_lag1': 0.5, 'p_sigma_resid_lag1': 0.5, 'overlap_frac_resid_lag1': 0.0,
            'p_mean': 0.5, 'p_var': 0.5, 'p_MWU': 0.5, 'p_energy': np.nan, 'acf_absdiff_l1': 0.0
        }
        default_meta = {
            'id': id_value,
            'n_total_lag1': 0, 'n_p0_lag1': 0, 'n_p1_lag1': 0,
            'n_total_vol': 0, 'n_p0_vol': 0, 'n_p1_vol': 0,
            'n_period0': 0, 'n_period1': 0, 'n_total': 0
        }
        return default_preds, default_meta


def validate_input_dataframe(df: pd.DataFrame) -> bool:
    if isinstance(df.index, pd.MultiIndex):
        if df.index.names != ['id', 'time']:
            logger.warning("MultiIndex should have names ['id', 'time']")
            return False
    else:
        if not {'id', 'time'}.issubset(df.columns):
            logger.error("DataFrame must have 'id' and 'time' columns or MultiIndex [id, time]")
            return False
    required_cols = ['value', 'period']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"DataFrame must have columns: {required_cols}")
        return False
    if not pd.api.types.is_numeric_dtype(df['value']):
        logger.error("'value' column must be numeric")
        return False
    if not pd.api.types.is_numeric_dtype(df['period']):
        logger.error("'period' column must be numeric")
        return False
    unique_periods = df['period'].unique()
    if not all(p in [0, 1] for p in unique_periods):
        logger.error("'period' column must contain only 0 and 1")
        return False
    return True


def run_batch(input_parquet: str, out_pred_parquet: str, out_meta_parquet: str,
              n_jobs: int = None, B_boot: int = None, seed: int = None,
              energy_enable: bool = None, energy_B: int = None, energy_max_n: int = None,
              verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if n_jobs is None:
        n_jobs = N_JOBS
    if B_boot is None:
        B_boot = B_BOOT
    if seed is None:
        seed = SEED
    if energy_enable is None:
        energy_enable = ENERGY_ENABLE
    if energy_B is None:
        energy_B = ENERGY_B
    if energy_max_n is None:
        energy_max_n = ENERGY_MAX_N_PER_PERIOD

    logger.info(f'Loading {input_parquet}')
    df = pd.read_parquet(input_parquet)

    if not validate_input_dataframe(df):
        raise ValueError("Invalid input DataFrame structure")
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(['id', 'time']).sort_index()

    ids = df.index.get_level_values('id').unique()
    logger.info(f'Total series: {len(ids)}')

    iterator = tqdm(ids, desc='Extracting predictors') if verbose else ids
    results = Parallel(n_jobs=n_jobs, verbose=0, batch_size=1)(
        delayed(process_one_series)(idv, df.xs(idv, level='id', drop_level=False),
                                    B_boot, seed, energy_enable, energy_B, energy_max_n)
        for idv in iterator
    )

    pred_rows = [r[0] for r in results]
    meta_rows = [r[1] for r in results]
    pred_df = pd.DataFrame(pred_rows).set_index('id').sort_index()
    meta_df = pd.DataFrame(meta_rows).set_index('id').sort_index()

    # Save outputs based on file extension (Parquet by default, CSV if requested)
    def _save_df(df: pd.DataFrame, path: str) -> None:
        lower = str(path).lower()
        if lower.endswith(".csv"):
            df.to_csv(path, index=True)
        else:
            df.to_parquet(path, index=True)

    _save_df(pred_df, out_pred_parquet)
    _save_df(meta_df, out_meta_parquet)
    logger.info(f'Saved predictors to: {out_pred_parquet}')
    logger.info(f'Saved metadata to: {out_meta_parquet}')

    return pred_df, meta_df


def get_processing_summary(pred_df: pd.DataFrame, meta_df: pd.DataFrame) -> Dict[str, Any]:
    summary = {
        'n_series': len(pred_df),
        'n_predictors': len(pred_df.columns),
        'n_metadata': len(meta_df.columns),
        'predictor_columns': list(pred_df.columns),
        'metadata_columns': list(meta_df.columns),
        'missing_values': {
            'predictors': pred_df.isnull().sum().to_dict(),
            'metadata': meta_df.isnull().sum().to_dict()
        }
    }
    numeric_cols = pred_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['predictor_stats'] = {
            'mean': pred_df[numeric_cols].mean().to_dict(),
            'std': pred_df[numeric_cols].std().to_dict(),
            'min': pred_df[numeric_cols].min().to_dict(),
            'max': pred_df[numeric_cols].max().to_dict()
        }
    return summary
