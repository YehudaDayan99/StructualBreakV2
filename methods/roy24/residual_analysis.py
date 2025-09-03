"""
Residual analysis and state preparation for ADIA structural breakpoint detection.
"""

import numpy as np
from typing import Tuple, Optional
from .config import EPSILON


def ar1_residuals(vals: np.ndarray) -> np.ndarray:
    if len(vals) < 2:
        raise ValueError("Time series must have at least 2 observations")
    y = vals - np.mean(vals)
    y1 = y[1:]
    y0 = y[:-1]
    denom = np.dot(y0, y0) + EPSILON
    phi = float(np.dot(y0, y1) / denom)
    resid = y1 - phi * y0
    return resid


def prepare_state_arrays(vals: np.ndarray, per: np.ndarray, state: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    if state == 'lag1':
        if len(vals) < 2:
            return None
        Y = vals[1:]
        X = vals[:-1]
        perY = per[1:]
    elif state == 'vol':
        if len(vals) < 3:
            return None
        dY = np.diff(vals)
        X = np.abs(dY[:-1])
        Y = vals[2:]
        perY = per[2:]
    else:
        raise ValueError("state must be 'lag1' or 'vol'")
    if not (np.any(perY == 0) and np.any(perY == 1)):
        return None
    idx_split = int(np.where(perY == 0)[0].max())
    return X.astype(float), Y.astype(float), perY, idx_split


def validate_state_arrays(X: np.ndarray, Y: np.ndarray, perY: np.ndarray, idx_split: int) -> bool:
    if len(X) != len(Y) or len(Y) != len(perY):
        return False
    if idx_split < 0 or idx_split >= len(X):
        return False
    if not np.all(np.isin(perY, [0, 1])):
        return False
    if not (np.any(perY == 0) and np.any(perY == 1)):
        return False
    return True


def get_state_summary(vals: np.ndarray, per: np.ndarray, state: str) -> dict:
    prep = prepare_state_arrays(vals, per, state)
    if prep is None:
        return {
            'state': state,
            'valid': False,
            'n_total': 0,
            'n_period0': 0,
            'n_period1': 0,
            'idx_split': -1
        }
    X, Y, perY, idx_split = prep
    if not validate_state_arrays(X, Y, perY, idx_split):
        return {
            'state': state,
            'valid': False,
            'n_total': 0,
            'n_period0': 0,
            'n_period1': 0,
            'idx_split': -1
        }
    return {
        'state': state,
        'valid': True,
        'n_total': len(X),
        'n_period0': idx_split + 1,
        'n_period1': len(X) - idx_split - 1,
        'idx_split': idx_split,
        'X_range': (float(np.min(X)), float(np.max(X))),
        'Y_range': (float(np.min(Y)), float(np.max(Y))),
        'period0_frac': float(np.mean(perY == 0)),
        'period1_frac': float(np.mean(perY == 1))
    }


def compute_all_states(vals: np.ndarray, per: np.ndarray) -> dict:
    states = ['lag1', 'vol']
    summaries = {}
    for state in states:
        summaries[state] = get_state_summary(vals, per, state)
    if len(vals) >= 3:
        resid = ar1_residuals(vals)
        per_r = per[1:]
        summaries['resid_lag1'] = get_state_summary(resid, per_r, 'lag1')
    else:
        summaries['resid_lag1'] = {
            'state': 'resid_lag1',
            'valid': False,
            'n_total': 0,
            'n_period0': 0,
            'n_period1': 0,
            'idx_split': -1
        }
    return summaries
