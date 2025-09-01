"""
Predictor extraction for ADIA structural breakpoint detection.
"""

import numpy as np
from typing import Tuple, Dict, Any
from .core_statistics import (
    make_grid, robust_standardize, mean_variance_test, mann_whitney_p,
    energy_distance, permutation_pvalue, autocorrelation
)
from .conditional_tests import conditional_test_summary
from .residual_analysis import ar1_residuals, prepare_state_arrays
from .config import (
    B_BOOT, ENERGY_ENABLE, ENERGY_B, ENERGY_MAX_N_PER_PERIOD, SEED,
    ACF_LAGS, get_bandwidth
)


def compute_predictors_for_values(vals: np.ndarray, per: np.ndarray, 
                                B_boot: int = None, seed: int = None,
                                energy_enable: bool = None, energy_B: int = None, 
                                energy_max_n: int = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
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

    predictors: Dict[str, float] = {}
    meta: Dict[str, Any] = {}

    def conditional_block(state: str, label: str):
        prep = prepare_state_arrays(vals, per, state)
        if prep is None:
            predictors[f'p_mu_{label}'] = 0.5
            predictors[f'p_sigma_{label}'] = 0.5
            predictors[f'overlap_frac_{label}'] = 0.0
            meta[f'n_total_{label}'] = 0
            meta[f'n_p0_{label}'] = 0
            meta[f'n_p1_{label}'] = 0
            return
        X, Y, perY, idx_split = prep
        n = len(X)
        h = get_bandwidth(n)
        x_grid = make_grid(X, h)
        test_results = conditional_test_summary(x_grid, X, Y, idx_split, h, B_boot, seed)
        predictors[f'p_mu_{label}'] = test_results['p_mu']
        predictors[f'p_sigma_{label}'] = test_results['p_sigma']
        predictors[f'overlap_frac_{label}'] = test_results['overlap_mu']
        meta[f'n_total_{label}'] = test_results['n_total']
        meta[f'n_p0_{label}'] = test_results['n_before']
        meta[f'n_p1_{label}'] = test_results['n_after']

    conditional_block('lag1', 'lag1')
    conditional_block('vol', 'vol')

    resid = ar1_residuals(vals)
    per_r = per[1:]
    if len(resid) >= 3 and len(per_r) >= 2:
        Xr = resid[:-1]
        Yr = resid[1:]
        perYr = per_r[1:]
        if np.any(perYr == 0) and np.any(perYr == 1):
            idx_split_r = int(np.where(perYr == 0)[0].max())
            n_r = len(Xr)
            h_r = get_bandwidth(n_r)
            x_grid_r = make_grid(Xr, h_r)
            test_results_r = conditional_test_summary(x_grid_r, Xr, Yr, idx_split_r, h_r, B_boot, seed)
            predictors['p_mu_resid_lag1'] = test_results_r['p_mu']
            predictors['p_sigma_resid_lag1'] = test_results_r['p_sigma']
            predictors['overlap_frac_resid_lag1'] = test_results_r['overlap_mu']
        else:
            predictors['p_mu_resid_lag1'] = 0.5
            predictors['p_sigma_resid_lag1'] = 0.5
            predictors['overlap_frac_resid_lag1'] = 0.0
    else:
        predictors['p_mu_resid_lag1'] = 0.5
        predictors['p_sigma_resid_lag1'] = 0.5
        predictors['overlap_frac_resid_lag1'] = 0.0

    prep_lag1 = prepare_state_arrays(vals, per, 'lag1')
    if prep_lag1 is None:
        predictors['p_mean'] = 0.5
        predictors['p_var'] = 0.5
        predictors['p_MWU'] = 0.5
        predictors['p_energy'] = np.nan
        meta['n_period0'] = 0
        meta['n_period1'] = 0
        meta['n_total'] = 0
    else:
        Xl, Yl, perY_lag1, idx = prep_lag1
        y0 = Yl[:idx+1]
        y1 = Yl[idx+1:]
        Yz = robust_standardize(Yl)
        y0z = Yz[:idx+1]
        y1z = Yz[idx+1:]
        p_mean, p_var = mean_variance_test(y0z, y1z)
        predictors['p_mean'] = p_mean
        predictors['p_var'] = p_var
        predictors['p_MWU'] = mann_whitney_p(y0z, y1z)
        if energy_enable:
            def subsample(a, m):
                if len(a) <= m:
                    return a
                idxs = np.random.default_rng(seed).choice(len(a), size=m, replace=False)
                return a[idxs]
            a0 = subsample(y0z, energy_max_n)
            a1 = subsample(y1z, energy_max_n)
            E_obs = energy_distance(a0, a1)
            predictors['p_energy'] = permutation_pvalue(E_obs, a0, a1, lambda u, v: energy_distance(u, v), B=energy_B, seed=seed)
        else:
            predictors['p_energy'] = np.nan
        acf0 = np.array([autocorrelation(y0z, k) for k in ACF_LAGS])
        acf1 = np.array([autocorrelation(y1z, k) for k in ACF_LAGS])
        predictors['acf_absdiff_l1'] = float(np.sum(np.abs(acf0 - acf1)))
        meta['n_period0'] = idx + 1
        meta['n_period1'] = len(Yl) - idx - 1
        meta['n_total'] = len(Yl)

    return predictors, meta


def validate_predictors(predictors: Dict[str, float]) -> bool:
    required_keys = [
        'p_mu_lag1', 'p_sigma_lag1', 'overlap_frac_lag1',
        'p_mu_vol', 'p_sigma_vol', 'overlap_frac_vol',
        'p_mu_resid_lag1', 'p_sigma_resid_lag1', 'overlap_frac_resid_lag1',
        'p_mean', 'p_var', 'p_MWU', 'acf_absdiff_l1'
    ]
    for key in required_keys:
        if key not in predictors:
            return False
    p_value_keys = [k for k in predictors.keys() if k.startswith('p_') and k != 'p_energy']
    for key in p_value_keys:
        if not (0 <= predictors[key] <= 1):
            return False
    overlap_keys = [k for k in predictors.keys() if k.startswith('overlap_frac_')]
    for key in overlap_keys:
        if not (0 <= predictors[key] <= 1):
            return False
    return True


def get_predictor_summary(predictors: Dict[str, float]) -> Dict[str, Any]:
    if not validate_predictors(predictors):
        return {'valid': False, 'error': 'Invalid predictors'}
    p_values = {k: v for k, v in predictors.items() if k.startswith('p_') and k != 'p_energy'}
    overlap_fracs = {k: v for k, v in predictors.items() if k.startswith('overlap_frac_')}
    summary = {
        'valid': True,
        'n_predictors': len(predictors),
        'p_value_stats': {
            'mean': float(np.mean(list(p_values.values()))),
            'std': float(np.std(list(p_values.values()))),
            'min': float(np.min(list(p_values.values()))),
            'max': float(np.max(list(p_values.values())))
        },
        'overlap_stats': {
            'mean': float(np.mean(list(overlap_fracs.values()))),
            'std': float(np.std(list(overlap_fracs.values()))),
            'min': float(np.min(list(overlap_fracs.values()))),
            'max': float(np.max(list(overlap_fracs.values())))
        },
        'acf_absdiff': predictors.get('acf_absdiff_l1', np.nan)
    }
    return summary
