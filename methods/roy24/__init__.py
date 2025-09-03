"""
Roy24 method for structural breakpoint detection.

Implementation of the nonparametric method based on Roy et al. 2024.
"""

from .roy24_method import Roy24Method
from .config import (
    B_BOOT, ENERGY_ENABLE, ENERGY_B, ENERGY_MAX_N_PER_PERIOD,
    SEED, N_JOBS, get_bandwidth, validate_config
)
from .core_statistics import (
    epanechnikov_kernel, make_grid, nw_mean, jackknife_mu, nw_variance,
    robust_standardize, mean_variance_test, ranks_with_ties, tie_counts,
    mann_whitney_p, energy_distance, permutation_pvalue, autocorrelation
)
from .conditional_tests import (
    T_mu_statistic, T_sigma_statistic, circular_shift_bootstrap,
    conditional_test_summary
)
from .residual_analysis import (
    ar1_residuals, prepare_state_arrays, validate_state_arrays,
    get_state_summary, compute_all_states
)
from .predictor_extractor import (
    compute_predictors_for_values, validate_predictors, get_predictor_summary
)
from .batch_processor import (
    process_one_series, validate_input_dataframe, run_batch,
    get_processing_summary
)

__all__ = [
    'Roy24Method',
    'B_BOOT', 'ENERGY_ENABLE', 'ENERGY_B', 'ENERGY_MAX_N_PER_PERIOD',
    'SEED', 'N_JOBS', 'get_bandwidth', 'validate_config',
    'epanechnikov_kernel', 'make_grid', 'nw_mean', 'jackknife_mu', 'nw_variance',
    'robust_standardize', 'mean_variance_test', 'ranks_with_ties', 'tie_counts',
    'mann_whitney_p', 'energy_distance', 'permutation_pvalue', 'autocorrelation',
    'T_mu_statistic', 'T_sigma_statistic', 'circular_shift_bootstrap',
    'conditional_test_summary',
    'ar1_residuals', 'prepare_state_arrays', 'validate_state_arrays',
    'get_state_summary', 'compute_all_states',
    'compute_predictors_for_values', 'validate_predictors', 'get_predictor_summary',
    'process_one_series', 'validate_input_dataframe', 'run_batch',
    'get_processing_summary'
]
