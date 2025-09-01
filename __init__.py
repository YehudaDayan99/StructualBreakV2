"""
ADIA - Structural Breakpoint Detection Package
"""

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

__version__ = "1.0.0"
__author__ = "ADIA Team"
__description__ = "Structural Breakpoint Detection Package"


def quick_setup(energy_enable: bool = False, B_boot: int = 80, n_jobs: int = None, seed: int = 123) -> dict:
    from .config import N_JOBS
    if n_jobs is None:
        n_jobs = N_JOBS
    return {
        'energy_enable': energy_enable,
        'B_boot': B_boot,
        'n_jobs': n_jobs,
        'seed': seed
    }


def process_time_series(input_path: str, output_pred_path: str, output_meta_path: str, **kwargs) -> tuple:
    return run_batch(input_path, output_pred_path, output_meta_path, **kwargs)
