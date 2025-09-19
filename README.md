# Structural Breakpoint Detection Package

A modular package for detecting structural breakpoints in time series data using multiple advanced methods.

## Features

- **Multiple Methods**: Support for Roy24 and Wavelet21 detection methods
- **Modular Architecture**: Easy to add new detection methods
- **Batch Processing**: Efficient processing of multiple time series
- **Parallel Processing**: Multi-core support for faster computation
- **Comprehensive Output**: Detailed predictors and metadata for each series

## Available Methods

### Roy24 Method
Nonparametric method based on Roy et al. 2024, featuring:
- Kernel smoothing with Epanechnikov kernel
- Conditional statistical tests
- Bootstrap-based significance testing
- Energy distance analysis
- Autocorrelation analysis

### Wavelet21 Method
Wavelet-based method featuring:
- Multi-resolution wavelet decomposition
- Frequency domain analysis
- Breakpoint localization
- Frequency band energy analysis

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Using Roy24 Method (Default)

```python
from StructualBreak import run_batch

pred_df, meta_df = run_batch(
    input_parquet='X_train.parquet',
    out_pred_parquet='X_train_predictors.parquet',
    out_meta_parquet='X_train_metadata.parquet',
    method='roy24',
    B_boot=80,
    energy_enable=False,
    n_jobs=4,
    verbose=True
)
```

### Using Wavelet21 Method

```python
from StructualBreakV2 import run_batch

# Residual-first + period-aware contrasts (reconstruction engine)
pred_df, meta_df = run_batch(
    input_parquet='X_train.parquet',
    out_pred_parquet='X_train_predictors.parquet',
    out_meta_parquet='X_train_metadata.parquet',
    method='wavelet21',
    # Wavelet21 accepts method-specific parameters via the config dict:
    config={
        'wavelet': 'sym4',
        'J': 3,
        'alpha': 0.05,
        'use_residuals': True,
        'contrast_engine': 'recon',  # or 'swt' for legacy contrasts
        # null_model uses defaults; override via nested fields if needed
    },
    n_jobs=2,
    verbose=True,
)
```

### Wavelet21 – advanced options (residual-first uplift)

These options are opt-in via the `config` dict (defaults preserve legacy behavior).

```python
config={
  'wavelet': 'sym4',        # mother wavelet (e.g., 'sym4', 'db8')
  'J': 3,                   # levels (capped by signal length)
  'alpha': 0.05,            # exceedance/MC level
  'use_residuals': True,    # residual-first pipeline (H0 fit on full series)
  'contrast_engine': 'recon',   # 'recon' (DWT reconstruction) or 'coef_swt' (SWT; shift-invariant)
  'wavelets': ('sym4','db8'),   # multi-family sweep (optional)
  'windows': (16,32,64),    # boundary-local windows (optional)
}
```

New columns (per wavelet family `{w}` and level `L{j}`):
- Contrasts: `wav_{w}_L{j}_var_logratio`, `wav_{w}_L{j}_mad_logratio`, `wav_{w}_L{j}_energy_logratio`
- Boundary-local: `wav_{w}_L{j}_localmax_w{16|32|64}`, `wav_{w}_L{j}_exceed_w{16|32|64}` (MC-calibrated; cached)
- Coefficient-domain tests: `wav_{w}_L{j}_ttest_stat_signed`, `wav_{w}_L{j}_ttest_neglog10p`, `wav_{w}_L{j}_ftest_neglog10p`

Residual diagnostics (with `use_residuals=True`):
- `h0_ljungbox_p`, `h0_archlm_p`, `h0_err_is_t`, `h0_t_nu`, plus `h0_lb_p_resid2`, `h0_kurtosis_resid`

Notes:
- `'coef_swt'` engine computes contrasts on SWT detail coefficients (shift-invariant). If SWT is unusable on a series, it safely falls back to `'recon'` to preserve columns.
- `ThresholdCache` is re-used to store MC thresholds for boundary features, keyed by `(n=window, J, wavelet, alpha)`.

### Single Series Analysis

```python
from StructualBreak import compute_predictors_for_values
import numpy as np

# Generate sample data with structural break
np.random.seed(42)
n = 120
break_point = 60
values = np.concatenate([
    np.random.normal(0, 1.0, break_point),
    np.random.normal(1.5, 1.5, n - break_point),
])
periods = np.concatenate([np.zeros(break_point), np.ones(n - break_point)])

# Analyze with Roy24 method
preds, meta = compute_predictors_for_values(
    values, periods, 
    method='roy24',
    B_boot=20, 
    energy_enable=False
)

# Analyze with Wavelet21 method
preds_wavelet, meta_wavelet = compute_predictors_for_values(
    values, periods, 
    method='wavelet21',
    wavelet_type='db4',
    decomposition_levels=4
)
```

## Command Line Interface

### Roy24 Method

```bash
python main.py --method roy24 \
    --input data.parquet \
    --output-pred predictors.csv \
    --output-meta metadata.csv \
    --bootstrap 80 \
    --energy-enable \
    --n-jobs 4
```

### Wavelet21 Method

```bash
python main.py --method wavelet21 \
    --input data.parquet \
    --output-pred predictors.csv \
    --output-meta metadata.csv \
    --wavelet-type db4 \
    --decomposition-levels 4 \
    --threshold-factor 0.1 \
    --n-jobs 4
```

## Project Structure

```
StructualBreakV2/
├── methods/                          # All modeling techniques
│   ├── base/                         # Common interfaces and utilities
│   │   ├── base_method.py           # Abstract base class
│   │   ├── common_config.py         # Shared configuration
│   │   └── utils.py                 # Common utilities
│   ├── roy24/                       # Roy24 implementation
│   │   ├── config.py
│   │   ├── core_statistics.py
│   │   ├── conditional_tests.py
│   │   ├── residual_analysis.py
│   │   ├── predictor_extractor.py
│   │   ├── batch_processor.py
│   │   └── roy24_method.py
│   └── wavelet21/                   # Wavelet21 implementation
│       ├── config.py
│       ├── wavelet_analysis.py
│       ├── feature_extractor.py
│       ├── batch_processor.py
│       └── wavelet21_method.py
├── examples/                        # Example notebooks
│   ├── roy24_example.ipynb
│   ├── getting_started.ipynb
│   └── comparison_example.ipynb
├── tests/                          # Unit tests
├── main.py                         # CLI with method selection
├── config.py                       # Global configuration
├── requirements.txt
└── README.md
```

## Method Comparison

| Feature | Roy24 | Wavelet21 |
|---------|-------|-----------|
| Approach | Nonparametric kernel smoothing | Wavelet decomposition |
| Strengths | Robust to noise, well-tested | Multi-resolution analysis |
| Best for | General time series | Frequency domain changes |
| Parameters | Bootstrap reps, energy tests | Wavelet type, decomposition levels |
| Speed | Moderate | Fast |

## Adding New Methods

To add a new detection method:

1. Create a new directory under `methods/`
2. Implement the `BaseMethod` interface
3. Add method-specific modules (config, analysis, batch processing)
4. Update `methods/__init__.py` to include the new method
5. Add CLI support in `main.py`

See the existing `roy24` and `wavelet21` implementations for examples.

## Examples

Check the `examples/` directory for:
- `roy24_example.ipynb`: Roy24 method demonstration
- `getting_started.ipynb`: General introduction
- `comparison_example.ipynb`: Comparing both methods

## Requirements

- Python 3.7+
- numpy>=1.21.0
- pandas>=1.3.0
- scipy>=1.7.0
- joblib>=1.1.0
- tqdm>=4.62.0
- pyarrow>=6.0.0

## License

This project is licensed under the MIT License.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{structural_break_detection,
  title={Structural Breakpoint Detection Package},
  author={Structural Breakpoint Detection Team},
  year={2024},
  url={https://github.com/your-repo/StructualBreakV2}
}
```

## Contributing

Contributions are welcome! Please see our contributing guidelines for details on how to submit pull requests, report issues, or suggest new features.