# ADIA - Structural Breakpoint Detection Package

A modular package for generating features and tests detecting structural breakpoints in time series.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
from StructualBreak import run_batch

pred_df, meta_df = run_batch(
    input_parquet='X_train.parquet',
    out_pred_parquet='X_train_predictors.csv',  # supports .csv or .parquet
    out_meta_parquet='X_train_metadata.csv',
    B_boot=80,
    energy_enable=False,
    n_jobs=1,
    verbose=True,
)
```

## CLI

```bash
python main.py --input data.parquet --output-pred predictors.csv --output-meta metadata.csv
```
