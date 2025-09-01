#!/usr/bin/env python3
"""
Main entry point for ADIA structural breakpoint detection.
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adia_refactored import (
    run_batch, quick_setup, validate_config, get_processing_summary
)
from adia_refactored.config import (
    DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PRED_PATH, DEFAULT_OUTPUT_META_PATH
)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADIA - Structural Breakpoint Detection")
    parser.add_argument('--input', '-i', type=str, default=DEFAULT_INPUT_PATH, help='Input parquet file path')
    parser.add_argument('--output-pred', '-op', type=str, default=DEFAULT_OUTPUT_PRED_PATH, help='Output predictors parquet file path')
    parser.add_argument('--output-meta', '-om', type=str, default=DEFAULT_OUTPUT_META_PATH, help='Output metadata parquet file path')
    parser.add_argument('--bootstrap', '-b', type=int, default=80, help='Bootstrap replicates')
    parser.add_argument('--energy-enable', action='store_true', help='Enable energy distance tests')
    parser.add_argument('--energy-perm', '-ep', type=int, default=40, help='Permutations for energy tests')
    parser.add_argument('--energy-max-n', '-emn', type=int, default=400, help='Max sample size for energy tests')
    parser.add_argument('--n-jobs', '-j', type=int, default=None, help='Parallel jobs')
    parser.add_argument('--seed', '-s', type=int, default=123, help='Random seed')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--validate-only', action='store_true', help='Validate config and exit')
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    try:
        validate_config()
        if args.validate_only:
            logger.info("Validation only - exiting")
            return 0
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file does not exist: {input_path}")
            return 1
        output_pred_path = Path(args.output_pred)
        output_meta_path = Path(args.output_meta)
        output_pred_path.parent.mkdir(parents=True, exist_ok=True)
        output_meta_path.parent.mkdir(parents=True, exist_ok=True)
        config = quick_setup(energy_enable=args.energy_enable, B_boot=args.bootstrap, n_jobs=args.n_jobs, seed=args.seed)
        logger.info(f"Config: {config}")
        pred_df, meta_df = run_batch(
            input_parquet=str(input_path),
            out_pred_parquet=str(output_pred_path),
            out_meta_parquet=str(output_meta_path),
            B_boot=args.bootstrap,
            seed=args.seed,
            energy_enable=args.energy_enable,
            energy_B=args.energy_perm,
            energy_max_n=args.energy_max_n,
            n_jobs=config['n_jobs'],
            verbose=True
        )
        summary = get_processing_summary(pred_df, meta_df)
        logger.info(f"Processed {summary['n_series']} series")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
