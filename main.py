#!/usr/bin/env python3
"""
Main entry point for Structural Breakpoint Detection Package.

Supports multiple methods: Roy24 and Wavelet21.
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from methods import Roy24Method, Wavelet21Method
from methods.base import CommonConfig


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Structural Breakpoint Detection - Multi-Method Package")
    
    # Method selection
    parser.add_argument('--method', '-m', type=str, default='roy24', 
                       choices=['roy24', 'wavelet21'],
                       help='Detection method to use')
    
    # Input/Output paths
    parser.add_argument('--input', '-i', type=str, 
                       default=CommonConfig.DEFAULT_INPUT_PATH,
                       help='Input parquet file path')
    parser.add_argument('--output-pred', '-op', type=str, 
                       default=CommonConfig.DEFAULT_OUTPUT_PRED_PATH,
                       help='Output predictors file path')
    parser.add_argument('--output-meta', '-om', type=str, 
                       default=CommonConfig.DEFAULT_OUTPUT_META_PATH,
                       help='Output metadata file path')
    
    # Common parameters
    parser.add_argument('--n-jobs', '-j', type=int, default=None, 
                       help='Number of parallel jobs')
    parser.add_argument('--seed', '-s', type=int, default=123, 
                       help='Random seed')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose logging')
    parser.add_argument('--validate-only', action='store_true', 
                       help='Validate config and exit')
    
    # Roy24-specific parameters
    parser.add_argument('--bootstrap', '-b', type=int, default=80, 
                       help='Bootstrap replicates (Roy24)')
    parser.add_argument('--energy-enable', action='store_true', 
                       help='Enable energy distance tests (Roy24)')
    parser.add_argument('--energy-perm', '-ep', type=int, default=40, 
                       help='Permutations for energy tests (Roy24)')
    parser.add_argument('--energy-max-n', '-emn', type=int, default=400, 
                       help='Max sample size for energy tests (Roy24)')
    
    # Wavelet21-specific parameters
    parser.add_argument('--wavelet-type', type=str, default='db4', 
                       help='Wavelet type (Wavelet21)')
    parser.add_argument('--decomposition-levels', type=int, default=4, 
                       help='Number of decomposition levels (Wavelet21)')
    parser.add_argument('--threshold-factor', type=float, default=0.1, 
                       help='Threshold factor for breakpoint detection (Wavelet21)')
    
    return parser.parse_args()


def get_method_config(args: argparse.Namespace) -> dict:
    """Get method-specific configuration from arguments."""
    config = {
        'seed': args.seed,
        'n_jobs': args.n_jobs or CommonConfig.N_JOBS
    }
    
    if args.method == 'roy24':
        config.update({
            'b_boot': args.bootstrap,
            'energy_enable': args.energy_enable,
            'energy_b': args.energy_perm,
            'energy_max_n': args.energy_max_n
        })
    elif args.method == 'wavelet21':
        config.update({
            'wavelet_type': args.wavelet_type,
            'decomposition_levels': args.decomposition_levels,
            'threshold_factor': args.threshold_factor
        })
    
    return config


def run_batch_processing(args: argparse.Namespace) -> int:
    """Run batch processing with selected method."""
    logger = logging.getLogger(__name__)
    
    # Get method configuration
    config = get_method_config(args)
    
    # Initialize method
    if args.method == 'roy24':
        method = Roy24Method(config)
        from methods.roy24.batch_processor import run_batch
    elif args.method == 'wavelet21':
        method = Wavelet21Method(config)
        from methods.wavelet21.batch_processor import run_wavelet_batch as run_batch
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    # Validate configuration
    method.validate_config()
    if args.validate_only:
        logger.info("Validation only - exiting")
        return 0
    
    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return 1
    
    # Create output directories
    output_pred_path = Path(args.output_pred)
    output_meta_path = Path(args.output_meta)
    output_pred_path.parent.mkdir(parents=True, exist_ok=True)
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run batch processing
    logger.info(f"Running {args.method} method with config: {config}")
    
    try:
        pred_df, meta_df = run_batch(
            input_parquet=str(input_path),
            out_pred_parquet=str(output_pred_path),
            out_meta_parquet=str(output_meta_path),
            config=config,
            n_jobs=config['n_jobs'],
            verbose=True
        )
        
        # Get summary
        if args.method == 'roy24':
            from methods.roy24.batch_processor import get_processing_summary
        else:
            from methods.wavelet21.batch_processor import get_wavelet_summary as get_processing_summary
        
        summary = get_processing_summary(pred_df, meta_df)
        logger.info(f"Processed {summary['n_series']} series")
        logger.info(f"Success: {summary.get('n_successful', 'N/A')}, Failed: {summary.get('n_failed', 'N/A')}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting structural breakpoint detection with {args.method} method")
        return run_batch_processing(args)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())