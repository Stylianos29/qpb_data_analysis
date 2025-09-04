#!/usr/bin/env python3
"""
PCAC Mass Plateau Extraction Script

This script extracts plateau PCAC mass values from PCAC mass time series
using jackknife analysis and robust plateau detection methods.

The script processes HDF5 files from calculate_PCAC_mass.py, detects
plateau regions, and exports results to CSV.

Usage:
    python extract_plateau_PCAC_mass.py -i pcac_mass_analysis.h5 -o
    output_dir
"""

import os
import sys
from pathlib import Path
from typing import Optional

import click

# Import library components
from library.utils.logging_utilities import create_script_logger
from library.validation.click_validators import (
    hdf5_file,
    directory,
    validate_log_filename,
)

# Import configuration
from src.analysis.plateau_extraction._plateau_extraction_shared_config import (
    PLATEAU_DETECTION_SIGMA_THRESHOLDS,
    MIN_PLATEAU_SIZE,
    CSV_OUTPUT_CONFIG,
)
from src.analysis.plateau_extraction._pcac_plateau_config import (
    INPUT_DATASETS,
    TIME_OFFSET,
    APPLY_SYMMETRIZATION,
    SYMMETRIZATION_TRUNCATION,
    PLATEAU_SEARCH_RANGE,
    DEFAULT_OUTPUT_HDF5_FILENAME,
    DEFAULT_OUTPUT_CSV_FILENAME,
    OUTPUT_COLUMN_PREFIX,
    validate_pcac_config,
)

# Import core functions
from src.analysis.plateau_extraction._plateau_extraction_core import (
    process_all_groups,
    export_to_csv,
)


@click.command()
@click.option(
    "-i",
    "--input_hdf5_file",
    required=True,
    callback=hdf5_file.input,
    help="Path to input HDF5 file containing PCAC mass analysis results.",
)
@click.option(
    "-o",
    "--output_directory",
    required=True,
    callback=directory.can_create,
    help="Directory for output CSV file.",
)
@click.option(
    "-out_h5",
    "--output_hdf5_filename",
    default=DEFAULT_OUTPUT_HDF5_FILENAME,
    help=f"Output HDF5 filename. Default: {DEFAULT_OUTPUT_HDF5_FILENAME}",
)
@click.option(
    "-out_csv",
    "--output_csv_filename",
    default=DEFAULT_OUTPUT_CSV_FILENAME,
    help=f"Output CSV filename. Default: {DEFAULT_OUTPUT_CSV_FILENAME}",
)
@click.option(
    "--enable_logging/--no_logging",
    default=True,
    help="Enable or disable logging to file.",
)
@click.option(
    "--log_directory",
    callback=directory.can_create,
    help="Directory for log files. Default: same as output directory.",
)
@click.option(
    "--log_filename",
    callback=validate_log_filename,
    help="Custom log filename. Default: auto-generated.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose console output.",
)
def main(
    input_hdf5_file: str,
    output_directory: str,
    output_hdf5_filename: str,
    output_csv_filename: str,
    enable_logging: bool,
    log_directory: Optional[str],
    log_filename: Optional[str],
    verbose: bool,
) -> None:
    """
    Extract plateau PCAC mass values from PCAC mass time series.

    This script processes PCAC mass jackknife samples, detects plateau
    regions, and exports results to CSV format.
    """
    # Validate configuration
    if not validate_pcac_config():
        click.echo("❌ Invalid configuration detected.", err=True)
        sys.exit(1)

    # Setup logging
    if enable_logging:
        log_dir = log_directory or output_directory
    else:
        log_dir = None

    logger = create_script_logger(
        log_directory=log_dir,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=verbose,
    )

    logger.log_script_start("PCAC mass plateau extraction")

    try:
        # Log parameters
        logger.info(f"Input file: {input_hdf5_file}")
        logger.info(f"Output directory: {output_directory}")
        logger.info(f"Output CSV: {output_csv_filename}")

        # Process all groups
        results = process_all_groups(
            input_hdf5_file,
            INPUT_DATASETS,
            APPLY_SYMMETRIZATION,
            SYMMETRIZATION_TRUNCATION,
            PLATEAU_DETECTION_SIGMA_THRESHOLDS,
            MIN_PLATEAU_SIZE,
            PLATEAU_SEARCH_RANGE,
            "PCAC mass",
            logger,
            verbose,
        )

        if not results:
            logger.warning("No results obtained from processing")
            click.echo("⚠️ No results to export", err=True)
            sys.exit(1)

        # Export to CSV
        output_path = os.path.join(output_directory, output_csv_filename)
        export_to_csv(
            results,
            output_path,
            OUTPUT_COLUMN_PREFIX,
            TIME_OFFSET,
            CSV_OUTPUT_CONFIG,
            logger,
        )

        # Report summary
        n_success = sum(1 for r in results if r["success"])
        n_total = len(results)

        logger.log_script_end(f"Extraction complete: {n_success}/{n_total} successful")
        click.echo(
            f"✅ Plateau extraction complete: {n_success}/{n_total} successful\n"
            f"   Results saved to: {output_path}"
        )

    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.log_script_end("PCAC mass plateau extraction failed")
        click.echo(f"❌ ERROR: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
