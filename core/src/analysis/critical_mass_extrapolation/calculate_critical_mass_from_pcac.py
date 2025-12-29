#!/usr/bin/env python3
"""
Critical mass calculation from PCAC plateau estimates.

Usage: python calculate_critical_mass_from_pcac.py \
    -i plateau_pcac.csv \
    -o output.csv
"""

from pathlib import Path
import sys

import click

from library.validation.click_validators import (
    csv_file,
    directory,
    validate_log_filename,
)
from library.utils.logging_utilities import create_script_logger

from src.analysis.critical_mass_extrapolation._pcac_critical_mass_config import (
    REQUIRED_COLUMNS,
    COLUMN_MAPPING,
    QUADRATIC_FIT_CONFIG,
    PLATEAU_MASS_POWER,
    DEFAULT_OUTPUT_FILENAME,
    validate_pcac_critical_config,
    get_fit_range_config,
)
from src.analysis.critical_mass_extrapolation._critical_mass_core import (
    process_critical_mass_analysis,
    InsufficientDataError,
)


@click.command()
@click.option(
    "-i",
    "--input_csv",
    required=True,
    callback=csv_file.input,
    help="Input CSV file with PCAC plateau estimates",
)
@click.option(
    "-o",
    "--output_csv",
    default=DEFAULT_OUTPUT_FILENAME,
    callback=csv_file.output,
    help=f"Output CSV file (name or path). Default: {DEFAULT_OUTPUT_FILENAME}",
)
@click.option(
    "-log_on",
    "--enable_logging",
    is_flag=True,
    help="Enable logging",
)
@click.option(
    "-log_dir",
    "--log_directory",
    callback=directory.can_create,
    help="Directory for log files",
)
@click.option(
    "-log",
    "--log_filename",
    callback=validate_log_filename,
    help="Log filename",
)
def main(input_csv, output_csv, enable_logging, log_directory, log_filename):
    """Calculate critical bare mass from PCAC plateau estimates."""

    validate_pcac_critical_config()

    # Determine output directory from output_csv
    output_csv_path = Path(output_csv)
    if output_csv_path.is_absolute():
        output_directory = output_csv_path.parent
    else:
        output_directory = Path(input_csv).parent
        output_csv = str(output_directory / output_csv)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_directory = str(output_directory)

    # Set up logging
    log_dir = (
        log_directory if log_directory else output_directory if enable_logging else None
    )

    logger = create_script_logger(
        log_directory=log_dir,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
    )

    try:
        logger.log_script_start("PCAC critical mass calculation")

        # Get fit range configuration
        fit_range_config = get_fit_range_config()

        output_path = process_critical_mass_analysis(
            input_csv,
            output_csv,
            "pcac",
            COLUMN_MAPPING,
            REQUIRED_COLUMNS,
            QUADRATIC_FIT_CONFIG,
            fit_range_config,
            PLATEAU_MASS_POWER,
            logger,
        )

        click.echo(f"✓ PCAC critical mass calculation complete: {output_path}")
        logger.log_script_end("PCAC critical mass calculation completed")

    except InsufficientDataError as e:
        # Graceful skip - not enough data for analysis (not an error)
        logger.warning(f"Insufficient data: {e}")
        logger.log_script_end(
            "PCAC critical mass calculation skipped (insufficient data)"
        )
        click.echo(f"⚠ Insufficient data for PCAC critical mass calculation", err=True)
        click.echo(f"  → {e}", err=True)
        sys.exit(2)  # Exit code 2 = graceful skip

    except Exception as e:
        # Actual error
        logger.error(f"Script failed: {e}")
        logger.log_script_end("PCAC critical mass calculation failed")
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)  # Exit code 1 = error


if __name__ == "__main__":
    main()
