#!/usr/bin/env python3
"""
Critical mass calculation from Pion plateau estimates.

Usage: python calculate_critical_mass_from_pion.py \
    -i plateau_pion.csv \
    -o output_dir
"""

from pathlib import Path

import click

from library.validation.click_validators import (
    csv_file,
    directory,
    validate_log_filename,
)
from library.utils.logging_utilities import create_script_logger

from src.analysis.critical_mass_extrapolation._pion_critical_mass_config import (
    REQUIRED_COLUMNS,
    OUTPUT_FILENAME,
    validate_pion_critical_config,
)
from src.analysis.critical_mass_extrapolation._critical_mass_shared_config import (
    GROUPING_PARAMETERS,
)
from src.analysis.critical_mass_extrapolation._critical_mass_core import (
    load_and_validate_plateau_data,
    group_data_by_parameters,
    calculate_critical_mass_for_group,
    export_results_to_csv,
)


def validate_pion_input_data(df, logger):
    """Validate Pion plateau data for critical mass calculation."""
    required_cols = REQUIRED_COLUMNS
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for sufficient data points
    if len(df) < 3:
        raise ValueError("Need at least 3 data points for extrapolation")

    logger.info(f"Validated {len(df)} Pion plateau data points")


def process_pion_critical_mass(input_csv_path, output_directory, logger):
    """Process Pion plateau data to calculate critical mass values."""
    # Load and validate input data
    df = load_and_validate_plateau_data(input_csv_path, "pion")
    validate_pion_input_data(df, logger)

    # Group data by lattice parameters
    grouped_data = group_data_by_parameters(df, GROUPING_PARAMETERS)
    logger.info(f"Processing {len(grouped_data)} parameter groups")

    # Calculate critical mass for each group
    results = []
    for group_id, group_df in grouped_data:
        try:
            result = calculate_critical_mass_for_group(group_id, group_df, "pion")
            if result:
                results.append(result)
        except Exception as e:
            logger.warning(f"Failed to process group {group_id}: {e}")
            continue

    # Export results
    if results:
        output_path = export_results_to_csv(results, output_directory, OUTPUT_FILENAME)
        logger.info(f"Exported {len(results)} critical mass values to {output_path}")
        return output_path
    else:
        raise ValueError("No valid critical mass calculations completed")


@click.command()
@click.option(
    "-i",
    "--input_csv",
    required=True,
    callback=csv_file.input,
    help="Input CSV file with Pion plateau estimates",
)
@click.option(
    "-o",
    "--output_directory",
    default=None,
    callback=directory.can_create,
    help="Output directory for results",
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
def main(input_csv, output_directory, enable_logging, log_directory, log_filename):
    """Calculate critical bare mass from Pion plateau estimates."""
    # Validate configuration
    validate_pion_critical_config()

    # Set fallback for output directory
    if output_directory is None:
        output_directory = str(Path(input_csv).parent)

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
        logger.log_script_start("Pion critical mass calculation")

        # Process data
        output_path = process_pion_critical_mass(input_csv, output_directory, logger)

        click.echo(f"✓ Pion critical mass calculation complete: {output_path}")
        logger.log_script_end("Pion critical mass calculation completed")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.log_script_end("Pion critical mass calculation failed")
        raise


if __name__ == "__main__":
    main()
