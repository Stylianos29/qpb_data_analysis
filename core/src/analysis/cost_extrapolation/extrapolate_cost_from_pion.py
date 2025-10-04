#!/usr/bin/env python3
"""
Computational Cost Extrapolation from Pion Mass

Extrapolates computational costs (core-hours per spinor per
configuration) to a reference pion mass value using two-step process:
  1. Convert reference pion mass to bare mass via fit (typically
     quadratic)
  2. Extrapolate computational cost using shifted power law

This script performs calculations only - no visualization. Use the
separate visualize_cost_extrapolation_analysis.py script for plotting.

Key features:
    - Pion mass to bare mass conversion with uncertainty propagation
    - Configuration averaging across simulation runs
    - Group-specific bare mass derivation
    - Shifted power law fitting for cost extrapolation
    - CSV export with detailed fit parameters and uncertainties

Usage:
    python extrapolate_cost_from_pion.py \
        -i_cost processed_parameter_values.csv \
        -i_pion plateau_pion_mass_estimates.csv \
        -o output_directory
"""

from pathlib import Path

import click

from library.validation.click_validators import (
    csv_file,
    directory,
    validate_log_filename,
)
from library.utils.logging_utilities import create_script_logger

from src.analysis.cost_extrapolation._pion_cost_extrapolation_config import (
    COLUMN_MAPPING,
    DEFAULT_OUTPUT_FILENAME,
    validate_pion_cost_config,
    get_required_columns,
)
from src.analysis.cost_extrapolation._cost_extrapolation_core import (
    process_cost_extrapolation_analysis,
)


@click.command()
@click.option(
    "-i_cost",
    "--input_cost_csv",
    required=True,
    callback=csv_file.input,
    help="Input CSV file with computational cost data (processed_parameter_values.csv)",
)
@click.option(
    "-i_pion",
    "--input_pion_csv",
    required=True,
    callback=csv_file.input,
    help="Input CSV file with pion mass plateau estimates",
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
    help="Enable detailed logging to file",
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
    help="Custom log filename",
)
def main(
    input_cost_csv,
    input_pion_csv,
    output_csv,
    enable_logging,
    log_directory,
    log_filename,
):
    """
    Extrapolate computational costs from pion mass analysis.

    Workflow:
      1. Load pion plateau estimates
      2. Fit pion mass vs bare mass (typically m_π² = a * m_bare + b)
      3. Invert fit to get reference bare mass from reference pion mass
      4. Load computational cost data
      5. Average costs across configurations
      6. Fit cost vs bare mass (shifted power law)
      7. Extrapolate cost at reference bare mass
      8. Export results to CSV
    """
    # Validate configurations
    validate_pion_cost_config()

    # Determine output directory from output_csv
    output_csv_path = Path(output_csv)
    if output_csv_path.is_absolute():
        output_directory = output_csv_path.parent
    else:
        output_directory = Path(input_cost_csv).parent
        output_csv = str(output_directory / output_csv)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_directory = str(output_directory)

    # Setup logging
    log_dir = (
        log_directory if log_directory else output_directory if enable_logging else None
    )

    logger = create_script_logger(
        log_directory=log_dir,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
    )

    try:
        logger.log_script_start("Pion cost extrapolation")
        logger.info(f"Input cost CSV: {input_cost_csv}")
        logger.info(f"Input pion CSV: {input_pion_csv}")
        logger.info(f"Output CSV: {output_csv}")

        # Process the analysis
        output_path = process_cost_extrapolation_analysis(
            cost_csv_path=input_cost_csv,
            mass_csv_path=input_pion_csv,
            output_csv_path=output_csv,
            analysis_type="pion",
            column_mapping=COLUMN_MAPPING,
            required_columns=get_required_columns(),
            logger=logger,
        )

        click.echo(f"✓ Pion cost extrapolation complete: {output_path}")
        logger.log_script_end("Pion cost extrapolation completed successfully")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.log_script_end("Pion cost extrapolation failed")
        raise


if __name__ == "__main__":
    main()
