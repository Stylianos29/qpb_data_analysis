#!/usr/bin/env python3
"""
Computational Cost Extrapolation Script

Extrapolates computational costs (core-hours per spinor per
configuration) using DataPlotter for automatic grouping, curve fitting,
and visualization.

Key features:
    - Automatic parameter detection and grouping via DataFrameAnalyzer
    - Configuration averaging using
      DataFrameAnalyzer.group_by_multivalued_tunable_parameters()
    - Curve fitting with shifted power law function a/(x-b)+c via
      DataPlotter
    - Professional visualization with fit diagnostics
    - CSV export with detailed results and metadata

Usage:
    python extrapolate_computational_cost.py \
        -i_proc processed_parameter_values.csv \
        -o output_dir [options]
"""

import sys
from pathlib import Path

import click
import pandas as pd

# Configure matplotlib backend
import matplotlib

matplotlib.use("Agg")

# Import library components
from library.utils.logging_utilities import create_script_logger
from library.validation.click_validators import (
    csv_file,
    directory,
    validate_log_filename,
)

# Import from auxiliary modules
from src.analysis._cost_extrapolation_config import CONFIG, validate_config
from src.analysis._cost_extrapolation_methods import (
    load_and_prepare_data,
    create_cost_plotter,
    perform_cost_extrapolation,
    export_results,
    get_plotting_config,
)


@click.command()
@click.option(
    "-i_proc",
    "--input_processed_csv",
    required=True,
    callback=csv_file.input,
    help="Path to input CSV file containing processed parameter values.",
)
@click.option(
    "-o",
    "--output_directory",
    required=True,
    callback=directory.must_exist,
    help="Directory for output files.",
)
@click.option(
    "-p",
    "--plots_directory",
    default=None,
    callback=directory.can_create,
    help="Directory for output plots. If not specified, uses output_directory/plots.",
)
@click.option(
    "--output_csv_filename",
    default=CONFIG["output"]["csv_filename"],
    help=f"Output CSV filename (default: {CONFIG['output']['csv_filename']}).",
)
@click.option(
    "-log_on",
    "--enable_logging",
    is_flag=True,
    default=False,
    help="Enable detailed logging to file.",
)
@click.option(
    "-log_dir",
    "--log_directory",
    default=None,
    callback=directory.can_create,
    help="Directory for log files. Default: output directory",
)
@click.option(
    "-log_name",
    "--log_filename",
    default=None,
    callback=validate_log_filename,
    help="Custom name for log file. Default: auto-generated",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose console output.",
)
def main(
    input_processed_csv,
    output_directory,
    plots_directory,
    output_csv_filename,
    enable_logging,
    log_directory,
    log_filename,
    verbose,
):
    """
    Extrapolate computational costs using DataPlotter for automatic
    grouping and fitting.

    This script leverages the DataFrameAnalyzer for automatic parameter
    detection and grouping, then uses DataPlotter for curve fitting with
    the shifted power law function a/(x-b)+c and professional
    visualizations.
    """
    # Validate configuration
    if not validate_config():
        click.echo("Configuration validation failed. Exiting.", err=True)
        sys.exit(1)

    # Setup directories
    output_directory = Path(output_directory)
    if plots_directory is None:
        plots_directory = output_directory / "plots"
    else:
        plots_directory = Path(plots_directory)

    if log_directory is None and enable_logging:
        log_directory = output_directory
    else:
        log_directory = Path(log_directory) if log_directory else None

    # Create directories
    plots_directory.mkdir(parents=True, exist_ok=True)
    if log_directory:
        log_directory.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = create_script_logger(
        log_directory=str(log_directory) if log_directory else None,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=verbose,
        verbose=verbose,
    )

    logger.log_script_start("Computational cost extrapolation")
    logger.info(f"Input processed CSV: {input_processed_csv}")
    logger.info(f"Output directory: {output_directory}")
    logger.info(f"Plots directory: {plots_directory}")

    try:
        # Phase 1: Data Loading and Preparation
        logger.info("=== PHASE 1: Data Loading and Preparation ===")
        prepared_df = load_and_prepare_data(input_processed_csv, logger)

        if len(prepared_df) == 0:
            logger.error("No valid data remaining after preparation. Exiting.")
            sys.exit(1)

        # Phase 2: DataPlotter Initialization
        logger.info("=== PHASE 2: DataPlotter Initialization ===")
        cost_plotter = create_cost_plotter(prepared_df, plots_directory, logger)

        # Log detected parameters
        logger.info(
            f"Detected multivalued parameters: {cost_plotter.list_of_multivalued_tunable_parameter_names}"
        )
        logger.info(
            f"Detected single-valued parameters: {cost_plotter.list_of_single_valued_tunable_parameter_names}"
        )

        # Check for sufficient grouping parameters
        if not cost_plotter.list_of_multivalued_tunable_parameter_names:
            logger.warning(
                "No multivalued parameters detected - extrapolation will be performed on entire dataset"
            )

        # Phase 3: Cost Extrapolation and Fitting
        logger.info("=== PHASE 3: Cost Extrapolation and Fitting ===")
        extrapolation_results = perform_cost_extrapolation(cost_plotter, logger)

        if not extrapolation_results:
            logger.error("Extrapolation failed to produce results. Exiting.")
            sys.exit(1)

        # Phase 4: Result Export
        logger.info("=== PHASE 4: Result Export ===")
        results_df = export_results(
            extrapolation_results, output_directory, output_csv_filename, logger
        )

        logger.log_script_end("Computational cost extrapolation completed successfully")

        # Get plotting configuration
        plotting_config = get_plotting_config()

        # Final success message
        success_msg = "✓ Computational cost extrapolation completed successfully!"
        success_msg += (
            f"\n  • Results saved to: {output_directory / output_csv_filename}"
        )
        success_msg += f"\n  • Analyzed {len(results_df)} parameter groups"
        success_msg += f"\n  • Total data points: {len(prepared_df)}"
        success_msg += (
            "\n  • Plots saved to: "
            f"{plots_directory / plotting_config.get(
                'base_subdirectory', 'Computational_cost_extrapolation')}"
        )

        click.echo(success_msg)

    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        logger.log_script_error(e)
        click.echo(f"✗ Script failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
