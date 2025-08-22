#!/usr/bin/env python3
"""
Computational Cost Extrapolation Script

Extrapolates computational costs (core-hours per spinor per
configuration) using DataPlotter for automatic grouping, curve fitting,
and visualization.

Automatically detects and applies the appropriate extrapolation method:
    - Fixed bare mass: When only processed data is provided (-i_proc)
    - Fixed PCAC mass: When both processed and PCAC data are provided
      (-i_proc + -i_pcac)

Key features:
    - Automatic method detection based on input files
    - Automatic parameter detection and grouping via DataFrameAnalyzer
    - Configuration averaging using
      DataFrameAnalyzer.group_by_multivalued_tunable_parameters()
    - Curve fitting with linear (PCAC) and shifted power law (cost)
      functions
    - Professional visualization with fit diagnostics and uncertainty
      bands
    - CSV export with detailed results and metadata

Usage:
    # Fixed bare mass method (automatic) python
    extrapolate_computational_cost.py \
        -i_proc processed_parameter_values.csv \
        -o output_dir [options]
    
    # Fixed PCAC mass method (automatic) python
    extrapolate_computational_cost.py \
        -i_proc processed_parameter_values.csv \
        -i_pcac plateau_PCAC_mass_estimates.csv \
        -o output_dir [options]
"""

import sys
from pathlib import Path

import click

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
from src.analysis._cost_extrapolation_config import (
    validate_config,
    get_shared_config,
    get_reference_pcac_mass,
    get_reference_bare_mass,
)
from src.analysis._cost_extrapolation_methods import (
    extrapolate_computational_cost,
    detect_extrapolation_method,
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
    "-i_pcac",
    "--input_pcac_csv",
    default=None,
    callback=csv_file.input,
    help="Path to input CSV file containing PCAC mass data (enables fixed_pcac_mass method).",
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
    required=True,
    callback=directory.must_exist,
    help="Directory for output plots.",
)
@click.option(
    "--output_csv_filename",
    default=lambda: get_shared_config()["output"]["csv_filename"],
    help="Output CSV filename (default: from configuration).",
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
    input_pcac_csv,
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

    This script supports both fixed bare mass and fixed PCAC mass
    methods. The method is determined by configuration settings and
    provided input files. Uses DataFrameAnalyzer for automatic parameter
    detection and DataPlotter for curve fitting with professional
    visualizations.
    """
    # Validate configuration
    if not validate_config():
        click.echo("Configuration validation failed. Exiting.", err=True)
        sys.exit(1)

    # Setup directories
    output_directory = Path(output_directory)
    plots_directory = Path(plots_directory)

    if log_directory is None and enable_logging:
        log_directory = output_directory
    else:
        log_directory = Path(log_directory) if log_directory else None

    # Create directories
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
    if input_pcac_csv:
        logger.info(f"Input PCAC CSV: {input_pcac_csv}")
    logger.info(f"Output directory: {output_directory}")
    logger.info(f"Plots directory: {plots_directory}")

    try:
        # Validate method configuration
        logger.info("=== CONFIGURATION VALIDATION ===")
        method = detect_extrapolation_method(
            str(input_pcac_csv) if input_pcac_csv else None
        )
        logger.info(f"Extrapolation method: {method}")

        if method == "fixed_bare_mass":
            target = get_reference_bare_mass()
            logger.info(f"Target bare mass: {target}")

            if input_pcac_csv:
                logger.warning(
                    "PCAC data provided but method is 'fixed_bare_mass'. "
                    "PCAC data will be ignored. "
                    "Change method to 'fixed_pcac_mass' to use PCAC data."
                )

        elif method == "fixed_pcac_mass":
            reference = get_reference_pcac_mass()
            logger.info(f"Reference PCAC mass: {reference}")

            if input_pcac_csv is None:
                logger.error(
                    "Configuration method is 'fixed_pcac_mass' but no PCAC "
                    "data file provided. Either provide PCAC data file with "
                    "-i_pcac or change method to 'fixed_bare_mass'."
                )
                sys.exit(1)
        else:
            logger.error(f"Unknown extrapolation method: {method}")
            sys.exit(1)

        # Phase 1: Computational Cost Extrapolation (Unified)
        logger.info("=== PHASE 1: Computational Cost Extrapolation ===")

        extrapolation_results = extrapolate_computational_cost(
            processed_csv_path=str(input_processed_csv),
            output_directory=output_directory,
            plots_directory=plots_directory,
            logger=logger,
            pcac_csv_path=str(input_pcac_csv) if input_pcac_csv else None,
        )

        if not extrapolation_results:
            logger.error("Extrapolation failed to produce results. Exiting.")
            sys.exit(1)

        logger.info(f"✓ Generated {len(extrapolation_results)} fit results")

        logger.log_script_end("Computational cost extrapolation completed successfully")

        # Final success message
        base_subdir = get_shared_config()["base_subdirectory"]
        success_msg = "✓ Computational cost extrapolation completed successfully!"
        success_msg += (
            f"\n  • Results saved to: {output_directory / output_csv_filename}"
        )
        success_msg += f"\n  • Analyzed {len(extrapolation_results)} parameter groups"
        success_msg += f"\n  • Method used: {method}"

        if method == "fixed_bare_mass":
            success_msg += f"\n  • Target bare mass: {get_reference_bare_mass()}"
        elif method == "fixed_pcac_mass":
            success_msg += f"\n  • Reference PCAC mass: {get_reference_pcac_mass()}"

        success_msg += f"\n  • Plots saved to: {plots_directory / base_subdir}"

        click.echo(success_msg)

    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        logger.log_script_error(e)
        click.echo(f"✗ Script failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
