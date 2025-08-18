#!/usr/bin/env python3
"""
Computational Cost Analysis Script using DataPlotter Integration

This script analyzes computational costs (core-hours per spinor per configuration)
using the DataPlotter class for automatic grouping, curve fitting, and visualization.
The DataPlotter automatically detects multivalued parameters for grouping and applies
sophisticated curve fitting with the shifted power law function a/(x-b)+c.

Key features:
    - Automatic parameter detection and grouping via DataPlotter
    - Sophisticated curve fitting with shifted power law function
    - Professional visualization with fit diagnostics
    - Comprehensive statistical analysis and validation
    - CSV export with detailed results and metadata

Place this file as:
qpb_data_analysis/core/src/analysis/estimate_computational_cost_at_target_bare_mass.py

Usage:
    python estimate_computational_cost_at_target_bare_mass.py \
        -i_proc processed_parameter_values.csv \
        -o output_dir [options]
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import click
import numpy as np
import pandas as pd

# Configure matplotlib to use non-interactive backend
import matplotlib

matplotlib.use("Agg")

# Import library components
from library.utils.logging_utilities import create_script_logger
from library.validation.click_validators import (
    csv_file,
    directory,
    validate_log_filename,
)
from library import constants

# Import our modules
from src.analysis._cost_estimation_config import (
    PROCESSED_PARAMS_CSV_COLUMNS,
    OUTPUT_CSV_CONFIG,
    REFERENCE_CONFIG,
    validate_config,
)
from src.analysis._cost_estimation_methods import (
    load_and_prepare_data,
    create_cost_plotter,
    perform_cost_analysis,
    compile_final_results,
    validate_results,
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
    default=OUTPUT_CSV_CONFIG["default_filename"],
    help=f"Output CSV filename (default: {OUTPUT_CSV_CONFIG['default_filename']}).",
)
@click.option(
    "--enable_plotting",
    is_flag=True,
    default=True,
    help="Enable generation of diagnostic plots.",
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
    enable_plotting,
    enable_logging,
    log_directory,
    log_filename,
    verbose,
):
    """
    Analyze computational costs using DataPlotter for automatic grouping and fitting.

    This script leverages the DataPlotter class to automatically detect multivalued
    parameters for grouping, apply sophisticated curve fitting with the shifted
    power law function a/(x-b)+c, and generate professional visualizations.
    """

    # Get reference PCAC mass from config
    reference_pcac_mass = REFERENCE_CONFIG["default_reference_pcac_mass"]

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

    # Create directories if needed
    if enable_plotting:
        plots_directory.mkdir(parents=True, exist_ok=True)
    if log_directory:
        log_directory.mkdir(parents=True, exist_ok=True)

    # Get script name for reference
    script_name = Path(__file__).stem

    # Setup logging
    logger = create_script_logger(
        log_directory=str(log_directory) if log_directory else None,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=verbose,
        verbose=verbose,
    )

    logger.log_script_start("Computational cost analysis using DataPlotter")
    logger.info(f"Input processed CSV: {input_processed_csv}")
    logger.info(f"Reference PCAC mass: {reference_pcac_mass}")
    logger.info(f"Output directory: {output_directory}")
    logger.info(f"Plots directory: {plots_directory}")
    logger.info(f"Enable plotting: {enable_plotting}")

    try:
        # Load and prepare data
        logger.info("=== PHASE 1: Data Loading and Preparation ===")
        processed_df = load_and_prepare_data(input_processed_csv, logger)

        if len(processed_df) == 0:
            logger.error("No valid data remaining after filtering. Exiting.")
            sys.exit(1)

        # Create DataPlotter for cost analysis
        logger.info("=== PHASE 2: DataPlotter Initialization ===")
        cost_plotter = create_cost_plotter(processed_df, plots_directory, logger)

        # Log detected parameters
        logger.info(
            f"Detected multivalued parameters: {cost_plotter.list_of_multivalued_tunable_parameter_names}"
        )
        logger.info(
            f"Detected single-valued parameters: {cost_plotter.list_of_single_valued_tunable_parameter_names}"
        )
        logger.info(
            f"Detected output quantities: {cost_plotter.list_of_output_quantity_names_from_dataframe}"
        )

        # Check for sufficient grouping parameters
        if not cost_plotter.list_of_multivalued_tunable_parameter_names:
            logger.warning(
                "No multivalued parameters detected - analysis will be performed on entire dataset"
            )

        # Perform cost analysis
        logger.info("=== PHASE 3: Cost Analysis and Fitting ===")
        if enable_plotting:
            analysis_results, group_results = perform_cost_analysis(
                cost_plotter, logger
            )
        else:
            # Skip plotting but still do analysis
            logger.info("Plotting disabled - performing statistical analysis only")
            analysis_results, group_results = perform_statistical_analysis_only(
                cost_plotter, logger
            )

        # Compile results
        logger.info("=== PHASE 4: Result Compilation ===")
        final_results_df = compile_final_results(
            analysis_results, group_results, logger
        )

        if final_results_df.empty:
            logger.error("No results to export. Analysis failed.")
            sys.exit(1)

        # Validate results
        logger.info("=== PHASE 5: Result Validation ===")
        validation_passed = validate_results(final_results_df, logger)
        if not validation_passed:
            logger.warning(
                "Some results failed validation checks - review output carefully"
            )

        # Export results
        logger.info("=== PHASE 6: Result Export ===")
        export_results_to_csv(
            final_results_df, output_directory, output_csv_filename, logger
        )

        # Generate summary
        generate_analysis_summary(analysis_results, final_results_df, logger)

        logger.log_script_end("Computational cost analysis completed successfully")

        # Final success message
        success_msg = f"✓ Computational cost analysis completed successfully!"
        success_msg += (
            f"\n  • Results saved to: {output_directory / output_csv_filename}"
        )
        success_msg += f"\n  • Analyzed {len(final_results_df)} parameter groups"
        success_msg += (
            f"\n  • Total data points: {analysis_results['total_data_points']}"
        )
        if enable_plotting:
            success_msg += f"\n  • Plots saved to: {plots_directory}"

        click.echo(success_msg)

    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        logger.log_script_error(e)
        click.echo(f"✗ Script failed: {e}", err=True)
        sys.exit(1)


def perform_statistical_analysis_only(cost_plotter, logger):
    """Perform analysis without plotting."""
    logger.info("Performing statistical analysis without plotting...")

    # Extract analysis results without plotting
    from src.analysis._cost_estimation_methods import (
        extract_analysis_results,
        extract_group_results,
    )

    analysis_results = extract_analysis_results(cost_plotter, logger)
    group_results = extract_group_results(cost_plotter, logger)

    return analysis_results, group_results


def export_results_to_csv(
    results_df: pd.DataFrame, output_directory: Path, output_csv_filename: str, logger
) -> None:
    """Export results to CSV file."""

    try:
        # Save to CSV
        csv_path = output_directory / output_csv_filename
        results_df.to_csv(csv_path, index=False)

        logger.info(f"Exported {len(results_df)} results to {csv_path}")

        # Log CSV column info
        logger.info(f"CSV contains {len(results_df.columns)} columns:")
        for col in results_df.columns:
            logger.debug(f"  - {col}")

    except Exception as e:
        logger.error(f"Failed to export results to CSV: {e}")
        raise


def generate_analysis_summary(
    analysis_results: Dict[str, Any], results_df: pd.DataFrame, logger
) -> None:
    """Generate and log comprehensive analysis summary."""

    logger.info("=== ANALYSIS SUMMARY ===")

    # Overall statistics
    logger.info(f"Total data points analyzed: {analysis_results['total_data_points']}")
    logger.info(f"Number of parameter groups: {len(results_df)}")
    logger.info(f"Reference PCAC mass: {analysis_results['reference_pcac_mass']}")

    # Parameter information
    mv_params = analysis_results["multivalued_parameters"]
    sv_params = analysis_results["single_valued_parameters"]
    logger.info(f"Multivalued parameters ({len(mv_params)}): {mv_params}")
    logger.info(f"Single-valued parameters ({len(sv_params)}): {sv_params}")

    # Cost statistics
    cost_stats = analysis_results["cost_statistics"]
    logger.info("Overall cost statistics (core-hours per spinor per configuration):")
    logger.info(f"  Mean: {cost_stats['mean']:.3f} ± {cost_stats['std']:.3f}")
    logger.info(f"  Range: [{cost_stats['min']:.3f}, {cost_stats['max']:.3f}]")
    logger.info(f"  Median: {cost_stats['median']:.3f}")

    # Bare mass range
    mass_range = analysis_results["bare_mass_range"]
    logger.info(f"Bare mass range: [{mass_range['min']:.6f}, {mass_range['max']:.6f}]")
    logger.info(f"Average bare mass: {mass_range['mean']:.6f}")

    # Group-level statistics
    if len(results_df) > 0:
        group_sizes = results_df["n_data_points"]
        logger.info("Group size statistics:")
        logger.info(f"  Average points per group: {group_sizes.mean():.1f}")
        logger.info(f"  Group size range: [{group_sizes.min()}, {group_sizes.max()}]")

        # Cost variation across groups
        group_costs = results_df[
            "avg_core_hours_per_configuration"
        ]  # Updated column name
        logger.info("Cost variation across groups:")
        logger.info(
            f"  Group cost range: [{group_costs.min():.3f}, {group_costs.max():.3f}]"
        )
        logger.info(f"  Cost variation (std): {group_costs.std():.3f}")

        # Configuration information
        if "total_configurations" in results_df.columns:
            total_configs = results_df["total_configurations"].sum()
            logger.info(f"Total configurations analyzed: {total_configs}")
            avg_configs_per_group = results_df["total_configurations"].mean()
            logger.info(
                f"Average configurations per group: {avg_configs_per_group:.1f}"
            )

    logger.info("=== END SUMMARY ===")


if __name__ == "__main__":
    main()
