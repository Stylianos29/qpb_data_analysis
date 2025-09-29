#!/usr/bin/env python3
"""
Cost Extrapolation Analysis Visualization Script

Creates visualization plots for computational cost extrapolation
analysis, showing both mass-to-bare-mass conversion and cost
extrapolation. Supports both PCAC and pion mass-based analyses with
appropriate transformations (PCAC: mass¹, Pion: mass²).

The script combines cost extrapolation results with original mass and
cost data to visualize:
  1. Mass vs bare mass linear fit with derived reference bare mass
  2. Cost vs bare mass shifted power law fit with extrapolated cost

Features:
    - Automatic analysis-type detection or manual specification
    - Configurable plot clearing and directory management
    - Group-wise plotting for different lattice parameter combinations
    - Professional styling with fit lines, extrapolation markers, and
      annotations
    - Comprehensive error handling with detailed logging

Usage Examples:
    # Basic PCAC analysis visualization
    python visualize_cost_extrapolation_analysis.py \
        -t pcac \
        -r computational_cost_extrapolation_from_pcac.csv \
        -m plateau_PCAC_mass_estimates.csv \
        -c processed_parameter_values.csv \
        -o plots_output

    # Pion analysis with plot clearing and logging
    python visualize_cost_extrapolation_analysis.py \
        -t pion \
        -r computational_cost_extrapolation_from_pion.csv \
        -m plateau_pion_mass_estimates.csv \
        -c processed_parameter_values.csv \
        -o plots_output \
        --clear_existing \
        -log_on \
        -log_dir logs

Input Requirements:
    - Results CSV: Output from extrapolate_cost_from_{pcac,pion}.py
    - Mass CSV: Output from extract_plateau_{PCAC,pion}_mass.py
    - Cost CSV: Processed parameter values with cost data
    - Analysis type: Must match the type used in calculation
"""

from pathlib import Path

import click

from library.validation.click_validators import (
    csv_file,
    directory,
    validate_log_filename,
)
from library.utils.logging_utilities import create_script_logger
from library.visualization.builders.title_builder import PlotTitleBuilder
from library.visualization.managers.file_manager import PlotFileManager
from library.constants.labels import TITLE_LABELS_DICTIONARY

from src.analysis.cost_extrapolation._cost_extrapolation_visualization_config import (
    validate_visualization_config,
    get_plot_subdirectory,
)
from src.analysis.cost_extrapolation._cost_extrapolation_visualization_core import (
    load_and_validate_results_data,
    load_and_validate_mass_data,
    load_and_validate_cost_data,
    group_data_for_visualization,
    create_cost_extrapolation_plots,
)


def process_cost_extrapolation_visualization(
    results_csv_path: str,
    mass_csv_path: str,
    cost_csv_path: str,
    plots_directory: str,
    analysis_type: str,
    clear_existing_plots: bool,
    logger,
) -> int:
    """
    Process cost extrapolation data and create visualization plots.

    Loads cost extrapolation results, original mass data, and cost data,
    groups by lattice parameters, and generates two types of plots per
    group:
      1. Mass vs bare mass linear fit plot
      2. Cost vs bare mass shifted power law fit with extrapolation

    Args:
        results_csv_path: Path to cost extrapolation results CSV
        mass_csv_path: Path to mass plateau estimates CSV cost_csv_path:
        Path to cost data CSV plots_directory: Directory for saving
        plots analysis_type: "pcac" or "pion" clear_existing_plots:
        Whether to clear existing plots logger: Logger instance

    Returns:
        Number of plot pairs successfully created
    """
    # Load and validate data
    results_df = load_and_validate_results_data(results_csv_path, logger)
    mass_df = load_and_validate_mass_data(mass_csv_path, analysis_type, logger)
    cost_df = load_and_validate_cost_data(cost_csv_path, logger)

    # Set up visualization infrastructure
    title_builder = PlotTitleBuilder(TITLE_LABELS_DICTIONARY)

    # Create plots directory structure
    plot_base_name = get_plot_subdirectory(analysis_type)

    file_manager = PlotFileManager(base_directory=str(plots_directory))
    plots_subdir_path = file_manager.prepare_subdirectory(
        plot_base_name, clear_existing=clear_existing_plots, confirm_clear=False
    )

    if clear_existing_plots:
        logger.info(f"Cleared existing plots in {plot_base_name} subdirectory")

    # Group data for visualization
    grouped_data = group_data_for_visualization(
        results_df, mass_df, cost_df, analysis_type, logger
    )

    logger.info(f"Creating plots for {len(grouped_data)} parameter groups")

    # Create plots
    plots_created = 0

    for group_info in grouped_data:
        try:
            mass_plot_path, cost_plot_path = create_cost_extrapolation_plots(
                group_info,
                Path(plots_subdir_path),
                file_manager,
                title_builder,
                logger,
            )

            if mass_plot_path and cost_plot_path:
                plots_created += 1

        except Exception as e:
            logger.warning(
                f"Failed to create plots for group {group_info.get('group_id', 'unknown')}: {e}"
            )
            continue

    logger.info(
        f"Successfully created {plots_created * 2} plots ({plots_created} groups)"
    )

    return plots_created


@click.command()
@click.option(
    "-r",
    "--results_csv",
    required=True,
    callback=csv_file.input,
    help="Cost extrapolation results CSV file",
)
@click.option(
    "-m",
    "--mass_csv",
    required=True,
    callback=csv_file.input,
    help="Mass plateau estimates CSV file (PCAC or pion)",
)
@click.option(
    "-c",
    "--cost_csv",
    required=True,
    callback=csv_file.input,
    help="Cost data CSV file (processed_parameter_values.csv)",
)
@click.option(
    "-o",
    "--plots_directory",
    default=None,
    callback=directory.can_create,
    help="Output directory for plots (default: same as results CSV)",
)
@click.option(
    "-t",
    "--analysis_type",
    type=click.Choice(["pcac", "pion"], case_sensitive=False),
    required=True,
    help="Analysis type for cost extrapolation (PCAC mass or pion mass)",
)
@click.option(
    "-clear",
    "--clear_existing",
    is_flag=True,
    default=False,
    help="Clear existing plots in output subdirectory before creating new ones",
)
@click.option(
    "-log_on",
    "--enable_logging",
    is_flag=True,
    help="Enable logging to file",
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
    results_csv,
    mass_csv,
    cost_csv,
    plots_directory,
    analysis_type,
    clear_existing,
    enable_logging,
    log_directory,
    log_filename,
):
    """Create cost extrapolation visualization plots."""

    # Validate configuration
    validate_visualization_config()

    # Setup plots directory (default to results CSV parent directory)
    if plots_directory is None:
        plots_directory = Path(results_csv).parent / "plots"
        plots_directory.mkdir(parents=True, exist_ok=True)
    else:
        plots_directory = Path(plots_directory)

    # Setup logging
    log_dir = (
        log_directory
        if log_directory
        else str(plots_directory) if enable_logging else None
    )

    logger = create_script_logger(
        log_directory=log_dir,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
    )

    try:
        logger.log_script_start(
            f"{analysis_type.upper()} cost extrapolation visualization"
        )
        logger.info(f"Results CSV: {results_csv}")
        logger.info(f"Mass CSV: {mass_csv}")
        logger.info(f"Cost CSV: {cost_csv}")
        logger.info(f"Plots directory: {plots_directory}")
        logger.info(f"Analysis type: {analysis_type}")

        # Process visualization
        plots_created = process_cost_extrapolation_visualization(
            results_csv_path=results_csv,
            mass_csv_path=mass_csv,
            cost_csv_path=cost_csv,
            plots_directory=str(plots_directory),
            analysis_type=analysis_type.lower(),
            clear_existing_plots=clear_existing,
            logger=logger,
        )

        if plots_created == 0:
            logger.warning("No plots were created")
            click.echo("⚠ Warning: No plots were created. Check logs for details.")
        else:
            click.echo(
                f"✓ Cost extrapolation visualization complete: {plots_created * 2} plots created"
            )

        logger.log_script_end(
            f"{analysis_type.upper()} cost extrapolation visualization completed"
        )

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        logger.log_script_end(
            f"{analysis_type.upper()} cost extrapolation visualization failed"
        )
        raise


if __name__ == "__main__":
    main()
