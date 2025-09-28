#!/usr/bin/env python3
"""
Critical mass analysis visualization script.

Usage: python visualize_critical_mass_analysis.py \
    --analysis_type "pcac" \
    -r results.csv \
    -p plateau.csv \
    -o plots_dir
"""

from typing import Optional, List
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

from src.analysis.critical_mass_extrapolation._critical_mass_shared_config import (
    OUTPUT_COLUMN_NAMES,
)
from src.analysis.critical_mass_extrapolation._critical_mass_visualization_config import (
    validate_visualization_config,
    get_plateau_column_mapping,
)
from src.analysis.critical_mass_extrapolation._critical_mass_visualization_core import (
    create_critical_mass_extrapolation_plots,
    load_and_validate_results_data,
    load_and_validate_plateau_data,
    group_data_for_visualization,
)


def process_critical_mass_visualization(
    results_csv_path,
    plateau_csv_path,
    plots_directory,
    analysis_type,
    clear_existing_plots,
    logger,
):
    """Process critical mass data and create visualizations."""

    plateau_column_mapping = get_plateau_column_mapping(analysis_type)

    # Load and validate data using NEW configurable functions
    results_df = load_and_validate_results_data(
        results_csv_path, OUTPUT_COLUMN_NAMES, logger
    )
    plateau_df = load_and_validate_plateau_data(
        plateau_csv_path, plateau_column_mapping
    )

    # Set up visualization infrastructure
    title_builder = PlotTitleBuilder(TITLE_LABELS_DICTIONARY)

    # Create plots directory structure
    plot_base_name = f"critical_mass_extrapolation_{analysis_type}"

    file_manager = PlotFileManager(base_directory=str(plots_directory))
    # Clear and prepare subdirectory
    plots_subdir_path = file_manager.prepare_subdirectory(
        plot_base_name, clear_existing=clear_existing_plots, confirm_clear=False
    )

    if clear_existing_plots:
        logger.info(f"Cleared existing plots in {plot_base_name} subdirectory")

    # Group data for visualization
    grouped_data = group_data_for_visualization(results_df, plateau_df, analysis_type)
    logger.info(f"Creating plots for {len(grouped_data)} parameter groups")

    # Create plots
    plots_created = 0
    for group_info in grouped_data:
        try:
            plot_path = create_critical_mass_extrapolation_plots(
                group_info,
                title_builder,
                file_manager,
                plots_subdir_path,
                analysis_type,
            )
            if plot_path:
                plots_created += 1
        except Exception as e:
            logger.warning(
                f"Failed to create plot for group {group_info.get('group_id', 'unknown')}: {e}"
            )
            continue

    logger.info(
        f"Successfully created {plots_created} critical mass extrapolation plots"
    )
    return plots_created


@click.command()
@click.option(
    "-r",
    "--results_csv",
    required=True,
    callback=csv_file.input,
    help="Critical mass results CSV file",
)
@click.option(
    "-p",
    "--plateau_csv",
    required=True,
    callback=csv_file.input,
    help="Plateau mass estimates CSV file",
)
@click.option(
    "-o",
    "--plots_directory",
    default=None,
    callback=directory.can_create,
    help="Output directory for plots",
)
@click.option(
    "-t",
    "--analysis_type",
    type=click.Choice(["pcac", "pion"], case_sensitive=False),
    required=True,
    help=("Analysis type for critical mass visualization " "(PCAC mass or pion mass)"),
)
@click.option(
    "-c",
    "--clear_existing",
    is_flag=True,
    default=False,
    help="Clear existing plots in output subdirectory before creating new ones",
)
@click.option("-log_on", "--enable_logging", is_flag=True, help="Enable logging")
@click.option(
    "-log_dir",
    "--log_directory",
    callback=directory.can_create,
    help="Directory for log files",
)
@click.option(
    "-log", "--log_filename", callback=validate_log_filename, help="Log filename"
)
def main(
    results_csv,
    plateau_csv,
    plots_directory,
    analysis_type,
    clear_existing,
    enable_logging,
    log_directory,
    log_filename,
):
    """Create critical mass extrapolation visualization plots."""

    # Validate configuration
    validate_visualization_config()

    # Set fallback for plots directory
    if plots_directory is None:
        plots_directory = str(Path(results_csv).parent)

    # Set up logging
    log_dir = (
        log_directory
        if log_directory
        else str(Path(results_csv).parent) if enable_logging else None
    )

    logger = create_script_logger(
        log_directory=log_dir,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
    )

    try:
        logger.log_script_start(f"Critical mass {analysis_type.upper()} visualization")

        plots_created = process_critical_mass_visualization(
            results_csv,
            plateau_csv,
            plots_directory,
            analysis_type,
            clear_existing,
            logger,
        )

        click.echo(f"✓ Created {plots_created} critical mass extrapolation plots")
        logger.log_script_end(
            f"Critical mass {analysis_type.upper()} visualization completed"
        )

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        logger.log_script_end(
            f"Critical mass {analysis_type.upper()} visualization failed"
        )
        raise


if __name__ == "__main__":
    main()
