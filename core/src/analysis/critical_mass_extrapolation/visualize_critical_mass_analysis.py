#!/usr/bin/env python3
"""
Critical Mass Analysis Visualization Script

Creates linear extrapolation plots for critical mass analysis, showing
plateau mass vs bare mass data with fitted lines and critical mass
annotations. Supports both PCAC mass and pion effective mass analyses
with automatic power transformation (PCAC: mass¹, Pion: mass²).

The script combines critical mass calculation results with original
plateau data to visualize the linear extrapolation to the chiral limit,
generating publication-quality plots with comprehensive annotations, fit
quality metrics, and error propagation.

Features:
    - Automatic analysis-type detection or manual specification
    - Configurable plot clearing and directory management  
    - Group-wise plotting for different lattice parameter combinations
    - Professional styling with error bars, fit lines, and annotations
    - Robust error handling with detailed logging

Usage Examples:
    # Basic PCAC analysis visualization
    python visualize_critical_mass_analysis.py \
        -t pcac \
        -r critical_bare_mass_from_pcac.csv \
        -p plateau_PCAC_mass_estimates.csv \
        -o plots_output

    # Pion analysis with plot clearing and logging
    python visualize_critical_mass_analysis.py \
        -t pion \
        -r critical_bare_mass_from_pion.csv \
        -p plateau_pion_mass_estimates.csv \
        -o plots_output \
        --clear_existing \
        -log_on \
        -log_dir logs

Input Requirements:
    - Results CSV: Output from
      calculate_critical_mass_from_{pcac,pion}.py
    - Plateau CSV: Output from extract_plateau_{PCAC,pion}_mass.py
    - Analysis type: Must match the type used in calculation
"""

from pathlib import Path
import shutil

import click


from library.validation.click_validators import (
    csv_file,
    directory,
    validate_log_filename,
)
from library.utils.logging_utilities import create_script_logger
from library.visualization.builders.title_builder import PlotTitleBuilder
from library.visualization.managers.file_manager import PlotFileManager
from library.constants.labels import TITLE_LABELS_BY_COLUMN_NAME

from src.analysis.critical_mass_extrapolation._critical_mass_visualization_config import (
    validate_visualization_config,
    get_results_column_mapping,
    get_plateau_column_mapping,
    get_plot_subdirectory_name,
)
from src.analysis.critical_mass_extrapolation._critical_mass_visualization_core import (
    create_critical_mass_extrapolation_plots,
    load_and_validate_results_data,
    load_and_validate_plateau_data,
    group_data_for_visualization,
)


def process_critical_mass_visualization(
    results_csv_path: str,
    plateau_csv_path: str,
    plots_directory: str,
    analysis_type: str,
    clear_existing_plots: bool,
    logger,
) -> int:
    """
    Process critical mass data and create comprehensive visualization
    plots.

    Loads critical mass calculation results and original plateau data,
    groups by lattice parameters, and generates linear extrapolation
    plots showing the determination of critical bare mass at the chiral
    limit. Handles both PCAC and pion analyses with appropriate power
    transformations.

    Args:
        - results_csv_path: Path to critical mass calculation results
          CSV file
        - plateau_csv_path: Path to plateau mass estimates CSV file
        - plots_directory: Directory path for saving generated plots
        - analysis_type: Type of analysis ("pcac" or "pion")
        - clear_existing_plots: Whether to clear existing plots before
          creation
        - logger: Logger instance from custom logging system for
          progress tracking

    Returns:
        Number of plots successfully created

    Raises:
        ValueError: If input files are empty, missing required columns,
        or
                   analysis_type is unsupported
        FileNotFoundError: If input CSV files do not exist Exception:
        For plot creation failures (logged as warnings, not fatal)

    Process Flow:
        1. Load and validate input CSV files with required columns
        2. Validate parameter consistency between results and plateau data
        3. Group data by lattice parameters using intelligent analysis
        4. Set up visualization infrastructure (file managers, title
           builders)
        5. Create individual plots for each parameter group combination
        6. Apply analysis-specific transformations (pion: square values)
        7. Generate plots with data points, fit lines, and critical mass
           annotations
        8. Save plots with descriptive filenames and return success
           count

    Plot Features:
        - Linear extrapolation lines extending to chiral limit
        - Data points with error bars from plateau estimates
        - Critical mass vertical line and annotation with uncertainty
        - Fit quality metrics (R²) displayed in legend
        - Professional styling with grids and appropriate axis ranges
    """

    results_column_mapping = get_results_column_mapping()
    plateau_column_mapping = get_plateau_column_mapping(analysis_type)

    logger.info("=" * 80)
    logger.info("LOADING AND VALIDATING DATA")
    logger.info("=" * 80)

    # Load and validate data using configurable functions
    logger.info(f"Loading results data from: {results_csv_path}")
    results_df = load_and_validate_results_data(
        results_csv_path, results_column_mapping, logger
    )
    logger.info(f"  Loaded {len(results_df)} results rows")

    logger.info(f"Loading plateau data from: {plateau_csv_path}")
    plateau_df = load_and_validate_plateau_data(
        plateau_csv_path, plateau_column_mapping
    )
    logger.info(f"  Loaded {len(plateau_df)} plateau rows")

    # Set up visualization infrastructure
    logger.info("\n" + "=" * 80)
    logger.info("SETTING UP VISUALIZATION INFRASTRUCTURE")
    logger.info("=" * 80)

    title_builder = PlotTitleBuilder(TITLE_LABELS_BY_COLUMN_NAME)

    file_manager = PlotFileManager(base_directory=str(plots_directory))

    # Get hierarchical directory structure
    parent_name, subdir_name = get_plot_subdirectory_name(analysis_type)

    if parent_name:
        # Hierarchical: parent/subdir/
        logger.info(f"Using directory structure: {parent_name}/{subdir_name}/")

        # Create parent directory
        parent_path = file_manager.prepare_subdirectory(
            parent_name, clear_existing=False, confirm_clear=False
        )

        # Create subdirectory within parent
        plots_subdir_path = Path(parent_path) / subdir_name
        plots_subdir_path.mkdir(parents=True, exist_ok=True)

        # Clear only this subdirectory if requested
        if clear_existing_plots:
            if plots_subdir_path.exists():
                shutil.rmtree(plots_subdir_path)
                plots_subdir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared existing plots in {parent_name}/{subdir_name}/")

        plots_subdir_path = str(plots_subdir_path)
    else:
        # Flat structure (backward compatibility)
        logger.info(f"Using directory structure: {subdir_name}/")
        plots_subdir_path = file_manager.prepare_subdirectory(
            subdir_name, clear_existing=clear_existing_plots, confirm_clear=False
        )
        if clear_existing_plots:
            logger.info(f"Cleared existing plots in {subdir_name}/")

    # Group data for visualization
    logger.info("\n" + "=" * 80)
    logger.info("GROUPING DATA FOR VISUALIZATION")
    logger.info("=" * 80)

    try:
        grouped_data = group_data_for_visualization(results_df, plateau_df)
    except ValueError as e:
        logger.error(f"Data grouping failed: {e}")
        raise

    logger.info(
        f"\n✓ Successfully grouped data into {len(grouped_data)} parameter combinations"
    )

    # Create plots
    logger.info("\n" + "=" * 80)
    logger.info("CREATING PLOTS")
    logger.info("=" * 80)

    plots_created = 0
    plots_failed = 0

    for i, group_info in enumerate(grouped_data, 1):
        group_id = group_info.get("group_id", "unknown")
        logger.info(f"\nPlot {i}/{len(grouped_data)}: {group_id}")

        try:
            plot_path = create_critical_mass_extrapolation_plots(
                group_info,
                title_builder,
                plots_subdir_path,
                analysis_type,
            )
            if plot_path:
                plots_created += 1
                logger.info(f"  ✓ Created: {plot_path}")
            else:
                plots_failed += 1
                logger.warning(f"  ✗ Failed to create plot (no path returned)")
        except Exception as e:
            plots_failed += 1
            logger.warning(f"  ✗ Failed to create plot: {e}")
            continue

    logger.info("\n" + "=" * 80)
    logger.info("VISUALIZATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total groups: {len(grouped_data)}")
    logger.info(f"Plots created: {plots_created}")
    logger.info(f"Plots failed: {plots_failed}")
    logger.info(
        f"Success rate: {plots_created}/{len(grouped_data)} ({100*plots_created/len(grouped_data):.1f}%)"
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
