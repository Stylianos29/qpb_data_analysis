#!/usr/bin/env python3
"""
Critical Bare Mass Calculation Script

This script calculates critical bare mass values from plateau PCAC mass 
estimates using linear extrapolation to the chiral limit (PCAC mass = 0).

The script processes CSV files containing plateau PCAC mass estimates,
groups data by lattice parameters, performs linear fits, and extrapolates
to find the critical bare mass where PCAC mass vanishes.

Key features:
    - Linear extrapolation to chiral limit using robust fitting methods
    - Comprehensive fit quality validation and physical reasonableness checks
    - Configurable data filtering and fitting parameters
    - Multi-panel visualization with fit diagnostics
    - CSV export with comprehensive metadata and diagnostics

Place this file as:
qpb_data_analysis/core/src/analysis/calculate_critical_bare_mass.py

Usage:
    python calculate_critical_bare_mass.py -i plateau_pcac_mass.csv -o 
    output_dir [options]
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
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import gvar as gv

# Import library components
from library.utils.logging_utilities import create_script_logger
from library.visualization.builders.title_builder import PlotTitleBuilder
from library.visualization.builders.filename_builder import PlotFilenameBuilder
from library.visualization.managers.file_manager import PlotFileManager
from library.visualization.plotters.data_plotter import DataPlotter
from library import constants
from library.validation.click_validators import (
    hdf5_file,
    directory,
    validate_log_filename,
)

# Import our modules
from src.analysis._critical_bare_mass_config import (
    INPUT_CSV_COLUMNS,
    GROUPING_PARAMETERS,
    FILENAME_EXCLUDED_PARAMETERS,
    OUTPUT_CSV_CONFIG,
    DATA_FILTERING,
    get_plotting_config,
    get_error_handling_config,
    validate_config,
)
from src.analysis._critical_bare_mass_methods import (
    load_and_validate_pcac_data,
    group_data_by_parameters,
    process_parameter_group,
    generate_fit_diagnostics,
    create_summary_statistics,
    linear_function,
)


@click.command()
@click.option(
    "-i",
    "--input_csv_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to input CSV file containing plateau PCAC mass estimates.",
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
    help="Directory for output plots. Default: output_directory/plots",
)
@click.option(
    "-out_csv",
    "--output_csv_filename",
    default=OUTPUT_CSV_CONFIG["default_filename"],
    help="Name for output CSV file with critical bare mass estimates.",
)
@click.option(
    "--enable_plotting",
    is_flag=True,
    default=True,
    help="Enable critical bare mass extrapolation plots.",
)
@click.option(
    "--clear_existing_plots",
    is_flag=True,
    default=False,
    help="Clear existing plots before generating new ones.",
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
    input_csv_file: str,
    output_directory: str,
    plots_directory: Optional[str],
    output_csv_filename: str,
    enable_plotting: bool,
    clear_existing_plots: bool,
    enable_logging: bool,
    log_directory: Optional[str],
    log_filename: Optional[str],
    verbose: bool,
) -> None:
    """
    Calculate critical bare mass values from plateau PCAC mass estimates
    using linear extrapolation to the chiral limit.

    This script processes PCAC mass plateau estimates, groups data by lattice
    parameters, and performs linear fits to extrapolate to the critical bare
    mass where PCAC mass = 0.
    """
    # Validate configuration
    if not validate_config():
        click.echo("❌ Invalid configuration detected. Please check config file.")
        sys.exit(1)

    # Set up logging
    if log_directory is None and enable_logging:
        log_directory = output_directory

    logger = create_script_logger(
        log_directory=log_directory if enable_logging else None,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=verbose,
    )

    logger.log_script_start("Critical bare mass calculation")

    try:
        # Log input parameters
        logger.info(f"Input CSV file: {input_csv_file}")
        logger.info(f"Output directory: {output_directory}")
        logger.info(f"Plots directory: {plots_directory}")
        logger.info(f"Plotting enabled: {enable_plotting}")

        file_manager_info = _prepare_output_directories(
            plots_directory,
            output_directory,
            enable_plotting,
            clear_existing_plots,
            logger,
        )

        # Process all groups
        calculation_results = _process_all_groups(
            input_csv_file, file_manager_info, enable_plotting, logger, verbose
        )

        # Export results to CSV
        csv_file_path = _export_results_to_csv(
            calculation_results, output_directory, output_csv_filename, logger
        )

        # Report final statistics
        _report_final_statistics(calculation_results, csv_file_path, logger)

        logger.log_script_end("Critical bare mass calculation completed successfully")
        click.echo(
            f"✓ Critical bare mass calculation complete. Results saved to: {csv_file_path}"
        )

    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.log_script_end("Critical bare mass calculation failed")
        raise


def _prepare_output_directories(
    plots_directory: Optional[str],
    output_directory: str,
    enable_plotting: bool,
    clear_existing_plots: bool,
    logger,
) -> Optional[Tuple[PlotFileManager, str]]:
    """
    Prepare output directories for plots and other files.

    Args:
        plots_directory: User-specified plots directory
        output_directory: Main output directory
        enable_plotting: Whether plotting is enabled
        clear_existing_plots: Whether to clear existing plots
        logger: Logger instance

    Returns:
        Tuple of (PlotFileManager, subdirectory_path) if plotting enabled, None otherwise
    """
    if not enable_plotting:
        logger.info("Plotting disabled - no plot directories will be created")
        return None

    # Use plots subdirectory if not specified
    if plots_directory is None:
        plots_directory = os.path.join(output_directory, "plots")

    plotting_config = get_plotting_config()
    plot_output_config = plotting_config["output"]

    # Create base plots directory
    base_plots_dir = os.path.join(plots_directory, plot_output_config["base_directory"])

    # Create plot file manager
    file_manager = PlotFileManager(base_plots_dir)

    # Create critical bare mass subdirectory
    critical_mass_subdir = file_manager.prepare_subdirectory(
        plot_output_config["subdirectory"],
        clear_existing=clear_existing_plots,
        confirm_clear=False,  # Don't require confirmation in script
    )

    if clear_existing_plots:
        logger.info("Cleared existing critical bare mass plots")

    logger.info(f"Plot output directory: {critical_mass_subdir}")
    return file_manager, critical_mass_subdir


def _process_all_groups(
    input_csv_file: str,
    file_manager_info: Optional[Tuple[PlotFileManager, str]],
    enable_plotting: bool,
    logger,
    verbose: bool,
) -> List[Dict[str, Any]]:
    """
    Process all parameter groups for critical bare mass calculation.

    Args:
        input_csv_file: Path to input CSV file
        file_manager_info: Tuple of (file_manager, subdirectory_path) if plotting enabled
        enable_plotting: Whether to generate plots
        logger: Logger instance
        verbose: Whether to show progress

    Returns:
        List of results from all processed groups
    """
    logger.info("Starting critical bare mass calculation workflow")

    # Load and validate data
    df = load_and_validate_pcac_data(input_csv_file, logger)

    # Group data by parameters
    groups = group_data_by_parameters(df, logger)

    # Process each group
    all_results = []
    total_groups = len(groups)

    for idx, (group_id, group_df) in enumerate(groups.items(), 1):
        if verbose:
            click.echo(f"Processing group {idx}/{total_groups}: {group_id}")

        logger.info(f"Processing group {idx}/{total_groups}: {group_id}")

        # Process this parameter group
        result = process_parameter_group(group_id, group_df, logger)
        all_results.append(result)

        # Generate plot if successful and plotting enabled
        if result["success"] and enable_plotting and file_manager_info is not None:
            try:
                file_manager, plot_subdir = file_manager_info
                _create_critical_mass_plot(
                    result, group_df, file_manager, plot_subdir, logger, df
                )
            except Exception as e:
                logger.warning(f"Failed to create plot for group {group_id}: {e}")

        # Log progress for this group
        if result["success"]:
            critical_mass = result["critical_bare_mass"]
            logger.info(f"Group {group_id} completed: critical mass = {critical_mass}")
        else:
            logger.warning(f"Group {group_id} failed: {result['error']}")

    return all_results


def _determine_multivalued_parameters(df: pd.DataFrame) -> List[str]:
    """
    Determine which grouping parameters actually have multiple values in the dataset.

    Args:
        df: DataFrame containing all data

    Returns:
        List of parameter names that have multiple unique values
    """
    multivalued_params = []

    for param in GROUPING_PARAMETERS:
        if param in df.columns and param not in FILENAME_EXCLUDED_PARAMETERS:
            unique_values = df[param].nunique()
            if unique_values > 1:
                multivalued_params.append(param)

    return multivalued_params


def _create_critical_mass_plot(
    result: Dict[str, Any],
    group_df: pd.DataFrame,
    file_manager: PlotFileManager,
    plot_subdir: str,
    logger,
    all_df: pd.DataFrame,  # Add reference to full dataset
) -> None:
    """
    Create critical bare mass extrapolation plot for a parameter group.

    Args:
        result: Processing results for this group
        group_df: Original data for this group
        file_manager: Plot file manager
        plot_subdir: Subdirectory path for plots
        logger: Logger instance
    """
    plotting_config = get_plotting_config()
    style_config = plotting_config["style"]
    content_config = plotting_config["content"]
    main_config = plotting_config["main"]

    # Prepare data for plotting
    bare_mass_col = INPUT_CSV_COLUMNS["bare_mass"]
    pcac_mean_col = INPUT_CSV_COLUMNS["pcac_mass_mean"]
    pcac_error_col = INPUT_CSV_COLUMNS["pcac_mass_error"]

    # Get all data points (not just fitted ones)
    x_all = group_df[bare_mass_col].to_numpy()
    y_all = group_df[pcac_mean_col].to_numpy()
    yerr_all = group_df[pcac_error_col].to_numpy()

    # Apply filtering to identify fitted points
    from src.analysis._critical_bare_mass_methods import filter_data_for_fitting

    filtered_df, _ = filter_data_for_fitting(group_df, logger)
    x_fitted = filtered_df[bare_mass_col].to_numpy()
    y_fitted = filtered_df[pcac_mean_col].to_numpy()

    # Get fit parameters
    slope = result["slope"]
    intercept = result["intercept"]
    critical_mass = result["critical_bare_mass"]

    # Create figure
    fig, ax = plt.subplots(figsize=main_config["default_figure_size"])

    # Plot all data points (if different from fitted)
    if len(x_all) > len(x_fitted):
        ax.errorbar(
            x_all,
            y_all,
            yerr=yerr_all,
            fmt="o",
            color="lightblue",
            alpha=0.5,
            markersize=main_config["marker_size"],
            capsize=main_config["capsize"],
            label="Excluded data points",
        )

    # Plot fitted data points
    fitted_mask = np.isin(x_all, x_fitted)
    ax.errorbar(
        x_all[fitted_mask],
        y_all[fitted_mask],
        yerr=yerr_all[fitted_mask],
        fmt="o",
        color=style_config["data_points"]["color"],
        alpha=style_config["data_points"]["alpha"],
        markersize=main_config["marker_size"],
        capsize=main_config["capsize"],
        label=style_config["data_points"]["label"],
    )

    # Create fit line data
    x_min, x_max = np.min(x_fitted), np.max(x_fitted)
    data_range = x_max - x_min
    extension = data_range * 0.3  # Extend 30% beyond data range

    x_line = np.linspace(gv.mean(critical_mass) - extension, x_max + extension, 100)
    y_line = linear_function(x_line, [slope, intercept])

    # Plot fit line
    ax.plot(
        x_line,
        gv.mean(y_line),
        color=style_config["linear_fit"]["color"],
        linestyle=style_config["linear_fit"]["linestyle"],
        linewidth=main_config["line_width"],
        alpha=style_config["linear_fit"]["alpha"],
        label=style_config["linear_fit"]["label"],
    )

    # Plot fit uncertainty band
    y_err = gv.sdev(y_line)
    ax.fill_between(
        x_line,
        gv.mean(y_line) - y_err,
        gv.mean(y_line) + y_err,
        color=style_config["fit_uncertainty"]["color"],
        alpha=style_config["fit_uncertainty"]["alpha"],
    )

    # Mark critical bare mass
    ax.axvline(
        gv.mean(critical_mass),
        color=style_config["critical_mass_line"]["color"],
        linestyle=style_config["critical_mass_line"]["linestyle"],
        alpha=style_config["critical_mass_line"]["alpha"],
        label=style_config["critical_mass_line"]["label"],
    )

    # Add zero lines
    ax.axhline(
        0,
        color=style_config["zero_lines"]["color"],
        linestyle=style_config["zero_lines"]["linestyle"],
        alpha=style_config["zero_lines"]["alpha"],
        linewidth=style_config["zero_lines"]["linewidth"],
    )
    ax.axvline(
        0,
        color=style_config["zero_lines"]["color"],
        linestyle=style_config["zero_lines"]["linestyle"],
        alpha=style_config["zero_lines"]["alpha"],
        linewidth=style_config["zero_lines"]["linewidth"],
    )

    # Configure axes
    ax.set_xlabel(
        constants.AXES_LABELS_BY_COLUMN_NAME["Bare_mass"],
        fontsize=main_config["font_size"],
    )
    ax.set_ylabel(r"a$m_{\text{PCAC}}$", fontsize=main_config["font_size"])

    if content_config["include_grid"]:
        ax.grid(True, alpha=0.3)

    # Add fit information text box
    if content_config["show_fit_equation"] or content_config["show_chi2"]:
        info_text = []

        if content_config["show_fit_equation"]:
            info_text.append(f"Linear fit:")
            info_text.append(f"slope = {slope:.4f}")
            info_text.append(f"intercept = {intercept:.4f}")

        if content_config["show_chi2"]:
            chi2_per_dof = result["chi2_per_dof"]
            dof = result["dof"]
            info_text.append(f"χ²/dof = {chi2_per_dof:.3f}")
            info_text.append(f"dof = {dof}")

        if content_config["show_critical_mass_value"]:
            info_text.append(f"Critical bare mass:")
            info_text.append(f"{critical_mass:.5f}")

        ax.text(
            0.02,
            0.98,
            "\n".join(info_text),
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=main_config["font_size"] - 2,
        )

    # Create plot title
    title_builder = PlotTitleBuilder(constants.TITLE_LABELS_BY_COLUMN_NAME)
    # Get tunable parameters from grouping parameters
    tunable_params = [
        col for col in GROUPING_PARAMETERS if col in result["group_metadata"]
    ]
    title = title_builder.build(
        metadata_dict=result["group_metadata"],
        tunable_params=tunable_params,
        leading_substring="Critical Bare Mass Extrapolation",
        wrapping_length=content_config["title_max_width"],
    )
    ax.set_title(title, fontsize=main_config["font_size"], pad=15)

    # Add legend
    ax.legend(fontsize=main_config["font_size"] - 2)

    # Generate filename and save
    filename_builder = PlotFilenameBuilder(constants.FILENAME_LABELS_BY_COLUMN_NAME)
    # Get multivalued parameters from the full dataset, not just grouping parameters
    multivalued_params = _determine_multivalued_parameters(all_df)
    plot_filename = filename_builder.build(
        metadata_dict=result["group_metadata"],
        base_name=get_plotting_config()["output"]["base_filename"],
        multivalued_params=multivalued_params,
    )

    plot_path = file_manager.plot_path(
        plot_subdir, plot_filename, format=get_plotting_config()["output"]["format"]
    )

    plt.tight_layout()
    fig.savefig(plot_path, dpi=main_config["default_dpi"], bbox_inches="tight")
    plt.close(fig)

    logger.debug(f"Created plot: {plot_path}")


def _export_results_to_csv(
    calculation_results: List[Dict[str, Any]],
    output_directory: str,
    output_csv_filename: str,
    logger,
) -> str:
    """
    Export critical bare mass calculation results to CSV file.

    Args:
        calculation_results: List of results from all parameter groups
        output_directory: Output directory path
        output_csv_filename: Name for output CSV file
        logger: Logger instance

    Returns:
        Path to the created CSV file
    """
    logger.info("Exporting results to CSV")

    csv_records = []

    for result in calculation_results:
        # Only include successful calculations
        if result["success"]:
            # Create record for successful calculation
            record = result["group_metadata"].copy()  # Start with group metadata

            # Add critical bare mass results
            critical_mass = result["critical_bare_mass"]
            record.update(
                {
                    "critical_bare_mass_mean": gv.mean(critical_mass),
                    "critical_bare_mass_error": gv.sdev(critical_mass),
                    "slope_mean": gv.mean(result["slope"]),
                    "slope_error": gv.sdev(result["slope"]),
                    "intercept_mean": gv.mean(result["intercept"]),
                    "intercept_error": gv.sdev(result["intercept"]),
                    "slope_significance": result["slope_significance"],
                    "n_data_points": result["n_data_points"],
                    "data_range_min": result["data_range"][0],
                    "data_range_max": result["data_range"][1],
                    "calculation_success": True,
                }
            )

            # Add fit diagnostics if enabled
            if OUTPUT_CSV_CONFIG["include_fit_diagnostics"]:
                record.update(
                    {
                        "chi2": result["chi2"],
                        "chi2_per_dof": result["chi2_per_dof"],
                        "dof": result["dof"],
                        "Q_value": result["Q_value"],
                        "fit_quality_passed": result["fit_validation"][
                            "quality_passed"
                        ],
                        "quality_warnings": "; ".join(
                            result["fit_validation"]["warnings"]
                        ),
                    }
                )

            # Add filtering information
            filtering_info = result["filtering_info"]
            record.update(
                {
                    "original_data_points": filtering_info["original_count"],
                    "filtered_data_points": filtering_info["final_count"],
                }
            )

            csv_records.append(record)

    if not csv_records:
        logger.warning("No results to export")
        raise ValueError("No successful calculations to export")

    # Create DataFrame and export
    df = pd.DataFrame(csv_records)
    csv_file_path = os.path.join(output_directory, output_csv_filename)

    # Round floating point values to specified precision
    float_precision = OUTPUT_CSV_CONFIG["float_precision"]
    float_columns = df.select_dtypes(include=[np.number]).columns
    df[float_columns] = df[float_columns].round(float_precision)

    df.to_csv(csv_file_path, index=False)

    logger.info(f"Exported {len(df)} records to: {csv_file_path}")
    return csv_file_path


def _report_final_statistics(
    calculation_results: List[Dict[str, Any]], csv_file_path: str, logger
) -> None:
    """
    Report final statistics and summary information.

    Args:
        calculation_results: List of results from all parameter groups
        csv_file_path: Path to exported CSV file
        logger: Logger instance
    """
    logger.info("Generating final statistics")

    # Create summary statistics
    summary = create_summary_statistics(calculation_results)

    # Log summary
    logger.info(f"Processing Summary:")
    logger.info(f"  Total parameter groups: {summary['total_groups']}")
    logger.info(f"  Successful calculations: {summary['successful_groups']}")
    logger.info(f"  Failed calculations: {summary['failed_groups']}")
    logger.info(f"  Success rate: {summary['success_rate']:.2%}")

    if summary["successful_groups"] > 0:
        logger.info(
            f"  Critical mass range: [{summary['critical_mass_range'][0]:.6f}, {summary['critical_mass_range'][1]:.6f}]"
        )
        logger.info(f"  Mean critical mass: {summary['critical_mass_mean']:.6f}")
        logger.info(f"  Median error: {summary['median_error']:.6f}")
        logger.info(f"  Median χ²/dof: {summary['median_chi2_per_dof']:.3f}")
        logger.info(
            f"  Good quality fits: {summary['good_quality_fits']}/{summary['successful_groups']}"
        )

    # Console output
    click.echo(f"\n📊 Calculation Summary:")
    click.echo(f"   Total groups processed: {summary['total_groups']}")
    click.echo(f"   Successful calculations: {summary['successful_groups']}")
    click.echo(f"   Success rate: {summary['success_rate']:.1%}")

    if summary["successful_groups"] > 0:
        click.echo(
            f"   Critical mass range: [{summary['critical_mass_range'][0]:.6f}, {summary['critical_mass_range'][1]:.6f}]"
        )
        click.echo(
            f"   Good quality fits: {summary['good_quality_fits']}/{summary['successful_groups']}"
        )

    click.echo(f"\n📁 Output files:")
    click.echo(f"   Results CSV: {os.path.basename(csv_file_path)}")

    # Report any systematic issues
    failed_results = [r for r in calculation_results if not r["success"]]
    if failed_results:
        error_messages = [r["error"] for r in failed_results]
        unique_errors = list(set(error_messages))

        logger.warning("Common failure reasons:")
        for error in unique_errors:
            count = error_messages.count(error)
            logger.warning(f"  '{error}': {count} groups")


if __name__ == "__main__":
    main()
