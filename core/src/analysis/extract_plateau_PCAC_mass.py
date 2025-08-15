#!/usr/bin/env python3
"""
PCAC Mass Plateau Extraction Script

This script extracts plateau PCAC mass values from PCAC mass time series
using jackknife analysis and robust plateau detection methods.

The script processes HDF5 files from calculate_PCAC_mass.py, detects plateau
regions in individual time series, and estimates jackknife averaged plateau
values for each parameter group.

Key features:
    - Robust plateau detection with multiple sigma thresholds
    - Individual sample plateau extraction with error tracking
    - Multi-panel visualization showing individual fits
    - CSV export with comprehensive diagnostics
    - Configurable estimation methods and quality control

Place this file as: qpb_data_analysis/core/src/analysis/extract_plateau_PCAC_mass.py

Usage:
    python extract_plateau_PCAC_mass.py -i pcac_mass_analysis.h5 -o output_dir
    [options]
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import click
import numpy as np
import pandas as pd
import h5py

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
from library import constants
from library.validation.click_validators import (
    hdf5_file,
    directory,
    validate_log_filename,
)

# Import our modules
from src.analysis._plateau_extraction_config import (
    INPUT_DATASETS,
    METADATA_DATASETS,
    OUTPUT_CSV_CONFIG,
    get_plotting_config,
    get_error_handling_config,
    validate_config,
)
from src.analysis._plateau_fitting_methods import (
    extract_plateau_from_group,
    calculate_jackknife_average,
)


@click.command()
@click.option(
    "-i",
    "--input_hdf5_file",
    required=True,
    callback=hdf5_file.input,
    help="Path to input HDF5 file containing PCAC mass analysis results.",
)
@click.option(
    "-o",
    "--output_directory",
    required=True,
    callback=directory.must_exist,
    help="Directory for output files.",
)
@click.option(
    "-out_csv",
    "--output_csv_filename",
    default=OUTPUT_CSV_CONFIG["default_filename"],
    help="Name for output CSV file with plateau estimates.",
)
@click.option(
    "--enable_plotting",
    is_flag=True,
    default=True,
    help="Enable multi-panel plateau extraction plots.",
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
    input_hdf5_file: str,
    output_directory: str,
    output_csv_filename: str,
    enable_plotting: bool,
    clear_existing_plots: bool,
    enable_logging: bool,
    log_directory: Optional[str],
    log_filename: Optional[str],
    verbose: bool,
) -> None:
    """
    Extract plateau PCAC mass values from PCAC mass time series analysis.

    This script processes PCAC mass jackknife samples, detects plateau regions,
    and extracts plateau values with uncertainties for each parameter group.
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

    logger.log_script_start("PCAC mass plateau extraction")

    try:
        # Log input parameters
        logger.info(f"Input HDF5 file: {input_hdf5_file}")
        logger.info(f"Output directory: {output_directory}")
        logger.info(f"Plotting enabled: {enable_plotting}")

        # Prepare output directories
        plots_dir = _prepare_output_directories(
            output_directory, enable_plotting, clear_existing_plots, logger
        )

        # Process PCAC mass data
        extraction_results = _process_all_groups(
            input_hdf5_file, plots_dir, enable_plotting, logger, verbose
        )

        # Export results to CSV
        csv_file_path = _export_results_to_csv(
            extraction_results, output_directory, output_csv_filename, logger
        )

        # Report final statistics
        _report_final_statistics(extraction_results, csv_file_path, logger)

        logger.log_script_end("PCAC mass plateau extraction completed successfully")
        click.echo(f"✓ Plateau extraction complete. Results saved to: {csv_file_path}")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.log_script_end("PCAC mass plateau extraction failed")
        raise


def _prepare_output_directories(
    output_directory: str, enable_plotting: bool, clear_existing_plots: bool, logger
) -> Optional[str]:
    """
    Prepare output directories for plots and other files.

    Returns:
        Path to plots directory if plotting enabled, None otherwise
    """
    plots_dir = None

    if enable_plotting:
        plot_config = get_plotting_config()
        plots_dir = os.path.join(
            output_directory, plot_config["output"]["base_directory"]
        )

        if clear_existing_plots and os.path.exists(plots_dir):
            logger.info(f"Clearing existing plots directory: {plots_dir}")
            import shutil

            shutil.rmtree(plots_dir)

        os.makedirs(plots_dir, exist_ok=True)
        logger.info(f"Plots directory: {plots_dir}")

    return plots_dir


def _process_all_groups(
    input_hdf5_file: str,
    plots_dir: Optional[str],
    enable_plotting: bool,
    logger,
    verbose: bool,
) -> List[Dict[str, Any]]:
    """
    Process all groups in the HDF5 file for plateau extraction.

    Returns:
        List of extraction results for all groups
    """
    extraction_results = []

    with h5py.File(input_hdf5_file, "r") as hdf5_file:
        # Find all groups containing PCAC mass data
        pcac_groups = _find_pcac_mass_groups(hdf5_file, logger)

        if not pcac_groups:
            logger.warning("No groups with PCAC mass data found")
            return extraction_results

        logger.info(f"Found {len(pcac_groups)} groups with PCAC mass data")

        for group_idx, group_path in enumerate(pcac_groups):
            if verbose:
                click.echo(
                    f"  Processing group {group_idx + 1}/{len(pcac_groups)}: {group_path}"
                )

            try:
                result = _process_single_group(
                    hdf5_file,
                    group_path,
                    plots_dir,
                    enable_plotting,
                    logger,
                    # , verbose
                )
                extraction_results.append(result)

                status = "✓" if result["success"] else "✗"
                logger.info(
                    f"Group {group_path}: {status} - {result.get('summary', 'Processed')}"
                )

            except Exception as e:
                logger.error(f"Error processing group {group_path}: {e}")
                # Create failed result entry
                result = {
                    "success": False,
                    "group_name": os.path.basename(group_path),
                    "group_path": group_path,
                    "error_message": f"Processing error: {e}",
                    "summary": f"Failed - {e}",
                }
                extraction_results.append(result)
                continue

    return extraction_results


def _find_pcac_mass_groups(hdf5_file: h5py.File, logger) -> List[str]:
    """
    Find all groups containing PCAC mass datasets.

    Returns:
        List of group paths containing PCAC mass data
    """
    pcac_groups = []
    required_dataset = INPUT_DATASETS["jackknife_samples"]

    def find_groups(name, obj):
        if isinstance(obj, h5py.Group) and required_dataset in obj:
            pcac_groups.append(name)

    hdf5_file.visititems(find_groups)
    logger.debug(f"Found {len(pcac_groups)} groups with PCAC mass data")

    return pcac_groups


def _process_single_group(
    hdf5_file: h5py.File,
    group_path: str,
    plots_dir: Optional[str],
    enable_plotting: bool,
    logger,
) -> Dict[str, Any]:
    """
    Process a single group for plateau extraction.

    Returns:
        Dictionary containing extraction results
    """
    group = hdf5_file[group_path]
    group_name = os.path.basename(group_path)

    # Load PCAC mass data
    jackknife_samples = group[INPUT_DATASETS["jackknife_samples"]][()]
    pcac_mean = group[INPUT_DATASETS["mean_values"]][()]
    pcac_error = group[INPUT_DATASETS["error_values"]][()]

    # Load configuration labels
    config_labels = _load_configuration_labels(
        group, jackknife_samples.shape[0], logger
    )

    # Load group metadata for CSV export
    group_metadata = _extract_group_metadata(group, group_path, hdf5_file, logger)

    # Validate data
    _validate_group_data(jackknife_samples, pcac_mean, pcac_error, group_path, logger)

    # Extract plateau from group
    extraction_result = extract_plateau_from_group(
        jackknife_samples, config_labels, group_name, logger
    )

    # Add metadata and paths to result
    extraction_result.update(
        {
            "group_path": group_path,
            "group_metadata": group_metadata,
            "input_data_shape": jackknife_samples.shape,
        }
    )

    # Create summary
    if extraction_result["success"]:
        plateau_value = extraction_result["plateau_value"]
        n_successful = extraction_result["n_successful"]
        n_total = extraction_result["n_total_samples"]
        extraction_result["summary"] = (
            f"Plateau: {plateau_value.mean:.5f}±{plateau_value.sdev:.5f} "
            f"({n_successful}/{n_total} samples)"
        )
    else:
        extraction_result["summary"] = extraction_result.get("error_message", "Failed")

    # Create plots if enabled and extraction was successful
    if enable_plotting and extraction_result["success"] and plots_dir:
        try:
            _create_plateau_extraction_plots(
                jackknife_samples,
                pcac_mean,
                pcac_error,
                config_labels,
                extraction_result,
                group_name,
                group_metadata,  # THIS WAS MISSING - FIXED!
                plots_dir,
                logger,
            )
        except Exception as e:
            logger.warning(f"Plot creation failed for group {group_name}: {e}")

    return extraction_result


def _load_configuration_labels(group: h5py.Group, n_samples: int, logger) -> List[str]:
    """
    Load configuration labels from the group.

    Returns:
        List of configuration labels
    """
    try:
        if "gauge_configuration_labels" in group:
            config_labels = [
                label.decode() if isinstance(label, bytes) else str(label)
                for label in group["gauge_configuration_labels"][()]
            ]
        else:
            config_labels = [f"config_{i:03d}" for i in range(n_samples)]
            logger.warning("No gauge_configuration_labels found, using default labels")

        # Ensure we have enough labels
        if len(config_labels) < n_samples:
            logger.warning(
                f"Insufficient config labels ({len(config_labels)}) for samples ({n_samples})"
            )
            config_labels.extend(
                [f"config_{i:03d}" for i in range(len(config_labels), n_samples)]
            )

        return config_labels

    except Exception as e:
        logger.warning(f"Error loading configuration labels: {e}")
        return [f"config_{i:03d}" for i in range(n_samples)]


def _extract_group_metadata(
    group: h5py.Group, group_path: str, hdf5_file: h5py.File, logger
) -> Dict[str, Any]:
    """
    Extract metadata from group and parent groups for CSV export.

    Returns:
        Dictionary of metadata parameters
    """
    metadata = {}

    # Extract group attributes
    for attr_name, attr_value in group.attrs.items():
        metadata[attr_name] = attr_value

    # Extract parent group attributes
    parent_path = "/".join(group_path.split("/")[:-1])
    if parent_path and parent_path in hdf5_file:
        parent_group = hdf5_file[parent_path]
        for attr_name, attr_value in parent_group.attrs.items():
            if attr_name not in metadata:  # Don't override group-specific attributes
                metadata[attr_name] = attr_value

    # Add group identification
    metadata["group_name"] = os.path.basename(group_path)
    metadata["group_path"] = group_path

    logger.debug(
        f"Extracted {len(metadata)} metadata parameters for group {group_path}"
    )

    return metadata


def _validate_group_data(
    jackknife_samples: np.ndarray,
    pcac_mean: np.ndarray,
    pcac_error: np.ndarray,
    group_path: str,
    logger,
) -> None:
    """
    Validate PCAC mass data dimensions and consistency.
    """
    n_samples, n_time_points = jackknife_samples.shape

    if pcac_mean.shape[0] != n_time_points:
        raise ValueError(
            f"Group {group_path}: Mean values length ({pcac_mean.shape[0]}) "
            f"doesn't match time points ({n_time_points})"
        )

    if pcac_error.shape[0] != n_time_points:
        raise ValueError(
            f"Group {group_path}: Error values length ({pcac_error.shape[0]}) "
            f"doesn't match time points ({n_time_points})"
        )

    # Check for invalid values
    if np.any(~np.isfinite(jackknife_samples)):
        raise ValueError(f"Group {group_path}: Invalid values in jackknife samples")

    logger.debug(
        f"Group {group_path}: Validation passed - "
        f"{n_samples} jackknife samples, {n_time_points} time points"
    )


def _create_plateau_extraction_plots(
    jackknife_samples: np.ndarray,
    pcac_mean: np.ndarray,
    pcac_error: np.ndarray,
    config_labels: List[str],
    extraction_result: Dict[str, Any],
    group_name: str,
    group_metadata: Dict[str, Any],
    plots_dir: str,
    logger,
) -> None:
    """
    Create multi-panel plots showing plateau extraction for individual samples.
    """
    plot_config = get_plotting_config()
    samples_per_plot = plot_config["config"]["samples_per_plot"]

    successful_extractions = extraction_result["successful_extractions"]
    plateau_bounds = extraction_result["plateau_bounds"]

    if not successful_extractions:
        logger.warning(f"No successful extractions to plot for group {group_name}")
        return

    n_successful = len(successful_extractions)
    n_plots = (n_successful + samples_per_plot - 1) // samples_per_plot

    for plot_idx in range(n_plots):
        start_idx = plot_idx * samples_per_plot
        end_idx = min(start_idx + samples_per_plot, n_successful)

        plot_extractions = successful_extractions[start_idx:end_idx]
        n_panels = len(plot_extractions)

        try:
            _create_single_multi_panel_plot(
                jackknife_samples,
                pcac_mean,
                pcac_error,
                plot_extractions,
                plateau_bounds,
                group_name,
                group_metadata,
                plot_idx,
                start_idx,  # For filename
                end_idx - 1,  # For filename
                n_panels,
                plots_dir,
                plot_config,
                logger,
            )
        except Exception as e:
            logger.error(
                f"Failed to create plot {plot_idx} for group {group_name}: {e}"
            )


def _create_single_multi_panel_plot(
    jackknife_samples: np.ndarray,
    pcac_mean: np.ndarray,
    pcac_error: np.ndarray,
    plot_extractions: List[Dict],
    plateau_bounds: Tuple[int, int],
    group_name: str,
    group_metadata: Dict[str, Any],
    plot_idx: int,
    start_sample_idx: int,
    end_sample_idx: int,
    n_panels: int,
    plots_dir: str,
    plot_config: Dict,
    logger,
) -> None:
    """
    Create a single multi-panel plot showing plateau extraction results.
    """
    config = plot_config["config"]
    subplot_style = plot_config["subplot_style"]
    time_series_style = plot_config["time_series_style"]
    plateau_style = plot_config["plateau_fit_style"]
    labels = plot_config["labels"]
    data_range = plot_config.get("data_range", {})

    # Create figure with subplots
    fig, axes = plt.subplots(
        n_panels, 1, figsize=config["figure_size"], sharex=config["share_x_axis"]
    )

    if n_panels == 1:
        axes = [axes]  # Ensure axes is always a list

    plt.subplots_adjust(hspace=config["subplot_spacing"])

    # Create title using PlotTitleBuilder
    title_builder = PlotTitleBuilder(constants.TITLE_LABELS_BY_COLUMN_NAME)

    try:
        # Create proper title with metadata
        plot_title = title_builder.build(
            metadata_dict=group_metadata,
            tunable_params=list(group_metadata.keys()),
            leading_substring="PCAC Mass Plateau Extraction",
            wrapping_length=80,
        )
    except Exception as e:
        # Fallback title
        logger.warning(f"Failed to build title with PlotTitleBuilder: {e}")
        plot_title = f"PCAC Mass Plateau Extraction - {group_name}"

    # Set main title on the figure
    fig.suptitle(plot_title, fontsize=subplot_style["font_size"] + 2, y=0.98)

    plateau_start, plateau_end = plateau_bounds

    # Apply data trimming if configured
    trim_start = (
        data_range.get("trim_start_points", 0)
        if data_range.get("apply_trimming", False)
        else 0
    )
    trim_end = (
        data_range.get("trim_end_points", 0)
        if data_range.get("apply_trimming", False)
        else 0
    )

    if trim_end > 0:
        plot_end_idx = len(pcac_mean) - trim_end
    else:
        plot_end_idx = len(pcac_mean)

    # Create time index with offset and trimming
    full_time_index = (
        np.arange(len(pcac_mean)) + 2
    )  # Add time offset (PCAC starts at t=2)
    plot_time_index = full_time_index[trim_start:plot_end_idx]

    for panel_idx, (ax, extraction) in enumerate(zip(axes, plot_extractions)):
        sample_idx = extraction["sample_index"]
        config_label = extraction["config_label"]
        plateau_value = extraction["plateau_value"]

        # Get sample data with trimming
        full_sample_data = jackknife_samples[sample_idx, :]
        plot_sample_data = full_sample_data[trim_start:plot_end_idx]

        # Plot time series data
        # Clean style dict for individual samples (no error bars)
        plot_style = time_series_style.copy()
        plot_style.pop("capsize", None)  # Remove capsize since yerr=None

        # Use config label as legend title
        legend_title = labels["legend_title_template"].format(config_label=config_label)

        ax.errorbar(
            plot_time_index,
            plot_sample_data,
            yerr=None,  # Individual samples don't have error bars
            label=legend_title,
            **plot_style,
        )

        # Plot plateau fit line (only within actual fitting range)
        plateau_y = plateau_value.mean
        plateau_err = plateau_value.sdev

        # Convert plateau bounds to time coordinates with offset
        fit_start_time = plateau_start + 2  # Add time offset
        fit_end_time = plateau_end + 2

        # Only show plateau line if it's within the plotted range
        plot_start_time = plot_time_index[0]
        plot_end_time = plot_time_index[-1]

        if fit_start_time <= plot_end_time and fit_end_time >= plot_start_time:
            # Clip to visible range
            visible_fit_start = max(fit_start_time, plot_start_time)
            visible_fit_end = min(fit_end_time, plot_end_time)

            # Plot plateau line
            ax.hlines(
                y=plateau_y,
                xmin=visible_fit_start,
                xmax=visible_fit_end,
                colors=plateau_style["line_color"],
                linestyles=plateau_style["line_style"],
                linewidth=plateau_style["line_width"],
                label="Plateau fit",
            )

            # Plot uncertainty band
            ax.fill_between(
                [visible_fit_start, visible_fit_end],
                plateau_y - plateau_err,
                plateau_y + plateau_err,
                color=plateau_style["band_color"],
                alpha=plateau_style["band_alpha"],
            )

            # Highlight actual fitting range (not entire plateau region)
            ax.axvspan(
                visible_fit_start,
                visible_fit_end,
                alpha=0.1,
                color="green",
                label="Fitting range",
            )

        # Formatting
        if subplot_style["grid"]:
            ax.grid(True, alpha=subplot_style["grid_alpha"])

        # Y-axis label
        ax.set_ylabel(labels["y_label"], fontsize=subplot_style["font_size"])

        # Fit info text
        fit_info = labels["fit_info_template"].format(
            value=plateau_value.mean, error=plateau_value.sdev
        )
        ax.text(
            0.02,
            0.98,
            fit_info,
            transform=ax.transAxes,
            fontsize=subplot_style["font_size"] - 1,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Legend for each panel (with config label as title)
        legend = ax.legend(fontsize=subplot_style["font_size"] - 1, loc="upper right")
        if hasattr(legend, "set_title"):
            legend.set_title("")  # Remove legend title since we include it in the label

    # X-axis label on bottom panel
    axes[-1].set_xlabel(labels["x_label"], fontsize=subplot_style["font_size"])

    # Save plot with correct filename format (matching visualize_jackknife_samples.py)
    filename = f"{group_name}_{start_sample_idx}-{end_sample_idx}.{plot_config['output']['file_format']}"
    output_path = os.path.join(plots_dir, filename)

    fig.savefig(
        output_path,
        dpi=plot_config["output"]["dpi"],
        bbox_inches=plot_config["output"]["bbox_inches"],
    )
    plt.close(fig)

    logger.debug(f"Created plot: {filename}")


def _export_results_to_csv(
    extraction_results: List[Dict[str, Any]],
    output_directory: str,
    output_csv_filename: str,
    logger,
) -> str:
    """
    Export extraction results to CSV file.

    Returns:
        Path to the created CSV file
    """
    csv_records = []

    for result in extraction_results:
        if result["success"]:
            # Create record for successful extraction
            record = result["group_metadata"].copy()  # Start with group metadata

            # Add plateau extraction results
            plateau_value = result["plateau_value"]
            record.update(
                {
                    "plateau_PCAC_mass_mean": plateau_value.mean,
                    "plateau_PCAC_mass_error": plateau_value.sdev,
                    "plateau_start_time": result["plateau_bounds"][0]
                    + 2,  # Add time offset
                    "plateau_end_time": result["plateau_bounds"][1] + 2,
                    "plateau_n_points": result["plateau_bounds"][1]
                    - result["plateau_bounds"][0],
                    "sample_size": result["sample_size"],
                    "n_total_samples": result["n_total_samples"],
                    "n_failed_samples": result["n_failed"],
                    "extraction_success": True,
                }
            )

            # Add diagnostics if enabled
            if OUTPUT_CSV_CONFIG["include_diagnostics"]:
                diagnostics = result["final_diagnostics"]
                record.update(
                    {
                        "estimation_method": diagnostics["method"],
                        "use_inverse_variance": diagnostics["use_inverse_variance"],
                        "avg_correlation": diagnostics.get("avg_correlation", None),
                    }
                )

            # Add failed configuration labels if any
            if result["failed_extractions"]:
                failed_labels = [
                    fe["config_label"] for fe in result["failed_extractions"]
                ]
                record["failed_config_labels"] = ";".join(failed_labels)
            else:
                record["failed_config_labels"] = ""

            csv_records.append(record)

        else:
            # Optionally include failed groups (based on config)
            error_config = get_error_handling_config()
            if error_config["failed_group_action"] == "include_nan":
                record = result.get("group_metadata", {}).copy()
                record.update(
                    {
                        "plateau_PCAC_mass_mean": np.nan,
                        "plateau_PCAC_mass_error": np.nan,
                        "plateau_start_time": np.nan,
                        "plateau_end_time": np.nan,
                        "plateau_n_points": np.nan,
                        "sample_size": result.get("n_successful", 0),
                        "n_total_samples": result.get("n_total_samples", 0),
                        "n_failed_samples": result.get("n_failed", 0),
                        "extraction_success": False,
                        "error_message": result.get("error_message", "Unknown error"),
                    }
                )
                csv_records.append(record)

    if not csv_records:
        logger.warning("No successful extractions to export")
        # Create empty CSV with headers
        empty_df = pd.DataFrame(
            columns=[
                "group_name",
                "plateau_PCAC_mass_mean",
                "plateau_PCAC_mass_error",
                "extraction_success",
                "error_message",
            ]
        )
        csv_file_path = os.path.join(output_directory, output_csv_filename)
        empty_df.to_csv(csv_file_path, index=False)
        return csv_file_path

    # Create DataFrame and export
    df = pd.DataFrame(csv_records)

    # Sort by group name for consistency
    df = df.sort_values("group_name", ignore_index=True)

    # Round floating point values
    float_cols = df.select_dtypes(include=[np.floating]).columns
    df[float_cols] = df[float_cols].round(OUTPUT_CSV_CONFIG["float_precision"])

    csv_file_path = os.path.join(output_directory, output_csv_filename)
    df.to_csv(csv_file_path, index=False)

    logger.info(f"Exported {len(df)} results to CSV: {csv_file_path}")

    return csv_file_path


def _report_final_statistics(
    extraction_results: List[Dict[str, Any]], csv_file_path: str, logger
) -> None:
    """
    Report final statistics about the extraction process.
    """
    # Group statistics
    total_groups = len(extraction_results)
    successful_groups = sum(1 for r in extraction_results if r["success"])
    failed_groups = total_groups - successful_groups

    # Sample statistics
    total_samples = sum(r.get("n_total_samples", 0) for r in extraction_results)
    successful_samples = sum(r.get("n_successful", 0) for r in extraction_results)
    failed_samples = sum(r.get("n_failed", 0) for r in extraction_results)

    # Log statistics
    logger.info("=== EXTRACTION SUMMARY ===")
    logger.info(f"Total groups processed: {total_groups}")
    logger.info(f"Successful groups: {successful_groups}")
    logger.info(f"Failed groups: {failed_groups}")
    if total_groups > 0:  # Add check
        logger.info(f"Group success rate: {successful_groups/total_groups*100:.1f}%")
    logger.info("")
    logger.info(f"Total samples processed: {total_samples}")
    logger.info(f"Successful sample extractions: {successful_samples}")
    logger.info(f"Failed sample extractions: {failed_samples}")
    if total_samples > 0:  # Add check
        logger.info(f"Sample success rate: {successful_samples/total_samples*100:.1f}%")
    logger.info("")
    logger.info(f"Results exported to: {csv_file_path}")

    # Console output
    click.echo(f"\n📊 Extraction Summary:")
    if total_groups > 0:  # Add check
        click.echo(
            f"   Groups: {successful_groups}/{total_groups} successful ({successful_groups/total_groups*100:.1f}%)"
        )
    else:
        click.echo(f"   Groups: {successful_groups}/{total_groups} successful")

    if total_samples > 0:  # Add check
        click.echo(
            f"   Samples: {successful_samples}/{total_samples} successful ({successful_samples/total_samples*100:.1f}%)"
        )
    else:
        click.echo(f"   Samples: No samples processed")

    # Report groups with issues
    failed_group_results = [r for r in extraction_results if not r["success"]]
    if failed_group_results:
        click.echo(f"\n⚠️  Failed groups:")
        for result in failed_group_results[:5]:  # Show first 5
            group_name = result.get("group_name", "Unknown")
            error_msg = result.get("error_message", "Unknown error")
            click.echo(f"   - {group_name}: {error_msg}")

        if len(failed_group_results) > 5:
            click.echo(
                f"   ... and {len(failed_group_results) - 5} more (see log file)"
            )


if __name__ == "__main__":
    main()
