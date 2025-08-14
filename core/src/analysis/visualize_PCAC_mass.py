#!/usr/bin/env python3
"""
PCAC Mass Visualization Script

This script creates plots for PCAC mass jackknife samples alongside
their statistical averages from HDF5 PCAC mass analysis results.

The script processes the PCAC_mass_jackknife_samples dataset and creates
plots showing both individual jackknife samples and the corresponding
jackknife average with error bars.

Key features:
    - Multi-sample plots with configurable samples per plot
    - Jackknife average overlay with error bars
    - Dataset-specific styling and axis configuration
    - Hierarchical output directory structure
    - Comprehensive logging and error handling

Place this file as:
qpb_data_analysis/core/src/analysis/visualize_PCAC_mass.py

Usage:
    python visualize_PCAC_mass.py -i pcac_mass_analysis.h5 -o plots_dir
    [options]
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import click
import numpy as np

# Configure matplotlib to use non-interactive backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import h5py

# Import library components
from library.data.hdf5_analyzer import HDF5Analyzer
from library.visualization.managers.file_manager import PlotFileManager
from library.visualization.managers.layout_manager import PlotLayoutManager
from library.visualization.managers.style_manager import PlotStyleManager
from library.visualization.builders.filename_builder import PlotFilenameBuilder
from library.visualization.builders.title_builder import PlotTitleBuilder
from library import constants
from library.utils.logging_utilities import create_script_logger
from library.validation.click_validators import (
    hdf5_file,
    directory,
    validate_log_filename,
)

# Import configuration
from src.analysis._pcac_mass_visualization_config import (
    PCAC_MASS_DATASET_TO_PLOT,
    PCAC_MASS_PLOTS_BASE_DIRECTORY,
    SAMPLE_PLOT_STYLE,
    AVERAGE_PLOT_STYLE,
    DEFAULT_FIGURE_SIZE,
    DEFAULT_FONT_SIZE,
    SAMPLES_PER_PLOT,
    PLOT_QUALITY,
    get_pcac_mass_plot_config,
    get_sample_color,
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
    help="Directory for output plots.",
)
@click.option(
    "--clear_existing",
    is_flag=True,
    default=False,
    help="Clear all existing plots created by this script before generating new ones.",
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
    clear_existing: bool,
    enable_logging: bool,
    log_directory: Optional[str],
    log_filename: Optional[str],
    verbose: bool,
) -> None:
    """
    Visualize PCAC mass jackknife samples from HDF5 analysis results.

    This script creates plots showing PCAC mass jackknife samples along
    with their statistical averages and error bars.
    """
    # Set up logging
    if log_directory is None and enable_logging:
        log_directory = output_directory

    logger = create_script_logger(
        log_directory=log_directory if enable_logging else None,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=verbose,
    )

    logger.log_script_start("PCAC mass visualization")

    try:
        # Log input parameters
        logger.info(f"Input HDF5 file: {input_hdf5_file}")
        logger.info(f"Output directory: {output_directory}")
        logger.info(f"Samples per plot: {SAMPLES_PER_PLOT}")

        # Setup visualization managers
        file_manager = PlotFileManager(output_directory)
        layout_manager = PlotLayoutManager(constants)
        style_manager = PlotStyleManager(constants)
        filename_builder = PlotFilenameBuilder(constants.FILENAME_LABELS_BY_COLUMN_NAME)
        title_builder = PlotTitleBuilder(constants.TITLE_LABELS_BY_COLUMN_NAME)

        # Prepare base output directory
        base_plots_dir = _prepare_base_output_directory(
            output_directory, clear_existing, file_manager, logger
        )

        # Process PCAC mass data
        total_plots = _process_pcac_mass_data(
            input_hdf5_file,
            base_plots_dir,
            file_manager,
            layout_manager,
            style_manager,
            filename_builder,
            title_builder,
            logger,
            verbose,
        )

        # Report results
        if total_plots > 0:
            logger.log_script_end(f"Successfully created {total_plots} plots")
            click.echo(
                "✓ PCAC mass visualization complete. Created "
                f"{total_plots} plots in: {base_plots_dir}"
            )
        else:
            logger.warning("No plots were created")
            click.echo("⚠ No plots created. Check input file and logs for details.")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.log_script_end("PCAC mass visualization failed")
        raise


def _prepare_base_output_directory(
    output_directory: str, clear_existing: bool, file_manager: PlotFileManager, logger
) -> str:
    """
    Prepare the base output directory for plots.

    Returns:
        Path to base plots directory
    """
    base_plots_dir = os.path.join(output_directory, PCAC_MASS_PLOTS_BASE_DIRECTORY)

    if clear_existing and os.path.exists(base_plots_dir):
        logger.info(f"Clearing existing plots directory: {base_plots_dir}")
        import shutil

        shutil.rmtree(base_plots_dir)

    os.makedirs(base_plots_dir, exist_ok=True)
    logger.info(f"Base plots directory: {base_plots_dir}")

    return base_plots_dir


def _process_pcac_mass_data(
    input_hdf5_file: str,
    base_plots_dir: str,
    file_manager: PlotFileManager,
    layout_manager: PlotLayoutManager,
    style_manager: PlotStyleManager,
    filename_builder: PlotFilenameBuilder,
    title_builder: PlotTitleBuilder,
    logger,
    verbose: bool,
) -> int:
    """
    Process PCAC mass data and create visualization plots.

    Returns:
        Total number of plots created
    """
    total_plots = 0

    # Initialize HDF5Analyzer for metadata extraction
    analyzer = HDF5Analyzer(input_hdf5_file)

    try:
        with h5py.File(input_hdf5_file, "r") as hdf5_file:
            # Find all groups containing PCAC mass data
            pcac_mass_groups = _find_pcac_mass_groups(hdf5_file, logger)

            if not pcac_mass_groups:
                logger.warning("No groups with PCAC mass data found")
                return 0

            logger.info(f"Found {len(pcac_mass_groups)} groups with PCAC mass data")

            for group_path in pcac_mass_groups:
                if verbose:
                    click.echo(f"  Processing group: {group_path}")

                try:
                    plots_created = _process_single_pcac_mass_group(
                        hdf5_file,
                        group_path,
                        base_plots_dir,
                        analyzer,
                        file_manager,
                        layout_manager,
                        style_manager,
                        filename_builder,
                        title_builder,
                        logger,
                        verbose,
                    )

                    total_plots += plots_created
                    logger.info(
                        f"Created {plots_created} plots for group: {group_path}"
                    )

                except Exception as e:
                    logger.error(f"Error processing group {group_path}: {e}")
                    continue

    finally:
        analyzer.close()

    return total_plots


def _find_pcac_mass_groups(hdf5_file: h5py.File, logger) -> List[str]:
    """
    Find all groups containing PCAC mass datasets.

    Returns:
        List of group paths containing PCAC mass data
    """
    pcac_mass_groups = []

    def find_groups(name, obj):
        if isinstance(obj, h5py.Group) and PCAC_MASS_DATASET_TO_PLOT in obj:
            pcac_mass_groups.append(name)

    hdf5_file.visititems(find_groups)
    logger.debug(f"Found {len(pcac_mass_groups)} groups with PCAC mass data")

    return pcac_mass_groups


def _process_single_pcac_mass_group(
    hdf5_file: h5py.File,
    group_path: str,
    base_plots_dir: str,
    analyzer: HDF5Analyzer,
    file_manager: PlotFileManager,
    layout_manager: PlotLayoutManager,
    style_manager: PlotStyleManager,
    filename_builder: PlotFilenameBuilder,
    title_builder: PlotTitleBuilder,
    logger,
    verbose: bool,
) -> int:
    """
    Process a single group containing PCAC mass data.

    Returns:
        Number of plots created for this group
    """
    # Verify that group_path points to an HDF5 group
    try:
        group = hdf5_file[group_path]
        if not isinstance(group, h5py.Group):
            logger.error(
                f"Path '{group_path}' does not point to an HDF5 group (type: {type(group)})"
            )
            return 0
    except KeyError:
        logger.error(f"Group path '{group_path}' not found in HDF5 file")
        return 0
    except Exception as e:
        logger.error(f"Error accessing group '{group_path}': {e}")
        return 0

    # Verify and load PCAC mass jackknife samples dataset
    try:
        pcac_mass_obj = group[PCAC_MASS_DATASET_TO_PLOT]
        if not isinstance(pcac_mass_obj, h5py.Dataset):
            logger.error(
                f"'{PCAC_MASS_DATASET_TO_PLOT}' is not a dataset (type: {type(pcac_mass_obj)})"
            )
            return 0
        pcac_mass_samples = pcac_mass_obj[()]
    except KeyError:
        logger.error(
            f"Dataset '{PCAC_MASS_DATASET_TO_PLOT}' not found in group '{group_path}'"
        )
        return 0
    except Exception as e:
        logger.error(f"Error loading dataset '{PCAC_MASS_DATASET_TO_PLOT}': {e}")
        return 0

    # Verify and load corresponding mean values dataset
    try:
        pcac_mass_mean_obj = group["PCAC_mass_mean_values"]
        if not isinstance(pcac_mass_mean_obj, h5py.Dataset):
            logger.error(
                f"'PCAC_mass_mean_values' is not a dataset (type: {type(pcac_mass_mean_obj)})"
            )
            return 0
        pcac_mass_mean = pcac_mass_mean_obj[()]
    except KeyError:
        logger.error(
            f"Dataset 'PCAC_mass_mean_values' not found in group '{group_path}'"
        )
        return 0
    except Exception as e:
        logger.error(f"Error loading dataset 'PCAC_mass_mean_values': {e}")
        return 0

    # Verify and load corresponding error values dataset
    try:
        pcac_mass_error_obj = group["PCAC_mass_error_values"]
        if not isinstance(pcac_mass_error_obj, h5py.Dataset):
            logger.error(
                f"'PCAC_mass_error_values' is not a dataset (type: {type(pcac_mass_error_obj)})"
            )
            return 0
        pcac_mass_error = pcac_mass_error_obj[()]
    except KeyError:
        logger.error(
            f"Dataset 'PCAC_mass_error_values' not found in group '{group_path}'"
        )
        return 0
    except Exception as e:
        logger.error(f"Error loading dataset 'PCAC_mass_error_values': {e}")
        return 0

    # Load configuration labels
    config_labels = _load_configuration_labels(
        group, pcac_mass_samples.shape[0], logger
    )

    # Validate data dimensions
    _validate_pcac_mass_data(
        pcac_mass_samples, pcac_mass_mean, pcac_mass_error, group_path, logger
    )

    # Create group-specific output directory
    group_name = os.path.basename(group_path)
    group_plots_dir = os.path.join(base_plots_dir, group_name)
    os.makedirs(group_plots_dir, exist_ok=True)

    # Create multi-sample plots
    plots_created = _create_multi_sample_plots(
        pcac_mass_samples,
        pcac_mass_mean,
        pcac_mass_error,
        config_labels,
        group_name,
        group_path,
        group_plots_dir,
        analyzer,
        file_manager,
        layout_manager,
        style_manager,
        filename_builder,
        title_builder,
        logger,
        verbose,
    )

    return plots_created


def _load_configuration_labels(group: h5py.Group, n_samples: int, logger) -> List[str]:
    """
    Load configuration labels from the group.

    Returns:
        List of configuration labels
    """
    try:
        # Try to load gauge configuration labels
        if "gauge_configuration_labels" in group:
            # Verify it's a dataset before accessing
            config_obj = group["gauge_configuration_labels"]
            if not isinstance(config_obj, h5py.Dataset):
                logger.warning(
                    "'gauge_configuration_labels' is not a dataset "
                    f"(type: {type(config_obj)}), using default labels"
                )
                return [f"config_{i}" for i in range(n_samples)]

            config_data = config_obj[()]
            config_labels = [
                label.decode() if isinstance(label, bytes) else str(label)
                for label in config_data
            ]
        else:
            # Fallback to default labels
            config_labels = [f"config_{i}" for i in range(n_samples)]
            logger.warning("No gauge_configuration_labels found, using default labels")

        # Ensure we have enough labels
        if len(config_labels) < n_samples:
            logger.warning(
                f"Insufficient config labels ({len(config_labels)}) for samples ({n_samples})"
            )
            config_labels.extend(
                [f"config_{i}" for i in range(len(config_labels), n_samples)]
            )

        return config_labels

    except Exception as e:
        logger.warning(f"Error loading configuration labels: {e}")
        return [f"config_{i}" for i in range(n_samples)]


def _validate_pcac_mass_data(
    pcac_mass_samples: np.ndarray,
    pcac_mass_mean: np.ndarray,
    pcac_mass_error: np.ndarray,
    group_path: str,
    logger,
) -> None:
    """
    Validate PCAC mass data dimensions and consistency.
    """
    n_samples, n_time_points = pcac_mass_samples.shape

    if pcac_mass_mean.shape[0] != n_time_points:
        raise ValueError(
            f"Group {group_path}: Mean values length ({pcac_mass_mean.shape[0]}) "
            f"doesn't match time points ({n_time_points})"
        )

    if pcac_mass_error.shape[0] != n_time_points:
        raise ValueError(
            f"Group {group_path}: Error values length ({pcac_mass_error.shape[0]}) "
            f"doesn't match time points ({n_time_points})"
        )

    logger.debug(
        f"Group {group_path}: Validation passed - "
        f"{n_samples} jackknife samples, {n_time_points} time points"
    )


def _create_multi_sample_plots(
    pcac_mass_samples: np.ndarray,
    pcac_mass_mean: np.ndarray,
    pcac_mass_error: np.ndarray,
    config_labels: List[str],
    group_name: str,
    group_path: str,
    group_plots_dir: str,
    analyzer: HDF5Analyzer,
    file_manager: PlotFileManager,
    layout_manager: PlotLayoutManager,
    style_manager: PlotStyleManager,
    filename_builder: PlotFilenameBuilder,
    title_builder: PlotTitleBuilder,
    logger,
    verbose: bool,
) -> int:
    """
    Create multi-sample plots for PCAC mass data.

    Returns:
        Number of plots created
    """
    n_samples, n_time_points = pcac_mass_samples.shape
    n_plots = (n_samples + SAMPLES_PER_PLOT - 1) // SAMPLES_PER_PLOT
    plots_created = 0

    if verbose:
        click.echo(
            f"    Creating {n_plots} multi-sample plots ({SAMPLES_PER_PLOT} samples each)"
        )

    for plot_idx in range(n_plots):
        start_sample = plot_idx * SAMPLES_PER_PLOT
        end_sample = min(start_sample + SAMPLES_PER_PLOT, n_samples)

        try:
            # Extract samples for this plot
            samples_data = pcac_mass_samples[start_sample:end_sample, :]
            sample_labels = config_labels[start_sample:end_sample]

            # Create time index
            time_index = np.arange(n_time_points)

            # Create the multi-sample plot
            fig, _ = _create_pcac_mass_plot(
                time_index,
                samples_data,
                sample_labels,
                pcac_mass_mean,
                pcac_mass_error,
                group_name,
                group_path,
                analyzer,
                (start_sample, end_sample - 1),
                title_builder,
                logger,
            )

            # Generate filename
            filename = _generate_plot_filename(group_name, start_sample, end_sample - 1)

            # Save plot
            full_path = os.path.join(group_plots_dir, filename)
            fig.savefig(full_path, **PLOT_QUALITY)
            plt.close(fig)

            plots_created += 1

            if verbose:
                click.echo(f"      Created plot {plot_idx + 1}/{n_plots}: {filename}")

        except Exception as e:
            logger.error(
                f"Error creating plot {plot_idx} for samples "
                f"{start_sample}-{end_sample-1}: {e}"
            )
            continue

    return plots_created


def _create_pcac_mass_plot(
    time_index: np.ndarray,
    samples_data: np.ndarray,
    sample_labels: List[str],
    mean_values: np.ndarray,
    error_values: np.ndarray,
    group_name: str,
    group_path: str,
    analyzer: HDF5Analyzer,
    sample_indices: Tuple[int, int],
    title_builder: PlotTitleBuilder,
    logger,
) -> Tuple[Figure, Axes]:
    """
    Create a PCAC mass plot with multiple samples and average.

    Returns:
        Tuple of (figure, axes)
    """
    # Get plot configuration
    plot_config = get_pcac_mass_plot_config()

    # Apply dataset-specific slicing
    sliced_time, sliced_samples, sliced_mean, sliced_error = apply_dataset_slicing(
        time_index, samples_data, mean_values, error_values, plot_config
    )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    # Plot each sample
    sample_style = SAMPLE_PLOT_STYLE.copy()
    sample_style.pop("label_suffix", None)  # Remove non-matplotlib parameter

    for i, (sample_data, sample_label) in enumerate(zip(sliced_samples, sample_labels)):
        color = get_sample_color(sample_indices[0] + i)  # Use consistent coloring
        ax.plot(
            sliced_time,
            sample_data,
            label=sample_label,
            color=color,
            **sample_style,
        )

    # Plot average with error bars
    average_style = AVERAGE_PLOT_STYLE.copy()
    avg_label = average_style.pop("label", "Jackknife average")

    # Check if we have valid data to plot
    if sliced_time is None or sliced_mean is None or sliced_error is None:
        raise ValueError("One or more data arrays is None - cannot create plot")

    if len(sliced_time) == 0 or len(sliced_mean) == 0:
        raise ValueError("Empty data arrays - cannot create plot")

    ax.errorbar(
        sliced_time,
        sliced_mean,
        yerr=sliced_error,
        label=avg_label,
        **average_style,
    )

    # Add zero line if configured
    if plot_config.get("show_zero_line", False):
        zero_style = plot_config["zero_line_style"]
        ax.axhline(y=0, **zero_style)

    # Set axis properties
    ax.set_yscale(plot_config["y_scale"])
    ax.set_xlabel(plot_config["x_label"], fontsize=DEFAULT_FONT_SIZE)
    ax.set_ylabel(plot_config["y_label"], fontsize=DEFAULT_FONT_SIZE)

    # Set x-axis to always show from 0 for perspective if configured
    if plot_config.get("show_full_time_range", False):
        max_time = (
            max(sliced_time)
            if len(sliced_time) > 0
            else len(mean_values) + plot_config.get("time_offset", 0)
        )
        ax.set_xlim(0, max_time + 1)  # Add small buffer

    # Add title using proper title builder
    title = generate_sample_plot_title(
        sample_label=f"Samples {sample_indices[0]}-{sample_indices[1]}",
        dataset_name=PCAC_MASS_DATASET_TO_PLOT,
        group_metadata=get_group_metadata(analyzer, group_path),
        title_builder=title_builder,
    )
    ax.set_title(title, fontsize=DEFAULT_FONT_SIZE + 2)

    # Add legend and grid
    ax.legend(
        fontsize=DEFAULT_FONT_SIZE - 2, loc=plot_config.get("legend_location", "best")
    )
    ax.grid(True, alpha=plot_config.get("grid_alpha", 0.3))

    # Tight layout
    plt.tight_layout()

    return fig, ax


def apply_dataset_slicing(
    time_index: np.ndarray,
    sample_data: np.ndarray,
    mean_values: np.ndarray,
    error_values: np.ndarray,
    dataset_config: dict,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Apply dataset-specific slicing to time index and data arrays.

    Args:
        - time_index: Array of time indices
        - sample_data: Sample data array (can be 1D or 2D)
        - mean_values: Mean values array
        - error_values: Error values array
        - dataset_config: Configuration dict with x_start_index,
          x_end_offset, and time_offset

    Returns:
        Tuple of sliced (time_index, sample_data, mean_values,
        error_values)
    """
    start_idx = dataset_config.get("x_start_index", 0)
    end_offset = dataset_config.get("x_end_offset", 0)
    time_offset = dataset_config.get("time_offset", 0)

    # Calculate end index for data slicing
    if end_offset > 0:
        end_idx = len(time_index) - end_offset 
    else:
        end_idx = len(time_index)

    # Apply slicing to data arrays Handle both 1D and 2D sample data
    if sample_data.ndim == 1:
        sliced_sample_data = sample_data[start_idx:end_idx]
    else:  # 2D case (multiple samples)
        sliced_sample_data = sample_data[:, start_idx:end_idx]

    sliced_mean_values = (
        mean_values[start_idx:end_idx] if mean_values is not None else None
    )
    sliced_error_values = (
        error_values[start_idx:end_idx] if error_values is not None else None
    )

    # Create time index for the sliced data with proper offset The
    # actual data points start at time_offset + start_idx
    data_length = (
        len(sliced_mean_values)
        if sliced_mean_values is not None
        else sliced_sample_data.shape[-1]
    )
    sliced_time_index = np.arange(
        time_offset + start_idx, time_offset + start_idx + data_length
    )

    return (
        sliced_time_index,
        sliced_sample_data,
        sliced_mean_values,
        sliced_error_values,
    )


def get_group_metadata(analyzer: HDF5Analyzer, group_path: str) -> Dict:
    """
    Get group metadata for use in titles and filenames.

    Returns:
        Dictionary of group metadata
    """
    try:
        # Temporarily restrict to single group to get its metadata
        original_groups = analyzer.active_groups.copy()
        analyzer.active_groups = {group_path}

        # Create a temporary dataframe to extract metadata
        temp_df = analyzer.to_dataframe(
            datasets=[PCAC_MASS_DATASET_TO_PLOT],
            add_time_column=False,
            flatten_arrays=False,
        )

        # Extract unique parameter values as metadata
        metadata = {}
        for column in temp_df.columns:
            if column != PCAC_MASS_DATASET_TO_PLOT:  # Skip the actual data column
                unique_vals = temp_df[column].unique()
                if len(unique_vals) == 1:
                    metadata[column] = unique_vals[0]
                else:
                    metadata[column] = list(unique_vals)

        # Restore original active groups
        analyzer.active_groups = original_groups

        return metadata
    except Exception as e:
        # Restore original active groups on error
        if "original_groups" in locals():
            analyzer.active_groups = original_groups
        return {}


def generate_sample_plot_title(
    sample_label: str,
    dataset_name: str,
    group_metadata: Dict,
    title_builder: PlotTitleBuilder,
) -> str:
    """
    Generate a descriptive title for the sample plot using the title
    builder.
    """
    # Clean dataset name for title
    clean_dataset_name = "PCAC Mass"

    # Base title
    base_title = f"{clean_dataset_name} - {sample_label}"

    # Add metadata if available
    if group_metadata:
        try:
            metadata_title = title_builder.build(
                metadata_dict=group_metadata,
                tunable_params=list(group_metadata.keys()),
                leading_substring=base_title,
                wrapping_length=80,
            )
            return metadata_title
        except Exception:
            return base_title

    return base_title


def _generate_plot_filename(group_name: str, start_sample: int, end_sample: int) -> str:
    """
    Generate filename for the plot.

    Returns:
        Plot filename
    """
    return f"PCAC_mass_{group_name}_samples_{start_sample:02d}_{end_sample:02d}.png"


if __name__ == "__main__":
    main()
