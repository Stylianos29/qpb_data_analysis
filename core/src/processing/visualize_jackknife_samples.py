#!/usr/bin/env python3
"""
Jackknife Visualization Script

This script creates individual plots for each jackknife sample time
series alongside their statistical averages from HDF5 jackknife analysis
results.

The script processes 2D jackknife datasets and creates separate plots
for each sample (row) showing both the individual sample time series and
the corresponding gvar average with error bars.

Usage:
    python visualize_jackknife_samples.py -i input.h5 -o plots_dir
    [options]
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union, List

import click
import numpy as np
from numpy import ndarray

# Configure matplotlib to use non-interactive backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

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
from src.processing._jackknife_visualization_config import (
    JACKKNIFE_DATASETS_TO_PLOT,
    JACKKNIFE_PLOTS_BASE_DIRECTORY,
    SAMPLE_PLOT_STYLE,
    AVERAGE_PLOT_STYLE,
    DEFAULT_FIGURE_SIZE,
    DEFAULT_FONT_SIZE,
    SAMPLES_PER_PLOT,
    TITLE_EXCLUDED_PARAMETERS,
    get_dataset_plot_config,
    get_dataset_labels,
)


@click.command()
@click.option(
    "-i",
    "--input_hdf5_file",
    required=True,
    callback=hdf5_file.input,
    help="Path to input HDF5 file containing jackknife analysis results.",
)
@click.option(
    "-p",
    "--plots_directory",
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
    plots_directory: str,
    clear_existing: bool,
    enable_logging: bool,
    log_directory: Optional[str],
    log_filename: str,
    verbose: bool,
) -> None:
    """
    Visualize jackknife samples from HDF5 jackknife analysis results.

    This script creates individual plots for each jackknife sample
    showing both the sample time series and the corresponding
    statistical average with error bars.
    """
    # === SETUP AND VALIDATION ===

    # Setup logging
    logger = create_script_logger(
        log_directory=log_directory or plots_directory,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=verbose,  # Console logging follows verbose flag
        verbose=False,
    )

    logger.log_script_start("Jackknife samples visualization")

    logger.info(f"Input file: {input_hdf5_file}")
    logger.info(f"Output directory: {plots_directory}")
    logger.info(f"Datasets to process: {JACKKNIFE_DATASETS_TO_PLOT}")

    try:
        # === LOAD HDF5 DATA ===
        analyzer = HDF5Analyzer(input_hdf5_file)
        logger.info(f"Loaded HDF5 file: {input_hdf5_file}")
        logger.info(f"Found {len(analyzer.active_groups)} groups in HDF5 file")

        # === SETUP VISUALIZATION MANAGERS ===
        file_manager = PlotFileManager(plots_directory)
        layout_manager = PlotLayoutManager(constants)
        style_manager = PlotStyleManager(constants)
        filename_builder = PlotFilenameBuilder(constants.FILENAME_LABELS_BY_COLUMN_NAME)
        title_builder = PlotTitleBuilder(constants.TITLE_LABELS_BY_COLUMN_NAME)

        # === PREPARE BASE DIRECTORY ===
        base_plots_dir = file_manager.prepare_subdirectory(
            JACKKNIFE_PLOTS_BASE_DIRECTORY,
            clear_existing=clear_existing,
            confirm_clear=False,
        )

        logger.info(f"Base plots directory: {base_plots_dir}")

        # === PROCESS ALL DATASETS ===
        total_plots_created = process_jackknife_datasets(
            analyzer=analyzer,
            file_manager=file_manager,
            layout_manager=layout_manager,
            style_manager=style_manager,
            filename_builder=filename_builder,
            title_builder=title_builder,
            base_plots_dir=base_plots_dir,
            logger=logger,
            verbose=verbose,
        )

        logger.info(
            f"Successfully created {total_plots_created} jackknife sample plots"
        )

        # Success summary (console output for immediate feedback)
        if verbose or not enable_logging:
            click.echo(f"✓ Visualization complete!")
            click.echo(f"✓ Created {total_plots_created} plots")
            click.echo(f"✓ Results saved to: {base_plots_dir}")

    except Exception as e:
        logger.log_script_error(e)
        click.echo(f"ERROR: Critical failure during processing: {e}")
        sys.exit(1)

    finally:
        # Clean up
        if "analyzer" in locals():
            analyzer.close()
        logger.log_script_end(f"Created {total_plots_created} plots successfully")
        logger.close()


def process_jackknife_datasets(
    analyzer: HDF5Analyzer,
    file_manager: PlotFileManager,
    layout_manager: PlotLayoutManager,
    style_manager: PlotStyleManager,
    filename_builder: PlotFilenameBuilder,
    title_builder: PlotTitleBuilder,
    base_plots_dir: str,
    logger,
    verbose: bool,
) -> int:
    """
    Process all jackknife datasets and create visualization plots.

    Args:
        - analyzer: HDF5Analyzer instance
        - file_manager: PlotFileManager for directory operations
        - layout_manager: PlotLayoutManager for plot layout
        - style_manager: PlotStyleManager for plot styling
        - filename_builder: PlotFilenameBuilder for consistent naming
        - title_builder: PlotTitleBuilder for plot titles
        - base_plots_dir: Base directory for plots
        - logger: Logger instance
        - verbose: Whether to show verbose output

    Returns:
        Total number of plots created
    """
    total_plots_created = 0

    for dataset_name in JACKKNIFE_DATASETS_TO_PLOT:
        if verbose:
            click.echo(f"\nProcessing dataset: {dataset_name}")

        # Check if dataset exists in HDF5 file
        if dataset_name not in analyzer.list_of_output_quantity_names_from_hdf5:
            logger.warning(
                f"Dataset '{dataset_name}' not found in HDF5 file. Skipping."
            )
            if verbose:
                click.echo(f"⚠ Dataset '{dataset_name}' not found. Skipping.")
            continue

        # Create dataset subdirectory
        dataset_plots_dir = file_manager.prepare_subdirectory(
            os.path.join(base_plots_dir, dataset_name)
        )

        # Process each active group
        dataset_plots_count = 0
        for group_path in sorted(analyzer.active_groups):
            try:
                group_plots_count = process_group_jackknife_data(
                    analyzer=analyzer,
                    group_path=group_path,
                    dataset_name=dataset_name,
                    file_manager=file_manager,
                    layout_manager=layout_manager,
                    style_manager=style_manager,
                    filename_builder=filename_builder,
                    title_builder=title_builder,
                    dataset_plots_dir=dataset_plots_dir,
                    logger=logger,
                    verbose=verbose,
                )

                dataset_plots_count += group_plots_count
                total_plots_created += group_plots_count

            except Exception as e:
                logger.error(
                    f"Error processing group {group_path} for dataset {dataset_name}: {e}"
                )
                if verbose:
                    click.echo(f"⚠ Error processing group {group_path}: {e}")
                continue

        if verbose:
            click.echo(f"  Created {dataset_plots_count} plots for {dataset_name}")

    return total_plots_created


def process_group_jackknife_data(
    analyzer: HDF5Analyzer,
    group_path: str,
    dataset_name: str,
    file_manager: PlotFileManager,
    layout_manager: PlotLayoutManager,
    style_manager: PlotStyleManager,
    filename_builder: PlotFilenameBuilder,
    title_builder: PlotTitleBuilder,
    dataset_plots_dir: str,
    logger,
    verbose: bool,
) -> int:
    """
    Process jackknife data for a specific group and dataset. Updated to
    create multi-sample plots instead of individual plots.

    Args:
        - analyzer: HDF5Analyzer instance
        - group_path: Path to the HDF5 group
        - dataset_name: Name of the jackknife dataset
        - file_manager: PlotFileManager instance
        - layout_manager: PlotLayoutManager instance
        - style_manager: PlotStyleManager instance
        - filename_builder: PlotFilenameBuilder instance
        - title_builder: PlotTitleBuilder instance
        - dataset_plots_dir: Directory for this dataset's plots
        - logger: Logger instance
        - verbose: Whether to show verbose output

    Returns:
        Number of plots created for this group
    """
    # Extract group name for directory
    group_name = group_path.split("/")[-1]

    # Create group subdirectory
    group_plots_dir = file_manager.prepare_subdirectory(
        os.path.join(dataset_plots_dir, group_name)
    )

    if verbose:
        click.echo(f"    Processing group: {group_name}")

    try:
        # Load jackknife samples data (2D array)
        jackknife_data = analyzer.dataset_values(
            dataset_name, return_gvar=False, group_path=group_path
        )

        if not isinstance(jackknife_data, np.ndarray) or jackknife_data.ndim != 2:
            logger.warning(
                f"Dataset {dataset_name} in group {group_path} is not a 2D array. Skipping."
            )
            return 0

        # Load corresponding mean and error values for gvar average
        mean_values, error_values = load_corresponding_gvar_data(
            analyzer, dataset_name, group_path, logger
        )

        if mean_values is None or error_values is None:
            logger.warning(
                f"Could not load corresponding mean/error data for {dataset_name} in {group_path}"
            )
            return 0

        # Convert to numpy arrays if they're lists
        if isinstance(mean_values, list):
            if len(mean_values) == 1:
                mean_values = mean_values[0]  # Single array case
            else:
                mean_values = np.concatenate(mean_values)  # Multiple arrays case
                logger.info(f"Concatenated {len(mean_values)} mean value arrays")

        if isinstance(error_values, list):
            if len(error_values) == 1:
                error_values = error_values[0]  # Single array case
            else:
                error_values = np.concatenate(error_values)  # Multiple arrays case
                logger.info(f"Concatenated {len(error_values)} error value arrays")

        # Load gauge configuration labels
        config_labels = load_gauge_configuration_labels(
            analyzer, group_path, jackknife_data.shape[0], logger
        )

        # Create multi-sample plots instead of individual plots
        plots_created = create_multi_sample_plots(
            jackknife_data=jackknife_data,
            mean_values=mean_values,
            error_values=error_values,
            config_labels=config_labels,
            dataset_name=dataset_name,
            group_name=group_name,
            group_path=group_path,
            analyzer=analyzer,
            file_manager=file_manager,
            layout_manager=layout_manager,
            style_manager=style_manager,
            filename_builder=filename_builder,
            title_builder=title_builder,
            group_plots_dir=group_plots_dir,
            logger=logger,
            verbose=verbose,
        )

        return plots_created

    except Exception as e:
        logger.error(f"Error processing jackknife data for group {group_path}: {e}")
        return 0


def create_multi_sample_plots(
    jackknife_data: np.ndarray,
    mean_values: np.ndarray,
    error_values: np.ndarray,
    config_labels: List[str],
    dataset_name: str,
    group_name: str,
    group_path: str,
    analyzer: HDF5Analyzer,
    file_manager: PlotFileManager,
    layout_manager: PlotLayoutManager,
    style_manager: PlotStyleManager,
    filename_builder: PlotFilenameBuilder,
    title_builder: PlotTitleBuilder,
    group_plots_dir: str,
    logger,
    verbose: bool,
) -> int:
    """
    Create multi-sample plots for jackknife data.

    Args:
        - jackknife_data: 2D array of jackknife samples (n_samples,
          n_time)
        - mean_values: Array of mean values
        - error_values: Array of error values
        - config_labels: List of gauge configuration labels
        - dataset_name: Name of the dataset
        - group_name: Name of the group
        - group_path: Path to the HDF5 group
        - analyzer: HDF5Analyzer instance
        - file_manager: PlotFileManager instance
        - layout_manager: PlotLayoutManager instance
        - style_manager: PlotStyleManager instance
        - filename_builder: PlotFilenameBuilder instance
        - title_builder: PlotTitleBuilder instance
        - group_plots_dir: Directory for group plots
        - logger: Logger instance
        - verbose: Whether to show verbose output

    Returns:
        Number of plots created
    """
    n_samples, n_time_points = jackknife_data.shape
    plots_created = 0

    # Get dataset configuration for logging
    dataset_config = get_dataset_plot_config(dataset_name)
    if verbose:
        config_desc = dataset_config.get("description", "Default configuration")
        click.echo(f"      Using config: {config_desc}")
        click.echo(f"      Samples per plot: {SAMPLES_PER_PLOT}")

    # Ensure we have enough configuration labels
    if len(config_labels) < n_samples:
        logger.warning(
            f"Insufficient config labels ({len(config_labels)}) for samples ({n_samples})"
        )
        # Extend with defaults
        config_labels.extend(
            [f"config_{i}" for i in range(len(config_labels), n_samples)]
        )

    # Create time index
    time_index = np.arange(n_time_points)

    # Calculate number of plots needed
    n_plots = (n_samples + SAMPLES_PER_PLOT - 1) // SAMPLES_PER_PLOT  # Ceiling division

    if verbose:
        click.echo(f"      Creating {n_plots} plots for {n_samples} samples")

    for plot_idx in range(n_plots):
        try:
            # Calculate sample range for this plot
            start_sample = plot_idx * SAMPLES_PER_PLOT
            end_sample = min(start_sample + SAMPLES_PER_PLOT, n_samples)

            # Extract samples for this plot
            plot_samples = jackknife_data[start_sample:end_sample, :]
            plot_labels = config_labels[start_sample:end_sample]

            # Create the plot
            fig, ax = create_multi_sample_plot(
                time_index=time_index,
                samples_data=plot_samples,
                sample_labels=plot_labels,
                mean_values=mean_values,
                error_values=error_values,
                dataset_name=dataset_name,
                group_name=group_name,
                group_metadata=get_group_metadata(
                    analyzer, group_path
                ),  # Get the metadata
                sample_indices=(start_sample, end_sample - 1),
                title_builder=title_builder,  # Pass the title builder
            )

            # Generate filename
            filename = generate_multi_sample_plot_filename(
                group_name=group_name, sample_indices=(start_sample, end_sample - 1)
            )

            # Save plot
            full_path = file_manager.plot_path(group_plots_dir, filename)
            fig.savefig(full_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            plots_created += 1

            if verbose:
                click.echo(f"      Created plot {plot_idx + 1}/{n_plots}: {filename}")

        except Exception as e:
            logger.error(
                f"Error creating plot {plot_idx} for samples {start_sample}-{end_sample-1}: {e}"
            )
            continue

    return plots_created


def create_individual_sample_plots(
    jackknife_data: np.ndarray,
    mean_values: np.ndarray,
    error_values: np.ndarray,
    config_labels: List[str],
    dataset_name: str,
    group_name: str,
    group_path: str,
    analyzer: HDF5Analyzer,
    file_manager: PlotFileManager,
    layout_manager: PlotLayoutManager,
    style_manager: PlotStyleManager,
    filename_builder: PlotFilenameBuilder,
    title_builder: PlotTitleBuilder,
    group_plots_dir: str,
    logger,
    verbose: bool,
) -> int:
    """
    Create individual plots for each jackknife sample. Enhanced to
    support dataset-specific configurations.
    """
    n_samples, n_time_points = jackknife_data.shape
    plots_created = 0

    # Get dataset configuration for logging
    dataset_config = get_dataset_plot_config(dataset_name)
    if verbose:
        config_desc = dataset_config.get("description", "Default configuration")
        click.echo(f"      Using config: {config_desc}")

    # Ensure we have enough configuration labels
    if len(config_labels) < n_samples:
        logger.warning(
            f"Insufficient config labels ({len(config_labels)}) for samples ({n_samples})"
        )
        # Extend with defaults
        config_labels.extend(
            [f"config_{i}" for i in range(len(config_labels), n_samples)]
        )

    # Get group metadata for titles
    group_metadata = get_group_metadata(analyzer, group_path)

    for sample_idx in range(n_samples):
        try:
            # Extract sample data
            sample_data = jackknife_data[sample_idx, :]
            config_label = config_labels[sample_idx]

            # Create time index
            time_index = np.arange(n_time_points)

            # Create the plot with enhanced configuration support
            fig, _ = create_sample_plot(
                time_index=time_index,
                sample_data=sample_data,
                mean_values=mean_values,
                error_values=error_values,
                sample_label=config_label,
                dataset_name=dataset_name,
                group_metadata=group_metadata,
                layout_manager=layout_manager,
                style_manager=style_manager,
                title_builder=title_builder,
            )

            # Generate filename
            filename = generate_sample_plot_filename(
                config_label=config_label,
                dataset_name=dataset_name,
                group_name=group_name,
                group_metadata=group_metadata,
                filename_builder=filename_builder,
            )

            # Save plot
            full_path = file_manager.plot_path(group_plots_dir, filename)
            fig.savefig(full_path)
            plt.close(fig)

            plots_created += 1

            if verbose and sample_idx % 5 == 0:  # Progress update every 5 plots
                click.echo(f"      Created {sample_idx + 1}/{n_samples} plots...")

        except Exception as e:
            logger.error(
                f"Error creating plot for sample {sample_idx} ({config_label}): {e}"
            )
            continue

    return plots_created


def create_multi_sample_plot(
    time_index: np.ndarray,
    samples_data: np.ndarray,  # 2D array: (n_samples_in_plot, n_time)
    sample_labels: List[str],
    mean_values: np.ndarray,
    error_values: np.ndarray,
    dataset_name: str,
    group_name: str,
    group_metadata: Dict,  # Add this parameter
    sample_indices: Tuple[int, int],  # (start_idx, end_idx) for filename
    title_builder: PlotTitleBuilder,  # Add this parameter
) -> Tuple[Figure, Axes]:
    """
    Create a plot with multiple jackknife samples and their average.

    Args:
        - time_index: Array of time indices
        - samples_data: 2D array with sample data (n_samples_in_plot,
          n_time)
        - sample_labels: List of labels for each sample
        - mean_values: Mean values array
        - error_values: Error values array
        - dataset_name: Name of the dataset
        - group_name: Name of the group
        - sample_indices: Tuple of (start_idx, end_idx) for this plot

    Returns:
        Tuple of (figure, axes)
    """
    # Get dataset-specific configuration
    dataset_config = get_dataset_plot_config(dataset_name)

    # Apply dataset-specific slicing to all data
    sliced_time, sliced_samples, sliced_mean, sliced_error = apply_dataset_slicing(
        time_index, samples_data, mean_values, error_values, dataset_config
    )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    # Plot each sample with its own label
    sample_style = SAMPLE_PLOT_STYLE.copy()
    # Remove any non-matplotlib parameters
    sample_style.pop("label_suffix", None)  # Remove if it exists

    for i, (sample_data, sample_label) in enumerate(zip(sliced_samples, sample_labels)):
        ax.plot(
            sliced_time,
            sample_data,
            label=sample_label,  # Use the actual config label
            **sample_style,
        )

    # Plot average data with error bars if available
    if sliced_mean is not None and sliced_error is not None:
        average_style = AVERAGE_PLOT_STYLE.copy()
        # Extract and remove non-matplotlib parameters
        avg_label = average_style.pop("label", "Jackknife average")
        average_style.pop("label_suffix", None)  # Remove if it exists

        ax.errorbar(
            sliced_time,
            sliced_mean,
            yerr=sliced_error,
            label=avg_label,
            **average_style,
        )

    # Apply dataset-specific y-axis scale
    y_scale = dataset_config.get("y_scale", "linear")
    ax.set_yscale(y_scale)

    # Set axis labels using dataset-specific LaTeX notation
    labels = get_dataset_labels(dataset_name)
    ax.set_xlabel(labels["x_label"], fontsize=DEFAULT_FONT_SIZE)
    ax.set_ylabel(labels["y_label"], fontsize=DEFAULT_FONT_SIZE)

    # Use the proper title builder system with the existing
    # generate_sample_plot_title function
    try:
        # Use the existing function that properly handles the title
        # builder
        sample_label = f"Samples {sample_indices[0]}-{sample_indices[1]}"
        plot_title = generate_sample_plot_title(
            sample_label=sample_label,
            dataset_name=dataset_name,
            group_metadata=group_metadata,
            title_builder=title_builder,
        )
    except Exception as e:
        # Fallback title if title building fails
        plot_title = f"{dataset_name} - Samples {sample_indices[0]}-{sample_indices[1]}"

    ax.set_title(plot_title, fontsize=DEFAULT_FONT_SIZE + 2)

    # Add legend
    ax.legend(fontsize=DEFAULT_FONT_SIZE - 2, loc="best")

    # Grid for better readability
    ax.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    return fig, ax


def create_sample_plot(
    time_index: np.ndarray,
    sample_data: np.ndarray,
    mean_values: np.ndarray,
    error_values: np.ndarray,
    sample_label: str,
    dataset_name: str,
    group_metadata: Dict,
    layout_manager: PlotLayoutManager,
    style_manager: PlotStyleManager,
    title_builder: PlotTitleBuilder,
) -> Tuple[Figure, Axes]:
    """
    Create a single sample plot with both sample and average data.
    Enhanced to support dataset-specific axis configurations.
    """
    # Get dataset-specific configuration
    dataset_config = get_dataset_plot_config(dataset_name)

    # Apply dataset-specific slicing
    sliced_time, sliced_sample, sliced_mean, sliced_error = apply_dataset_slicing(
        time_index, sample_data, mean_values, error_values, dataset_config
    )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    # Plot sample data with dataset-specific slicing
    sample_style = SAMPLE_PLOT_STYLE.copy()
    sample_label_full = f"{sample_label}{sample_style.pop('label_suffix', '')}"

    ax.plot(sliced_time, sliced_sample, label=sample_label_full, **sample_style)

    # Plot average data with error bars if available
    if sliced_mean is not None and sliced_error is not None:
        average_style = AVERAGE_PLOT_STYLE.copy()
        avg_label_full = f"{dataset_name}{average_style.pop('label_suffix', '')}"

        ax.errorbar(
            sliced_time,
            sliced_mean,
            yerr=sliced_error,
            label=avg_label_full,
            **average_style,
        )

    # Apply dataset-specific y-axis scale
    y_scale = dataset_config.get("y_scale", "linear")
    ax.set_yscale(y_scale)

    # Set axis labels using dataset-specific LaTeX notation
    x_label = dataset_config.get("x_label", r"$t/a$")
    y_label = dataset_config.get("y_label", "Correlator Value")
    ax.set_xlabel(x_label, fontsize=DEFAULT_FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=DEFAULT_FONT_SIZE)

    # Add title
    title = generate_sample_plot_title(
        sample_label=sample_label,
        dataset_name=dataset_name,
        group_metadata=group_metadata,
        title_builder=title_builder,
    )
    ax.set_title(title, fontsize=DEFAULT_FONT_SIZE + 2)

    # Add legend
    ax.legend(fontsize=DEFAULT_FONT_SIZE - 1)

    # Grid for better readability
    ax.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    return fig, ax


def load_corresponding_gvar_data(
    analyzer: HDF5Analyzer, dataset_name: str, group_path: str, logger
) -> Tuple[
    Union[np.ndarray, List[np.ndarray], None], Union[np.ndarray, List[np.ndarray], None]
]:
    """
    Load corresponding mean and error values for the jackknife dataset.

    Args:
        - analyzer: HDF5Analyzer instance
        - dataset_name: Name of the jackknife dataset
        - group_path: Path to the HDF5 group
        - logger: Logger instance

    Returns:
        Tuple of (mean_values, error_values) or (None, None) if not
        found
    """
    # Derive mean and error dataset names
    base_name = dataset_name.replace("_jackknife_samples", "")
    mean_dataset = f"{base_name}_mean_values"
    error_dataset = f"{base_name}_error_values"

    try:
        mean_values = analyzer.dataset_values(
            mean_dataset, return_gvar=False, group_path=group_path
        )
        error_values = analyzer.dataset_values(
            error_dataset, return_gvar=False, group_path=group_path
        )

        return mean_values, error_values

    except Exception as e:
        logger.warning(f"Could not load gvar data {mean_dataset}/{error_dataset}: {e}")
        return None, None


def load_gauge_configuration_labels(
    analyzer: HDF5Analyzer, group_path: str, n_samples: int, logger
) -> List[str]:
    """
    Load gauge configuration labels for sample identification.

    Attempts to extract Configuration_label values from group metadata.
    Falls back to parsing filenames if direct labels unavailable.

    Args:
        - analyzer: HDF5Analyzer instance
        - group_path: Path to the HDF5 group
        - n_samples: Number of samples (for creating defaults)
        - logger: Logger instance

    Returns:
        List of configuration labels
    """
    try:
        # First try: Load from gauge_configuration_labels dataset
        labels = analyzer.dataset_values(
            "gauge_configuration_labels", return_gvar=False, group_path=group_path
        )

        # Convert to list of strings
        if isinstance(labels, np.ndarray):
            labels_list = [
                label.decode("utf-8") if isinstance(label, bytes) else str(label)
                for label in labels.flatten()
            ]
        elif isinstance(labels, list):
            labels_list = [str(label) for label in labels]
        else:
            raise ValueError("Unexpected format for gauge configuration labels")

        # Extract just the configuration part if labels are filenames
        # Look for pattern like "config0013800" and extract just the number
        clean_labels = []
        for label in labels_list:
            # Try to extract configuration number from filename pattern
            import re

            match = re.search(r"config(\d+)", label)
            if match:
                clean_labels.append(match.group(1))  # Just the number
            else:
                # If no match, use the original label
                clean_labels.append(label)

        return clean_labels

    except Exception as e:
        logger.warning(f"Could not load gauge configuration labels: {e}")
        # Create default labels
        return [f"config_{i}" for i in range(n_samples)]


def get_group_metadata(analyzer, group_path):
    """Extract metadata from a group for titles and filenames."""
    try:
        metadata = analyzer.parameters_for_group(group_path)
        return metadata
    except Exception:
        return {}


def generate_sample_plot_title(
    sample_label, dataset_name, group_metadata, title_builder
):
    """Generate a descriptive title for the sample plot using the title
    builder."""
    # Clean dataset name for title
    clean_dataset_name = (
        dataset_name.replace("_jackknife_samples", "").replace("_", " ").title()
    )

    # Base title
    base_title = (
        f"{clean_dataset_name} - {sample_label}"  # Modified to use samples range
    )

    # Add metadata if available
    if group_metadata:
        try:
            metadata_title = title_builder.build(
                metadata_dict=group_metadata,
                tunable_params=list(group_metadata.keys()),
                excluded=set(TITLE_EXCLUDED_PARAMETERS),
                # leading_substring=base_title,
                wrapping_length=80,
            )
            return metadata_title
        except Exception:
            return base_title

    return base_title


def generate_sample_plot_filename(
    config_label: str,
    dataset_name: str,
    group_name: str,
    group_metadata: Dict,
    filename_builder: PlotFilenameBuilder,
) -> str:
    """
    Generate a filename for the sample plot.

    Args:
        - config_label: Gauge configuration label
        - dataset_name: Name of the dataset
        - group_name: Name of the group
        - group_metadata: Metadata for the group
        - filename_builder: PlotFilenameBuilder instance

    Returns:
        Generated filename
    """
    # Clean dataset name for filename
    clean_dataset_name = dataset_name.replace("_jackknife_samples", "")

    # Base filename - include group name for uniqueness
    base_name = f"{clean_dataset_name}_{group_name}_sample_{config_label}"

    # Add metadata if available
    if group_metadata:
        try:
            # Extract multivalued parameters (if any)
            multivalued_params = [
                key
                for key in group_metadata.keys()
                if key not in ["Configuration_label", "MPI_geometry"]
            ]

            filename = filename_builder.build(
                metadata_dict=group_metadata,
                base_name=base_name,
                multivalued_params=multivalued_params,
            )
            return filename
        except Exception:
            return base_name

    return base_name


def generate_multi_sample_plot_filename(
    group_name: str, sample_indices: Tuple[int, int]
) -> str:
    """
    Generate filename for multi-sample plot.

    Args:
        group_name: Name of the group sample_indices: Tuple of
        (start_idx, end_idx) for this plot

    Returns:
        Generated filename (without extension - file manager adds it)
    """
    start_idx, end_idx = sample_indices
    return f"{group_name}_{start_idx}-{end_idx}"  # No .png extension here


def apply_dataset_slicing(
    time_index: np.ndarray,
    sample_data: np.ndarray,
    mean_values: np.ndarray,
    error_values: np.ndarray,
    dataset_config: dict,
) -> Tuple[ndarray, ndarray, Optional[ndarray], Optional[ndarray]]:
    """
    Apply dataset-specific slicing to time index and data arrays.

    Args:
        - time_index: Array of time indices
        - sample_data: Sample data array (can be 1D or 2D)
        - mean_values: Mean values array
        - error_values: Error values array
        - dataset_config: Configuration dict with x_start_index and
          x_end_offset

    Returns:
        Tuple of sliced (time_index, sample_data, mean_values,
        error_values)
    """
    start_idx = dataset_config.get("x_start_index", 0)
    end_offset = dataset_config.get("x_end_offset", 0)

    # Calculate end index
    if end_offset > 0:
        end_idx = len(time_index) - end_offset
    else:
        end_idx = len(time_index)

    # Apply slicing
    sliced_time_index = time_index[start_idx:end_idx]

    # Handle both 1D and 2D sample data
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

    return (
        sliced_time_index,
        sliced_sample_data,
        sliced_mean_values,
        sliced_error_values,
    )


if __name__ == "__main__":
    main()
