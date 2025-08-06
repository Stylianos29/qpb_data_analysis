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
from library import (
    constants,
    filesystem_utilities,
    validate_input_directory,
    validate_input_script_log_filename,
)

# Import configuration
from src.processing._jackknife_visualization_config import (
    JACKKNIFE_DATASETS_TO_PLOT,
    JACKKNIFE_PLOTS_BASE_DIRECTORY,
    SAMPLE_PLOT_STYLE,
    AVERAGE_PLOT_STYLE,
    DEFAULT_FIGURE_SIZE,
    DEFAULT_FONT_SIZE,
)


@click.command()
@click.option(
    "-i",
    "--input_hdf5_file",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Path to input HDF5 file containing jackknife analysis results.",
)
@click.option(
    "-o",
    "--output_directory",
    required=True,
    callback=validate_input_directory,
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
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    help="Directory for log files. Default: output directory",
)
@click.option(
    "-log_name",
    "--log_filename",
    default=None,
    callback=validate_input_script_log_filename,
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
    Visualize jackknife samples from HDF5 jackknife analysis results.

    This script creates individual plots for each jackknife sample
    showing both the sample time series and the corresponding
    statistical average with error bars.
    """
    # === SETUP AND VALIDATION ===

    # Setup logging
    logger = filesystem_utilities.LoggingWrapper(
        log_directory or output_directory, log_filename, enable_logging
    )
    logger.initiate_script_logging()

    if verbose:
        click.echo(f"Input file: {input_hdf5_file}")
        click.echo(f"Output directory: {output_directory}")
        click.echo(f"Datasets to process: {JACKKNIFE_DATASETS_TO_PLOT}")

    try:
        # === LOAD HDF5 DATA ===
        analyzer = HDF5Analyzer(input_hdf5_file)
        logger.info(f"Loaded HDF5 file: {input_hdf5_file}")

        if verbose:
            click.echo(f"Found {len(analyzer.active_groups)} groups in HDF5 file")

        # === SETUP VISUALIZATION MANAGERS ===
        file_manager = PlotFileManager(output_directory)
        layout_manager = PlotLayoutManager(constants)
        style_manager = PlotStyleManager(constants)
        filename_builder = PlotFilenameBuilder(constants.FILENAME_LABELS_BY_COLUMN_NAME)
        title_builder = PlotTitleBuilder(constants.TITLE_LABELS_BY_COLUMN_NAME)

        # === PREPARE BASE DIRECTORY ===
        base_plots_dir = file_manager.prepare_subdirectory(
            JACKKNIFE_PLOTS_BASE_DIRECTORY, clear_existing=clear_existing
        )

        if verbose:
            click.echo(f"Base plots directory: {base_plots_dir}")

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

        if verbose:
            click.echo(f"\n✓ Visualization complete!")
            click.echo(f"✓ Created {total_plots_created} plots")
            click.echo(f"✓ Results saved to: {base_plots_dir}")

        logger.info(
            f"Successfully created {total_plots_created} jackknife sample plots"
        )

    except Exception as e:
        logger.error(f"Critical error during processing: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    finally:
        # Clean up
        if "analyzer" in locals():
            analyzer.close()
        logger.terminate_script_logging()


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
    Process jackknife data for a specific group and dataset.

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

        if mean_values is None or error_values is None:
            logger.warning(
                f"Could not load corresponding mean/error data for {dataset_name} in {group_path}"
            )
            return 0

        # Load gauge configuration labels
        config_labels = load_gauge_configuration_labels(
            analyzer, group_path, jackknife_data.shape[0], logger
        )

        # Create individual plots for each sample
        plots_created = create_individual_sample_plots(
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


def load_corresponding_gvar_data(
    analyzer: HDF5Analyzer, dataset_name: str, group_path: str, logger
) -> Tuple[Union[ndarray, List[ndarray], None], Union[ndarray, List[ndarray], None]]:
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

    Args:
        - analyzer: HDF5Analyzer instance
        - group_path: Path to the HDF5 group
        - n_samples: Number of samples (for creating defaults)
        - logger: Logger instance

    Returns:
        List of configuration labels
    """
    try:
        labels = analyzer.dataset_values(
            "gauge_configuration_labels", return_gvar=False, group_path=group_path
        )

        # Convert to list of strings
        if isinstance(labels, np.ndarray):
            # Flattens the labels array and returns a list of strings,
            # decoding each label from bytes to UTF-8 if necessary.
            return [
                label.decode("utf-8") if isinstance(label, bytes) else str(label)
                for label in labels.flatten()
            ]
        elif isinstance(labels, list):
            return [str(label) for label in labels]
        else:
            raise ValueError("Unexpected format for gauge configuration labels")

    except Exception as e:
        logger.warning(f"Could not load gauge configuration labels: {e}")
        # Create default labels
        return [f"config_{i}" for i in range(n_samples)]


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
    Create individual plots for each jackknife sample.

    Args:
        - jackknife_data: 2D array of jackknife samples
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

            # Create the plot
            fig, ax = create_sample_plot(
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

    Args:
        - time_index: Array of time indices
        - sample_data: Individual sample time series
        - mean_values: Array of mean values
        - error_values: Array of error values
        - sample_label: Label for the sample
        - dataset_name: Name of the dataset
        - group_metadata: Metadata for the group
        - layout_manager: PlotLayoutManager instance
        - style_manager: PlotStyleManager instance
        - title_builder: PlotTitleBuilder instance

    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes
    fig, ax = layout_manager.create_figure(DEFAULT_FIGURE_SIZE)

    # Configure axes
    layout_manager.configure_existing_axes(
        ax=ax,
        x_variable="time_index",
        y_variable=dataset_name,
        font_size=DEFAULT_FONT_SIZE,
    )

    # Generate colors for sample and average
    colors = ["#1f77b4", "#ff7f0e"]  # Blue for sample, orange for average

    # Plot sample data
    sample_style = SAMPLE_PLOT_STYLE.copy()
    sample_style["color"] = colors[0]
    sample_style["label"] = f"{sample_label}{sample_style.pop('label_suffix')}"

    ax.plot(time_index, sample_data, **sample_style)

    # Plot average with error bars
    average_style = AVERAGE_PLOT_STYLE.copy()
    average_style["color"] = colors[1]
    average_style["label"] = f"Average{average_style.pop('label_suffix')}"

    ax.errorbar(time_index, mean_values, yerr=error_values, **average_style)

    # Add legend
    style_manager.configure_legend(
        ax=ax,
        include_legend=True,
        legend_location="upper right",
        font_size=DEFAULT_FONT_SIZE - 1,
    )

    # Add title
    title = generate_sample_plot_title(
        sample_label=sample_label,
        dataset_name=dataset_name,
        group_metadata=group_metadata,
        title_builder=title_builder,
    )
    ax.set_title(title, fontsize=DEFAULT_FONT_SIZE + 2)

    return fig, ax


def get_group_metadata(analyzer: HDF5Analyzer, group_path: str) -> Dict:
    """
    Extract metadata from a group for titles and filenames.

    Args:
        - analyzer: HDF5Analyzer instance
        - group_path: Path to the HDF5 group

    Returns:
        Dictionary of metadata
    """
    try:
        # Get parameters for this group
        metadata = analyzer._parameters_for_group(group_path)
        return metadata
    except Exception:
        return {}


def generate_sample_plot_title(
    sample_label: str,
    dataset_name: str,
    group_metadata: Dict,
    title_builder: PlotTitleBuilder,
) -> str:
    """
    Generate a descriptive title for the sample plot.

    Args:
        - sample_label: Label for the sample
        - dataset_name: Name of the dataset
        - group_metadata: Metadata for the group
        - title_builder: PlotTitleBuilder instance

    Returns:
        Generated title string
    """
    # Clean dataset name for title
    clean_dataset_name = (
        dataset_name.replace("_jackknife_samples", "").replace("_", " ").title()
    )

    # Base title
    base_title = f"{clean_dataset_name} - Sample: {sample_label}"

    # Add metadata if available
    if group_metadata:
        try:
            metadata_title = title_builder.build(
                metadata_dict=group_metadata,
                tunable_params=list(group_metadata.keys()),
                leading_substring=base_title,
                wrapping_length=100,
            )
            return metadata_title
        except Exception:
            return base_title

    return base_title


def generate_sample_plot_filename(
    config_label: str,
    dataset_name: str,
    group_metadata: Dict,
    filename_builder: PlotFilenameBuilder,
) -> str:
    """
    Generate a filename for the sample plot.

    Args:
        - config_label: Gauge configuration label
        - dataset_name: Name of the dataset
        - group_metadata: Metadata for the group
        - filename_builder: PlotFilenameBuilder instance

    Returns:
        Generated filename
    """
    # Clean dataset name for filename
    clean_dataset_name = dataset_name.replace("_jackknife_samples", "")

    # Base filename
    base_name = f"{clean_dataset_name}_sample_{config_label}"

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


if __name__ == "__main__":
    main()
