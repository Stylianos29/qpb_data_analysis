#!/usr/bin/env python3
"""
Unified Correlator Analysis Visualization Script

This script creates plots for correlator analysis results from both PCAC
mass and effective mass calculations. It processes jackknife samples
alongside their statistical averages from HDF5 analysis results.

This version follows the exact patterns from visualize_PCAC_mass.py but
adapts them to work for both analysis types.

Key features:
    - Supports both PCAC mass and effective mass visualization
    - Multi-sample plots with configurable samples per plot  
    - Jackknife average overlay with error bars
    - Analysis-specific styling and axis configuration
    - Uses project's visualization infrastructure properly
    - Comprehensive logging and error handling

Usage:
    python visualize_correlator_analysis.py \
        --analysis_type pcac_mass \
        -i analysis.h5 \
        -o plots_dir
    python visualize_correlator_analysis.py \
        --analysis_type effective_mass \
        -i analysis.h5 \
        -o plots_dir
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from matplotlib.figure import Figure
import click
import numpy as np
import h5py

# Configure matplotlib to use non-interactive backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import library components
from library.constants.paths import ROOT
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
from src.analysis.correlator_calculations._correlator_visualization_config import (
    get_analysis_config,
    validate_visualization_config,
    DEFAULT_FONT_SIZE,
)


@click.command()
@click.option(
    "--analysis_type",
    required=True,
    type=click.Choice(["pcac_mass", "effective_mass"]),
    help="Type of correlator analysis to visualize.",
)
@click.option(
    "-i",
    "--input_hdf5_file",
    required=True,
    callback=hdf5_file.input,
    help="Path to input HDF5 file containing correlator analysis results.",
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
    analysis_type: str,
    input_hdf5_file: str,
    plots_directory: str,
    clear_existing: bool,
    enable_logging: bool,
    log_directory: Optional[str],
    log_filename: Optional[str],
    verbose: bool,
) -> None:
    """
    Visualize correlator analysis jackknife samples from HDF5 analysis results.

    This script creates plots showing correlator jackknife samples along
    with their statistical averages and error bars.
    """

    # Validate configuration
    validate_visualization_config()

    # Setup logging
    log_dir = None
    if enable_logging:
        log_dir = log_directory or os.path.dirname(input_hdf5_file)

    logger = create_script_logger(
        log_directory=log_dir,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=verbose,
    )

    logger.log_script_start(f"Correlator visualization ({analysis_type})")

    try:
        # Get analysis configuration
        analysis_config = get_analysis_config(analysis_type)

        # Log input parameters
        logger.info(f"Analysis type: {analysis_type}")
        logger.info(f"Input HDF5 file: {input_hdf5_file}")
        logger.info(f"Plots directory: {plots_directory}")
        logger.info(f"Samples per plot: {analysis_config.get('samples_per_plot', 8)}")

        # Setup visualization managers (following exact pattern from visualize_PCAC_mass.py)
        file_manager = PlotFileManager(plots_directory)
        layout_manager = PlotLayoutManager(constants)
        style_manager = PlotStyleManager(constants)
        filename_builder = PlotFilenameBuilder(constants.FILENAME_LABELS_BY_COLUMN_NAME)
        title_builder = PlotTitleBuilder(constants.TITLE_LABELS_BY_COLUMN_NAME)

        # Prepare base output directory
        base_plots_dir = _prepare_base_output_directory(
            plots_directory, analysis_config, clear_existing, file_manager, logger
        )

        # Process correlator data
        total_plots = _process_correlator_data(
            input_hdf5_file,
            base_plots_dir,
            analysis_config,
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
                f"✅ {analysis_type.replace('_', ' ').title()} visualization complete. Created "
                f"{total_plots} plots in: {base_plots_dir}"
            )
        else:
            logger.warning("No plots were created")
            click.echo("⚠️  No plots created. Check input file and logs for details.")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.log_script_end(f"Correlator visualization ({analysis_type}) failed")
        click.echo(f"❌ ERROR: {e}", err=True)
        sys.exit(1)


def _prepare_base_output_directory(
    output_directory: str,
    analysis_config: Dict,
    clear_existing: bool,
    file_manager: PlotFileManager,
    logger,
) -> str:
    """
    Prepare the base output directory for plots.

    Returns:
        Path to base plots directory
    """
    base_plots_dir = os.path.join(
        output_directory, analysis_config["plot_base_directory"]
    )

    if clear_existing and os.path.exists(base_plots_dir):
        logger.info(f"Clearing existing plots directory: {base_plots_dir}")
        import shutil

        shutil.rmtree(base_plots_dir)

    os.makedirs(base_plots_dir, exist_ok=True)
    logger.info(f"Base plots directory: {base_plots_dir}")

    return base_plots_dir


def _process_correlator_data(
    input_hdf5_file: str,
    base_plots_dir: str,
    analysis_config: Dict,
    file_manager: PlotFileManager,
    layout_manager: PlotLayoutManager,
    style_manager: PlotStyleManager,
    filename_builder: PlotFilenameBuilder,
    title_builder: PlotTitleBuilder,
    logger,
    verbose: bool,
) -> int:
    """
    Process correlator data and create visualization plots.

    Returns:
        Total number of plots created
    """
    total_plots = 0

    # Initialize HDF5Analyzer for metadata extraction (following exact pattern)
    analyzer = HDF5Analyzer(input_hdf5_file)

    try:
        with h5py.File(input_hdf5_file, "r") as hdf5_file:
            # Find all groups containing correlator data
            correlator_groups = _find_correlator_groups(
                hdf5_file, analysis_config, logger
            )

            if not correlator_groups:
                logger.warning("No groups with correlator data found")
                return 0

            logger.info(f"Found {len(correlator_groups)} groups with correlator data")

            for group_path in correlator_groups:
                if verbose:
                    click.echo(f"  Processing group: {group_path}")

                try:
                    plots_created = _process_single_correlator_group(
                        hdf5_file,
                        group_path,
                        base_plots_dir,
                        analysis_config,
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


def _find_correlator_groups(
    hdf5_file: h5py.File, analysis_config: Dict, logger
) -> List[str]:
    """
    Find all groups containing correlator datasets.

    Returns:
        List of group paths containing correlator data
    """
    correlator_groups = []
    target_dataset = analysis_config["samples_dataset"]

    def find_groups(name, obj):
        if isinstance(obj, h5py.Group) and target_dataset in obj:
            correlator_groups.append(name)

    hdf5_file.visititems(find_groups)
    logger.debug(f"Found {len(correlator_groups)} groups with correlator data")

    return correlator_groups


def _process_single_correlator_group(
    hdf5_file: h5py.File,
    group_path: str,
    base_plots_dir: str,
    analysis_config: Dict,
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
    Process a single group containing correlator data.

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

    # Load correlator datasets
    try:
        samples_data, mean_data, error_data = _load_correlator_datasets(
            group, analysis_config, group_path, logger
        )
    except Exception as e:
        logger.error(f"Failed to load datasets from group {group_path}: {e}")
        return 0

    # Load configuration labels
    n_samples = samples_data.shape[0]
    config_labels = _load_configuration_labels(group, n_samples, logger)

    # Validate data dimensions
    _validate_correlator_data(samples_data, mean_data, error_data, group_path, logger)

    # Extract group metadata using analyzer
    try:
        # Get all attributes for this group
        group_metadata = {}
        for attr_name in group.attrs:
            group_metadata[attr_name] = group.attrs[attr_name]

        # Add basic group information
        group_name = os.path.basename(group_path)
        group_metadata["group_name"] = group_name

    except Exception as e:
        logger.warning(f"Could not extract group metadata: {e}")
        group_metadata = {"group_name": os.path.basename(group_path)}

    # Create plots for this group
    plots_created = _create_multi_sample_plots(
        samples_data,
        mean_data,
        error_data,
        config_labels,
        group_name,
        group_path,
        base_plots_dir,
        analysis_config,
        group_metadata,
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


def _load_correlator_datasets(
    group: h5py.Group, analysis_config: Dict, group_path: str, logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load correlator datasets (samples, mean, error) from group.

    Returns:
        Tuple of (samples_data, mean_data, error_data)
    """

    # Load jackknife samples dataset
    samples_dataset_name = analysis_config["samples_dataset"]
    try:
        samples_obj = group[samples_dataset_name]
        if not isinstance(samples_obj, h5py.Dataset):
            raise ValueError(f"'{samples_dataset_name}' is not a dataset")
        samples_data = samples_obj[()]
    except KeyError:
        raise ValueError(
            f"Dataset '{samples_dataset_name}' not found in group '{group_path}'"
        )

    # Load mean values dataset
    mean_dataset_name = analysis_config["mean_dataset"]
    try:
        mean_obj = group[mean_dataset_name]
        if not isinstance(mean_obj, h5py.Dataset):
            raise ValueError(f"'{mean_dataset_name}' is not a dataset")
        mean_data = mean_obj[()]
    except KeyError:
        raise ValueError(
            f"Dataset '{mean_dataset_name}' not found in group '{group_path}'"
        )

    # Load error values dataset
    error_dataset_name = analysis_config["error_dataset"]
    try:
        error_obj = group[error_dataset_name]
        if not isinstance(error_obj, h5py.Dataset):
            raise ValueError(f"'{error_dataset_name}' is not a dataset")
        error_data = error_obj[()]
    except KeyError:
        raise ValueError(
            f"Dataset '{error_dataset_name}' not found in group '{group_path}'"
        )

    return samples_data, mean_data, error_data


def _load_configuration_labels(group: h5py.Group, n_samples: int, logger) -> List[str]:
    """
    Load configuration labels with fallback to defaults.

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
                return [f"Sample_{i:03d}" for i in range(n_samples)]

            config_data = config_obj[()]
            config_labels = [
                label.decode() if isinstance(label, bytes) else str(label)
                for label in config_data
            ]
        else:
            # Fallback to default labels
            config_labels = [f"Sample_{i:03d}" for i in range(n_samples)]
            logger.warning("No gauge_configuration_labels found, using default labels")

        # Ensure we have enough labels
        if len(config_labels) < n_samples:
            logger.warning(
                f"Insufficient config labels ({len(config_labels)}) for samples ({n_samples})"
            )
            config_labels.extend(
                [f"Sample_{i:03d}" for i in range(len(config_labels), n_samples)]
            )

        return config_labels

    except Exception as e:
        logger.warning(f"Error loading configuration labels: {e}")
        return [f"Sample_{i:03d}" for i in range(n_samples)]


def _validate_correlator_data(
    samples_data: np.ndarray,
    mean_data: np.ndarray,
    error_data: np.ndarray,
    group_path: str,
    logger,
) -> None:
    """
    Validate correlator data dimensions and consistency.
    """
    n_samples, n_time_points = samples_data.shape

    if mean_data.shape[0] != n_time_points:
        raise ValueError(
            f"Group {group_path}: Mean values length ({mean_data.shape[0]}) "
            f"doesn't match time points ({n_time_points})"
        )

    if error_data.shape[0] != n_time_points:
        raise ValueError(
            f"Group {group_path}: Error values length ({error_data.shape[0]}) "
            f"doesn't match time points ({n_time_points})"
        )

    logger.debug(
        f"Group {group_path}: Validation passed - "
        f"{n_samples} jackknife samples, {n_time_points} time points"
    )


def _create_multi_sample_plots(
    samples_data: np.ndarray,
    mean_data: np.ndarray,
    error_data: np.ndarray,
    config_labels: List[str],
    group_name: str,
    group_path: str,
    base_plots_dir: str,
    analysis_config: Dict,
    group_metadata: Dict,
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
    Create multi-sample plots for correlator data.

    Returns:
        Number of plots created
    """
    n_samples, n_time_points = samples_data.shape
    samples_per_plot = analysis_config.get("samples_per_plot", 8)

    # Create time index with analysis-specific offset
    time_offset = analysis_config.get("time_offset", 0)
    time_index = np.arange(n_time_points) + time_offset

    # Create group subdirectory
    group_plots_dir = os.path.join(base_plots_dir, group_name)
    os.makedirs(group_plots_dir, exist_ok=True)

    # Calculate number of plots needed
    n_plots = (n_samples + samples_per_plot - 1) // samples_per_plot
    plots_created = 0

    if verbose:
        click.echo(f"    Creating {n_plots} plots ({samples_per_plot} samples each)")

    for plot_idx in range(n_plots):
        start_idx = plot_idx * samples_per_plot
        end_idx = min(start_idx + samples_per_plot, n_samples)

        try:
            # Extract samples for this plot
            plot_samples = samples_data[start_idx:end_idx]
            plot_labels = config_labels[start_idx:end_idx]

            # Create the plot
            fig = _create_single_correlator_plot(
                time_index,
                plot_samples,
                plot_labels,
                mean_data,
                error_data,
                group_name,
                (start_idx, end_idx - 1),
                analysis_config,
                group_metadata,
                layout_manager,
                style_manager,
                title_builder,
            )

            # Generate filename using filename builder
            try:
                plot_metadata = group_metadata.copy()
                plot_metadata["sample_start"] = start_idx
                plot_metadata["sample_end"] = end_idx - 1

                base_name = f"correlator_samples_{start_idx:03d}_{end_idx-1:03d}"
                filename = filename_builder.build(
                    metadata_dict=plot_metadata,
                    base_name=base_name,
                    multivalued_params=list(group_metadata.keys()),
                )
            except Exception as e:
                # Fallback filename if builder fails
                logger.warning(f"Filename builder failed: {e}, using fallback")
                filename = f"{group_name}_correlator_samples_{start_idx:03d}_{end_idx-1:03d}.png"

            # Save plot
            output_path = os.path.join(group_plots_dir, filename)
            fig.savefig(
                output_path,
                **analysis_config.get(
                    "plot_quality", {"dpi": 300, "bbox_inches": "tight"}
                ),
            )
            plt.close(fig)  # Critical: close figure to prevent memory leak

            plots_created += 1

            if verbose:
                click.echo(f"      Created: {filename}")

        except Exception as e:
            logger.error(
                f"Error creating plot {plot_idx} for samples {start_idx}-{end_idx-1}: {e}"
            )
            continue

    return plots_created


def _create_single_correlator_plot(
    time_index: np.ndarray,
    samples_data: np.ndarray,
    sample_labels: List[str],
    mean_data: np.ndarray,
    error_data: np.ndarray,
    group_name: str,
    sample_range: Tuple[int, int],
    analysis_config: Dict,
    group_metadata: Dict,
    layout_manager: PlotLayoutManager,
    style_manager: PlotStyleManager,
    title_builder: PlotTitleBuilder,
) -> Figure:
    """
    Create a single correlator plot with multiple samples and average.

    Returns:
        Matplotlib figure
    """

    # Create figure using layout manager
    fig, ax = layout_manager.create_figure()
    #     figsize=analysis_config.get("figure_size", (10, 7)),
    #     subplot_config="single"
    # )

    try:
        # Generate color/marker mapping using style manager
        marker_color_map = style_manager.generate_marker_color_map(
            sample_labels, index_shift=sample_range[0]
        )

        # Plot individual samples
        for i, (sample_data, label) in enumerate(zip(samples_data, sample_labels)):
            marker, color = marker_color_map.get(label, ("o", f"C{i}"))

            ax.plot(
                time_index,
                sample_data,
                label=label,
                color=color,
                marker=marker,
                markersize=6,
                alpha=0.7,
                linestyle="none",
            )

        # Plot jackknife average with error bars
        ax.errorbar(
            time_index,
            mean_data,
            yerr=error_data,
            label="Jackknife Average",
            color="red",
            marker="s",
            markersize=8,
            capsize=8,
            capthick=2,
            elinewidth=2,
            alpha=1.0,
        )

        # Configure plot appearance from analysis config
        plot_config = analysis_config["plot_config"]
        ax.set_xlabel(plot_config["x_label"], fontsize=DEFAULT_FONT_SIZE)
        ax.set_ylabel(plot_config["y_label"], fontsize=DEFAULT_FONT_SIZE)
        ax.set_yscale(plot_config["y_scale"])

        # Create title using title builder
        try:
            title_metadata = group_metadata.copy()
            title_metadata["sample_range"] = f"{sample_range[0]}-{sample_range[1]}"

            title = title_builder.build(
                metadata_dict=title_metadata,
                tunable_params=list(group_metadata.keys()),
                leading_substring=f"{group_name} - Samples {sample_range[0]} to {sample_range[1]}",
            )
        except Exception as e:
            # Fallback title if builder fails
            title = f"{group_name} - Samples {sample_range[0]} to {sample_range[1]}"

        ax.set_title(title, fontsize=DEFAULT_FONT_SIZE + 2)

        # # Configure legend using layout manager
        # layout_manager.configure_legend(ax, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"Error creating correlator plot: {e}")


if __name__ == "__main__":
    main()
