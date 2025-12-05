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

import click
import numpy as np
import h5py


# Import library components
from library.constants.paths import ROOT
from library.data.hdf5_analyzer import HDF5Analyzer
from library.visualization.managers.file_manager import PlotFileManager
from library.visualization.managers.layout_manager import PlotLayoutManager
from library.visualization.managers.style_manager import PlotStyleManager
from library.visualization.builders.title_builder import PlotTitleBuilder
from library import constants
from library.utils.logging_utilities import create_script_logger
from library.validation.click_validators import (
    hdf5_file,
    directory,
    validate_log_filename,
)

# Import from auxiliary files
from src.analysis.correlator_calculations._correlator_analysis_core import (
    find_analysis_groups,
)
from src.analysis.correlator_calculations._correlator_visualization_core import (
    _create_multi_sample_plots,
)
from src.analysis.correlator_calculations._correlator_visualization_config import (
    get_analysis_config,
    validate_visualization_config,
)


def _validate_correlator_data(
    samples_data: np.ndarray,
    mean_data: np.ndarray,
    error_data: np.ndarray,
) -> None:
    """Validate correlator data dimensions and consistency."""
    _, n_time_points = samples_data.shape

    if mean_data.shape[0] != n_time_points:
        raise ValueError(
            f"Mean values length ({mean_data.shape[0]}) "
            f"doesn't match time points ({n_time_points})"
        )

    if error_data.shape[0] != n_time_points:
        raise ValueError(
            f"Error values length ({error_data.shape[0]}) "
            f"doesn't match time points ({n_time_points})"
        )


def _load_configuration_labels(group: h5py.Group, n_samples: int) -> List[str]:
    """
    Load configuration labels and validate count matches samples.

    Extracts clean configuration numbers from filename strings.
    """
    import re

    config_obj = group["gauge_configuration_labels"]

    if not isinstance(config_obj, h5py.Dataset):
        raise ValueError("'gauge_configuration_labels' is not a dataset")

    labels = [
        label.decode() if isinstance(label, bytes) else str(label)
        for label in config_obj[()]
    ]

    if len(labels) != n_samples:
        raise ValueError(
            f"Label count mismatch: {len(labels)} labels vs {n_samples} samples"
        )

    # Extract just the configuration numbers from filenames
    clean_labels = []
    for label in labels:
        # Look for pattern like "config0013800" and extract just the number
        match = re.search(r"config(\d+)", label)
        if match:
            clean_labels.append(match.group(1))  # Just the number
        else:
            # If no match, use the original label
            clean_labels.append(label)

    return clean_labels


def _load_single_dataset(
    group: h5py.Group, dataset_name: str, group_path: str
) -> np.ndarray:
    """Load and validate a single dataset from group."""
    try:
        dataset_obj = group[dataset_name]
        if not isinstance(dataset_obj, h5py.Dataset):
            raise ValueError(f"'{dataset_name}' is not a dataset")
        return dataset_obj[()]
    except KeyError:
        raise ValueError(f"Dataset '{dataset_name}' not found in group '{group_path}'")


def _load_correlator_datasets(
    group: h5py.Group, analysis_config: Dict, group_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load correlator datasets (samples, mean, error) from group."""
    samples_data = _load_single_dataset(
        group, analysis_config["samples_dataset"], group_path
    )
    mean_data = _load_single_dataset(group, analysis_config["mean_dataset"], group_path)
    error_data = _load_single_dataset(
        group, analysis_config["error_dataset"], group_path
    )

    return samples_data, mean_data, error_data


def _process_single_correlator_group(
    hdf5_file: h5py.File,
    group_path: str,
    parent_metadata: Dict,
    base_plots_dir: str,
    analysis_config: Dict,
    file_manager: PlotFileManager,
    layout_manager: PlotLayoutManager,
    style_manager: PlotStyleManager,
    title_builder: PlotTitleBuilder,
    logger,
    verbose: bool,
) -> int:
    """
    Process a single group containing correlator data.

    Returns:
        Number of plots created for this group
    """
    try:
        group = hdf5_file[group_path]
        # Verify that group_path points to an HDF5 group
        if not isinstance(group, h5py.Group):
            logger.error(
                f"Path '{group_path}' does not point to an HDF5 group "
                f"(type: {type(group)})"
            )
            return 0

        # Load correlator datasets
        samples_data, mean_data, error_data = _load_correlator_datasets(
            group, analysis_config, group_path
        )

        n_samples = samples_data.shape[0]
        config_labels = _load_configuration_labels(group, n_samples)

        _validate_correlator_data(samples_data, mean_data, error_data)

        group_metadata = dict(group.attrs) | parent_metadata
        group_name = os.path.basename(group_path)

        # Create plots for this group
        plots_created = _create_multi_sample_plots(
            samples_data,
            mean_data,
            error_data,
            config_labels,
            group_name,
            base_plots_dir,
            analysis_config,
            group_metadata,
            file_manager,
            layout_manager,
            style_manager,
            title_builder,
            verbose,
        )

        return plots_created

    except Exception as e:
        logger.error(f"Failed to process group {group_path}: {e}")
        return 0  # Single point of failure handling


def _process_correlator_data(
    input_hdf5_file: str,
    base_plots_dir: str,
    analysis_config: Dict,
    file_manager: PlotFileManager,
    layout_manager: PlotLayoutManager,
    style_manager: PlotStyleManager,
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

    # Initialize HDF5Analyzer for metadata extraction
    analyzer = HDF5Analyzer(input_hdf5_file)

    try:
        with h5py.File(input_hdf5_file, "r") as hdf5_file:
            # Find all groups containing correlator data
            required_datasets = [
                analysis_config["samples_dataset"],
                analysis_config["mean_dataset"],
                analysis_config["error_dataset"],
            ]
            correlator_groups = find_analysis_groups(input_hdf5_file, required_datasets)

            if not correlator_groups:
                logger.warning("No groups with correlator data found")
                return 0

            logger.info(f"Found {len(correlator_groups)} groups with correlator data")

            # Cache parent metadata ONCE for all groups
            first_group_path = correlator_groups[0]
            parent_path = os.path.dirname(first_group_path)
            parent_group = hdf5_file[parent_path]
            parent_metadata = dict(parent_group.attrs)  # Load once, use for all groups
            logger.debug(f"Cached parent metadata from: {parent_path}")

            for group_path in correlator_groups:
                if verbose:
                    click.echo(f"  Processing group: {group_path}")

                try:
                    plots_created = _process_single_correlator_group(
                        hdf5_file,
                        group_path,
                        parent_metadata,
                        base_plots_dir,
                        analysis_config,
                        file_manager,
                        layout_manager,
                        style_manager,
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
    Visualize correlator analysis jackknife samples from HDF5 analysis
    results.

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

        # Setup visualization managers (following exact pattern from
        # visualize_PCAC_mass.py)
        file_manager = PlotFileManager(plots_directory)
        layout_manager = PlotLayoutManager(constants)
        style_manager = PlotStyleManager(constants)
        title_builder = PlotTitleBuilder(constants.TITLE_LABELS_BY_COLUMN_NAME)

        # Prepare plots base subdirectory
        base_plots_dir = file_manager.prepare_subdirectory(
            analysis_config["plot_base_directory"], clear_existing
        )

        # Process correlator data
        total_plots = _process_correlator_data(
            input_hdf5_file,
            base_plots_dir,
            analysis_config,
            file_manager,
            layout_manager,
            style_manager,
            title_builder,
            logger,
            verbose,
        )

        # Report results
        if total_plots > 0:
            logger.log_script_end(f"Successfully created {total_plots} plots")
            click.echo(
                f"✅ {analysis_type.replace('_', ' ').title()} visualization complete. Created "
                f"{total_plots} plots in: {Path(base_plots_dir).relative_to(ROOT)}"
            )
        else:
            logger.warning("No plots were created")
            click.echo("⚠️  No plots created. Check input file and logs for details.")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.log_script_end(f"Correlator visualization ({analysis_type}) failed")
        click.echo(f"❌ ERROR: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
