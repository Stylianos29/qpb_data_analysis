#!/usr/bin/env python3
"""
Unified Plateau Extraction Visualization Script

This script creates multi-panel visualizations of plateau extraction
results from both PCAC mass and pion effective mass analyses. It reads
extraction results from HDF5 files and generates plots showing
individual jackknife samples with their detected plateau regions.

Usage:
    python visualize_plateau_extraction.py \
        --analysis_type pcac_mass \
        -i plateau_results.h5 \
        -p plots_dir
        
    python visualize_plateau_extraction.py \
        --analysis_type pion_mass \
        -i plateau_results.h5 \
        -p plots_dir
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import click
import numpy as np
import h5py

# Import library components
from library.constants.paths import ROOT
from library.visualization.managers.file_manager import PlotFileManager
from library.visualization.managers.layout_manager import PlotLayoutManager
from library.visualization.builders.title_builder import PlotTitleBuilder
from library import constants
from library.utils.logging_utilities import create_script_logger
from library.validation.click_validators import (
    hdf5_file,
    directory,
    validate_log_filename,
)
from library.data.hdf5_analyzer import HDF5Analyzer

# Import visualization components
from src.analysis.plateau_extraction._plateau_visualization_config import (
    get_analysis_config,
    get_data_processing_config,
    validate_visualization_config,
    PLOT_STYLING,
)
from src.analysis.plateau_extraction._plateau_visualization_core import (
    create_multi_panel_figure,
    split_extractions_into_figures,
)


def _load_extraction_results(
    group: h5py.Group,
    analysis_config: Dict,
) -> List[Dict]:
    """
    Load plateau extraction results from an HDF5 group.

    Args:
        - group: HDF5 group containing extraction data
        - analysis_config: Analysis-specific configuration

    Returns:
        List of extraction result dictionaries
    """
    # Get dataset names from config
    datasets = analysis_config["input_datasets"]

    # Determine the plateau values dataset name based on analysis type
    if "pcac" in analysis_config.get("plot_subdirectory", ""):
        plateau_dataset_name = "PCAC_plateau_values"
        plateau_attr_prefix = "PCAC"
    else:
        plateau_dataset_name = "pion_plateau_values"
        plateau_attr_prefix = "pion"

    # Load jackknife samples (original time series)
    samples_data = group[datasets["samples"]][:]
    n_samples, n_time = samples_data.shape

    # Load plateau values if available
    plateau_values = None
    if plateau_dataset_name in group:
        plateau_values = group[plateau_dataset_name][:]

    # Load plateau bounds from attributes if available
    plateau_start = None
    plateau_end = None
    if f"{plateau_attr_prefix}_plateau_start" in group.attrs:
        plateau_start = group.attrs[f"{plateau_attr_prefix}_plateau_start"]
        plateau_end = group.attrs[f"{plateau_attr_prefix}_plateau_end"]
        # Convert back to array indices
        plateau_start = int(plateau_start - analysis_config["time_offset"])
        plateau_end = int(plateau_end - analysis_config["time_offset"])

    # Load configuration labels if available
    config_labels = []
    if "gauge_configuration_labels" in group:
        labels_data = group["gauge_configuration_labels"][:]
        config_labels = [
            label.decode("utf-8") if isinstance(label, bytes) else label
            for label in labels_data
        ]
    else:
        # Generate default labels
        config_labels = [f"Sample_{i+1:03d}" for i in range(n_samples)]

    # Create extraction results
    extraction_results = []
    for i in range(n_samples):
        result = {
            "sample_index": i,
            "config_label": (
                config_labels[i] if i < len(config_labels) else f"Sample_{i+1:03d}"
            ),
            "time_series": samples_data[i, :],
        }

        # Add plateau information if available
        if plateau_values is not None and not np.isnan(plateau_values[i]):
            result["plateau_value"] = plateau_values[i]
            # Estimate error from neighboring points if we have bounds
            if plateau_start is not None and plateau_end is not None:
                plateau_region = samples_data[i, plateau_start:plateau_end]
                result["plateau_error"] = np.std(plateau_region) / np.sqrt(
                    len(plateau_region)
                )
                result["plateau_bounds"] = (plateau_start, plateau_end)
            else:
                result["plateau_error"] = 0.0
                result["plateau_bounds"] = None
        else:
            # No plateau found for this sample
            result["plateau_value"] = None
            result["plateau_error"] = 0.0
            result["plateau_bounds"] = None

    return extraction_results


def _extract_group_metadata(group: h5py.Group) -> Dict:
    """Extract metadata from HDF5 group for plot titles."""
    metadata = {}

    # Extract attributes
    for key, value in group.attrs.items():
        metadata[key] = value

    # Extract key metadata datasets if present
    metadata_keys = ["Number_of_gauge_configurations"]
    for key in metadata_keys:
        if key in group:
            data = group[key][()]
            # Handle scalar vs array
            metadata[key] = data.item() if hasattr(data, "item") else data

    return metadata


def _process_single_group(
    hdf5_file: h5py.File,
    group_path: str,
    analysis_config: Dict,
    file_manager: PlotFileManager,
    title_builder: PlotTitleBuilder,
    logger,
    verbose: bool,
) -> int:
    """
    Process and visualize extraction results for a single parameter
    group.

    Returns:
        Number of plots created
    """
    group = hdf5_file[group_path]
    group_name = os.path.basename(group_path)

    # Load extraction results
    extraction_results = _load_extraction_results(group, analysis_config)

    if not extraction_results:
        logger.warning(f"No extraction results found in group: {group_path}")
        return 0

    # Extract metadata for titles
    group_metadata = _extract_group_metadata(group)
    group_metadata["group_name"] = group_name

    # Get data processing config
    data_config = get_data_processing_config()
    max_panels = min(
        data_config["max_samples_per_figure"], PLOT_STYLING["figure"]["max_panels"]
    )

    # Split into batches if needed
    batches = split_extractions_into_figures(extraction_results, max_panels)

    plots_created = 0
    for batch_idx, batch in enumerate(batches):
        # Create multi-panel figure
        fig = create_multi_panel_figure(
            batch,
            group_metadata,
            analysis_config,
            title_builder,
        )

        # Generate filename
        if len(batches) > 1:
            filename_base = f"{group_name}_batch_{batch_idx+1:02d}"
        else:
            filename_base = group_name

        # Save figure
        plot_path = file_manager.plot_path(
            analysis_config["plot_subdirectory"],
            filename_base,
        )

        fig.savefig(
            plot_path,
            dpi=PLOT_STYLING["output"]["dpi"],
            bbox_inches=PLOT_STYLING["output"]["bbox_inches"],
            format=PLOT_STYLING["output"]["format"],
        )

        import matplotlib.pyplot as plt

        plt.close(fig)

        plots_created += 1

        if verbose:
            click.echo(f"  Created: {os.path.basename(plot_path)}")

    return plots_created


def _find_groups_with_data(
    hdf5_file: h5py.File,
    required_datasets: List[str],
) -> List[str]:
    """Find all groups containing required datasets."""
    valid_groups = []

    def check_group(name, obj):
        if isinstance(obj, h5py.Group):
            if all(dataset in obj for dataset in required_datasets):
                valid_groups.append(name)

    hdf5_file.visititems(check_group)
    return valid_groups


def _process_visualization(
    input_hdf5_file: str,
    analysis_config: Dict,
    file_manager: PlotFileManager,
    title_builder: PlotTitleBuilder,
    logger,
    verbose: bool,
) -> int:
    """
    Process all groups in the HDF5 file and create visualizations.

    Returns:
        Total number of plots created
    """
    total_plots = 0

    with h5py.File(input_hdf5_file, "r") as hdf5_file:
        # Find groups with required data
        required = [analysis_config["input_datasets"]["samples"]]
        valid_groups = _find_groups_with_data(hdf5_file, required)

        if not valid_groups:
            logger.warning(f"No groups found with required dataset: {required[0]}")
            return 0

        logger.info(f"Found {len(valid_groups)} groups with extraction data")

        for group_path in valid_groups:
            if verbose:
                click.echo(f"Processing group: {group_path}")

            plots_created = _process_single_group(
                hdf5_file,
                group_path,
                analysis_config,
                file_manager,
                title_builder,
                logger,
                verbose,
            )

            total_plots += plots_created
            logger.info(f"Created {plots_created} plots for group: {group_path}")

    return total_plots


@click.command()
@click.option(
    "--analysis_type",
    required=True,
    type=click.Choice(["pcac_mass", "pion_mass"]),
    help="Type of plateau extraction to visualize.",
)
@click.option(
    "-i",
    "--input_hdf5_file",
    required=True,
    callback=hdf5_file.input,
    help="Path to HDF5 file containing extraction results.",
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
    help="Clear existing plots before generating new ones.",
)
@click.option(
    "-log_on",
    "--enable_logging",
    is_flag=True,
    default=False,
    help="Enable logging to file.",
)
@click.option(
    "-log_dir",
    "--log_directory",
    default=None,
    callback=directory.can_create,
    help="Directory for log files.",
)
@click.option(
    "-log_name",
    "--log_filename",
    default=None,
    callback=validate_log_filename,
    help="Custom log filename.",
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
    Visualize plateau extraction results from HDF5 files.

    This script creates multi-panel plots showing individual jackknife
    samples with their detected plateau regions for both PCAC mass and
    pion effective mass analyses.
    """
    # Validate configuration
    validate_visualization_config()

    # Setup logging
    if enable_logging:
        log_dir = log_directory or os.path.dirname(input_hdf5_file)
    else:
        log_dir = None

    logger = create_script_logger(
        log_directory=log_dir,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=verbose,
    )

    logger.log_script_start(f"Plateau extraction visualization ({analysis_type})")

    try:
        # Get analysis configuration
        analysis_config = get_analysis_config(analysis_type)

        # Log parameters
        logger.info(f"Analysis type: {analysis_type}")
        logger.info(f"Input file: {input_hdf5_file}")
        logger.info(f"Plots directory: {plots_directory}")

        # Setup visualization managers
        file_manager = PlotFileManager(plots_directory)
        layout_manager = PlotLayoutManager(constants)
        title_builder = PlotTitleBuilder(constants.TITLE_LABELS_BY_COLUMN_NAME)

        # Prepare plots subdirectory
        plots_subdir = file_manager.prepare_subdirectory(
            analysis_config["plot_subdirectory"],
            clear_existing,
        )

        # Process and create visualizations
        total_plots = _process_visualization(
            input_hdf5_file,
            analysis_config,
            file_manager,
            title_builder,
            logger,
            verbose,
        )

        # Report results
        if total_plots > 0:
            logger.log_script_end(f"Successfully created {total_plots} plots")
            click.echo(
                f"✅ Visualization complete. Created {total_plots} plots in: "
                f"{Path(plots_subdir).relative_to(ROOT)}"
            )
        else:
            logger.warning("No plots were created")
            click.echo("⚠️ No plots created. Check input file for valid data.")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.log_script_end(f"Visualization failed")
        click.echo(f"❌ ERROR: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
