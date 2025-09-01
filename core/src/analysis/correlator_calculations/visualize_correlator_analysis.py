#!/usr/bin/env python3
"""
Unified Correlator Analysis Visualization Script

This script creates plots for correlator analysis results from both PCAC
mass and effective mass calculations. It processes jackknife samples
alongside their statistical averages from HDF5 analysis results.

Key features:
    - Supports both PCAC mass and effective mass visualization
    - Multi-sample plots with configurable samples per plot  
    - Jackknife average overlay with error bars
    - Analysis-specific styling and axis configuration
    - Hierarchical output directory structure
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

import click
import h5py

# Configure matplotlib to use non-interactive backend
import matplotlib

matplotlib.use("Agg")

from library.constants.paths import ROOT
from library.validation.click_validators import (
    hdf5_file,
    directory,
    validate_log_filename,
)
from library.utils.logging_utilities import create_script_logger

# Import configuration and core functions
from src.analysis.correlator_calculations._correlator_visualization_config import (
    get_analysis_config,
    validate_visualization_config,
)
from src.analysis.correlator_calculations._correlator_visualization_core import (
    find_correlator_groups,
    prepare_base_output_directory,
    process_correlator_group,
)


def process_correlator_visualization(
    input_hdf5_file, output_directory, analysis_type, clear_existing, logger, verbose
):
    """Process correlator visualization for specified analysis type."""
    # Get analysis-specific configuration
    analysis_config = get_analysis_config(analysis_type)

    # Prepare base output directory
    with h5py.File(input_hdf5_file, "r") as hdf5_file:
        base_plots_dir = prepare_base_output_directory(
            output_directory, analysis_config, clear_existing, logger
        )

        # Find valid groups
        valid_groups = find_correlator_groups(hdf5_file, analysis_config, logger)

        if not valid_groups:
            logger.warning("No groups with required datasets found")
            return 0

        logger.info(
            f"Processing {len(valid_groups)} groups for {analysis_type} visualization"
        )

        # Process each group
        total_plots = 0
        for group_path in valid_groups:
            plots_created = process_correlator_group(
                group_path,
                hdf5_file,
                base_plots_dir,
                analysis_config,
                clear_existing,
                logger,
                verbose,
            )
            total_plots += plots_created

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
    help="Enable verbose console output showing detailed progress.",
)
def main(
    analysis_type: str,
    input_hdf5_file: str,
    output_directory: str,
    clear_existing: bool,
    enable_logging: bool,
    log_directory: str,
    log_filename: str,
    verbose: bool,
) -> None:
    """Create visualization plots for correlator analysis results."""

    # Validate configuration
    validate_visualization_config()

    # Setup logging
    log_dir = None
    if enable_logging:
        log_dir = log_directory or output_directory

    logger = create_script_logger(
        log_directory=log_dir,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=False,
    )

    try:
        logger.log_script_start(f"Correlator visualization ({analysis_type})")

        if verbose:
            click.echo(f"Starting {analysis_type} visualization...")
            click.echo(f"  Input: {Path(input_hdf5_file).relative_to(ROOT)}")
            click.echo(f"  Output: {Path(output_directory).relative_to(ROOT)}")

        # Process visualization
        total_plots = process_correlator_visualization(
            input_hdf5_file,
            output_directory,
            analysis_type,
            clear_existing,
            logger,
            verbose,
        )

        if total_plots > 0:
            # Get analysis config for output directory name
            analysis_config = get_analysis_config(analysis_type)
            plots_dir = os.path.join(
                output_directory, analysis_config["plot_base_directory"]
            )

            click.echo(
                f"✓ {analysis_type.replace('_', ' ').title()} visualization complete"
            )
            click.echo(f"  Created: {total_plots} plots")
            click.echo(f"  Output: {Path(plots_dir).relative_to(ROOT)}")
        else:
            logger.warning("No plots were created")
            click.echo("⚠ No plots created. Check input file and logs for details.")

        logger.log_script_end(f"Correlator visualization completed ({analysis_type})")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        click.echo(f"ERROR: {e}", err=True)
        logger.log_script_end(f"Correlator visualization failed ({analysis_type})")
        sys.exit(1)


if __name__ == "__main__":
    main()
