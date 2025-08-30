#!/usr/bin/env python3
"""
PCAC mass calculation script.

Usage: python calculate_PCAC_mass.py -i input.h5 -o output.h5
"""

import os
import sys
from typing import Optional

import click
import numpy as np
import h5py

from library.validation.click_validators import (
    hdf5_file,
    directory,
    validate_log_filename,
)
from library.utils.logging_utilities import create_script_logger

from src.analysis.correlator_calculations._pcac_mass_config import (
    PCAC_MASS_FACTOR,
    TRUNCATE_START,
    TRUNCATE_END,
    REQUIRED_DATASETS,
    OUTPUT_DATASETS,
    validate_pcac_config,
)
from src.analysis.correlator_calculations._correlator_analysis_shared_config import (
    MIN_JACKKNIFE_SAMPLES,
    METADATA_DATASETS,
)
from src.analysis.correlator_calculations._correlator_analysis_core import (
    calculate_jackknife_statistics,
    find_analysis_groups,
    safe_divide,
    copy_metadata,
    copy_parent_attributes,
)


def calculate_pcac_mass(g4g5g5_derivative, g5g5_samples):
    """Calculate PCAC mass: factor * derivative / g5g5_truncated."""
    # Truncate g5g5 to match derivative length
    if TRUNCATE_END > 0:
        g5g5_truncated = g5g5_samples[:, TRUNCATE_START:-TRUNCATE_END]
    else:
        g5g5_truncated = g5g5_samples[:, TRUNCATE_START:]

    # Calculate PCAC mass
    return safe_divide(PCAC_MASS_FACTOR * g4g5g5_derivative, g5g5_truncated)


def process_pcac_file(input_path, output_path, logger):
    """Process PCAC analysis for all groups."""
    # Find valid groups
    analysis_groups = find_analysis_groups(input_path, REQUIRED_DATASETS)
    if not analysis_groups:
        raise ValueError(f"No groups with required datasets: {REQUIRED_DATASETS}")

    logger.info(f"Processing {len(analysis_groups)} groups")

    successful = 0
    processed_parents = set()  # Track which parent groups we've processed

    with h5py.File(input_path, "r") as input_file, h5py.File(
        output_path, "w"
    ) as output_file:
        # No file-level attributes - clean structure like input

        for group_path in analysis_groups:
            try:
                input_item = input_file[group_path]
                if not isinstance(input_item, h5py.Group):
                    continue

                # Copy parent group attributes (second-to-deepest level)
                copy_parent_attributes(
                    input_file, output_file, group_path, processed_parents
                )

                # Read data with type-safe access
                g4g5g5_item = input_item[REQUIRED_DATASETS[0]]
                g5g5_item = input_item[REQUIRED_DATASETS[1]]

                if not isinstance(g4g5g5_item, h5py.Dataset):
                    logger.warning(
                        f"Skipping {group_path}: {REQUIRED_DATASETS[0]} is not a dataset"
                    )
                    continue

                if not isinstance(g5g5_item, h5py.Dataset):
                    logger.warning(
                        f"Skipping {group_path}: {REQUIRED_DATASETS[1]} is not a dataset"
                    )
                    continue

                g4g5g5_derivative = g4g5g5_item[:]
                g5g5_samples = g5g5_item[:]

                # Basic validation
                if g4g5g5_derivative.shape[0] < MIN_JACKKNIFE_SAMPLES:
                    logger.warning(f"Skipping {group_path}: insufficient samples")
                    continue

                # Calculate PCAC mass
                pcac_mass = calculate_pcac_mass(g4g5g5_derivative, g5g5_samples)
                mean_values, error_values = calculate_jackknife_statistics(pcac_mass)

                # Create output group
                output_group = output_file.create_group(group_path)

                # Save results
                output_group.create_dataset(
                    OUTPUT_DATASETS["samples"],
                    data=pcac_mass,
                )
                output_group.create_dataset(OUTPUT_DATASETS["mean"], data=mean_values)
                output_group.create_dataset(OUTPUT_DATASETS["error"], data=error_values)

                # Copy metadata (datasets + deepest group attributes)
                copy_metadata(input_item, output_group, METADATA_DATASETS)

                successful += 1

            except Exception as e:
                logger.error(f"Failed to process {group_path}: {e}")
                continue

    return successful, len(analysis_groups) - successful


@click.command()
@click.option(
    "-i",
    "--input_hdf5_file",
    required=True,
    callback=hdf5_file.input,
    help="Path to input HDF5 file containing jackknife analysis results.",
)
@click.option(
    "-o",
    "--output_hdf5_file",
    required=True,
    callback=hdf5_file.output,
    help="Path for output HDF5 file with PCAC mass results.",
)
@click.option(
    "-out_dir",
    "--output_directory",
    default=None,
    callback=directory.must_exist,
    help="Directory for output files. If not specified, uses input file directory.",
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
def main(
    input_hdf5_file: str,
    output_hdf5_file: str,
    output_directory: Optional[str],
    enable_logging: bool,
    log_directory: Optional[str],
    log_filename: Optional[str],
) -> None:
    """Calculate PCAC mass from jackknife-analyzed correlator data."""

    # Validate configuration
    validate_pcac_config()

    # Setup paths
    if output_directory:
        output_path = os.path.join(output_directory, output_hdf5_file)
    else:
        output_path = os.path.join(os.path.dirname(input_hdf5_file), output_hdf5_file)

    # Setup logging
    if enable_logging:
        log_dir = log_directory or output_directory or os.path.dirname(output_path)
    else:
        log_dir = None

    logger = create_script_logger(
        log_directory=log_dir,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=False,
    )

    try:
        logger.log_script_start("PCAC mass calculation")

        successful, failed = process_pcac_file(input_hdf5_file, output_path, logger)

        click.echo(f"✓ PCAC mass calculation complete")
        click.echo(f"  Processed: {successful}/{successful + failed} groups")
        click.echo(f"  Output: {output_path}")

        if failed > 0:
            click.echo(f"  ⚠ {failed} groups failed", err=True)

        logger.log_script_end("PCAC mass calculation completed")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
