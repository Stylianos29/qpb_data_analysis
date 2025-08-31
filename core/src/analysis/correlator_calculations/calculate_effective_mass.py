#!/usr/bin/env python3
"""
Effective mass calculation script.

Usage: python calculate_effective_mass.py -i input.h5 -o output.h5
"""

import os
import sys
from typing import Optional
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import h5py

from library.constants.paths import ROOT
from library.validation.click_validators import (
    hdf5_file,
    directory,
    validate_log_filename,
)
from library.utils.logging_utilities import create_script_logger

from src.analysis.correlator_calculations._effective_mass_config import (
    APPLY_SYMMETRIZATION,
    TRUNCATE_HALF,
    LOWERING_FACTOR,
    REQUIRED_DATASETS,
    OUTPUT_DATASETS,
    ANALYSIS_DOCUMENTATION,
    validate_effective_config,
)
from src.analysis.correlator_calculations._correlator_analysis_shared_config import (
    METADATA_DATASETS,
)
from src.analysis.correlator_calculations._correlator_analysis_core import (
    calculate_jackknife_statistics,
    find_analysis_groups,
    safe_log,
    safe_sqrt,
    symmetrize_correlator,
    copy_metadata,
    copy_parent_attributes,
)


def calculate_effective_mass(g5g5_samples):
    """Calculate effective mass using two-state periodic formula."""
    # Apply symmetrization if requested
    if APPLY_SYMMETRIZATION:
        g5g5_processed = symmetrize_correlator(g5g5_samples)
    else:
        g5g5_processed = g5g5_samples

    # Truncate to half if requested (for periodic BC)
    if TRUNCATE_HALF:
        half_length = g5g5_processed.shape[-1] // 2
        g5g5_processed = g5g5_processed[:, :half_length]

    # Two-state periodic effective mass calculation
    middle = LOWERING_FACTOR * np.min(g5g5_processed, axis=-1, keepdims=True)

    # Calculate for t=1 to T-2
    c_prev = g5g5_processed[:, :-2]  # C(t-1)
    c_next = g5g5_processed[:, 2:]  # C(t+1)

    # Two-state formula
    sqrt_prev = safe_sqrt(c_prev**2 - middle**2)
    sqrt_next = safe_sqrt(c_next**2 - middle**2)

    numerator = c_prev + sqrt_prev
    denominator = c_next + sqrt_next

    return 0.5 * safe_log(numerator / denominator)


def process_effective_file(input_path, output_path, logger):
    """Process effective mass analysis for all groups."""
    # Find valid groups
    analysis_groups = find_analysis_groups(input_path, REQUIRED_DATASETS)
    if not analysis_groups:
        raise ValueError(f"No groups with required datasets: {REQUIRED_DATASETS}")

    # Validate file consistency once
    g5g5_length, expected_effective_length = validate_effective_mass_file_consistency(
        input_path, REQUIRED_DATASETS, logger
    )

    logger.info(
        f"Processing {len(analysis_groups)} groups with g5g5 length {g5g5_length},"
        f" expected effective mass length {expected_effective_length}"
    )

    # Choose output dataset names from config
    output_names = OUTPUT_DATASETS

    successful = 0
    processed_parents = set()  # Track which parent groups we've processed

    with h5py.File(input_path, "r") as input_file, h5py.File(
        output_path, "w"
    ) as output_file:

        for group_path in analysis_groups:
            input_item = input_file[group_path]
            if not isinstance(input_item, h5py.Group):
                continue

            # Copy parent group attributes (second-to-deepest level)
            copy_parent_attributes(
                input_file, output_file, group_path, processed_parents
            )

            # Read data with type-safe access
            g5g5_item = input_item[REQUIRED_DATASETS[0]]

            if not isinstance(g5g5_item, h5py.Dataset):
                logger.warning(
                    f"Skipping {group_path}: {REQUIRED_DATASETS[0]} is not a dataset"
                )
                continue

            g5g5_samples = g5g5_item[:]

            # Calculate effective mass
            effective_mass = calculate_effective_mass(g5g5_samples)
            mean_values, error_values = calculate_jackknife_statistics(effective_mass)

            # Add analysis methodology documentation to output file
            output_file.attrs["analysis_date"] = datetime.now().isoformat()
            output_file.attrs["source_file"] = os.path.basename(input_path)

            # Add documentation from config
            for key, value in ANALYSIS_DOCUMENTATION.items():
                output_file.attrs[key] = value

            # Create output group
            output_group = output_file.create_group(group_path)

            # Save results
            output_group.create_dataset(
                output_names["samples"],
                data=effective_mass,
            )
            output_group.create_dataset(output_names["mean"], data=mean_values)
            output_group.create_dataset(output_names["error"], data=error_values)

            # Copy metadata (datasets + deepest group attributes)
            copy_metadata(input_item, output_group, METADATA_DATASETS)

    return len(analysis_groups)


def validate_effective_mass_file_consistency(
    input_file_path, required_datasets, logger
):
    """
    Validate effective mass data consistency once per file using first
    valid group. Returns the established lengths for the entire file.
    """
    # Find first valid group as representative
    analysis_groups = find_analysis_groups(input_file_path, required_datasets)
    if not analysis_groups:
        raise ValueError("No groups with required effective mass datasets found")

    representative_group = analysis_groups[0]

    with h5py.File(input_file_path, "r") as f:
        group_item = f[representative_group]
        if not isinstance(group_item, h5py.Group):
            raise ValueError(
                f"Expected Group at {representative_group}, got {type(group_item)}"
            )
        group = group_item

        # Get actual g5g5 length with type checking
        g5g5_item = group[required_datasets[0]]  # g5g5_jackknife_samples
        if not isinstance(g5g5_item, h5py.Dataset):
            raise ValueError(
                f"Expected Dataset at {required_datasets[0]}, " f"got {type(g5g5_item)}"
            )

        g5g5_length = g5g5_item.shape[-1]

        # Calculate expected effective mass length based on processing
        # parameters
        if TRUNCATE_HALF:
            # For periodic BC: use first half, then t=1 to T/2-1
            expected_effective_length = (g5g5_length // 2) - 1
        else:
            # Full length: t=1 to T-2
            expected_effective_length = g5g5_length - 2

    logger.info(
        f"Validated effective mass file structure: g5g5={g5g5_length}, "
        f"expected_effective={expected_effective_length}"
    )
    return g5g5_length, expected_effective_length


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
    default="correlators_effective_mass_analysis.h5",
    callback=hdf5_file.output,
    help="Path for output HDF5 file with pion effective mass calculation results. Default: correlators_effective_mass_analysis.h5",
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
    enable_logging: bool,
    log_directory: Optional[str],
    log_filename: Optional[str],
) -> None:
    """Calculate effective mass from jackknife-analyzed g5-g5 correlator
    data."""

    # Validate configuration
    validate_effective_config()

    # Setup output path
    if os.path.dirname(output_hdf5_file):
        # Full path given - use as-is
        output_path = output_hdf5_file
    else:
        # Filename only - put in input file's directory
        output_path = os.path.join(os.path.dirname(input_hdf5_file), output_hdf5_file)

    # Setup logging
    log_dir = None
    if enable_logging:
        log_dir = log_directory or os.path.dirname(output_path)

    logger = create_script_logger(
        log_directory=log_dir,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=False,
    )

    try:
        logger.log_script_start("Effective mass calculation")

        total_groups = process_effective_file(input_hdf5_file, output_path, logger)

        click.echo(f"✓ Effective mass calculation complete")
        click.echo(f"  Processed: {total_groups} groups")
        click.echo(f"  Output: {Path(output_path).relative_to(ROOT)}")

        logger.log_script_end("Effective mass calculation completed")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
