#!/usr/bin/env python3
"""
PCAC mass calculation script for QPB correlator data analysis.

This script calculates PCAC (Partially Conserved Axial Current) mass
from jackknife-analyzed correlator data stored in HDF5 format. The PCAC
mass is calculated as:

    PCAC_mass = 0.5 * g4g5g5_derivative / g5g5_truncated

where g5g5 correlators are truncated to match the length of the
derivative correlators. The script processes each deepest-level group
independently and exports jackknife samples, means, and errors in a new
HDF5 file.

Key features:
    - Configurable truncation parameters
    - Jackknife error propagation
    - Preservation of group hierarchy and metadata
    - Comprehensive logging and validation

Place this file as:
qpb_data_analysis/core/src/analysis/calculate_PCAC_mass.py

Usage:
    python calculate_PCAC_mass.py -i jackknife_analysis.h5 -o
    pcac_mass.h5 [options]
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import h5py

# Import library components
from library.data.hdf5_analyzer import HDF5Analyzer
from library.validation.click_validators import (
    hdf5_file,
    directory,
    validate_log_filename,
)
from library.utils.logging_utilities import create_script_logger

# Import configuration
from src.analysis._pcac_mass_config import (
    TRUNCATE_START,
    TRUNCATE_END,
    EXPECTED_G5G5_LENGTH,
    EXPECTED_DERIVATIVE_LENGTH,
    EXPECTED_PCAC_LENGTH,
    REQUIRED_INPUT_DATASETS,
    PCAC_MASS_DATASETS,
    METADATA_DATASETS,
    PCAC_MASS_FACTOR,
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
    """
    Calculate PCAC mass from jackknife-analyzed correlator data.

    This script processes QPB correlator data that has undergone
    jackknife analysis and calculates the PCAC (Partially Conserved
    Axial Current) mass using the formula:

        PCAC_mass = 0.5 * g4g5g5_derivative / g5g5_truncated

    The script preserves the group hierarchy and essential metadata
    while creating a clean output file containing only PCAC mass
    results.
    """
    # Handle default directories
    if output_directory is None:
        output_directory = os.path.dirname(input_hdf5_file)

    if log_directory is None and enable_logging:
        log_directory = output_directory

    # Create full output path
    output_file_path = os.path.join(output_directory, output_hdf5_file)

    # Set up logging
    logger = create_script_logger(
        log_directory=log_directory if enable_logging else None,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=False,
    )

    logger.log_script_start("PCAC mass calculation")

    try:
        # Log input parameters
        logger.info(f"Input HDF5 file: {input_hdf5_file}")
        logger.info(f"Output HDF5 file: {output_file_path}")
        logger.info(
            f"Truncation parameters: start={TRUNCATE_START}, end={TRUNCATE_END}"
        )
        logger.info(f"PCAC mass factor: {PCAC_MASS_FACTOR}")

        # Create output HDF5 file
        logger.info("Starting PCAC mass calculation...")
        _create_output_hdf5_structure(input_hdf5_file, output_file_path, logger)

        logger.log_script_end("PCAC mass calculation completed successfully")
        click.echo(f"✓ PCAC mass calculation complete. Output: {output_file_path}")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.log_script_end("PCAC mass calculation failed")
        raise


def _create_output_hdf5_structure(
    input_file_path: str, output_file_path: str, logger
) -> None:
    """
    Create output HDF5 file with PCAC mass results.

    Args:
        - input_file_path: Path to input HDF5 file
        - output_file_path: Path for output HDF5 file
        - logger: Logger instance
    """
    logger.info("Creating output HDF5 file structure")

    with h5py.File(input_file_path, "r") as input_file, h5py.File(
        output_file_path, "w"
    ) as output_file:

        # Find all deepest-level groups that contain required datasets
        analysis_groups = _find_analysis_groups(input_file, logger)

        if not analysis_groups:
            raise ValueError("No groups with required PCAC mass datasets found")

        logger.info(f"Found {len(analysis_groups)} groups to process")

        # Process each group
        for group_path in analysis_groups:
            logger.debug(f"Creating output structure for: {group_path}")

            # Create corresponding group structure in output file
            output_group = output_file.create_group(group_path)

            # Copy parent group attributes (second-to-deepest level)
            _copy_parent_group_attributes(input_file, output_file, group_path, logger)

            # Process the group
            _process_single_group(group_path, input_file, output_group, logger)

        logger.info(f"Successfully created output file: {output_file_path}")


def _find_analysis_groups(input_file: h5py.File, logger) -> List[str]:
    """
    Find all groups containing required datasets for PCAC mass
    calculation.

    Args:
        - input_file: Input HDF5 file
        - logger: Logger instance

    Returns:
        List of group paths containing required datasets
    """
    analysis_groups = []

    def find_groups(name, obj):
        if isinstance(obj, h5py.Group):
            # Check if this group contains the required datasets
            has_required = all(
                dataset_name in obj for dataset_name in REQUIRED_INPUT_DATASETS
            )
            if has_required:
                analysis_groups.append(name)

    input_file.visititems(find_groups)
    logger.debug(f"Found {len(analysis_groups)} groups with required datasets")

    return analysis_groups


def _copy_parent_group_attributes(
    input_file: h5py.File, output_file: h5py.File, group_path: str, logger
) -> None:
    """
    Copy parent group attributes to output file.

    Args:
        - input_file: Input HDF5 file
        - output_file: Output HDF5 file
        - group_path: Current group path
        - logger: Logger instance
    """
    # Get parent path (second-to-deepest level)
    parent_path = "/".join(group_path.split("/")[:-1])

    if parent_path and parent_path != "":
        # Ensure parent group exists in output
        parent_group = output_file.require_group(parent_path)

        # Copy attributes if not already copied
        if len(parent_group.attrs) == 0:
            input_parent = input_file[parent_path]
            for attr_name, attr_value in input_parent.attrs.items():
                parent_group.attrs[attr_name] = attr_value

            logger.debug(
                f"Copied {len(input_parent.attrs)} attributes to parent group: {parent_path}"
            )


def _process_single_group(
    group_path: str, input_file: h5py.File, output_group: h5py.Group, logger
) -> None:
    """
    Process a single deepest-level group to calculate PCAC mass.

    Args:
        - group_path: Path to the group in input HDF5
        - input_file: Input HDF5 file
        - output_group: Corresponding group in output HDF5 file
        - logger: Logger instance
    """
    logger.info(f"Processing group: {group_path}")

    try:
        input_item = input_file[group_path]

        # Validate that the path points to a group
        if not isinstance(input_item, h5py.Group):
            raise ValueError(
                f"Path {group_path} is not a group, it's a {type(input_item)}"
            )

        input_group = input_item

        # Validate that required items are datasets
        g4g5g5_derivative_item = input_group["g4g5g5_derivative_jackknife_samples"]
        g5g5_samples_item = input_group["g5g5_jackknife_samples"]

        if not isinstance(g4g5g5_derivative_item, h5py.Dataset):
            raise ValueError(
                f"g4g5g5_derivative_jackknife_samples is not a dataset, it's a {type(g4g5g5_derivative_item)}"
            )

        if not isinstance(g5g5_samples_item, h5py.Dataset):
            raise ValueError(
                f"g5g5_jackknife_samples is not a dataset, it's a {type(g5g5_samples_item)}"
            )

        # Read required datasets
        g4g5g5_derivative = g4g5g5_derivative_item[()]
        g5g5_samples = g5g5_samples_item[()]

        # Validate dimensions
        _validate_dataset_dimensions(
            g4g5g5_derivative, g5g5_samples, group_path, logger
        )

        # Truncate g5g5 samples
        g5g5_truncated = _truncate_g5g5_samples(g5g5_samples)

        logger.debug(
            f"Truncated g5g5 from {g5g5_samples.shape[1]} to {g5g5_truncated.shape[1]} elements"
        )

        # Calculate PCAC mass jackknife samples
        pcac_mass_samples = _calculate_pcac_mass_jackknife_samples(
            g4g5g5_derivative, g5g5_truncated
        )

        # Calculate statistics
        pcac_mass_mean, pcac_mass_error = _calculate_jackknife_statistics(
            pcac_mass_samples
        )

        # Write datasets to output
        output_group.create_dataset(
            PCAC_MASS_DATASETS["jackknife_samples"],
            data=pcac_mass_samples,
            compression="gzip",
            compression_opts=6,
        )
        output_group.create_dataset(
            PCAC_MASS_DATASETS["mean_values"],
            data=pcac_mass_mean,
            compression="gzip",
            compression_opts=6,
        )
        output_group.create_dataset(
            PCAC_MASS_DATASETS["error_values"],
            data=pcac_mass_error,
            compression="gzip",
            compression_opts=6,
        )

        # Copy metadata datasets
        for metadata_name in METADATA_DATASETS:
            if metadata_name in input_group:
                metadata_item = input_group[metadata_name]
                if isinstance(metadata_item, h5py.Dataset):
                    metadata_data = metadata_item[()]
                    output_group.create_dataset(metadata_name, data=metadata_data)
                else:
                    logger.warning(
                        f"Skipping {metadata_name}: not a dataset, it's a {type(metadata_item)}"
                    )

        # Copy group attributes
        for attr_name, attr_value in input_group.attrs.items():
            output_group.attrs[attr_name] = attr_value

        logger.info(
            f"Successfully calculated PCAC mass for group {group_path} - "
            f"{pcac_mass_samples.shape[0]} jackknife samples, "
            f"{pcac_mass_samples.shape[1]} time points"
        )

    except Exception as e:
        logger.error(f"Failed to process group {group_path}: {e}")
        raise


def _validate_dataset_dimensions(
    g4g5g5_derivative: np.ndarray, g5g5_samples: np.ndarray, group_name: str, logger
) -> None:
    """
    Validate that input datasets have expected dimensions.

    Args:
        - g4g5g5_derivative: Derivative jackknife samples array
        - g5g5_samples: G5G5 jackknife samples array
        - group_name: Name of current group for error reporting
        - logger: Logger instance

    Raises:
        ValueError: If dimensions don't match expectations
    """
    # Check g4g5g5_derivative dimensions
    if g4g5g5_derivative.shape[1] != EXPECTED_DERIVATIVE_LENGTH:
        raise ValueError(
            f"Group {group_name}: g4g5g5_derivative has {g4g5g5_derivative.shape[1]} "
            f"elements, expected {EXPECTED_DERIVATIVE_LENGTH}"
        )

    # Check g5g5 dimensions
    if g5g5_samples.shape[1] != EXPECTED_G5G5_LENGTH:
        raise ValueError(
            f"Group {group_name}: g5g5_jackknife_samples has {g5g5_samples.shape[1]} "
            f"elements, expected {EXPECTED_G5G5_LENGTH}"
        )

    # Check that jackknife sample counts match
    if g4g5g5_derivative.shape[0] != g5g5_samples.shape[0]:
        raise ValueError(
            f"Group {group_name}: Mismatch in jackknife sample count - "
            f"g4g5g5_derivative: {g4g5g5_derivative.shape[0]}, "
            f"g5g5: {g5g5_samples.shape[0]}"
        )

    logger.debug(
        f"Group {group_name}: Validation passed - "
        f"{g4g5g5_derivative.shape[0]} jackknife samples, "
        f"g4g5g5_derivative: {g4g5g5_derivative.shape[1]} elements, "
        f"g5g5: {g5g5_samples.shape[1]} elements"
    )


def _truncate_g5g5_samples(g5g5_samples: np.ndarray) -> np.ndarray:
    """
    Truncate g5g5 jackknife samples to match derivative length.

    Args:
        g5g5_samples: Original g5g5 jackknife samples [n_samples, 48]

    Returns:
        Truncated g5g5 samples [n_samples, 44]
    """
    end_index = g5g5_samples.shape[1] - TRUNCATE_END
    return g5g5_samples[:, TRUNCATE_START:end_index]


def _calculate_pcac_mass_jackknife_samples(
    g4g5g5_derivative: np.ndarray,
    g5g5_truncated: np.ndarray,
) -> np.ndarray:
    """
    Calculate PCAC mass jackknife samples.

    Args:
        - g4g5g5_derivative: Derivative jackknife samples [n_samples,
          44]
        - g5g5_truncated: Truncated g5g5 jackknife samples [n_samples,
          44]

    Returns:
        PCAC mass jackknife samples [n_samples, 44]
    """
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        pcac_mass_samples = PCAC_MASS_FACTOR * g4g5g5_derivative / g5g5_truncated

    return pcac_mass_samples


def _calculate_jackknife_statistics(
    jackknife_samples: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and error from jackknife samples.

    Args:
        jackknife_samples: Jackknife samples array [n_samples, n_points]

    Returns:
        Tuple of (mean_values, error_values)
    """
    # Calculate mean across jackknife samples
    mean_values = np.mean(jackknife_samples, axis=0)

    # Calculate jackknife error
    n_samples = jackknife_samples.shape[0]
    sample_means = jackknife_samples
    variance = ((n_samples - 1) / n_samples) * np.sum(
        (sample_means - mean_values[np.newaxis, :]) ** 2, axis=0
    )
    error_values = np.sqrt(variance)

    return mean_values, error_values


if __name__ == "__main__":
    main()
