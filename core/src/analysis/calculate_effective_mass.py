#!/usr/bin/env python3
"""
Effective mass calculation script for QPB correlator data analysis.

This script calculates effective mass from jackknife-analyzed g5-g5
correlator data stored in HDF5 format. The effective mass is calculated
using the two-state periodic formula from individual g5-g5 correlator
jackknife samples.

The calculation applies symmetrization to g5-g5 correlators and uses the
mathematical formula:
    effective_mass =
    calculate_two_state_periodic_effective_mass_correlator(g5g5_correlator)

Key features:
    - Configurable symmetrization and truncation parameters
    - Jackknife error propagation
    - Preservation of group hierarchy and metadata
    - Comprehensive logging and validation
    - Physical behavior validation

Place this file as:
qpb_data_analysis/core/src/analysis/calculate_effective_mass.py

Usage:
    python calculate_effective_mass.py -i jackknife_analysis.h5 -o
    effective_mass.h5 [options]
"""

import os
from typing import List, Optional, Tuple

import click
import numpy as np
import h5py

# Import library components
from library.validation.click_validators import (
    hdf5_file,
    directory,
    validate_log_filename,
)
from library.utils.logging_utilities import create_script_logger

# Import configuration
from src.analysis._effective_mass_config import (
    APPLY_SYMMETRIZATION,
    TRUNCATE_HALF,
    LOWERING_FACTOR,
    EXPECTED_G5G5_LENGTH,
    EXPECTED_EFFECTIVE_MASS_LENGTH,
    REQUIRED_INPUT_DATASETS,
    ALTERNATIVE_INPUT_DATASETS,
    EFFECTIVE_MASS_DATASETS,
    METADATA_DATASETS,
    EFFECTIVE_MASS_CALCULATION_PARAMS,
    VALIDATION_PARAMS,
    ERROR_HANDLING,
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
    help="Path for output HDF5 file with effective mass results.",
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
    Calculate effective mass from jackknife-analyzed g5-g5 correlator data.

    This script processes QPB correlator data that has undergone
    jackknife analysis and calculates effective mass using the two-state
    periodic effective mass formula. The script applies symmetrization
    to the input g5-g5 correlators and preserves the group hierarchy
    and essential metadata.
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

    logger.log_script_start("Effective mass calculation")

    try:
        # Log input parameters
        logger.info(f"Input HDF5 file: {input_hdf5_file}")
        logger.info(f"Output HDF5 file: {output_file_path}")
        logger.info(f"Apply symmetrization: {APPLY_SYMMETRIZATION}")
        logger.info(f"Truncate half: {TRUNCATE_HALF}")
        logger.info(f"Lowering factor: {LOWERING_FACTOR}")

        # Create output HDF5 file
        logger.info("Starting effective mass calculation...")
        _create_output_hdf5_structure(input_hdf5_file, output_file_path, logger)

        logger.log_script_end("Effective mass calculation completed successfully")
        click.echo(f"✓ Effective mass calculation complete. Output: {output_file_path}")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.log_script_end("Effective mass calculation failed")
        raise


def _create_output_hdf5_structure(
    input_file_path: str, output_file_path: str, logger
) -> None:
    """
    Create output HDF5 file with effective mass results.

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
            raise ValueError("No groups with required effective mass datasets found")

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
    Find all groups containing required datasets for effective mass
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
            # Check for primary dataset names first
            has_required = any(
                dataset_name in obj for dataset_name in REQUIRED_INPUT_DATASETS
            )

            # Check for alternative dataset names if primary not found
            if not has_required:
                has_required = any(
                    dataset_name in obj for dataset_name in ALTERNATIVE_INPUT_DATASETS
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
        - group_path: Path to the current group being processed
        - logger: Logger instance
    """
    # Extract parent path (second-to-deepest level)
    parent_path = "/".join(group_path.split("/")[:-1])

    if parent_path and parent_path in input_file:
        parent_group = input_file[parent_path]

        # Create parent group in output if it doesn't exist
        if parent_path not in output_file:
            output_parent = output_file.create_group(parent_path)

            # Copy all attributes from input parent group
            for attr_name, attr_value in parent_group.attrs.items():
                output_parent.attrs[attr_name] = attr_value

            logger.debug(f"Copied attributes to parent group: {parent_path}")


def _process_single_group(
    group_path: str, input_file: h5py.File, output_group: h5py.Group, logger
) -> None:
    """
    Process a single group containing g5-g5 correlator data.

    Args:
        - group_path: Path to the group being processed
        - input_file: Input HDF5 file
        - output_group: Output group for results
        - logger: Logger instance
    """
    input_obj = input_file[group_path]

    # Ensure we have a Group object
    if not isinstance(input_obj, h5py.Group):
        raise ValueError(f"Path {group_path} is not a Group object")

    input_group = input_obj

    # Find the correct input dataset name
    g5g5_dataset_name = _find_g5g5_dataset_name(input_group, logger)

    if g5g5_dataset_name is None:
        raise ValueError(f"Group {group_path}: No g5-g5 correlator dataset found")

    # Load g5-g5 jackknife samples
    g5g5_dataset = input_group[g5g5_dataset_name]

    # Ensure we have a Dataset object
    if not isinstance(g5g5_dataset, h5py.Dataset):
        raise ValueError(
            f"Group {group_path}: {g5g5_dataset_name} is not a Dataset object"
        )

    g5g5_samples = g5g5_dataset[:]

    logger.info(
        f"Group {group_path}: Loaded {g5g5_dataset_name} with shape {g5g5_samples.shape}"
    )

    # Validate input data
    _validate_input_data(g5g5_samples, group_path, logger)

    # Apply symmetrization if enabled
    if APPLY_SYMMETRIZATION:
        g5g5_samples = _apply_symmetrization(g5g5_samples, logger)
        logger.debug(f"Group {group_path}: Applied symmetrization to g5-g5 correlators")

    # Calculate effective mass for each jackknife sample
    effective_mass_samples = _calculate_effective_mass_jackknife_samples(
        g5g5_samples, group_path, logger
    )

    # Calculate jackknife statistics
    effective_mass_mean, effective_mass_error = _calculate_jackknife_statistics(
        effective_mass_samples
    )

    # Store results in output group
    _store_effective_mass_results(
        output_group,
        effective_mass_samples,
        effective_mass_mean,
        effective_mass_error,
        logger,
    )

    # Copy metadata datasets
    _copy_metadata_datasets(input_group, output_group, logger)

    # Copy group attributes
    for attr_name, attr_value in input_group.attrs.items():
        output_group.attrs[attr_name] = attr_value

    logger.info(
        f"Group {group_path}: Effective mass calculation completed - "
        f"output shape: {effective_mass_samples.shape}"
    )


def _find_g5g5_dataset_name(input_group: h5py.Group, logger) -> Optional[str]:
    """
    Find the correct g5-g5 dataset name in the input group.

    Args:
        - input_group: Input HDF5 group
        - logger: Logger instance

    Returns:
        Name of g5-g5 dataset if found, None otherwise
    """
    # Check primary dataset names first
    for dataset_name in REQUIRED_INPUT_DATASETS:
        if dataset_name in input_group:
            return dataset_name

    # Check alternative dataset names
    for dataset_name in ALTERNATIVE_INPUT_DATASETS:
        if dataset_name in input_group:
            return dataset_name

    return None


def _validate_input_data(g5g5_samples: np.ndarray, group_name: str, logger) -> None:
    """
    Validate g5-g5 correlator input data.

    Args:
        - g5g5_samples: G5-G5 jackknife samples array
        - group_name: Name of current group for error reporting
        - logger: Logger instance

    Raises:
        ValueError: If dimensions don't match expectations or data is invalid
    """
    # Check g5g5 dimensions
    if g5g5_samples.shape[1] != EXPECTED_G5G5_LENGTH:
        raise ValueError(
            f"Group {group_name}: g5g5_samples has {g5g5_samples.shape[1]} "
            f"elements, expected {EXPECTED_G5G5_LENGTH}"
        )

    # Check for non-zero values
    if VALIDATION_PARAMS["check_non_zero"]:
        if np.any(np.abs(g5g5_samples) < VALIDATION_PARAMS["min_correlation_value"]):
            logger.warning(f"Group {group_name}: Found very small correlator values")

    # Check for decreasing behavior (physical expectation)
    if VALIDATION_PARAMS["check_decreasing"]:
        mean_correlator = np.mean(g5g5_samples, axis=0)
        # Check first half of time series for decreasing behavior
        first_half = mean_correlator[: len(mean_correlator) // 2]
        if not np.all(np.diff(first_half) <= 0):
            logger.warning(
                f"Group {group_name}: Correlator does not decrease monotonically"
            )

    logger.debug(
        f"Group {group_name}: Validation passed - "
        f"{g5g5_samples.shape[0]} jackknife samples, "
        f"correlator length: {g5g5_samples.shape[1]}"
    )


def _apply_symmetrization(g5g5_samples: np.ndarray, logger) -> np.ndarray:
    """
    Apply symmetrization to g5-g5 correlator samples.

    Uses the same symmetrization method as momentum_correlator.symmetrization().

    Args:
        - g5g5_samples: Original g5-g5 jackknife samples

    Returns:
        Symmetrized g5-g5 samples
    """
    symmetrized_samples = np.array(
        [_symmetrize_single_correlator(sample) for sample in g5g5_samples]
    )

    logger.debug("Applied symmetrization to all jackknife samples")
    return symmetrized_samples


def _symmetrize_single_correlator(correlator: np.ndarray) -> np.ndarray:
    """
    Symmetrize a single correlator using the same method as
    momentum_correlator.symmetrization().

    Args:
        correlator: 1D correlator array

    Returns:
        Symmetrized 1D correlator array
    """
    reverse = correlator[::-1]
    return 0.5 * (correlator + np.roll(reverse, shift=+1))


def _calculate_effective_mass_jackknife_samples(
    g5g5_samples: np.ndarray, group_name: str, logger
) -> np.ndarray:
    """
    Calculate effective mass jackknife samples from g5-g5 correlators.

    Args:
        - g5g5_samples: G5-G5 jackknife samples [n_samples, 48]
        - group_name: Name of current group for error reporting
        - logger: Logger instance

    Returns:
        Effective mass jackknife samples [n_samples, ~23]
    """
    effective_mass_samples = []
    n_samples = g5g5_samples.shape[0]

    for i, g5g5_correlator in enumerate(g5g5_samples):
        try:
            effective_mass_correlator = (
                _calculate_two_state_periodic_effective_mass_correlator(g5g5_correlator)
            )
            effective_mass_samples.append(effective_mass_correlator)

        except Exception as e:
            if ERROR_HANDLING["log_invalid_calculations"]:
                logger.warning(
                    f"Group {group_name}: Error calculating effective mass for sample {i}: {e}"
                )

            # Create array of replacement values
            if len(effective_mass_samples) > 0:
                # Use same length as previous successful calculation
                length = len(effective_mass_samples[0])
            else:
                # Use expected length
                length = EXPECTED_EFFECTIVE_MASS_LENGTH

            replacement_array = np.full(
                length, ERROR_HANDLING["invalid_replacement_value"]
            )
            effective_mass_samples.append(replacement_array)

    effective_mass_array = np.array(effective_mass_samples)

    logger.info(
        f"Group {group_name}: Calculated effective mass for {n_samples} samples, "
        f"output shape: {effective_mass_array.shape}"
    )

    return effective_mass_array


def _calculate_two_state_periodic_effective_mass_correlator(
    g5g5_correlator_array: np.ndarray,
) -> np.ndarray:
    """
    Calculate two-state periodic effective mass correlator from g5-g5 correlator.

    This implements the same calculation as the function from the old effective_mass module:
    effective_mass.calculate_two_state_periodic_effective_mass_correlator()

    Args:
        g5g5_correlator_array: 1D g5-g5 correlator array

    Returns:
        Effective mass correlator array
    """
    lowering_factor = EFFECTIVE_MASS_CALCULATION_PARAMS["lowering_factor"]
    truncate_half = EFFECTIVE_MASS_CALCULATION_PARAMS["truncate_half"]

    # Calculate middle value
    middle_value = np.min(g5g5_correlator_array) * lowering_factor

    temporal_lattice_size = len(g5g5_correlator_array)

    # Create shifted arrays
    shifted_backward_array = np.roll(g5g5_correlator_array, shift=+1)
    shifted_forward_array = np.roll(g5g5_correlator_array, shift=-1)

    # Remove extreme elements
    if EFFECTIVE_MASS_CALCULATION_PARAMS["remove_extreme_points"]:
        shifted_backward_array = shifted_backward_array[1:-1]
        shifted_forward_array = shifted_forward_array[1:-1]

    # Truncate to half if requested
    if truncate_half:
        upper_index_cut = (temporal_lattice_size - 2) // 2
        shifted_backward_array = shifted_backward_array[:upper_index_cut]
        shifted_forward_array = shifted_forward_array[:upper_index_cut]

    # Handle potential numerical issues
    with np.errstate(invalid="ignore", divide="ignore"):
        # Calculate numerator and denominator
        numerator = shifted_backward_array + np.sqrt(
            np.square(shifted_backward_array) - middle_value**2
        )
        denominator = shifted_forward_array + np.sqrt(
            np.square(shifted_forward_array) - middle_value**2
        )

        # Calculate effective mass
        ratio = numerator / denominator
        effective_mass = 0.5 * np.log(ratio)

    # Handle invalid values if error handling is enabled
    if (
        ERROR_HANDLING["handle_division_by_zero"]
        or ERROR_HANDLING["handle_negative_sqrt"]
    ):
        effective_mass = np.where(
            np.isfinite(effective_mass),
            effective_mass,
            ERROR_HANDLING["invalid_replacement_value"],
        )

    return effective_mass


def _calculate_jackknife_statistics(
    jackknife_samples: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and error from jackknife samples.

    Args:
        - jackknife_samples: Jackknife samples array [n_samples, n_time]

    Returns:
        Tuple of (mean_values, error_values)
    """
    n_samples = jackknife_samples.shape[0]

    # Calculate jackknife mean
    mean_values = np.mean(jackknife_samples, axis=0)

    # Calculate jackknife error
    # Standard formula: sqrt((n-1) * sum((x_i - mean)^2) / n)
    deviations_squared = np.square(jackknife_samples - mean_values[np.newaxis, :])
    variance = np.sum(deviations_squared, axis=0) / n_samples
    error_values = np.sqrt((n_samples - 1) * variance)

    return mean_values, error_values


def _store_effective_mass_results(
    output_group: h5py.Group,
    effective_mass_samples: np.ndarray,
    effective_mass_mean: np.ndarray,
    effective_mass_error: np.ndarray,
    logger,
) -> None:
    """
    Store effective mass results in output HDF5 group.

    Args:
        - output_group: Output HDF5 group
        - effective_mass_samples: Jackknife samples
        - effective_mass_mean: Mean values
        - effective_mass_error: Error values
        - logger: Logger instance
    """
    # Store jackknife samples
    output_group.create_dataset(
        EFFECTIVE_MASS_DATASETS["jackknife_samples"],
        data=effective_mass_samples,
        compression="gzip",
        compression_opts=6,
    )

    # Store mean values
    output_group.create_dataset(
        EFFECTIVE_MASS_DATASETS["mean_values"],
        data=effective_mass_mean,
        compression="gzip",
        compression_opts=6,
    )

    # Store error values
    output_group.create_dataset(
        EFFECTIVE_MASS_DATASETS["error_values"],
        data=effective_mass_error,
        compression="gzip",
        compression_opts=6,
    )

    logger.debug(
        f"Stored effective mass results: samples {effective_mass_samples.shape}, "
        f"mean/error length {len(effective_mass_mean)}"
    )


def _copy_metadata_datasets(
    input_group: h5py.Group, output_group: h5py.Group, logger
) -> None:
    """
    Copy metadata datasets from input to output group.

    Args:
        - input_group: Input HDF5 group
        - output_group: Output HDF5 group
        - logger: Logger instance
    """
    copied_datasets = []

    for dataset_name in METADATA_DATASETS:
        if dataset_name in input_group:
            # Get the dataset object and ensure it's actually a Dataset
            input_obj = input_group[dataset_name]

            if not isinstance(input_obj, h5py.Dataset):
                logger.warning(
                    f"Metadata item {dataset_name} is not a Dataset, skipping"
                )
                continue

            input_dataset = input_obj

            # Copy dataset with same compression settings
            output_group.create_dataset(
                dataset_name,
                data=input_dataset[:],
                compression="gzip",
                compression_opts=6,
            )
            copied_datasets.append(dataset_name)

    if copied_datasets:
        logger.debug(f"Copied metadata datasets: {copied_datasets}")


if __name__ == "__main__":
    main()
