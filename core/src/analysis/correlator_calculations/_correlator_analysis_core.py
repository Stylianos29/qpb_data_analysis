#!/usr/bin/env python3
"""
Core utilities for correlator analysis scripts.

This module provides common functionality specific to PCAC mass and
effective mass calculations that isn't already covered by the existing
library infrastructure (HDF5Analyzer, click validators, logging
utilities).

The functions here focus on:
    - Mathematical operations specific to correlator analysis
    - Jackknife statistics calculations
    - Correlator-specific data transformations
    - Physics-specific validation
"""

from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np
import h5py


# =============================================================================
# JACKKNIFE STATISTICS (Mathematical Operations)
# =============================================================================


def calculate_jackknife_statistics(
    jackknife_samples: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and error from jackknife samples.

    This is the mathematical calculation only - HDF5 I/O is handled by
    HDF5Analyzer.

    Args:
        jackknife_samples: Array of jackknife samples [n_samples, ...]

    Returns:
        Tuple of (mean_values, error_values) with same shape as
        samples[0]
    """
    n_samples = jackknife_samples.shape[0]

    # Calculate jackknife average
    mean_values = np.mean(jackknife_samples, axis=0)

    # Calculate jackknife error Formula: sqrt((n-1) * mean((samples -
    # mean)^2))
    deviations = jackknife_samples - mean_values
    variance = np.mean(deviations**2, axis=0)
    error_values = np.sqrt((n_samples - 1) * variance)

    return mean_values, error_values


def validate_jackknife_consistency(
    datasets: Dict[str, np.ndarray], group_name: str = "unnamed"
) -> int:
    """
    Validate that all jackknife sample arrays have consistent
    dimensions.

    Args:
        - datasets: Dictionary of dataset names to arrays
        - group_name: Name of the group for error reporting

    Returns:
        Number of jackknife samples

    Raises:
        ValueError: If jackknife dimensions are inconsistent or no
        datasets provided
    """
    if not datasets:
        raise ValueError(f"Group '{group_name}': No datasets provided for validation")

    n_samples: Optional[int] = None

    for name, data in datasets.items():
        if n_samples is None:
            n_samples = data.shape[0]
        elif data.shape[0] != n_samples:
            raise ValueError(
                f"Group '{group_name}': Inconsistent jackknife samples - "
                f"'{name}' has {data.shape[0]} samples, expected {n_samples}"
            )

    # n_samples is guaranteed to be int here since we checked datasets
    # is not empty
    assert n_samples is not None
    return n_samples


# =============================================================================
# CORRELATOR-SPECIFIC OPERATIONS
# =============================================================================


def truncate_correlator(
    correlator: np.ndarray, truncate_start: int, truncate_end: int
) -> np.ndarray:
    """
    Truncate correlator array by removing elements from start and end.

    Args:
        - correlator: Input correlator array [..., time_length]
        - truncate_start: Number of elements to remove from start
        - truncate_end: Number of elements to remove from end

    Returns:
        Truncated correlator array
    """
    if truncate_end > 0:
        return correlator[..., truncate_start:-truncate_end]
    else:
        return correlator[..., truncate_start:]


def symmetrize_correlator(correlator: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Symmetrize a correlator using C(t) = 0.5 * (C(t) + C(T-t)).

    This is the standard symmetrization for periodic boundary
    conditions.

    Args:
        - correlator: Input correlator array
        - axis: Axis along which to symmetrize (default: last axis)

    Returns:
        Symmetrized correlator
    """
    reversed_correlator = np.flip(correlator, axis=axis)
    shifted_reversed = np.roll(reversed_correlator, shift=1, axis=axis)
    return 0.5 * (correlator + shifted_reversed)


def safe_divide_correlators(
    numerator: np.ndarray,
    denominator: np.ndarray,
    fill_value: float = 0.0,
    min_denominator: float = 1e-15,
) -> np.ndarray:
    """
    Safely divide correlator arrays, handling near-zero denominators.

    This is commonly needed for PCAC mass and effective mass
    calculations.

    Args:
        - numerator: Numerator correlator array
        - denominator: Denominator correlator array
        - fill_value: Value to use where denominator is too small
        - min_denominator: Minimum absolute value for denominator

    Returns:
        Result of division with safe handling of small denominators
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            np.abs(denominator) > min_denominator, numerator / denominator, fill_value
        )

    return result


# =============================================================================
# EFFECTIVE MASS SPECIFIC CALCULATIONS
# =============================================================================


def calculate_two_state_periodic_effective_mass(
    g5g5_correlator: np.ndarray,
    lowering_factor: float = 0.99,
    truncate_half: bool = True,
) -> np.ndarray:
    """
    Calculate two-state periodic effective mass from g5g5 correlator.

    This implements the formula used in library.effective_mass but
    operates on individual jackknife samples.

    Args:
        - g5g5_correlator: Input g5g5 correlator [time_length] or
          [n_samples, time_length]
        - lowering_factor: Factor for middle value calculation
        - truncate_half: Whether to truncate to half the temporal extent

    Returns:
        Effective mass array with shape adjusted based on truncation
    """
    # Handle both 1D and 2D inputs
    if g5g5_correlator.ndim == 1:
        g5g5_correlator = g5g5_correlator[np.newaxis, :]
        squeeze_output = True
    else:
        squeeze_output = False

    # Calculate middle value
    middle_value = np.min(g5g5_correlator, axis=-1, keepdims=True) * lowering_factor

    # Shift arrays
    shifted_backward = np.roll(g5g5_correlator, shift=+1, axis=-1)
    shifted_forward = np.roll(g5g5_correlator, shift=-1, axis=-1)

    # Remove extreme elements
    shifted_backward = shifted_backward[..., 1:-1]
    shifted_forward = shifted_forward[..., 1:-1]
    middle_value_expanded = np.broadcast_to(middle_value, shifted_backward.shape)

    # Apply truncation if requested
    if truncate_half:
        temporal_size = shifted_backward.shape[-1]
        upper_index = temporal_size // 2
        shifted_backward = shifted_backward[..., :upper_index]
        shifted_forward = shifted_forward[..., :upper_index]
        middle_value_expanded = middle_value_expanded[..., :upper_index]

    # Calculate effective mass
    with np.errstate(invalid="ignore"):
        numerator = shifted_backward + np.sqrt(
            np.maximum(
                0, np.square(shifted_backward) - np.square(middle_value_expanded)
            )
        )
        denominator = shifted_forward + np.sqrt(
            np.maximum(0, np.square(shifted_forward) - np.square(middle_value_expanded))
        )

        effective_mass = 0.5 * np.log(safe_divide_correlators(numerator, denominator))

    if squeeze_output:
        effective_mass = np.squeeze(effective_mass, axis=0)

    return effective_mass


# =============================================================================
# PCAC MASS SPECIFIC CALCULATIONS
# =============================================================================


def calculate_pcac_mass(
    g4g5g5_derivative: np.ndarray, g5g5_correlator: np.ndarray, pcac_factor: float = 0.5
) -> np.ndarray:
    """
    Calculate PCAC mass from derivative and correlator.

    Args:
        - g4g5g5_derivative: Derivative correlator array
        - g5g5_correlator: G5G5 correlator array (must be same shape)
        - pcac_factor: Multiplicative factor (default 0.5)

    Returns:
        PCAC mass array
    """
    return pcac_factor * safe_divide_correlators(g4g5g5_derivative, g5g5_correlator)


# =============================================================================
# VALIDATION UTILITIES FOR CORRELATOR DATA
# =============================================================================


def validate_correlator_dimensions(
    correlator: np.ndarray,
    expected_length: int,
    correlator_name: str,
    group_name: str = "unnamed",
) -> None:
    """
    Validate that a correlator has the expected temporal length.

    Args:
        - correlator: Correlator array to validate
        - expected_length: Expected length along time axis
        - correlator_name: Name of the correlator for error messages
        - group_name: Name of the group for error messages

    Raises:
        ValueError: If length doesn't match expectation
    """
    actual_length = correlator.shape[-1]  # Time is always last axis

    if actual_length != expected_length:
        raise ValueError(
            f"Group '{group_name}': {correlator_name} has temporal length {actual_length}, "
            f"expected {expected_length}"
        )


def check_correlator_physicality(
    correlator: np.ndarray,
    correlator_name: str,
    check_positive: bool = True,
    check_decreasing: bool = False,
    min_value: float = 1e-15,
) -> List[str]:
    """
    Check correlator for physical validity.

    Args:
        - correlator: Correlator array to check
        - correlator_name: Name for error messages
        - check_positive: Check that all values are positive
        - check_decreasing: Check that correlator decreases with time
        - min_value: Minimum allowed value

    Returns:
        List of issue descriptions (empty if no issues)
    """
    issues = []

    # Check for NaN or inf
    if np.any(np.isnan(correlator)):
        count = np.sum(np.isnan(correlator))
        issues.append(f"{correlator_name} contains {count} NaN values")

    if np.any(np.isinf(correlator)):
        count = np.sum(np.isinf(correlator))
        issues.append(f"{correlator_name} contains {count} infinite values")

    # Check positivity (common for correlators)
    if check_positive and np.any(correlator <= 0):
        count = np.sum(correlator <= 0)
        min_val = np.min(correlator)
        issues.append(
            f"{correlator_name} contains {count} non-positive values (min: {min_val:.2e})"
        )

    # Check minimum value threshold
    if np.any(np.abs(correlator) < min_value):
        count = np.sum(np.abs(correlator) < min_value)
        issues.append(
            f"{correlator_name} contains {count} values below threshold {min_value:.2e}"
        )

    # Check decreasing behavior (for g5g5 correlators)
    if check_decreasing and correlator.ndim >= 1:
        # Check along time axis (last axis)
        if correlator.shape[-1] > 1:
            # Check first half (before symmetrization effects)
            half_point = correlator.shape[-1] // 2
            for i in range(min(10, half_point - 1)):  # Check first 10 points
                if np.any(
                    correlator[..., i + 1] > correlator[..., i] * 1.01
                ):  # 1% tolerance
                    issues.append(
                        f"{correlator_name} is not monotonically decreasing at t={i+1}"
                    )
                    break

    return issues


# =============================================================================
# GROUP PROCESSING UTILITIES
# =============================================================================


def process_correlator_group(
    input_group: h5py.Group,
    output_group: h5py.Group,
    required_datasets: List[str],
    calculation_function: Callable,
    output_dataset_names: Dict[str, str],
    metadata_datasets: List[str],
    logger: Optional[Any] = None,
) -> None:
    """
    Generic processing function for a single correlator group.

    This template function handles the common pattern for both PCAC and
    effective mass calculations.

    Args:
        - input_group: Input HDF5 group
        - output_group: Output HDF5 group
        - required_datasets: List of required input dataset names
        - calculation_function: Function to calculate output from inputs
        - output_dataset_names: Mapping of output types to dataset names
        - metadata_datasets: List of metadata datasets to copy
        - logger: Optional logger instance
    """
    if input_group.name is None:
        raise ValueError("Input group must have a name")
    group_name = input_group.name

    try:
        # Read required datasets
        input_data = {}
        for dataset_name in required_datasets:
            if dataset_name not in input_group:
                raise KeyError(f"Required dataset '{dataset_name}' not found")

            # Check that it's actually a dataset, not a subgroup
            item = input_group[dataset_name]
            if not isinstance(item, h5py.Dataset):
                raise TypeError(
                    f"Expected '{dataset_name}' to be a dataset, "
                    f"but found {type(item).__name__}"
                )

            input_data[dataset_name] = item[:]

        # Validate consistency
        n_samples = validate_jackknife_consistency(input_data, group_name)

        if logger:
            logger.debug(f"Processing {group_name} with {n_samples} jackknife samples")

        # Apply calculation function
        result_samples = calculation_function(**input_data)

        # Calculate statistics
        mean_values, error_values = calculate_jackknife_statistics(result_samples)

        # Save results
        output_group.create_dataset(
            output_dataset_names["jackknife_samples"], data=result_samples
        )
        output_group.create_dataset(
            output_dataset_names["mean_values"], data=mean_values
        )
        output_group.create_dataset(
            output_dataset_names["error_values"], data=error_values
        )

        # Copy metadata
        for metadata_name in metadata_datasets:
            if metadata_name in input_group:
                item = input_group[metadata_name]
                if isinstance(item, h5py.Dataset):
                    output_group.create_dataset(metadata_name, data=item[:])

        # Copy group attributes
        for attr_name, attr_value in input_group.attrs.items():
            output_group.attrs[attr_name] = attr_value

        if logger:
            logger.debug(f"Successfully processed {group_name}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to process {group_name}: {e}")
        raise


# =============================================================================
# OUTPUT STRUCTURE HELPERS
# =============================================================================


def copy_parent_attributes(
    input_file: h5py.File, output_file: h5py.File, group_path: str
) -> None:
    """
    Copy attributes from parent groups to maintain hierarchy metadata.

    Args:
        - input_file: Input HDF5 file
        - output_file: Output HDF5 file
        - group_path: Path to the group being processed
    """
    # Split path into components
    path_parts = group_path.strip("/").split("/")

    # Copy attributes for each level of the hierarchy
    current_input_path = ""
    current_output_path = ""

    for i, part in enumerate(path_parts[:-1]):  # Exclude the deepest level
        current_input_path = (
            f"{current_input_path}/{part}" if current_input_path else part
        )
        current_output_path = (
            f"{current_output_path}/{part}" if current_output_path else part
        )

        # Create group if it doesn't exist
        if current_output_path not in output_file:
            output_file.create_group(current_output_path)

        # Copy attributes
        if current_input_path in input_file:
            input_group = input_file[current_input_path]
            output_group = output_file[current_output_path]

            for attr_name, attr_value in input_group.attrs.items():
                output_group.attrs[attr_name] = attr_value
