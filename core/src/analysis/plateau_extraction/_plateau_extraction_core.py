#!/usr/bin/env python3
"""
Core utilities for plateau extraction.

This module provides shared functions for detecting and extracting
plateau values from time series data using jackknife analysis.

The correct procedure:
    1. Calculate jackknife average (with uncertainties) from all samples
    2. Detect plateau region on the averaged time series  
    3. Apply the common bounds to extract individual plateau values
    4. Calculate jackknife statistics from individual plateau values
"""

import os
from typing import Tuple, Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import h5py
import click
import gvar as gv

from src.analysis.correlator_calculations._correlator_analysis_core import (
    copy_metadata,
)
from library.data.hdf5_analyzer import HDF5Analyzer
from library.constants import (
    PARAMETERS_WITH_EXPONENTIAL_FORMAT,
    PARAMETERS_OF_INTEGER_VALUE,
)


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


class NoSuccessfulExtractionsError(Exception):
    """
    Raised when plateau extraction completes but no groups have
    successful results.

    This is not a programming error but a data quality limitation - for
    example, when the signal-to-noise ratio is too low for reliable
    plateau detection in all parameter groups.
    """

    pass


# =============================================================================
# LOW-LEVEL UTILITY FUNCTIONS
# =============================================================================


def symmetrize_time_series(data: np.ndarray) -> np.ndarray:
    """
    Symmetrize time series data: C_sym(t) = 0.5 * (C(t) + C(T-t)).

    Args:
        data: 1D or 2D array (samples x time for 2D)

    Returns:
        Symmetrized array of same shape

    Raises:
        ValueError: If data has unsupported dimensions
    """
    if data.ndim == 1:
        reverse = data[::-1]
        return 0.5 * (data + np.roll(reverse, shift=1))
    elif data.ndim == 2:
        # Handle 2D case (multiple samples)
        reverse = data[:, ::-1]
        return 0.5 * (data + np.roll(reverse, shift=1, axis=1))
    else:
        raise ValueError(
            f"Unsupported array dimensions: {data.ndim}. "
            "Only 1D and 2D arrays are supported."
        )


def calculate_jackknife_statistics(
    jackknife_samples: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and error from jackknife samples.

    Args:
        jackknife_samples: Array of shape (N_samples, N_time)

    Returns:
        Tuple of (mean_values, error_values)
    """
    n_samples = jackknife_samples.shape[0]
    mean_values = np.mean(jackknife_samples, axis=0)

    # Jackknife error formula
    deviations = jackknife_samples - mean_values
    variance = np.mean(deviations**2, axis=0)
    error_values = np.sqrt((n_samples - 1) * variance)

    return mean_values, error_values


def calculate_jackknife_average_with_covariance(
    jackknife_samples: np.ndarray,
) -> np.ndarray:
    """
    Calculate jackknife average with full covariance matrix.

    Args:
        jackknife_samples: Array of shape (N_samples, N_time)

    Returns:
        Array of gvar objects with jackknife means and covariances
    """
    n_samples, n_time = jackknife_samples.shape

    # Calculate mean
    jk_mean = np.mean(jackknife_samples, axis=0)

    # Calculate covariance matrix
    covariance_matrix = np.zeros((n_time, n_time))
    for i in range(n_time):
        for j in range(n_time):
            values_i = jackknife_samples[:, i]
            values_j = jackknife_samples[:, j]
            mean_i = jk_mean[i]
            mean_j = jk_mean[j]
            cov_ij = (
                (n_samples - 1)
                / n_samples
                * np.sum((values_i - mean_i) * (values_j - mean_j))
            )
            covariance_matrix[i, j] = cov_ij

    # Create gvar array with full covariance
    return gv.gvar(jk_mean, covariance_matrix)


def detect_plateau_region_weighted_range(
    time_series_gvar: np.ndarray,
    sigma_threshold: float,
    min_plateau_size: int,
    search_range: Dict[str, Any],
) -> Optional[Tuple[int, int]]:
    """
    Detect plateau region using weighted range test on gvar time series.

    Args:
        - time_series_gvar: Array of gvar objects (from jackknife
          average)
        - sigma_threshold: Number of sigma for plateau criterion
        - min_plateau_size: Minimum number of points in plateau
        - search_range: Search range configuration

    Returns:
        (start_index, end_index) of plateau region, or None if not found
    """
    n_points = len(time_series_gvar)
    min_start = search_range.get("min_start", 0)
    max_end = search_range.get("max_end", -1)

    if max_end < 0:
        max_end = n_points + max_end

    # Extract means and standard deviations
    means = np.array([val.mean for val in time_series_gvar])
    stds = np.array([val.sdev for val in time_series_gvar])

    # Search for plateau regions
    for start in range(min_start, max_end - min_plateau_size + 1):
        for end in range(start + min_plateau_size, max_end + 1):
            plateau_means = means[start:end]
            plateau_stds = stds[start:end]

            # Weighted range test
            weights = 1.0 / (plateau_stds**2 + 1e-12)  # Avoid division by zero
            weighted_mean = np.sum(weights * plateau_means) / np.sum(weights)
            weighted_error = 1.0 / np.sqrt(np.sum(weights))

            # Check if all points are within sigma_threshold of weighted
            # mean
            deviations = np.abs(plateau_means - weighted_mean) / plateau_stds
            if np.all(deviations <= sigma_threshold):
                return (start, end)

    return None


# =============================================================================
# MAIN PLATEAU EXTRACTION FUNCTIONS
# =============================================================================


def extract_plateau_from_jackknife_samples(
    jackknife_samples: np.ndarray,
    config_labels: List[str],
    sigma_thresholds: List[float],
    min_plateau_size: int,
    search_range: Dict[str, Any],
    group_name: str,
    logger,
) -> Dict[str, Any]:
    """
    Extract plateau values using the CORRECT statistical procedure:
        1. Calculate jackknife average with covariances
        2. Detect plateau on averaged (less noisy) time series
        3. Apply common bounds to extract individual plateau values
        4. Calculate final jackknife statistics

    Args:
        - jackknife_samples: Array of shape (N_samples, N_time)
        - config_labels: List of configuration labels
        - sigma_thresholds: List of sigma thresholds to try
        - min_plateau_size: Minimum plateau size
        - search_range: Search range configuration
        - group_name: Name for logging
        - logger: Logger instance

    Returns:
        Dictionary with extraction results
    """
    n_samples, n_time = jackknife_samples.shape

    logger.info(
        f"Group {group_name}: Starting plateau extraction on {n_samples} samples"
    )

    # Step 1: Calculate jackknife average with full covariance
    try:
        gvar_time_series = calculate_jackknife_average_with_covariance(
            jackknife_samples
        )
        logger.debug(
            f"Group {group_name}: Calculated jackknife average with covariance"
        )
    except Exception as e:
        logger.error(f"Group {group_name}: Failed to calculate jackknife average: {e}")
        return {
            "success": False,
            "group_name": group_name,
            "error_message": f"Failed to calculate jackknife average: {e}",
            "n_total_samples": n_samples,
        }

    # Step 2: Detect plateau region on the averaged time series
    plateau_bounds = None
    sigma_used = None

    for sigma_threshold in sigma_thresholds:
        try:
            plateau_bounds = detect_plateau_region_weighted_range(
                gvar_time_series, sigma_threshold, min_plateau_size, search_range
            )

            if plateau_bounds is not None:
                sigma_used = sigma_threshold
                logger.info(
                    f"Group {group_name}: Plateau detected at σ={sigma_threshold}, "
                    f"bounds=[{plateau_bounds[0]}:{plateau_bounds[1]}]"
                )
                break

        except Exception as e:
            logger.debug(
                f"Group {group_name}: Plateau detection failed at σ={sigma_threshold}: {e}"
            )
            continue

    if plateau_bounds is None:
        max_sigma = max(sigma_thresholds)
        logger.warning(
            f"Group {group_name}: No plateau found with any σ threshold (max={max_sigma})"
        )
        return {
            "success": False,
            "group_name": group_name,
            "error_message": f"No plateau found with any sigma threshold (max={max_sigma})",
            "n_total_samples": n_samples,
        }

    plateau_start, plateau_end = plateau_bounds
    n_plateau_points = plateau_end - plateau_start

    # Step 3: Extract individual plateau values using common bounds
    individual_plateau_values = []
    individual_sigma_thresholds = []
    successful_config_labels = []
    successful_sample_indices = []

    for i, sample_data in enumerate(jackknife_samples):
        try:
            plateau_region = sample_data[plateau_start:plateau_end]

            # Check for invalid values
            if not np.all(np.isfinite(plateau_region)):
                logger.debug(
                    f"Group {group_name}: Sample {i} has invalid values in plateau region"
                )
                continue

            # Calculate plateau value for this sample
            individual_plateau = np.mean(plateau_region)
            individual_plateau_values.append(individual_plateau)

            # Store sigma threshold used (same for all successful
            # samples)
            individual_sigma_thresholds.append(sigma_used)

            # Keep track of successful indices and config labels
            successful_sample_indices.append(i)
            if i < len(config_labels):
                successful_config_labels.append(config_labels[i])
            else:
                successful_config_labels.append(f"sample_{i:03d}")

        except Exception as e:
            logger.debug(
                f"Group {group_name}: Failed to extract plateau for sample {i}: {e}"
            )
            continue

    n_successful = len(individual_plateau_values)
    n_failed = n_samples - n_successful

    # Check if we have enough successful extractions
    min_success_rate = 0.5  # At least 50% must succeed
    if n_successful < n_samples * min_success_rate:
        logger.warning(
            f"Group {group_name}: Insufficient successful extractions "
            f"({n_successful}/{n_samples} = {n_successful/n_samples:.1%})"
        )
        return {
            "success": False,
            "group_name": group_name,
            "error_message": f"Only {n_successful}/{n_samples} successful plateau extractions",
            "n_total_samples": n_samples,
            "n_successful": n_successful,
            "n_failed": n_failed,
            "plateau_bounds": plateau_bounds,
            "sigma_threshold": sigma_used,
        }

    # Step 4: Calculate final jackknife statistics from individual
    # plateau values
    plateau_values_array = np.array(individual_plateau_values)
    plateau_mean = np.mean(plateau_values_array)

    # Proper jackknife error calculation
    plateau_variance = (
        (n_successful - 1)
        / n_successful
        * np.sum((plateau_values_array - plateau_mean) ** 2)
    )
    plateau_error = np.sqrt(plateau_variance)

    # Create gvar result
    plateau_gvar = gv.gvar(plateau_mean, plateau_error)

    # Extract successful time series for HDF5 export (processed data
    # only)
    successful_time_series = jackknife_samples[successful_sample_indices, :]

    logger.info(
        f"Group {group_name}: Success! Plateau = {plateau_gvar}, "
        f"{n_successful}/{n_samples} samples, σ={sigma_used}"
    )

    return {
        "success": True,
        "group_name": group_name,
        "plateau_value": plateau_gvar,
        "plateau_bounds": plateau_bounds,
        "plateau_mean": plateau_mean,
        "plateau_error": plateau_error,
        "sigma_threshold": sigma_used,
        "n_total_samples": n_samples,
        "n_successful": n_successful,
        "n_failed": n_failed,
        "n_plateau_points": n_plateau_points,
        "individual_plateau_values": plateau_values_array,
        "individual_sigma_thresholds": np.array(individual_sigma_thresholds),
        "successful_config_labels": successful_config_labels,
        "successful_sample_indices": successful_sample_indices,
        "successful_time_series": successful_time_series,  # Processed data for HDF5
        "estimation_method": "jackknife_with_common_bounds",
    }


# =============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# =============================================================================


def load_dataset_array(group: h5py.Group, dataset_name: str) -> np.ndarray:
    """Validate and load dataset from HDF5 group as NumPy array."""
    if dataset_name not in group:
        raise ValueError(f"Dataset '{dataset_name}' not found in group")

    dataset = group[dataset_name]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"'{dataset_name}' is not an HDF5 dataset")

    data = dataset[()]
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    return data


def load_configuration_labels(group: h5py.Group) -> List[str]:
    """Load configuration labels from HDF5 group."""
    labels = []

    if "gauge_configuration_labels" in group:
        gauge_configuration_labels_dataset = group["gauge_configuration_labels"]
        if not isinstance(gauge_configuration_labels_dataset, h5py.Dataset):
            raise ValueError(
                f"'{gauge_configuration_labels_dataset}' is not an HDF5 dataset"
            )
        labels_data = gauge_configuration_labels_dataset[()]
        if isinstance(labels_data, np.ndarray):
            labels = [
                label.decode("utf-8") if isinstance(label, bytes) else str(label)
                for label in labels_data
            ]
        else:
            # Single label
            label = (
                labels_data.decode("utf-8")
                if isinstance(labels_data, bytes)
                else str(labels_data)
            )
            labels = [label]

    return labels


def apply_preprocessing(
    jackknife_samples: np.ndarray,
    mean_values: np.ndarray,
    error_values: np.ndarray,
    apply_symmetrization: bool,
    symmetrization_truncation: bool,
    data_type: str,
    logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply preprocessing steps (symmetrization, truncation) to time
    series data.

    Args:
        - jackknife_samples: Shape (N_samples, N_time)
        - mean_values: Shape (N_time,)
        - error_values: Shape (N_time,)
        - apply_symmetrization: Whether to symmetrize
        - symmetrization_truncation: Whether to truncate after
          symmetrization
        - data_type: "PCAC mass" or "pion effective mass" (for logging)
        - logger: Logger instance

    Returns:
        Tuple of processed (jackknife_samples, mean_values,
        error_values)
    """
    original_time_points = jackknife_samples.shape[1]

    if apply_symmetrization:
        logger.info(f"Applying symmetrization to {data_type} data")

        # Symmetrize all arrays
        jackknife_samples = symmetrize_time_series(jackknife_samples)
        mean_values = symmetrize_time_series(mean_values)
        error_values = symmetrize_time_series(error_values)

        if symmetrization_truncation:
            # CRITICAL: For PCAC mass, truncate to T/2 after
            # symmetrization
            truncate_length = original_time_points // 2
            logger.info(
                f"Truncating {data_type} to T/2 = {truncate_length} points after symmetrization"
            )

            jackknife_samples = jackknife_samples[:, :truncate_length]
            mean_values = mean_values[:truncate_length]
            error_values = error_values[:truncate_length]

    final_time_points = jackknife_samples.shape[1]
    logger.info(
        f"{data_type} preprocessing: {original_time_points} → {final_time_points} time points"
    )

    return jackknife_samples, mean_values, error_values


def extract_group_metadata(group: h5py.Group) -> Dict[str, Any]:
    """Extract metadata from HDF5 group."""
    metadata = {}

    # Validate that group is indeed an HDF5 group
    if not isinstance(group, h5py.Group):
        raise ValueError(f"Expected h5py.Group, got {type(group)}")

    # Extract all group attributes
    for key, value in group.attrs.items():
        metadata[key] = value

    # Extract key metadata datasets if present
    metadata_keys = ["Number_of_gauge_configurations"]
    for key in metadata_keys:
        if key in group:
            dataset = group[key]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"'{dataset}' is not an HDF5 dataset")
            data = dataset[()]
            metadata[key] = data.item() if hasattr(data, "item") else data

    return metadata


def extract_parent_group_metadata(
    hdf5_file: h5py.File, sample_group_path: str
) -> Dict[str, Any]:
    """Extract metadata from parent group."""
    parent_metadata = {}

    # Navigate to parent group
    path_parts = sample_group_path.split("/")
    if len(path_parts) > 1:
        parent_path = "/".join(path_parts[:-1])
        if parent_path in hdf5_file:
            parent_group = hdf5_file[parent_path]
            for key, value in parent_group.attrs.items():
                parent_metadata[key] = value

    return parent_metadata


# =============================================================================
# HIGH-LEVEL PROCESSING FUNCTIONS
# =============================================================================


def process_analysis_group(
    group: h5py.Group,
    group_name: str,
    input_datasets: Dict[str, str],
    apply_symmetrization: bool,
    symmetrization_truncation: bool,
    sigma_thresholds: List[float],
    min_plateau_size: int,
    search_range: Dict[str, Any],
    data_type: str,
    parent_metadata: Dict[str, Any],
    logger,
) -> Dict[str, Any]:
    """
    Process a single analysis group to extract plateau values.

    Args:
        - group: HDF5 group containing the data
        - group_name: Name of the group for logging
        - input_datasets: Dataset name mapping
        - apply_symmetrization: Whether to apply symmetrization
        - symmetrization_truncation: Whether to truncate after
          symmetrization
        - sigma_thresholds: List of sigma thresholds to try
        - min_plateau_size: Minimum plateau size
        - search_range: Search range configuration
        - data_type: Type of data being processed (for logging)
        - parent_metadata: Metadata from parent group
        - logger: Logger instance

    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing group: {group_name}")

    # Validate and load datasets
    try:
        jackknife_samples = load_dataset_array(group, input_datasets["samples"])
        mean_values = load_dataset_array(group, input_datasets["mean"])
        error_values = load_dataset_array(group, input_datasets["error"])
        config_labels = load_configuration_labels(group)
    except Exception as e:
        logger.error(f"Group {group_name}: Failed to load data: {e}")
        return {
            "success": False,
            "group_name": group_name,
            "error_message": f"Data loading error: {e}",
        }

    # Apply preprocessing
    try:
        jackknife_samples, mean_values, error_values = apply_preprocessing(
            jackknife_samples,
            mean_values,
            error_values,
            apply_symmetrization,
            symmetrization_truncation,
            data_type,
            logger,
        )
    except Exception as e:
        logger.error(f"Group {group_name}: Failed to preprocess data: {e}")
        return {
            "success": False,
            "group_name": group_name,
            "error_message": f"Preprocessing error: {e}",
        }

    # Extract plateau
    result = extract_plateau_from_jackknife_samples(
        jackknife_samples,
        config_labels,
        sigma_thresholds,
        min_plateau_size,
        search_range,
        group_name,
        logger,
    )

    # Add metadata
    group_metadata = extract_group_metadata(group)
    combined_metadata = {**parent_metadata, **group_metadata}
    result["metadata"] = combined_metadata

    return result


def process_all_groups(
    input_file: str,
    input_datasets: Dict[str, str],
    apply_symmetrization: bool,
    symmetrization_truncation: bool,
    sigma_thresholds: List[float],
    min_plateau_size: int,
    search_range: Dict[str, Any],
    data_type: str,
    logger,
    verbose: bool,
) -> List[Dict]:
    """Process all analysis groups in the HDF5 file."""
    results = []

    with h5py.File(input_file, "r") as hdf5_file:
        # Find groups with required datasets
        analyzer = HDF5Analyzer(input_file)

        try:
            valid_groups = []
            for group_path in analyzer.active_groups:
                group = hdf5_file[group_path]
                if not isinstance(group, h5py.Group):
                    logger.debug(f"Skipping non-group: {group_path}")
                    continue
                if all(dataset in group for dataset in input_datasets.values()):
                    valid_groups.append(group_path)

            if not valid_groups:
                logger.warning("No groups found with required datasets")
                return results

            logger.info(f"Found {len(valid_groups)} groups to process")

            # Extract parent metadata once
            parent_metadata = extract_parent_group_metadata(hdf5_file, valid_groups[0])

            # Process each group
            for group_path in valid_groups:
                group_name = os.path.basename(group_path)

                if verbose:
                    click.echo(f"Processing group: {group_name}")

                logger.info(f"Processing group: {group_path}")

                group = hdf5_file[group_path]
                # Validate that group is indeed an HDF5 group
                if not isinstance(group, h5py.Group):
                    raise ValueError(f"Expected h5py.Group, got {type(group)}")

                result = process_analysis_group(
                    group,
                    group_name,
                    input_datasets,
                    apply_symmetrization,
                    symmetrization_truncation,
                    sigma_thresholds,
                    min_plateau_size,
                    search_range,
                    data_type,
                    parent_metadata,
                    logger,
                )
                results.append(result)

                if result["success"]:
                    logger.info(
                        f"Successfully extracted plateau for {group_name}: "
                        f"{result['plateau_value']}"
                    )
                else:
                    logger.warning(
                        f"Failed to extract plateau for {group_name}: "
                        f"{result.get('error_message', 'Unknown error')}"
                    )

        finally:
            analyzer.close()

    return results


# =============================================================================
# CSV EXPORT FUNCTIONS
# =============================================================================


def export_to_csv(
    results: List[Dict],
    output_csv_path: str,
    output_column_prefix: str,
    delimiter: str,
    logger,
) -> None:
    """Export extraction results to CSV file."""
    if not results:
        logger.warning("No results to export to CSV")
        return

    # Filter successful results
    successful_results = [r for r in results if r.get("success", False)]

    if not successful_results:
        logger.warning("No successful results to export to CSV")
        return

    logger.info(f"Exporting {len(successful_results)} successful results to CSV")

    # Prepare data for CSV
    csv_data = []

    for result in successful_results:
        row_data = {}

        # Add metadata
        metadata = result.get("metadata", {})
        for key, value in metadata.items():
            # Format special parameters
            if key in PARAMETERS_WITH_EXPONENTIAL_FORMAT:
                row_data[key] = f"{value:.2e}"
            elif key in PARAMETERS_OF_INTEGER_VALUE:
                row_data[key] = int(value) if not np.isnan(float(value)) else value
            else:
                row_data[key] = value

        # Add extraction results
        row_data[f"{output_column_prefix}_plateau_mean"] = result["plateau_mean"]
        row_data[f"{output_column_prefix}_plateau_error"] = result["plateau_error"]
        row_data[f"{output_column_prefix}_plateau_start_time"] = result[
            "plateau_bounds"
        ][0]
        row_data[f"{output_column_prefix}_plateau_end_time"] = result["plateau_bounds"][
            1
        ]
        row_data[f"{output_column_prefix}_plateau_n_points"] = result[
            "n_plateau_points"
        ]
        row_data[f"{output_column_prefix}_n_successful_samples"] = result[
            "n_successful"
        ]
        row_data[f"{output_column_prefix}_n_total_samples"] = result["n_total_samples"]
        row_data[f"{output_column_prefix}_n_failed_samples"] = result["n_failed"]
        row_data[f"{output_column_prefix}_sigma_threshold_used"] = result[
            "sigma_threshold"
        ]
        row_data[f"{output_column_prefix}_estimation_method"] = result[
            "estimation_method"
        ]

        csv_data.append(row_data)

    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv_path, index=False, sep=delimiter, float_format=f"%.6f")

    logger.info(f"CSV export complete: {output_csv_path}")


# =============================================================================
# HDF5 EXPORT FUNCTIONS
# =============================================================================


def _copy_parent_group_structure(
    input_file: h5py.File,
    output_file: h5py.File,
    parent_path: str,
    logger,
) -> None:
    """Copy parent group structure and attributes to output file."""
    try:
        if parent_path in input_file:
            input_parent = input_file[parent_path]

            # Create parent group in output file
            output_parent = output_file.require_group(parent_path)

            # Copy all parent group attributes (the constant
            # parameters!)
            for attr_name, attr_value in input_parent.attrs.items():
                try:
                    output_parent.attrs[attr_name] = attr_value
                except (TypeError, ValueError) as e:
                    logger.warning(f"Could not copy attribute {attr_name}: {e}")

            logger.debug(
                f"Copied {len(input_parent.attrs)} attributes from parent group: {parent_path}"
            )

    except Exception as e:
        logger.error(f"Error copying parent group structure for {parent_path}: {e}")


def export_to_hdf5(
    results: List[Dict],
    input_hdf5_file: str,
    output_hdf5_file: str,
    input_datasets: Dict[str, str],
    output_column_prefix: str,
    logger,
) -> None:
    """Export results to HDF5 format optimized for visualization."""
    if not results:
        logger.warning("No results to export to HDF5")
        return

    # Filter successful results only
    successful_results = [r for r in results if r.get("success", False)]

    if not successful_results:
        logger.warning("No successful results to export to HDF5")
        return

    logger.info(f"Exporting {len(successful_results)} successful results to HDF5")

    # Create output datasets configuration
    output_datasets = {
        "time_series": f"{output_column_prefix}_time_series_samples",
        "plateau_estimates": f"{output_column_prefix}_plateau_estimates",
        "sigma_thresholds": f"{output_column_prefix}_individual_sigma_thresholds",
        "config_labels": "gauge_configuration_labels",
    }

    with h5py.File(input_hdf5_file, "r") as input_file, h5py.File(
        output_hdf5_file, "w"
    ) as output_file:

        # Add file-level documentation
        output_file.attrs["analysis_type"] = (
            f"{output_column_prefix} plateau extraction results"
        )
        output_file.attrs["analysis_date"] = datetime.now().isoformat()
        output_file.attrs["source_file"] = os.path.basename(input_hdf5_file)
        output_file.attrs["description"] = (
            f"Visualization-optimized HDF5 file containing processed time series, "
            f"individual plateau estimates, sigma thresholds, and configuration labels for {output_column_prefix} analysis"
        )

        # Use HDF5Analyzer to get structure
        analyzer = HDF5Analyzer(input_hdf5_file)

        # Map results to group paths
        result_mapping = {r["group_name"]: r for r in successful_results}

        # Track processed parent groups to avoid duplication
        processed_parents = set()

        # Process each successful result
        for group_path in analyzer.active_groups:
            group_name = os.path.basename(group_path)

            if group_name not in result_mapping:
                continue  # Skip unsuccessful groups

            result = result_mapping[group_name]
            input_group = input_file[group_path]

            # Copy parent group structure and attributes
            parent_path = "/".join(group_path.split("/")[:-1])
            if parent_path and parent_path not in processed_parents:
                _copy_parent_group_structure(
                    input_file, output_file, parent_path, logger
                )
                processed_parents.add(parent_path)

            # Create corresponding output group structure
            output_group = output_file.require_group(group_path)

            # Use processed time series data (already
            # symmetrized/truncated)
            processed_time_series = result["successful_time_series"]

            # Get individual plateau estimates and sigma thresholds
            individual_estimates = result["individual_plateau_values"]
            individual_sigmas = result["individual_sigma_thresholds"]

            # Get successful config labels
            successful_labels = result["successful_config_labels"]

            logger.debug(
                f"Group {group_name}: Exporting {len(individual_estimates)} successful samples, "
                f"time series shape: {processed_time_series.shape}"
            )

            # Save processed time series samples (smaller file size)
            output_group.create_dataset(
                output_datasets["time_series"],
                data=processed_time_series,
                compression="gzip",
                compression_opts=6,
            )

            # Save individual plateau estimates
            output_group.create_dataset(
                output_datasets["plateau_estimates"],
                data=individual_estimates,
                compression="gzip",
                compression_opts=6,
            )

            # Save individual sigma thresholds
            output_group.create_dataset(
                output_datasets["sigma_thresholds"],
                data=individual_sigmas,
                compression="gzip",
                compression_opts=6,
            )

            # Save configuration labels
            dt = h5py.string_dtype(encoding="utf-8")
            output_group.create_dataset(
                output_datasets["config_labels"],
                data=[label.encode("utf-8") for label in successful_labels],
                dtype=dt,
            )

            # Copy essential metadata
            metadata_datasets = ["mpi_geometry_values", "qpb_log_filenames"]
            copy_metadata(input_group, output_group, metadata_datasets)

            # Add plateau extraction attributes
            output_group.attrs["plateau_extraction_success"] = True
            output_group.attrs["n_samples"] = len(individual_estimates)
            output_group.attrs["n_time_points"] = processed_time_series.shape[1]
            output_group.attrs["plateau_mean"] = result["plateau_mean"]
            output_group.attrs["plateau_error"] = result["plateau_error"]
            output_group.attrs["plateau_start"] = result["plateau_bounds"][0]
            output_group.attrs["plateau_end"] = result["plateau_bounds"][1]
            output_group.attrs["sigma_threshold_used"] = result["sigma_threshold"]

            logger.debug(f"Exported group: {group_name}")

        analyzer.close()

    logger.info(f"HDF5 export complete: {output_hdf5_file}")
