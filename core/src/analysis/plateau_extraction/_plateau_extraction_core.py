#!/usr/bin/env python3
"""
Core utilities for plateau extraction.

This module provides shared functions for detecting and extracting
plateau values from time series data using jackknife analysis.
"""

import os
from typing import Tuple, Dict, List, Optional, Any
import numpy as np
import pandas as pd
import h5py
import click
import gvar as gv

from library.data.hdf5_analyzer import HDF5Analyzer


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


def detect_plateau_region(
    time_series: np.ndarray,
    errors: np.ndarray,
    sigma_threshold: float,
    min_plateau_size: int,
    search_range: Dict[str, Any],
) -> Optional[Tuple[int, int]]:
    """
    Detect plateau region in time series using weighted range test.

    Args:
        time_series: 1D array of time series values errors: 1D array of
        corresponding errors sigma_threshold: Number of sigma for
        plateau criterion min_plateau_size: Minimum number of points in
        plateau search_range: Search range configuration

    Returns:
        (start_index, end_index) of plateau region, or None if not found
    """
    n_points = len(time_series)
    min_start = search_range.get("min_start", 0)
    max_end = search_range.get("max_end", -1)

    if max_end < 0:
        max_end = n_points + max_end

    # Search for plateau regions
    for start in range(min_start, max_end - min_plateau_size + 1):
        for end in range(start + min_plateau_size, max_end + 1):
            plateau_data = time_series[start:end]
            plateau_errors = errors[start:end]

            # Weighted range test
            weights = 1.0 / (plateau_errors**2)
            weighted_mean = np.sum(weights * plateau_data) / np.sum(weights)
            weighted_error = 1.0 / np.sqrt(np.sum(weights))

            # Check if all points are within sigma_threshold of weighted
            # mean
            deviations = np.abs(plateau_data - weighted_mean) / plateau_errors
            if np.all(deviations <= sigma_threshold):
                return (start, end)

    return None


def process_single_group(
    jackknife_samples: np.ndarray,
    mean_values: np.ndarray,
    error_values: np.ndarray,
    config_labels: List[str],
    sigma_thresholds: List[float],
    min_plateau_size: int,
    search_range: Dict[str, Any],
    logger,
) -> Dict[str, Any]:
    """
    Process a single group to extract plateau from jackknife samples.

    Args:
        jackknife_samples: 2D array (n_samples x n_time) mean_values: 1D
        array of mean values error_values: 1D array of error values
        config_labels: List of configuration labels sigma_thresholds:
        List of sigma thresholds to try min_plateau_size: Minimum
        plateau size search_range: Search range configuration logger:
        Logger instance

    Returns:
        Dictionary with extraction results
    """
    n_samples, n_time = jackknife_samples.shape
    sample_results = []

    # Try to extract plateau from each individual sample
    for i, sample_data in enumerate(jackknife_samples):
        sample_result = None
        for sigma in sigma_thresholds:
            plateau_bounds = detect_plateau_region(
                sample_data, error_values, sigma, min_plateau_size, search_range
            )
            if plateau_bounds is not None:
                start, end = plateau_bounds
                plateau_value = np.mean(sample_data[start:end])
                sample_result = {
                    "plateau_value": plateau_value,
                    "plateau_bounds": plateau_bounds,
                    "sigma_threshold": sigma,
                    "config_label": (
                        config_labels[i] if i < len(config_labels) else f"sample_{i}"
                    ),
                }
                break

        if sample_result is not None:
            sample_results.append(sample_result)

    # Check if we have enough successful extractions
    n_successful = len(sample_results)
    if n_successful < n_samples * 0.5:  # At least 50% success
        return {
            "success": False,
            "n_samples": n_successful,
            "error_message": f"Only {n_successful}/{n_samples} samples had successful plateau extraction",
        }

    # Calculate jackknife average of plateau values
    plateau_values = [sr["plateau_value"] for sr in sample_results]
    plateau_gvar = gv.gvar(np.mean(plateau_values), np.std(plateau_values, ddof=1))

    # Use most common plateau bounds
    bounds_list = [sr["plateau_bounds"] for sr in sample_results]
    most_common_bounds = max(set(bounds_list), key=bounds_list.count)

    return {
        "success": True,
        "plateau_value": plateau_gvar,
        "plateau_bounds": most_common_bounds,
        "n_samples": n_successful,
        "sigma_threshold": sample_results[0]["sigma_threshold"],  # Most stringent used
        "diagnostics": {
            "method": "jackknife_average",
            "sample_results": sample_results,
        },
    }


# =============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# =============================================================================


def load_dataset_array(group: h5py.Group, dataset_name: str) -> np.ndarray:
    """Validate and load dataset from HDF5 group as NumPy array."""
    if dataset_name not in group:
        raise ValueError(f"Missing dataset: {dataset_name}")

    dataset_obj = group[dataset_name]
    if not isinstance(dataset_obj, h5py.Dataset):
        raise ValueError(f"Object '{dataset_name}' is not a dataset")

    return dataset_obj[:]


def load_configuration_labels(group: h5py.Group) -> List[str]:
    """Load and decode configuration labels from HDF5 group."""
    if "gauge_configuration_labels" not in group:
        return []

    labels_obj = group["gauge_configuration_labels"]
    if not isinstance(labels_obj, h5py.Dataset):
        return []

    labels_data = labels_obj[:]
    return [
        label.decode("utf-8") if isinstance(label, bytes) else label
        for label in labels_data
    ]


def extract_group_metadata(group: h5py.Group) -> Dict:
    """Extract metadata from HDF5 group for CSV output."""
    metadata = {}

    # Get group attributes
    for key, value in group.attrs.items():
        # Convert numpy types to Python types for CSV
        if hasattr(value, "item"):
            metadata[key] = value.item()
        else:
            metadata[key] = value

    return metadata


def apply_preprocessing(
    jackknife_samples: np.ndarray,
    mean_values: np.ndarray,
    error_values: np.ndarray,
    apply_symmetrization: bool,
    symmetrization_truncation: bool,
    data_type: str,
    logger,
) -> tuple:
    """Apply symmetrization and truncation if configured."""
    if apply_symmetrization:
        logger.info(f"Applying symmetrization to {data_type} data")
        jackknife_samples = symmetrize_time_series(jackknife_samples)
        mean_values = symmetrize_time_series(mean_values)
        error_values = symmetrize_time_series(error_values)

        if symmetrization_truncation:
            half_length = len(mean_values) // 2
            jackknife_samples = jackknife_samples[:, :half_length]
            mean_values = mean_values[:half_length]
            error_values = error_values[:half_length]
            logger.info(f"Truncated to half length: {half_length} points")

    return jackknife_samples, mean_values, error_values


# =============================================================================
# GROUP PROCESSING FUNCTIONS
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
    logger,
) -> Dict:
    """Process a single analysis group to extract plateau."""
    try:
        # Validate and load datasets
        jackknife_samples = load_dataset_array(group, input_datasets["samples"])
        mean_values = load_dataset_array(group, input_datasets["mean"])
        error_values = load_dataset_array(group, input_datasets["error"])
        config_labels = load_configuration_labels(group)
    except ValueError as e:
        return {"success": False, "error_message": str(e)}

    # Apply preprocessing
    jackknife_samples, mean_values, error_values = apply_preprocessing(
        jackknife_samples,
        mean_values,
        error_values,
        apply_symmetrization,
        symmetrization_truncation,
        data_type,
        logger,
    )

    # Extract plateau
    result = process_single_group(
        jackknife_samples,
        mean_values,
        error_values,
        config_labels,
        sigma_thresholds,
        min_plateau_size,
        search_range,
        logger,
    )

    # Add metadata
    result["group_name"] = group_name
    result["metadata"] = extract_group_metadata(group)

    return result


def create_csv_record(
    result: Dict,
    output_column_prefix: str,
    time_offset: int,
    csv_config: Dict[str, Any],
) -> Dict:
    """Create CSV record from extraction result."""
    record = {}

    # Add metadata
    metadata = result.get("metadata", {})
    for key in [
        "bare_mass",
        "kappa",
        "clover_coefficient",
        "kernel_operator_type",
        "solver_type",
    ]:
        record[key] = metadata.get(key, "")

    if result["success"]:
        plateau_value = result["plateau_value"]
        plateau_bounds = result["plateau_bounds"]

        # Add extraction results with column prefix
        record[f"{output_column_prefix}_plateau_mean"] = plateau_value.mean
        record[f"{output_column_prefix}_plateau_error"] = plateau_value.sdev
        record[f"{output_column_prefix}_plateau_start_time"] = (
            plateau_bounds[0] + time_offset
        )
        record[f"{output_column_prefix}_plateau_end_time"] = (
            plateau_bounds[1] + time_offset
        )
        record[f"{output_column_prefix}_plateau_n_points"] = (
            plateau_bounds[1] - plateau_bounds[0]
        )

        # Add statistics
        record["n_successful_samples"] = result["n_samples"]
        record["n_total_samples"] = result["n_samples"]
        record["n_failed_samples"] = 0

        # Add diagnostics if configured
        if csv_config["include_diagnostics"]:
            record["estimation_method"] = result["diagnostics"]["method"]
            record["sigma_threshold_used"] = result["sigma_threshold"]
    else:
        # Failed extraction
        record[f"{output_column_prefix}_plateau_mean"] = np.nan
        record[f"{output_column_prefix}_plateau_error"] = np.nan
        record[f"{output_column_prefix}_plateau_start_time"] = np.nan
        record[f"{output_column_prefix}_plateau_end_time"] = np.nan
        record[f"{output_column_prefix}_plateau_n_points"] = np.nan
        record["n_successful_samples"] = 0
        record["n_total_samples"] = result.get("n_samples", 0)
        record["n_failed_samples"] = result.get("n_samples", 0)
        record["error_message"] = result.get("error_message", "Unknown error")

    return record


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
                    logger.error(f"Invalid group at path: {group_path}")
                    continue
                if all(dataset in group for dataset in input_datasets.values()):
                    valid_groups.append(group_path)

            if not valid_groups:
                logger.warning("No groups found with required datasets")
                return results

            logger.info(f"Found {len(valid_groups)} groups to process")

            # Process each group
            for group_path in valid_groups:
                group_name = os.path.basename(group_path)

                if verbose:
                    click.echo(f"Processing group: {group_name}")

                logger.info(f"Processing group: {group_path}")

                group = hdf5_file[group_path]
                if not isinstance(group, h5py.Group):
                    continue  # Already logged error above
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
                    logger,
                )
                results.append(result)

                if result["success"]:
                    logger.info(
                        f"Successfully extracted plateau for {group_name}: "
                        f"{result['plateau_value']}"
                    )
                else:
                    logger.warning(f"Failed to extract plateau for {group_name}")

        finally:
            analyzer.close()

    return results


def export_to_csv(
    results: List[Dict],
    output_file: str,
    output_column_prefix: str,
    time_offset: int,
    csv_config: Dict[str, Any],
    logger,
) -> None:
    """Export extraction results to CSV file."""
    if not results:
        logger.warning("No results to export")
        return

    # Convert results to records
    records = [
        create_csv_record(result, output_column_prefix, time_offset, csv_config)
        for result in results
    ]

    # Create DataFrame
    df = pd.DataFrame(records)

    # Save to CSV with configured precision
    df.to_csv(
        output_file,
        index=False,
        float_format=f"%.{csv_config['float_precision']}f",
        sep=csv_config["delimiter"],
    )

    logger.info(f"Exported {len(records)} results to {output_file}")
