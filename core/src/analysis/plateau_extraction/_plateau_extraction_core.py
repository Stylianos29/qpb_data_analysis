#!/usr/bin/env python3
"""
Core utilities for plateau extraction.

This module provides shared functions for detecting and extracting
plateau values from time series data using jackknife analysis.
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
    copy_parent_attributes,
    copy_metadata,
)
from library.data.hdf5_analyzer import HDF5Analyzer
from library.constants import (
    PARAMETERS_WITH_EXPONENTIAL_FORMAT,
    PARAMETERS_OF_INTEGER_VALUE,
)

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
    error_values: np.ndarray,
    config_labels: List[str],
    sigma_thresholds: List[float],
    min_plateau_size: int,
    search_range: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Process a single group to extract plateau from jackknife samples.

    Args:
        - jackknife_samples: 2D array (n_samples x n_time) mean_values:
          1D
        - config_labels: List of configuration labels sigma_thresholds:
        - List of sigma thresholds to try min_plateau_size: Minimum
        - plateau size search_range: Search range configuration logger:
        - Logger instance

    Returns:
        Dictionary with extraction results
    """
    n_samples, _ = jackknife_samples.shape
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
    # TODO: Calculate the jackknife error properly
    plateau_gvar = gv.gvar(np.mean(plateau_values), np.std(plateau_values, ddof=1))

    # Use most common plateau bounds #TODO: The plateau bounds must be
    # common for all samples
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


def extract_parent_group_metadata(hdf5_file: h5py.File, group_path: str) -> Dict:
    """Extract shared metadata from parent group (second-to-deepest
    level)."""
    parent_path = os.path.dirname(group_path)

    if parent_path in hdf5_file:
        parent_group = hdf5_file[parent_path]
        if isinstance(parent_group, h5py.Group):
            return extract_group_metadata(parent_group)

    return {}


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
    parent_metadata: Dict,
    logger,
) -> Dict:
    """Process a single analysis group to extract plateau."""
    # Validate and load datasets
    try:
        jackknife_samples = load_dataset_array(group, input_datasets["samples"])
        mean_values = load_dataset_array(group, input_datasets["mean"])
        error_values = load_dataset_array(group, input_datasets["error"])
        config_labels = load_configuration_labels(group)
    except ValueError as e:
        return {"success": False, "error_message": str(e)}

    # Apply symmetrization and truncation if configured
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
        error_values,
        config_labels,
        sigma_thresholds,
        min_plateau_size,
        search_range,
    )

    result["group_name"] = group_name

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
                    logger.error(f"Invalid group at path: {group_path}")
                    continue
                if all(dataset in group for dataset in input_datasets.values()):
                    valid_groups.append(group_path)

            if not valid_groups:
                logger.warning("No groups found with required datasets")
                return results

            logger.info(f"Found {len(valid_groups)} groups to process")

            # Extract parent metadata once (using first group to
            # determine parent)
            parent_metadata = extract_parent_group_metadata(hdf5_file, valid_groups[0])

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
                    logger.warning(f"Failed to extract plateau for {group_name}")

        finally:
            analyzer.close()

    return results


# =============================================================================
# CSV EXPORT FUNCTIONS
# =============================================================================


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure priority columns come first in specified order."""
    priority_columns = ["Overlap_operator_method", "Kernel_operator_type"]

    # Only include priority columns that actually exist
    existing_priority = [col for col in priority_columns if col in df.columns]
    remaining_columns = [col for col in df.columns if col not in priority_columns]

    # Create new column order: priority first, then the rest
    new_column_order = existing_priority + remaining_columns

    # Reorder DataFrame
    return df.reindex(columns=new_column_order)


def _apply_export_formatting(df: pd.DataFrame) -> pd.DataFrame:
    """Apply proper formatting using existing constants."""
    available_columns = set(df.columns)

    for col in available_columns:
        if col in PARAMETERS_OF_INTEGER_VALUE:
            # Format as integers
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else "")

        elif col in PARAMETERS_WITH_EXPONENTIAL_FORMAT:
            # Format in exponential notation
            df[col] = df[col].apply(
                lambda x: f"{x:.1e}" if pd.notna(x) and x != 0 else "0.000000"
            )

    return df


def create_csv_record(
    result: Dict,
    output_column_prefix: str,
) -> Dict:
    """Create CSV record with only essential data."""
    # Start with ALL metadata (single-valued + multivalued parameters)
    record = result.get("metadata", {}).copy()

    if result["success"]:
        plateau_value = result["plateau_value"]
        # Plateau as tuple format: "(mean, uncertainty)" with 12 decimal
        # precision
        record[f"{output_column_prefix}_plateau"] = (
            f"({plateau_value.mean:.12f}, {plateau_value.sdev:.12f})"
        )

        # Extract failed configuration labels from sample results
        sample_results = result.get("diagnostics", {}).get("sample_results", [])
        failed_labels = []

        # Check if we have the full sample count vs successful count
        total_samples = result.get("n_samples", 0)
        if len(sample_results) < total_samples:
            # Some samples failed - we need to identify which ones
            successful_configs = {sr["config_label"] for sr in sample_results}
            # Note: We'd need access to original config_labels to
            # identify failed ones For now, just note that some failed
            failed_count = total_samples - len(sample_results)
            if failed_count > 0:
                failed_labels = [f"failed_samples_{failed_count}"]

        record["failed_config_labels"] = (
            ";".join(failed_labels) if failed_labels else ""
        )
    else:
        record[f"{output_column_prefix}_plateau"] = "NaN"
        record["failed_config_labels"] = result.get("error_message", "")

    return record


def export_to_csv(
    results: List[Dict],
    output_file: str,
    output_column_prefix: str,
    delimiter: str,
    logger,
) -> None:
    """Export extraction results to CSV file with proper formatting."""
    if not results:
        logger.warning("No results to export")
        return

    # Convert results to records
    records = [create_csv_record(result, output_column_prefix) for result in results]

    # Create DataFrame
    df = pd.DataFrame(records)

    # Apply column-specific formatting
    df = _apply_export_formatting(df)

    # Reorder columns
    df = _reorder_columns(df)

    # Save to CSV
    df.to_csv(
        output_file,
        index=False,
        sep=delimiter,
    )

    logger.info(f"Exported {len(records)} results to {output_file}")


# =============================================================================
# HDF5 EXPORT FUNCTIONS
# =============================================================================

# import os from datetime import datetime from typing import Dict, List
# import h5py import numpy as np

# from src.analysis.correlator_calculations._correlator_analysis_core
#     import ( copy_parent_attributes, copy_metadata, )


def _create_output_datasets_config(output_column_prefix: str) -> Dict[str, str]:
    """Create output dataset configuration for visualization."""
    return {
        "original_samples": f"{output_column_prefix}_time_series_samples",
        "plateau_estimates": f"{output_column_prefix}_plateau_estimates",
        "config_labels": "gauge_configuration_labels",
    }


def _map_results_to_groups(results: List[Dict]) -> Dict[str, Dict]:
    """Map successful results to their HDF5 group paths."""
    group_mapping = {}

    for result in results:
        if result["success"]:
            group_name = result["group_name"]
            group_mapping[group_name] = result

    return group_mapping


def _extract_visualization_data(
    input_group: h5py.Group,
    result: Dict,
    input_datasets: Dict[str, str],
    logger,
) -> tuple:
    """Extract data needed for visualization from input group and
    result."""
    try:
        # Load original time series data
        original_samples = load_dataset_array(input_group, input_datasets["samples"])
        config_labels = load_configuration_labels(input_group)

        # Extract individual plateau estimates from diagnostics
        sample_results = result["diagnostics"]["sample_results"]
        plateau_estimates = np.array([sr["plateau_value"] for sr in sample_results])

        # Ensure we have the right number of labels
        n_samples = len(sample_results)
        if len(config_labels) >= n_samples:
            config_labels = config_labels[:n_samples]
        else:
            # Pad with default labels if needed
            config_labels.extend(
                [f"sample_{i}" for i in range(len(config_labels), n_samples)]
            )

        return original_samples, plateau_estimates, config_labels

    except Exception as e:
        logger.error(f"Failed to extract visualization data: {e}")
        return None, None, None


def export_to_hdf5(
    results: List[Dict],
    input_hdf5_file: str,
    output_hdf5_file: str,
    input_datasets: Dict[str, str],
    output_column_prefix: str,
    logger,
) -> None:
    """Export plateau results to HDF5 for visualization."""
    if not results:
        logger.warning("No results to export to HDF5")
        return

    # Map results to group paths
    group_mapping = _map_results_to_groups(results)

    if not group_mapping:
        logger.warning("No successful results to export to HDF5")
        return

    # Get output dataset configuration
    output_datasets = _create_output_datasets_config(output_column_prefix)

    # Metadata datasets to copy (excluding ones we create ourselves)
    metadata_datasets = ["mpi_geometry_values", "qpb_log_filenames"]

    logger.info(f"Exporting {len(group_mapping)} successful results to HDF5")

    processed_parents = set()  # Track which parent groups we've processed

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
            f"Visualization-optimized HDF5 file containing original time series, "
            f"individual plateau estimates, and configuration labels for {output_column_prefix} analysis"
        )

        # Use HDF5Analyzer to find available groups (same as
        # process_all_groups)
        from library.data.hdf5_analyzer import HDF5Analyzer

        analyzer = HDF5Analyzer(input_hdf5_file)

        # Match full group paths with our results by basename
        available_groups = []
        for group_path in analyzer.active_groups:
            group_name = os.path.basename(group_path)
            if group_name in group_mapping:
                available_groups.append(group_path)
                group_mapping[group_name]["group_path"] = group_path

        analyzer.close()

        if not available_groups:
            logger.error("No matching group paths found between results and input file")
            return

        logger.info(f"Found {len(available_groups)} matching groups for export")

        # Process each group
        for group_path in available_groups:
            group_name = os.path.basename(group_path)
            result = group_mapping[group_name]

            logger.info(f"Processing group: {group_path}")

            # Get input group
            input_group = input_file[group_path]
            if not isinstance(input_group, h5py.Group):
                logger.warning(f"Skipping {group_path}: not a valid group")
                continue

            # Copy parent group attributes (second-to-deepest level)
            copy_parent_attributes(
                input_file, output_file, group_path, processed_parents
            )

            # Extract visualization data
            original_samples, plateau_estimates, config_labels = (
                _extract_visualization_data(input_group, result, input_datasets, logger)
            )

            if original_samples is None:
                logger.warning(
                    f"Skipping {group_path}: failed to extract visualization data"
                )
                continue

            # Create output group
            output_group = output_file.create_group(group_path)

            # Save visualization datasets
            output_group.create_dataset(
                output_datasets["original_samples"],
                data=original_samples,
                compression="gzip",
                compression_opts=6,
            )

            output_group.create_dataset(
                output_datasets["plateau_estimates"],
                data=plateau_estimates,
                compression="gzip",
                compression_opts=6,
            )

            # Save configuration labels as variable-length strings
            dt = h5py.string_dtype(encoding="utf-8")
            output_group.create_dataset(
                output_datasets["config_labels"],
                data=[
                    (
                        label.encode("utf-8")
                        if isinstance(label, str)
                        else str(label).encode("utf-8")
                    )
                    for label in config_labels
                ],
                dtype=dt,
            )

            # Copy essential metadata
            copy_metadata(input_group, output_group, metadata_datasets)

            # Add group-specific attributes
            output_group.attrs["plateau_extraction_success"] = True
            output_group.attrs["n_samples"] = len(plateau_estimates)
            output_group.attrs["n_time_points"] = original_samples.shape[1]

            plateau_stats = result["plateau_value"]
            output_group.attrs["plateau_mean"] = plateau_stats.mean
            output_group.attrs["plateau_error"] = plateau_stats.sdev
            output_group.attrs["sigma_threshold_used"] = result["sigma_threshold"]

            logger.info(f"Successfully exported group: {group_name}")

    logger.info(f"HDF5 export complete: {output_hdf5_file}")
