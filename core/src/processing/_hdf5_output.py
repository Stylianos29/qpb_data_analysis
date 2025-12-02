"""
HDF5 output module for jackknife analysis with processed parameter support.

This module creates custom HDF5 output files with proper structure and
processed parameter values from Stage 2A.
"""

from pathlib import Path
from typing import Dict, Optional, List
import os

import h5py
import numpy as np
import pandas as pd

# Import library components
from library.data.hdf5_analyzer import HDF5Analyzer

# Import from auxiliary modules
from src.processing._jackknife_config import (
    get_dataset_description,
)


def _create_filename_to_params_lookup(
    processed_csv_df: Optional[pd.DataFrame], logger
) -> Dict[str, Dict]:
    """
    Create dictionary mapping base filename (no extension) to processed parameters.

    This function strips the file extension from the Filename column in the CSV
    to enable matching with HDF5 group names that have different extensions.

    Args:
        processed_csv_df: DataFrame from processed_parameter_values.csv
        logger: Logger instance

    Returns:
        dict: {base_filename: {param_name: param_value, ...}}

    Example:
        Input CSV has: "KL_Brillouin_..._n1.txt"
        HDF5 has: "KL_Brillouin_..._n1.dat"
        Lookup key: "KL_Brillouin_..._n1"
    """
    if processed_csv_df is None:
        logger.warning("No processed parameters DataFrame provided")
        return {}

    lookup = {}
    for _, row in processed_csv_df.iterrows():
        # Strip extension (.txt) from CSV filename to match .dat in HDF5
        filename_with_ext = row["Filename"]
        base_filename = os.path.splitext(filename_with_ext)[0]

        # Store all parameter values for this file
        lookup[base_filename] = row.to_dict()

    logger.info(f"Created filename-to-parameters lookup with {len(lookup)} entries")
    return lookup


def _get_processed_parameters_for_group(
    group_name: str, filename_lookup: Dict[str, Dict], logger
) -> Optional[Dict]:
    """
    Get processed parameter values for an HDF5 group based on its filename.

    Strips the file extension from the HDF5 group name and looks up the
    corresponding processed parameters from the CSV-based lookup dictionary.

    Args:
        group_name: HDF5 group name (e.g., "KL_Brillouin_..._n1.dat")
        filename_lookup: Dictionary from _create_filename_to_params_lookup
        logger: Logger instance

    Returns:
        dict: Processed parameter values, or None if not found
    """
    if not filename_lookup:
        return None

    # Strip extension (.dat) from HDF5 group name
    base_filename = os.path.splitext(group_name)[0]

    if base_filename in filename_lookup:
        logger.debug(f"Found processed parameters for {base_filename}")
        return filename_lookup[base_filename]
    else:
        logger.warning(
            f"No processed parameters found for '{base_filename}' - using raw values"
        )
        return None


def _create_custom_hdf5_output(
    output_path: Path,
    all_processing_results: Dict,
    analyzer: HDF5Analyzer,
    processed_params_df: Optional[pd.DataFrame],
    compression: str,
    compression_level: int,
    logger,
) -> None:
    """
    Create custom HDF5 output file with proper structure and processed parameter values.

    This function writes jackknife analysis results to HDF5 while using processed
    parameter values from Stage 2A's CSV output instead of raw values from the
    input HDF5 file. This ensures consistency across all pipeline stages.

    Args:
        output_path: Path for output HDF5 file
        all_processing_results: Dictionary with all processing results
        analyzer: HDF5Analyzer instance for getting single-valued parameters
        processed_params_df: DataFrame from processed_parameter_values.csv (Stage 2A output)
        compression: Compression method
        compression_level: Compression level
        logger: Logger instance
    """
    # Prepare compression settings
    compression_opts = None if compression == "none" else compression_level
    final_compression = None if compression == "none" else compression

    # Create filename lookup for processed parameters
    filename_lookup = _create_filename_to_params_lookup(processed_params_df, logger)

    try:
        with h5py.File(output_path, "w") as output_file:
            # Recreate the same directory structure as input
            input_structure = _get_input_directory_structure(analyzer)

            # Create the group hierarchy
            parent_group = output_file
            for level_name in input_structure:
                parent_group = parent_group.create_group(level_name)

            # === ADD CONSTANT PARAMETERS (Second-to-deepest level) ===

            # Get constant parameters from processed CSV (first row since they're constant)
            if processed_params_df is not None:
                logger.info("Using constant parameters from processed CSV")
                first_row = processed_params_df.iloc[0]

                # Identify constant parameters (single-valued tunable parameters)
                for param_name in analyzer.list_of_tunable_parameter_names_from_hdf5:
                    if param_name in analyzer.unique_value_columns_dictionary:
                        # This is a constant parameter
                        if param_name in first_row:
                            parent_group.attrs[param_name] = first_row[param_name]
                            logger.debug(
                                f"Added constant parameter: {param_name} = {first_row[param_name]}"
                            )
            else:
                # Fallback: use raw values from analyzer
                logger.warning(
                    "No processed CSV - using raw constant parameters from HDF5"
                )
                single_valued_params = analyzer.unique_value_columns_dictionary
                for param_name, param_value in single_valued_params.items():
                    if param_name in analyzer.list_of_tunable_parameter_names_from_hdf5:
                        parent_group.attrs[param_name] = param_value

            # === CREATE JACKKNIFE ANALYSIS GROUPS ===

            for group_name, results in all_processing_results.items():
                jackknife_group = parent_group.create_group(group_name)

                # Get the first original HDF5 path from this group for parameter lookup
                # All paths in a group have the same parameter values
                group_paths = results.get("group_paths", [])
                sample_path = group_paths[0] if group_paths else group_name
                # Extract just the filename from the full path
                sample_filename = sample_path.split("/")[
                    -1
                ]  # e.g., "KL_Brillouin_...n7.dat"

                # Get processed parameters using the actual HDF5 filename
                processed_params = _get_processed_parameters_for_group(
                    sample_filename, filename_lookup, logger
                )

                # === ADD GROUP-SPECIFIC PARAMETERS (Deepest level) ===

                if processed_params:
                    # Use PROCESSED values from CSV
                    logger.info(f"Using processed parameters for group: {group_name}")

                    multivalued_params = (
                        analyzer.reduced_multivalued_tunable_parameter_names_list
                    )
                    # Always remove Configuration_label (it's handled separately)
                    actual_params = [
                        p for p in multivalued_params if p != "Configuration_label"
                    ]

                    for param_name in actual_params:
                        if param_name in processed_params:
                            jackknife_group.attrs[param_name] = processed_params[
                                param_name
                            ]
                            logger.debug(
                                f"  {param_name} = {processed_params[param_name]} (processed)"
                            )
                        else:
                            logger.warning(
                                f"Parameter {param_name} not found in processed CSV for {group_name}"
                            )

                else:
                    # FALLBACK: Use raw values from analyzer (current behavior)
                    logger.warning(f"Using raw parameters for group: {group_name}")

                    group_metadata = results["group_metadata"]
                    param_values = group_metadata.get("param_values", ())

                    multivalued_params = (
                        analyzer.reduced_multivalued_tunable_parameter_names_list
                    )
                    actual_params = [
                        p for p in multivalued_params if p != "Configuration_label"
                    ]

                    if isinstance(param_values, (tuple, list)) and len(
                        param_values
                    ) == len(actual_params):
                        for param_name, param_value in zip(actual_params, param_values):
                            jackknife_group.attrs[param_name] = param_value
                            logger.debug(f"  {param_name} = {param_value} (raw)")
                    elif (
                        not isinstance(param_values, (tuple, list))
                        and len(actual_params) == 1
                    ):
                        # Handle single parameter case
                        jackknife_group.attrs[actual_params[0]] = param_values
                        logger.debug(f"  {actual_params[0]} = {param_values} (raw)")

                # Add number of gauge configurations
                jackknife_results = results["jackknife_results"]
                n_configs = jackknife_results.get("n_gauge_configurations")
                if n_configs:
                    jackknife_group.attrs["Number_of_gauge_configurations"] = n_configs

                # Store jackknife results as datasets
                _store_jackknife_datasets(
                    jackknife_group,
                    jackknife_results,
                    final_compression,
                    compression_opts,
                    logger,
                )

                # Store metadata arrays
                config_metadata = results["config_metadata"]
                _store_metadata_arrays(
                    jackknife_group,
                    config_metadata,
                    final_compression,
                    compression_opts,
                    logger,
                )

        logger.info(f"Custom HDF5 output created: {output_path}")

    except Exception as e:
        logger.error(f"Failed to create custom HDF5 output: {e}")
        raise


def _get_input_directory_structure(analyzer: HDF5Analyzer) -> List[str]:
    """
    Extract the directory structure from the HDF5Analyzer.

    Args:
        analyzer: HDF5Analyzer instance

    Returns:
        List of group names representing the directory structure
    """
    # Get one of the active groups to determine structure
    if analyzer.active_groups:
        sample_group = list(analyzer.active_groups)[0]
        # Split the path and remove empty strings
        parts = [part for part in sample_group.split("/") if part]
        # Return all but the last part (which is the individual file group)
        return parts[:-1] if len(parts) > 1 else parts
    return []


def _store_jackknife_datasets(
    group: h5py.Group,
    jackknife_results: Dict,
    compression: Optional[str],
    compression_opts: Optional[int],
    logger,
) -> None:
    """
    Store jackknife analysis results as HDF5 datasets.

    Args:
        group: HDF5 group to store datasets in
        jackknife_results: Dictionary with jackknife results
        compression: Compression method
        compression_opts: Compression options
        logger: Logger instance
    """
    # Define which datasets to store
    dataset_names = [
        "g5g5_jackknife_samples",
        "g5g5_mean_values",
        "g5g5_error_values",
        "g4g5g5_jackknife_samples",
        "g4g5g5_mean_values",
        "g4g5g5_error_values",
        "g4g5g5_derivative_jackknife_samples",
        "g4g5g5_derivative_mean_values",
        "g4g5g5_derivative_error_values",
    ]

    for dataset_name in dataset_names:
        if dataset_name in jackknife_results:
            data = jackknife_results[dataset_name]

            # Create dataset with appropriate compression
            if compression and data.shape != ():  # No compression for scalars
                if compression == "lzf":
                    dataset = group.create_dataset(
                        dataset_name, data=data, compression=compression
                    )
                else:
                    dataset = group.create_dataset(
                        dataset_name,
                        data=data,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
            else:
                dataset = group.create_dataset(dataset_name, data=data)

            # Add description
            description = get_dataset_description(dataset_name)
            dataset.attrs["Description"] = description

            logger.debug(f"Stored dataset: {dataset_name} with shape {data.shape}")


def _store_metadata_arrays(
    group: h5py.Group,
    config_metadata: Dict,
    compression: Optional[str],
    compression_opts: Optional[int],
    logger,
) -> None:
    """
    Store configuration metadata as HDF5 datasets.

    Args:
        group: HDF5 group to store datasets in
        config_metadata: Dictionary with configuration metadata
        compression: Compression method
        compression_opts: Compression options
        logger: Logger instance
    """
    # Define metadata to store
    metadata_mappings = [
        ("configuration_labels", "gauge_configuration_labels"),
        ("qpb_filenames", "qpb_log_filenames"),
        ("mpi_geometries", "mpi_geometry_values"),
    ]

    for source_key, dataset_name in metadata_mappings:
        if source_key in config_metadata:
            data_list = config_metadata[source_key]

            logger.debug(f"Processing metadata {source_key}: {data_list}")

            if data_list:  # Only store if we have data
                # Convert to appropriate numpy array
                if source_key == "mpi_geometries":
                    # Handle potential None values and ensure string format
                    data_array = np.array(
                        [
                            (
                                str(val)
                                if val is not None and val != "unknown"
                                else "unknown"
                            )
                            for val in data_list
                        ],
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )
                else:
                    # For strings, use proper HDF5 string type
                    data_array = np.array(
                        data_list, dtype=h5py.string_dtype(encoding="utf-8")
                    )

                # Create dataset (strings usually don't benefit from compression)
                dataset = group.create_dataset(dataset_name, data=data_array)

                # Add description
                description = get_dataset_description(dataset_name)
                dataset.attrs["Description"] = description

                logger.debug(
                    f"Stored metadata: {dataset_name} with {len(data_array)} entries"
                )
            else:
                logger.warning(
                    f"Empty data for {source_key}, skipping dataset {dataset_name}"
                )
        else:
            logger.warning(
                f"Missing key {source_key} in config_metadata, skipping dataset {dataset_name}"
            )
