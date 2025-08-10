#!/usr/bin/env python3
"""
Jackknife analysis script for QPB correlator data preprocessing.

This script applies jackknife resampling to correlator data stored in
HDF5 format, focusing on processing tasks:
    - Jackknife resampling of g5-g5 correlators
    - Jackknife resampling of g4g5-g5 correlators
    - Calculation of g4g5-g5 derivative correlators using finite
      differences
    - Export of jackknife samples, means, and errors in clean HDF5
      format

The script uses HDF5Analyzer for modern data handling and maintains the
same hierarchical structure as the input file.

Usage:
    python apply_jackknife_analysis.py -i input.h5 -o output.h5
    [options]
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, List

import click
import numpy as np
import h5py
import re

# Import library components
from library.data.hdf5_analyzer import HDF5Analyzer
from library import (
    validate_input_directory,
    validate_input_script_log_filename,
    filesystem_utilities,
)
from library.constants import PARAMETERS_WITH_EXPONENTIAL_FORMAT, PARAMETER_LABELS

# Import our auxiliary modules
from src.processing._jackknife_config import (
    DEFAULT_DERIVATIVE_METHOD,
    DEFAULT_COMPRESSION,
    DEFAULT_COMPRESSION_LEVEL,
    EXCLUDED_FROM_GROUPING,
    INPUT_CORRELATOR_DATASETS,
    REQUIRED_INPUT_DATASETS,
    MIN_GAUGE_CONFIGURATIONS,
    get_dataset_description,
)

from src.processing._jackknife_processor import (
    JackknifeProcessor,
    extract_configuration_metadata,
)


@click.command()
@click.option(
    "-i",
    "--input_hdf5_file",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Path to input HDF5 file containing correlator data and parameters.",
)
@click.option(
    "-o",
    "--output_hdf5_file",
    required=True,
    type=click.Path(),
    help="Path for output HDF5 file with jackknife analysis results.",
)
@click.option(
    "-out_dir",
    "--output_directory",
    default=None,
    callback=validate_input_directory,
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
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    help="Directory for log files. Default: output directory",
)
@click.option(
    "-log_name",
    "--log_filename",
    default=None,
    callback=validate_input_script_log_filename,
    help="Custom name for log file. Default: auto-generated",
)
@click.option(
    "--min_configurations",
    default=MIN_GAUGE_CONFIGURATIONS,
    type=click.IntRange(2, None),
    help=(
        f"Minimum gauge configurations required. "
        f"Default: {MIN_GAUGE_CONFIGURATIONS}"
    ),
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose console output.",
)
def main(
    input_hdf5_file: str,
    output_hdf5_file: str,
    output_directory: Optional[str],
    enable_logging: bool,
    log_directory: Optional[str],
    log_filename: Optional[str],
    min_configurations: int,
    verbose: bool,
) -> None:
    """
    Apply jackknife analysis to correlator data in HDF5 format.

    This script processes correlator data by:
        1. Loading data using HDF5Analyzer
        2. Grouping by tunable parameters (excluding
           Configuration_label)
        3. Applying jackknife resampling to each group
        4. Computing finite difference derivatives
        5. Exporting results in clean, consistent format
    """
    # === SETUP AND VALIDATION ===

    # Handle output directory
    if output_directory is None:
        output_directory = os.path.dirname(input_hdf5_file)

    # Ensure output file is in the specified directory
    output_path = Path(output_directory) / Path(output_hdf5_file).name

    # Get configuration values from config file
    deriv_method = DEFAULT_DERIVATIVE_METHOD
    compression = DEFAULT_COMPRESSION
    compression_level = DEFAULT_COMPRESSION_LEVEL

    # Setup logging
    logger = filesystem_utilities.LoggingWrapper(
        log_directory or output_directory, log_filename, enable_logging
    )
    logger.initiate_script_logging()

    if verbose:
        click.echo(f"Input file: {input_hdf5_file}")
        click.echo(f"Output file: {output_path}")
        click.echo(f"Derivative method: {deriv_method.value}")

    # === LOAD AND ANALYZE INPUT DATA ===

    try:
        # Load HDF5 data using analyzer
        analyzer = HDF5Analyzer(input_hdf5_file)
        logger.info(f"Loaded HDF5 file: {input_hdf5_file}")

        if verbose:
            click.echo(f"Found {len(analyzer.active_groups)} groups")
            click.echo(
                f"Available datasets: {analyzer.list_of_output_quantity_names_from_hdf5[:5]}..."
            )

        # Validate required datasets are present
        missing_datasets = []
        for dataset in REQUIRED_INPUT_DATASETS:
            if dataset not in analyzer.list_of_output_quantity_names_from_hdf5:
                missing_datasets.append(dataset)

        if missing_datasets:
            error_msg = f"Missing required datasets: {missing_datasets}"
            logger.critical(error_msg, to_console=True)
            sys.exit(1)

        # === GROUP DATA BY PARAMETERS ===

        # Group by tunable parameters, excluding configuration labels
        grouped_data = analyzer.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=EXCLUDED_FROM_GROUPING, verbose=verbose
        )

        if not grouped_data:
            logger.critical("No valid parameter groupings found!", to_console=True)
            sys.exit(1)

        logger.info(f"Found {len(grouped_data)} parameter groupings for analysis")

        # === PROCESS EACH GROUP ===

        # Initialize jackknife processor
        processor = JackknifeProcessor(
            derivative_method=deriv_method, logger=logger.logger
        )

        # Store all processing results for custom HDF5 creation
        all_processing_results = {}
        processed_groups = 0

        for group_index, (param_values, group_paths) in enumerate(grouped_data.items()):
            group_name = _create_descriptive_group_name(
                param_values, analyzer.reduced_multivalued_tunable_parameter_names_list
            )

            if verbose:
                click.echo(
                    f"\nProcessing group {group_index + 1}/{len(grouped_data)}: {param_values}"
                )

            # === LOAD CORRELATOR DATA FOR THIS GROUP ===

            # Restrict analyzer to this specific group
            with analyzer:  # Use context manager to restore state after
                # Filter to just the groups in this parameter
                # combination
                analyzer._active_groups = set(group_paths)

                # Get group metadata
                if group_paths:
                    first_group_path = list(group_paths)[0]
                    group_metadata = analyzer._parameters_for_group(first_group_path)
                    group_metadata["group_name"] = group_name
                    group_metadata["param_values"] = param_values
                else:
                    continue

                # === ENSURE DETERMINISTIC ORDERING ===

                # Sort group paths by configuration label for consistent
                # ordering
                sorted_group_paths = sorted(group_paths)

                if verbose:
                    click.echo(
                        f"Processing {len(sorted_group_paths)} configurations in sorted order"
                    )

                # Load correlator datasets in deterministic order
                g5g5_data_list = []
                g4g5g5_data_list = []

                for group_path in sorted_group_paths:
                    try:
                        # Temporarily restrict to single group
                        analyzer._active_groups = {group_path}

                        g5g5_single = analyzer.dataset_values(
                            INPUT_CORRELATOR_DATASETS["g5g5"], return_gvar=False
                        )
                        g4g5g5_single = analyzer.dataset_values(
                            INPUT_CORRELATOR_DATASETS["g4g5g5"], return_gvar=False
                        )

                        # Handle single vs list returns
                        if isinstance(g5g5_single, list):
                            g5g5_data_list.extend(g5g5_single)
                            g4g5g5_data_list.extend(g4g5g5_single)
                        else:
                            g5g5_data_list.append(g5g5_single)
                            g4g5g5_data_list.append(g4g5g5_single)

                    except ValueError as e:
                        logger.warning(f"Group {group_path}: Missing datasets - {e}")
                        continue

                # Reset to full group set for metadata extraction
                analyzer._active_groups = set(sorted_group_paths)

                if not g5g5_data_list or not g4g5g5_data_list:
                    logger.warning(
                        f"Group {group_name}: No valid correlator data found"
                    )
                    continue

                # Stack data into 2D arrays (configs × time) - now in
                # sorted order
                g5g5_data = np.vstack(g5g5_data_list)
                g4g5g5_data = np.vstack(g4g5g5_data_list)

                # === EXTRACT CONFIGURATION METADATA ===

                # Get configuration labels and filenames IN SORTED ORDER
                config_metadata = {}

                # Try to get configuration labels from the analyzer
                try:
                    # Create metadata arrays in the same sorted order as
                    # data loading
                    config_metadata = _extract_ordered_configuration_metadata(
                        analyzer, sorted_group_paths, INPUT_CORRELATOR_DATASETS["g5g5"]
                    )

                    if verbose:
                        click.echo(
                            "Extracted metadata for "
                            f"{len(config_metadata.get('configuration_labels', []))} "
                            "configurations"
                        )

                except Exception as e:
                    logger.warning(f"Could not extract configuration metadata: {e}")
                    # Create default metadata in the same order
                    n_configs = g5g5_data.shape[0]
                    config_metadata = {
                        "configuration_labels": [
                            f"config_{i}" for i in range(n_configs)
                        ],
                        "qpb_filenames": [f"unknown_{i}.txt" for i in range(n_configs)],
                        "mpi_geometries": ["unknown" for _ in range(n_configs)],
                    }

                    if verbose:
                        click.echo(
                            f"Using default config metadata for {n_configs} configurations"
                        )

                # === APPLY JACKKNIFE PROCESSING ===

                # Process this group through complete jackknife analysis
                processing_results = processor.process_correlator_group(
                    g5g5_data=g5g5_data,
                    g4g5g5_data=g4g5g5_data,
                    group_metadata=group_metadata,
                    min_configurations=min_configurations,
                )

                if not processing_results:
                    logger.warning(f"Skipping group {group_name} - processing failed")
                    continue

                # Store results for later HDF5 creation
                all_processing_results[group_name] = {
                    "jackknife_results": processing_results,
                    "config_metadata": config_metadata,
                    "group_metadata": group_metadata,
                }

                processed_groups += 1

        # === SAVE RESULTS ===

        if processed_groups == 0:
            logger.critical("No groups were successfully processed!", to_console=True)
            sys.exit(1)

        logger.info(f"Successfully processed {processed_groups} groups")

        # === CREATE CUSTOM HDF5 OUTPUT ===

        # Create custom HDF5 file with proper structure
        _create_custom_hdf5_output(
            output_path=output_path,
            all_processing_results=all_processing_results,
            analyzer=analyzer,
            compression=compression,
            compression_level=compression_level,
            logger=logger,
        )

        if verbose:
            click.echo(f"\n✓ Analysis complete!")
            click.echo(f"✓ Processed {processed_groups} parameter groups")
            click.echo(f"✓ Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Critical error during processing: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    finally:
        # Clean up
        if "analyzer" in locals():
            analyzer.close()
        logger.terminate_script_logging()

        if verbose:
            click.echo("✓ Jackknife analysis completed successfully")


def _create_descriptive_group_name(param_values, multivalued_params, max_length=100):
    """
    Create a descriptive group name using parameter names and values.

    Args:
        param_values: Values of the multivalued parameters (tuple/list or single value)
        multivalued_params: List of parameter names
        max_length: Maximum length for the group name (default: 100)

    Returns:
        str: Descriptive group name like "jackknife_analysis_m0p06_n6_Brillouin"
    """
    if not multivalued_params:
        return "jackknife_analysis_default_group"

    # Handle single parameter case - convert to list
    if not isinstance(param_values, (tuple, list)):
        param_values = [param_values]

    # Handle the mismatch by filtering out Configuration_label
    # param_values contains: (Bare_mass, KL_diagonal_order, Kernel_operator_type)
    # multivalued_params may include Configuration_label which isn't in param_values
    if len(param_values) == 3 and len(multivalued_params) == 4:
        # Remove Configuration_label from multivalued_params since it's not in param_values
        filtered_params = [p for p in multivalued_params if p != "Configuration_label"]
        actual_params = filtered_params
    elif len(param_values) != len(multivalued_params):
        # Fallback for other mismatches
        param_str = str(param_values) + str(multivalued_params)
        hash_suffix = abs(hash(param_str)) % 10000
        return f"jackknife_analysis_mismatch_{hash_suffix:04d}"
    else:
        actual_params = multivalued_params

    name_parts = []
    for param_name, param_value in zip(actual_params, param_values):
        # Check if this parameter has a label or should use value directly
        if param_name in PARAMETER_LABELS:
            # Use label + formatted value (no underscore between them)
            label = PARAMETER_LABELS[param_name]
            formatted_value = _format_parameter_value(param_name, param_value)
            name_parts.append(f"{label}{formatted_value}")
        else:
            # Parameters without labels (Overlap_operator_method, Kernel_operator_type)
            # Use the value directly
            formatted_value = _format_parameter_value(param_name, param_value)
            name_parts.append(formatted_value)

    # Join with underscores between pairs only
    group_name_suffix = "_".join(name_parts)
    group_name = f"jackknife_analysis_{group_name_suffix}"

    # Ensure HDF5 compatibility (remove problematic characters except minus)
    group_name = re.sub(r"[^a-zA-Z0-9_.-]", "", group_name)

    # Truncate if too long but maintain uniqueness
    if len(group_name) > max_length:
        # Keep the first part and add a hash for uniqueness
        truncated = group_name[: max_length - 10]
        hash_suffix = abs(hash(str(param_values))) % 10000
        group_name = f"{truncated}_{hash_suffix:04d}"

    return group_name


def _format_parameter_value(param_name, value):
    """
    Format parameter value according to established rules.

    Args:
        param_name: Name of the parameter
        value: Value to format

    Returns:
        str: Formatted value suitable for group naming
    """
    if param_name in PARAMETERS_WITH_EXPONENTIAL_FORMAT:
        # Use exponential format for specified parameters
        if isinstance(value, (float, np.floating)):
            return f"{value:.0e}".replace("+", "")
        else:
            return str(value)
    elif isinstance(value, (float, np.floating)):
        # Format floats with consistent precision, replace decimal with 'p'
        formatted = f"{value:.6g}".replace(".", "p")
        return formatted
    elif isinstance(value, (int, np.integer)):
        return str(int(value))
    else:
        # Handle strings and other types
        clean_value = str(value).replace(" ", "").replace("_", "")
        # Apply specific transformations for kernel types
        if param_name == "Kernel_operator_type":
            # Convert Standard -> Wilson as per the patterns
            if clean_value.lower() == "standard":
                return "Wilson"
            else:
                return clean_value  # Keep original case (Brillouin)
        return clean_value  # Keep original case


def _extract_ordered_configuration_metadata(
    analyzer: HDF5Analyzer, sorted_group_paths: List[str], dataset_name: str
) -> Dict[str, List]:
    """
    Extract configuration metadata in the exact same order as sorted
    group paths.

    Args:
        - analyzer: HDF5Analyzer instance
        - sorted_group_paths: List of group paths in sorted order
        - dataset_name: Dataset name to use for DataFrame creation

    Returns:
        Dictionary with ordered configuration metadata
    """
    metadata = {
        "configuration_labels": [],
        "qpb_filenames": [],
        "mpi_geometries": [],
    }

    for group_path in sorted_group_paths:
        # Temporarily restrict to single group
        analyzer._active_groups = {group_path}

        try:
            # Create DataFrame for this single group
            temp_df = analyzer.to_dataframe(
                datasets=[dataset_name],
                add_time_column=False,
                flatten_arrays=False,
            )

            # Extract metadata for this group
            group_metadata = extract_configuration_metadata(temp_df)

            # Append to ordered lists (should be single values since
            # it's one group)
            for key in metadata.keys():
                source_key = key  # Direct mapping
                if source_key in group_metadata:
                    values = group_metadata[source_key]
                    if isinstance(values, list) and len(values) == 1:
                        metadata[key].append(values[0])
                    elif not isinstance(values, list):
                        metadata[key].append(values)
                    else:
                        # Handle multiple values per group (shouldn't
                        # happen but be safe)
                        metadata[key].extend(values)
                else:
                    # Add default value
                    if key == "configuration_labels":
                        # Extract from group path (filename)
                        filename = group_path.split("/")[-1]
                        metadata[key].append(filename)
                    elif key == "qpb_filenames":
                        metadata[key].append("unknown.txt")
                    elif key == "mpi_geometries":
                        metadata[key].append("unknown")

        except Exception as e:
            # Add default values for this group
            if "configuration_labels" not in metadata or len(
                metadata["configuration_labels"]
            ) < len(sorted_group_paths):
                filename = group_path.split("/")[-1]
                metadata["configuration_labels"].append(filename)
            if "qpb_filenames" not in metadata or len(metadata["qpb_filenames"]) < len(
                sorted_group_paths
            ):
                metadata["qpb_filenames"].append("unknown.txt")
            if "mpi_geometries" not in metadata or len(
                metadata["mpi_geometries"]
            ) < len(sorted_group_paths):
                metadata["mpi_geometries"].append("unknown")

    return metadata


def _create_custom_hdf5_output(
    output_path: Path,
    all_processing_results: Dict,
    analyzer: HDF5Analyzer,
    compression: str,
    compression_level: int,
    logger,
) -> None:
    """
    Create custom HDF5 output file with proper structure and only
    jackknife results.

    Args:
        - output_path: Path for output HDF5 file
        - all_processing_results: Dictionary with all processing results
        - analyzer: HDF5Analyzer instance for getting single-valued
          parameters
        - compression: Compression method
        - compression_level: Compression level
        - logger: Logger instance
    """
    # Prepare compression settings
    compression_opts = None if compression == "none" else compression_level
    final_compression = None if compression == "none" else compression

    try:
        with h5py.File(output_path, "w") as output_file:
            # Recreate the same directory structure as input Get the
            # relative path structure from the analyzer
            input_structure = _get_input_directory_structure(analyzer)

            # Create the group hierarchy
            parent_group = output_file
            for level_name in input_structure:
                parent_group = parent_group.create_group(level_name)

            # Add single-valued parameters as attributes to the parent
            # group
            single_valued_params = analyzer.unique_value_columns_dictionary
            for param_name, param_value in single_valued_params.items():
                # Only add tunable parameters to maintain consistency
                # with input
                if param_name in analyzer.list_of_tunable_parameter_names_from_hdf5:
                    parent_group.attrs[param_name] = param_value

            # Create jackknife analysis groups with proper names
            for group_name, results in all_processing_results.items():
                jackknife_group = parent_group.create_group(group_name)

                # Add group-specific parameters as attributes
                group_metadata = results["group_metadata"]
                param_values = group_metadata.get("param_values", ())

                # Add multivalued parameters for this specific group
                multivalued_params = (
                    analyzer.reduced_multivalued_tunable_parameter_names_list
                )
                if isinstance(param_values, (tuple, list)) and len(param_values) == len(
                    multivalued_params
                ):
                    for param_name, param_value in zip(
                        multivalued_params, param_values
                    ):
                        jackknife_group.attrs[param_name] = param_value
                elif (
                    not isinstance(param_values, (tuple, list))
                    and len(multivalued_params) == 1
                ):
                    # Handle single parameter case
                    jackknife_group.attrs[multivalued_params[0]] = param_values

                # Add number of gauge configurations
                jackknife_results = results["jackknife_results"]
                n_configs = jackknife_results.get("n_gauge_configurations")
                if n_configs:
                    jackknife_group.attrs["Number_of_gauge_configurations"] = n_configs

                # Store jackknife results as datasets
                jackknife_results = results["jackknife_results"]
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
        # Return all but the last part (which is the individual file
        # group)
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
        group: HDF5 group to store datasets in jackknife_results:
        Dictionary with jackknife results compression: Compression
        method compression_opts: Compression options logger: Logger
        instance
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
        - group: HDF5 group to store datasets in
        - config_metadata: Dictionary with configuration metadata
        - compression: Compression method
        - compression_opts: Compression options
        - logger: Logger instance
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

            # Debug logging
            logger.debug(f"Processing metadata {source_key}: {data_list}")

            if data_list:  # Only store if we have data
                # Convert to appropriate numpy array
                if source_key == "mpi_geometries":
                    # Handle potential None values and ensure string
                    # format
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

                # Create dataset (strings usually don't benefit from
                # compression)
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


if __name__ == "__main__":
    main()
