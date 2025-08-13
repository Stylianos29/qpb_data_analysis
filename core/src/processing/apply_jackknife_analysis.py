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
import re

# Import library components
from library.data.hdf5_analyzer import HDF5Analyzer
from library.constants import PARAMETERS_WITH_EXPONENTIAL_FORMAT, PARAMETER_LABELS
from library.validation.click_validators import (
    hdf5_file,
    directory,
    validate_log_filename,
)
from library.utils.logging_utilities import create_script_logger

# Import from auxiliary modules
from src.processing._jackknife_config import (
    DEFAULT_DERIVATIVE_METHOD,
    DEFAULT_COMPRESSION,
    DEFAULT_COMPRESSION_LEVEL,
    EXCLUDED_FROM_GROUPING,
    INPUT_CORRELATOR_DATASETS,
    REQUIRED_INPUT_DATASETS,
    MIN_GAUGE_CONFIGURATIONS,
)
from src.processing._jackknife_processor import (
    JackknifeProcessor,
    extract_configuration_metadata,
)
from src.processing._hdf5_output import _create_custom_hdf5_output


@click.command()
@click.option(
    "-i",
    "--input_hdf5_file",
    required=True,
    callback=hdf5_file.input,
    help="Path to input HDF5 file containing correlator data and parameters.",
)
@click.option(
    "-o",
    "--output_hdf5_file",
    required=True,
    callback=hdf5_file.output,
    help="Path for output HDF5 file with jackknife analysis results.",
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
    log_filename: str,
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

    # Ensure output file is in the specified directory. This way, we
    # avoid potential issues with relative paths and make the output
    # location clear.
    output_path = Path(output_directory) / Path(output_hdf5_file).name

    # Get configuration values from config file
    deriv_method = DEFAULT_DERIVATIVE_METHOD
    compression = DEFAULT_COMPRESSION
    compression_level = DEFAULT_COMPRESSION_LEVEL

    # Setup logging
    logger = create_script_logger(
        log_directory=log_directory or output_directory,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=verbose,  # Console logging follows verbose flag
        verbose=False,  # Don't use DEBUG level console output
    )

    # Script lifecycle logging
    logger.log_script_start("Jackknife analysis processing")

    # Log configuration details (always to file, optionally to console
    # if verbose)
    logger.info(f"Input file: {input_hdf5_file}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Derivative method: {deriv_method.value}")
    logger.info(f"Minimum configurations: {min_configurations}")

    # === LOAD AND ANALYZE INPUT DATA ===

    try:
        # Load HDF5 data using analyzer
        analyzer = HDF5Analyzer(input_hdf5_file)
        logger.info(f"Loaded HDF5 file with {len(analyzer.active_groups)} groups")
        logger.debug(
            f"Available datasets: {analyzer.list_of_output_quantity_names_from_hdf5}"
        )

        # Validate required datasets are present
        missing_datasets = []
        for dataset in REQUIRED_INPUT_DATASETS:
            if dataset not in analyzer.list_of_output_quantity_names_from_hdf5:
                missing_datasets.append(dataset)

        if missing_datasets:
            error_msg = f"Missing required datasets: {missing_datasets}"
            logger.critical(error_msg)
            # Use click.echo for immediate console feedback on critical
            # errors
            click.echo(f"ERROR: {error_msg}")
            sys.exit(1)

        # === GROUP DATA BY PARAMETERS ===

        # Group by tunable parameters, excluding configuration labels
        logger.info("Grouping data by tunable parameters")
        grouped_data = analyzer.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=EXCLUDED_FROM_GROUPING, verbose=verbose
        )

        if not grouped_data:
            error_msg = "No valid parameter groupings found"
            logger.critical(error_msg)
            click.echo(f"ERROR: {error_msg}")
            sys.exit(1)

        logger.info(f"Found {len(grouped_data)} parameter groups for processing")

        # === PROCESS EACH GROUP ===

        # Initialize jackknife processor
        processor = JackknifeProcessor(derivative_method=deriv_method, logger=logger)

        # Store all processing results for custom HDF5 creation
        all_processing_results = {}
        processed_groups = 0

        for group_index, (param_values, group_paths) in enumerate(grouped_data.items()):
            group_name = _create_descriptive_group_name(
                param_values, analyzer.reduced_multivalued_tunable_parameter_names_list
            )

            logger.info(
                f"Processing group {group_index + 1}/{len(grouped_data)}: {group_name}"
            )
            logger.debug(f"Group parameters: {param_values}")

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
                    logger.warning(f"No group paths found for {group_name}")
                    continue

                # === ENSURE DETERMINISTIC ORDERING ===

                # Sort group paths by configuration label for consistent
                # ordering
                sorted_group_paths = sorted(group_paths)

                logger.info(
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

                logger.debug(
                    f"Loaded data shapes: g5g5={g5g5_data.shape}, "
                    f"g4g5g5={g4g5g5_data.shape}"
                )

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

                    logger.info(
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

                    logger.info(
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
                logger.info(f"Successfully processed group {group_name}")

        # === VALIDATE RESULTS ===

        if processed_groups == 0:
            error_msg = "No groups were successfully processed"
            logger.critical(error_msg)
            click.echo(
                f"ERROR: {error_msg}"
            )  # Keep immediate console feedback for critical errors
            sys.exit(1)

        logger.info(
            f"Successfully processed {processed_groups}/{len(grouped_data)} groups"
        )

        # === CREATE CUSTOM HDF5 OUTPUT ===

        # Create custom HDF5 file with proper structure
        logger.info(f"Creating output HDF5 file: {output_path}")
        _create_custom_hdf5_output(
            output_path=output_path,
            all_processing_results=all_processing_results,
            analyzer=analyzer,
            compression=compression,
            compression_level=compression_level,
            logger=logger,
        )

        # Script completion with summary
        logger.log_script_end(
            f"Processed {processed_groups} groups, saved to {output_path.name}"
        )

        # Success summary (console output for immediate feedback)
        if verbose or not enable_logging:
            click.echo(f"✓ Jackknife analysis completed successfully")
            click.echo(
                f"✓ Processed {processed_groups}/{len(grouped_data)} parameter groups"
            )
            click.echo(f"✓ Results saved to: {output_path}")

    except Exception as e:
        logger.log_script_error(e)  # Use the enhanced error logging
        click.echo(f"ERROR: Critical failure during processing: {e}")
        sys.exit(1)

    finally:
        # Clean up resources
        if "analyzer" in locals():
            analyzer.close()
        logger.close()  # Proper logging cleanup


def _create_descriptive_group_name(param_values, multivalued_params, max_length=100):
    """
    Create a descriptive group name using parameter names and values.

    Args:
        - param_values: Values of the multivalued parameters (tuple/list
        or single value)
        - multivalued_params: List of parameter names
        - max_length: Maximum length for the group name (default: 100)

    Returns:
        str: Descriptive group name like
        "jackknife_analysis_m0p06_n6_Brillouin"
    """
    if not multivalued_params:
        return "jackknife_analysis_default_group"

    # Handle single parameter case - convert to list
    if not isinstance(param_values, (tuple, list)):
        param_values = [param_values]

    # Always remove Configuration_label if it exists (it's not used in
    # group names)
    actual_params = [p for p in multivalued_params if p != "Configuration_label"]

    # Check if lengths match after filtering
    if len(param_values) != len(actual_params):
        # Fallback for mismatches
        param_str = str(param_values) + str(multivalued_params)
        hash_suffix = abs(hash(param_str)) % 10000
        return f"jackknife_analysis_mismatch_{hash_suffix:04d}"

    name_parts = []
    for param_name, param_value in zip(actual_params, param_values):
        # Check if this parameter has a label or should use value
        # directly
        if param_name in PARAMETER_LABELS:
            # Use label + formatted value (no underscore between them)
            label = PARAMETER_LABELS[param_name]
            formatted_value = _format_parameter_value(param_name, param_value)
            name_parts.append(f"{label}{formatted_value}")
        else:
            # Parameters without labels (Overlap_operator_method,
            # Kernel_operator_type) Use the value directly
            formatted_value = _format_parameter_value(param_name, param_value)
            name_parts.append(formatted_value)

    # Join with underscores between pairs only
    group_name_suffix = "_".join(name_parts)
    group_name = f"jackknife_analysis_{group_name_suffix}"

    # Ensure HDF5 compatibility (remove problematic characters except
    # minus)
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
        param_name: Name of the parameter value: Value to format

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
        # Format floats with consistent precision, replace decimal with
        # 'p'
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


if __name__ == "__main__":
    main()
