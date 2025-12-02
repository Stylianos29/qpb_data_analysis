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
    python apply_jackknife_analysis.py -i input.h5 -csv processed.csv -o output.h5
    [options]
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, List, cast

import click
import numpy as np
import pandas as pd
import re

# Import library components
from library.data.hdf5_analyzer import HDF5Analyzer
from library.data import load_csv
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
    "-csv",
    "--processed_parameters_csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to processed_parameter_values.csv (from Stage 2A) for consistent parameter values.",
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
    processed_parameters_csv: str,
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
        2. Loading processed parameters from CSV (Stage 2A output)
        3. Grouping by tunable parameters (excluding
           Configuration_label)
        4. Applying jackknife resampling to each group
        5. Computing finite difference derivatives
        6. Exporting results with processed parameter values
    """
    # === SETUP AND VALIDATION ===

    # Handle output directory
    if output_directory is None:
        output_directory = os.path.dirname(input_hdf5_file)

    # Ensure output file is in the specified directory
    if not os.path.isabs(output_hdf5_file):
        output_path = Path(output_directory) / output_hdf5_file
    else:
        output_path = Path(output_hdf5_file)

    # Setup logging directory
    if enable_logging and log_directory is None:
        log_directory = output_directory

    # Initialize logger
    logger = create_script_logger(
        log_directory=log_directory,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=verbose,
    )

    try:
        logger.log_script_start("Jackknife analysis with processed parameters")

        # === LOAD PROCESSED PARAMETERS FROM CSV ===

        logger.info(f"Loading processed parameters from: {processed_parameters_csv}")
        try:
            processed_df = load_csv(processed_parameters_csv)
            logger.info(f"Loaded processed parameters: {len(processed_df)} rows")

            # Validate that CSV contains Filename column
            if "Filename" not in processed_df.columns:
                raise ValueError(
                    "Processed parameters CSV must contain 'Filename' column"
                )

        except Exception as e:
            logger.error(f"Failed to load processed parameters CSV: {e}")
            raise

        # === LOAD HDF5 INPUT ===

        logger.info(f"Loading correlator data from: {input_hdf5_file}")
        analyzer = HDF5Analyzer(input_hdf5_file)
        logger.info(f"HDF5 file loaded successfully")

        # === VALIDATE REQUIRED DATASETS ===

        available_datasets = analyzer.list_of_output_quantity_names_from_hdf5
        missing_datasets = [
            ds for ds in REQUIRED_INPUT_DATASETS if ds not in available_datasets
        ]

        if missing_datasets:
            error_msg = (
                f"Required correlator datasets not found in HDF5 file: "
                f"{missing_datasets}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("All required datasets found in input HDF5")

        # === GROUP DATA BY PARAMETERS ===

        logger.info("Grouping correlator data by tunable parameters...")
        grouped_data = analyzer.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=EXCLUDED_FROM_GROUPING, verbose=verbose
        )

        if not grouped_data:
            error_msg = "No valid parameter groups found in input data"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Found {len(grouped_data)} parameter groups to process")

        # === INITIALIZE PROCESSOR ===

        processor = JackknifeProcessor(
            derivative_method=DEFAULT_DERIVATIVE_METHOD, logger=logger
        )

        # === PROCESS EACH GROUP ===

        all_processing_results = {}
        processed_groups = 0
        compression = DEFAULT_COMPRESSION
        compression_level = DEFAULT_COMPRESSION_LEVEL

        for group_index, (param_values, group_paths) in enumerate(
            grouped_data.items(), 1
        ):
            # group_paths is List[str] of HDF5 paths

            # Create descriptive group name
            multivalued_params = (
                analyzer.reduced_multivalued_tunable_parameter_names_list
            )
            group_name = _create_descriptive_group_name(
                param_values, multivalued_params
            )

            logger.info(
                f"Processing group {group_index}/{len(grouped_data)}: {group_name}"
            )

            try:
                # === USE ORIGINAL APPROACH WITH CONTEXT MANAGER ===

                with analyzer:  # Context manager to restore state after
                    # Filter to just the groups in this parameter combination
                    analyzer.active_groups = set(group_paths)

                    # Sort paths for deterministic ordering
                    sorted_group_paths = sorted(group_paths)

                    logger.info(
                        f"Processing {len(sorted_group_paths)} configurations in sorted order"
                    )

                    # Load correlator data by looping through each path
                    g5g5_data_list = []
                    g4g5g5_data_list = []

                    for group_path in sorted_group_paths:
                        # Temporarily restrict to single group
                        analyzer.active_groups = {group_path}

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

                    # Reset to full group set
                    analyzer.active_groups = set(sorted_group_paths)

                    if not g5g5_data_list or not g4g5g5_data_list:
                        logger.warning(f"No valid correlator data for {group_name}")
                        continue

                    # Stack into 2D arrays (configs × time)
                    g5g5_group = np.vstack(g5g5_data_list)
                    g4g5g5_group = np.vstack(g4g5g5_data_list)

                    # Extract configuration metadata
                    # Use to_dataframe() within the filtered context
                    temp_df = analyzer.to_dataframe(
                        datasets=[INPUT_CORRELATOR_DATASETS["g5g5"]],
                        add_time_column=False,
                        flatten_arrays=False,
                    )
                    config_metadata = extract_configuration_metadata(temp_df)

                # Prepare group metadata
                group_metadata = {
                    "param_values": param_values,
                    "n_configurations": len(group_paths),
                    "group_name": group_name,
                }

                # Process with jackknife
                jackknife_results = processor.process_correlator_group(
                    g5g5_data=g5g5_group,
                    g4g5g5_data=g4g5g5_group,
                    group_metadata=group_metadata,
                    min_configurations=min_configurations,
                )

                # Store results
                all_processing_results[group_name] = {
                    "jackknife_results": jackknife_results,
                    "config_metadata": config_metadata,
                    "group_metadata": group_metadata,
                    "group_paths": sorted_group_paths,
                }

                processed_groups += 1
                logger.info(f"Successfully processed group {group_name}")

            except Exception as e:
                logger.warning(f"Failed to process group {group_name}: {e}")
                continue

        # === VALIDATE RESULTS ===

        if processed_groups == 0:
            error_msg = "No groups were successfully processed"
            logger.critical(error_msg)
            click.echo(f"ERROR: {error_msg}")
            sys.exit(1)

        logger.info(
            f"Successfully processed {processed_groups}/{len(grouped_data)} groups"
        )

        # === CREATE CUSTOM HDF5 OUTPUT ===

        logger.info(f"Creating output HDF5 file: {output_path}")
        _create_custom_hdf5_output(
            output_path=output_path,
            all_processing_results=all_processing_results,
            analyzer=analyzer,
            processed_params_df=processed_df,  # PASS PROCESSED PARAMETERS
            compression=compression,
            compression_level=compression_level,
            logger=logger,
        )

        # Script completion with summary
        logger.log_script_end(
            f"Processed {processed_groups} groups, saved to {output_path.name}"
        )

        # Success summary
        if verbose or not enable_logging:
            click.echo(f"✓ Jackknife analysis completed successfully")
            click.echo(
                f"✓ Processed {processed_groups}/{len(grouped_data)} parameter groups"
            )
            click.echo(
                f"✓ Used processed parameters from: {Path(processed_parameters_csv).name}"
            )
            click.echo(f"✓ Results saved to: {output_path}")

    except Exception as e:
        logger.log_script_error(e)
        click.echo(f"ERROR: Critical failure during processing: {e}")
        sys.exit(1)

    finally:
        # Clean up resources
        if "analyzer" in locals():
            analyzer.close()
        logger.close()


def _create_descriptive_group_name(param_values, multivalued_params, max_length=100):
    """
    Create a descriptive group name using parameter names and values.

    Args:
        - param_values: Tuple of parameter values
        - multivalued_params: List of parameter names
        - max_length: Maximum length for group name

    Returns:
        Descriptive string name for the group
    """
    # Filter out Configuration_label if present
    actual_params = [p for p in multivalued_params if p != "Configuration_label"]

    # Handle single vs multiple parameters
    if not isinstance(param_values, (tuple, list)):
        param_values = [param_values]

    # Create name parts
    parts = []
    for param_name, value in zip(actual_params, param_values):
        # Get short label if available
        label = PARAMETER_LABELS.get(param_name, param_name)

        # Format value
        if param_name in PARAMETERS_WITH_EXPONENTIAL_FORMAT:
            value_str = f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e")
        elif isinstance(value, float):
            value_str = f"{value:.4g}".replace(".", "p").replace("-", "m")
        else:
            value_str = str(value).replace(".", "p").replace("-", "m")

        parts.append(f"{label}{value_str}")

    # Combine parts
    name = "jackknife_analysis_" + "_".join(parts)

    # Truncate if needed
    if len(name) > max_length:
        name = name[: max_length - 3] + "..."

    return name


if __name__ == "__main__":
    main()

# TODO: Place main function definition at the bottom of the file and the
# rest of the function definitions above it.
# TODO: Set min_configurations parameter from the config file if
# available.
# TODO: Use the processed parameters Dataframe to update MPI_geometry
# parameter.
# TODO: For the jackknife analysis names, use the processed parameter
# values instead of the raw ones.
