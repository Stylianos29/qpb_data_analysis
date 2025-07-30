#!/usr/bin/env python3
"""
Jackknife analysis script for QPB correlator data preprocessing.

This script applies jackknife resampling to correlator data stored in
HDF5 format, focusing on preprocessing tasks:
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
from typing import Optional

import click
import numpy as np
import h5py

# Import library components
from library.data.hdf5_analyzer import HDF5Analyzer
from library import (
    validate_input_directory,
    validate_input_script_log_filename,
    filesystem_utilities,
)

# Import our auxiliary modules
from src.preprocessing.jackknife_config import (
    DEFAULT_DERIVATIVE_METHOD,
    DEFAULT_COMPRESSION,
    DEFAULT_COMPRESSION_LEVEL,
    EXCLUDED_FROM_GROUPING,
    INPUT_CORRELATOR_DATASETS,
    REQUIRED_INPUT_DATASETS,
    MIN_GAUGE_CONFIGURATIONS,
    get_dataset_description,
)

from src.preprocessing.jackknife_processor import (
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
    type=click.Path(),
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

        # Initialize analyzer for adding virtual datasets
        processed_groups = 0

        for group_index, (param_values, group_paths) in enumerate(grouped_data.items()):
            group_name = f"jackknife_analysis_{group_index}"

            if verbose:
                click.echo(
                    f"\nProcessing group {group_index + 1}/{len(grouped_data)}: {param_values}"
                )

            # === LOAD CORRELATOR DATA FOR THIS GROUP ===

            # Restrict analyzer to this specific group
            with analyzer:  # Use context manager to restore state after
                # Filter to just the groups in this parameter combination
                analyzer._active_groups = set(group_paths)

                # Get group metadata
                if group_paths:
                    first_group_path = list(group_paths)[0]
                    group_metadata = analyzer._parameters_for_group(first_group_path)
                    group_metadata["group_name"] = group_name
                    group_metadata["param_values"] = param_values
                else:
                    continue

                # Load correlator datasets for all groups in this parameter set
                try:
                    g5g5_data_list = analyzer.dataset_values(
                        INPUT_CORRELATOR_DATASETS["g5g5"], return_gvar=False
                    )
                    g4g5g5_data_list = analyzer.dataset_values(
                        INPUT_CORRELATOR_DATASETS["g4g5g5"], return_gvar=False
                    )
                except ValueError as e:
                    logger.warning(f"Group {group_name}: Missing datasets - {e}")
                    continue

                # Stack data into 2D arrays (configs × time)
                # Ensure we always have lists for consistent processing
                if not isinstance(g5g5_data_list, list):
                    g5g5_data_list = [g5g5_data_list]
                if not isinstance(g4g5g5_data_list, list):
                    g4g5g5_data_list = [g4g5g5_data_list]

                g5g5_data = np.vstack(g5g5_data_list)
                g4g5g5_data = np.vstack(g4g5g5_data_list)

                # === EXTRACT CONFIGURATION METADATA ===

                # Get configuration labels and filenames
                config_metadata = {} # TODO: Gather configuration metadata

                # Try to get configuration labels from the analyzer
                try:
                    # Create a temporary dataframe to extract metadata
                    temp_df = analyzer.to_dataframe(
                        datasets=[INPUT_CORRELATOR_DATASETS["g5g5"]],
                        add_time_column=False,
                        flatten_arrays=False,
                    )
                    config_metadata = extract_configuration_metadata(temp_df)
                except Exception as e:
                    logger.warning(f"Could not extract configuration metadata: {e}")
                    # Create default metadata
                    config_metadata = {
                        "configuration_labels": [
                            f"config_{i}" for i in range(g5g5_data.shape[0])
                        ],
                        "qpb_filenames": [
                            f"unknown_{i}.txt" for i in range(g5g5_data.shape[0])
                        ],
                    }

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

                # === ADD VIRTUAL DATASETS TO ANALYZER ===

                # Add all jackknife results as virtual datasets
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
                    if dataset_name in processing_results:
                        # Create a simple lambda that returns the data
                        # Note: We need to capture the value in the closure
                        data_value = processing_results[dataset_name]
                        transform_func = lambda x, val=data_value: val

                        analyzer.transform_dataset(
                            source_dataset=INPUT_CORRELATOR_DATASETS[
                                "g5g5"
                            ],  # Dummy source
                            transform_func=transform_func,
                            new_name=dataset_name,
                        )

                processed_groups += 1

        # === SAVE RESULTS ===

        if processed_groups == 0:
            logger.critical("No groups were successfully processed!", to_console=True)
            sys.exit(1)

        logger.info(f"Successfully processed {processed_groups} groups")

        # Prepare compression settings
        compression_opts = None if compression == "none" else compression_level
        final_compression = None if compression == "none" else compression

        # Save processed data using HDF5Analyzer's save method
        try:
            if final_compression is None:
                analyzer.save_transformed_data(
                    output_path=output_path,
                    include_virtual=True,
                )
            else:
                analyzer.save_transformed_data(
                    output_path=output_path,
                    include_virtual=True,
                    compression=final_compression,
                    compression_opts=compression_opts,  # type: ignore
                )

            logger.info(f"Results saved to: {output_path}")

            if verbose:
                click.echo(f"\n✓ Analysis complete!")
                click.echo(f"✓ Processed {processed_groups} parameter groups")
                click.echo(f"✓ Results saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

        # === ADD DATASET DESCRIPTIONS ===

        # Open the output file to add descriptions
        try:
            with h5py.File(output_path, "a") as output_file:
                _add_dataset_descriptions(output_file, logger)

        except Exception as e:
            logger.warning(f"Could not add dataset descriptions: {e}")

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


def _add_dataset_descriptions(hdf5_file: h5py.File, logger) -> None:
    """
    Add comprehensive descriptions to all datasets in the HDF5 file.

    Args:
        hdf5_file: Open HDF5 file handle
        logger: Logger instance
    """
    descriptions_added = 0

    def add_description(name, obj):
        nonlocal descriptions_added  # Allow modification of outer variable
        if isinstance(obj, h5py.Dataset):
            # Extract dataset name from full path
            dataset_name = name.split("/")[-1]

            # Get description for this dataset
            description = get_dataset_description(dataset_name)

            # Add description as attribute
            try:
                obj.attrs["Description"] = description
                descriptions_added += 1
            except Exception as e:
                logger.warning(f"Could not add description to {name}: {e}")

    # Visit all datasets in the file
    hdf5_file.visititems(add_description)

    logger.info(f"Added descriptions to {descriptions_added} datasets")


if __name__ == "__main__":
    main()
