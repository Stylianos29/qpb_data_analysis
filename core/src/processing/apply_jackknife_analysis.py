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
      format with processed parameter values

Key improvements in this version:
    - Uses processed parameters from Stage 2A CSV as single source of
      truth
    - CSV-driven grouping using DataFrameAnalyzer ensures ALL parameters
      are used
    - Integrates PlotFilenameBuilder for consistent group naming
    - Implements graceful error handling for filename mismatches
    - Provides detailed user feedback on processing statistics

Usage:
    python apply_jackknife_analysis.py -i input.h5 -csv processed.csv -o
    output.h5 [options]
"""

from asyncio.log import logger
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List

import click
import numpy as np
import pandas as pd

# Import library components
from library.data.hdf5_analyzer import HDF5Analyzer
from library.data.analyzer import DataFrameAnalyzer
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
    GROUPING_PARAMETERS,
    INPUT_CORRELATOR_DATASETS,
    REQUIRED_INPUT_DATASETS,
    MIN_GAUGE_CONFIGURATIONS,
)
from src.processing._jackknife_processor import JackknifeProcessor
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
        3. Grouping by tunable parameters using DataFrameAnalyzer
        4. Applying jackknife resampling to each group
        5. Computing finite difference derivatives
        6. Exporting results with processed parameter values from CSV
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
        logger.log_script_start("Jackknife analysis with CSV-driven grouping")

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
        input_hdf5_analyzer = HDF5Analyzer(input_hdf5_file)
        logger.info(f"HDF5 file loaded successfully")
        logger.info(
            f"Total HDF5 groups available: {len(input_hdf5_analyzer.active_groups)}"
        )

        # === VALIDATE REQUIRED DATASETS ===

        available_datasets = input_hdf5_analyzer.list_of_output_quantity_names_from_hdf5
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

        # === GROUP DATA BY PARAMETERS (CSV-DRIVEN) ===

        logger.info("=" * 80)
        logger.info("GROUPING DATA USING CSV PARAMETERS")
        logger.info("=" * 80)

        # Create analyzer for CSV
        processed_csv_analyzer = DataFrameAnalyzer(processed_df)

        # Filter GROUPING_PARAMETERS to only include parameters that
        # exist in this dataset
        available_multivalued = (
            processed_csv_analyzer.list_of_multivalued_tunable_parameter_names
        )
        actual_filter_params = [
            p for p in GROUPING_PARAMETERS if p in available_multivalued
        ]

        logger.info(
            f"Actual parameters to filter (present in dataset): {actual_filter_params}"
        )

        # Group CSV using DataFrameAnalyzer method
        csv_grouped = processed_csv_analyzer.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=actual_filter_params,  # Use filtered list
            verbose=False,
        )

        logger.info(f"Found {len(csv_grouped)} parameter groups in CSV")

        # Get the actual grouping parameter names used
        grouping_params = (
            processed_csv_analyzer.reduced_multivalued_tunable_parameter_names_list
        )
        logger.info(f"Grouping by parameters: {grouping_params}")

        # For each CSV group, collect corresponding HDF5 paths
        grouped_data = {}
        csv_filenames_not_found = []
        total_csv_files = 0

        for group_index, (group_values, group_df) in enumerate(csv_grouped, 1):
            # Create a tuple key for this group
            if len(grouping_params) == 1:
                group_key = (grouping_params[0], group_values)
            else:
                group_key = tuple(zip(grouping_params, group_values))

            # Get filenames from this CSV group
            csv_filenames = group_df["Filename"].tolist()
            total_csv_files += len(csv_filenames)

            # Convert to .dat extensions (correlator filenames)
            dat_filenames = [os.path.splitext(f)[0] + ".dat" for f in csv_filenames]

            # Create a set for fast lookup
            dat_filenames_set = set(dat_filenames)

            # Find corresponding HDF5 group paths
            hdf5_paths = []
            for path in input_hdf5_analyzer.active_groups:
                # Extract filename from HDF5 path
                hdf5_filename = path.split("/")[-1]

                # Check if this HDF5 group matches any CSV filename
                if hdf5_filename in dat_filenames_set:
                    hdf5_paths.append(path)

            if hdf5_paths:
                grouped_data[group_key] = {"paths": hdf5_paths, "dataframe": group_df}
                logger.info(
                    f"Group {group_index}/{len(csv_grouped)}: "
                    f"{dict(group_key) if isinstance(group_key, tuple) else group_key} → "
                    f"{len(hdf5_paths)} files matched"
                )
            else:
                logger.warning(
                    f"Group {group_index}/{len(csv_grouped)}: "
                    f"{dict(group_key) if isinstance(group_key, tuple) else group_key} → "
                    f"No matching HDF5 paths found (CSV has {len(dat_filenames)} files)"
                )
                csv_filenames_not_found.extend(csv_filenames)

        logger.info("=" * 80)
        logger.info(f"CSV-driven grouping complete:")
        logger.info(f"  Total CSV files: {total_csv_files}")
        logger.info(f"  Groups with HDF5 data: {len(grouped_data)}")
        logger.info(f"  CSV files without HDF5 match: {len(csv_filenames_not_found)}")
        logger.info("=" * 80)

        if not grouped_data:
            error_msg = "No valid parameter groups found with matching HDF5 data"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if csv_filenames_not_found:
            logger.warning(
                f"The following CSV files have no matching HDF5 groups "
                f"(first 10 shown):"
            )
            for filename in csv_filenames_not_found[:10]:
                logger.warning(f"  - {filename}")
            if len(csv_filenames_not_found) > 10:
                logger.warning(f"  ... and {len(csv_filenames_not_found) - 10} more")

        # === INITIALIZE PROCESSOR ===

        processor = JackknifeProcessor(
            derivative_method=DEFAULT_DERIVATIVE_METHOD, logger=logger
        )

        # === PROCESS EACH GROUP ===

        all_processing_results = {}
        processed_groups = 0
        skipped_insufficient_configs = 0
        compression = DEFAULT_COMPRESSION
        compression_level = DEFAULT_COMPRESSION_LEVEL

        for group_index, (param_values, group_data) in enumerate(
            grouped_data.items(), 1
        ):
            # Unpack the dictionary
            group_paths = group_data["paths"]
            group_df = group_data["dataframe"]
            # group_paths is List[str] of HDF5 paths

            # Note: Group naming is handled by _hdf5_output.py using
            # PlotFilenameBuilder. We use a temporary identifier here
            # for tracking
            temp_group_id = f"group_{group_index}"

            logger.info("")
            logger.info(
                f"Processing group {group_index}/{len(grouped_data)}: "
                f"{len(group_paths)} configurations"
            )
            logger.debug(f"Group parameters: {param_values}")

            # === LOAD CORRELATOR DATA ===

            logger.debug(f"Loading g5-g5 correlator data for group {group_index}")
            g5g5_data = []
            g4g5g5_data = []
            config_labels = []
            qpb_filenames = []

            for path in group_paths:
                try:
                    # Load g5-g5 correlator
                    g5g5_values = input_hdf5_analyzer.dataset_values(
                        INPUT_CORRELATOR_DATASETS["g5g5"], group_path=path
                    )
                    g5g5_data.append(g5g5_values)

                    # Load g4g5-g5 correlator
                    g4g5g5_values = input_hdf5_analyzer.dataset_values(
                        INPUT_CORRELATOR_DATASETS["g4g5g5"], group_path=path
                    )
                    g4g5g5_data.append(g4g5g5_values)

                    # Extract metadata
                    filename = path.split("/")[
                        -1
                    ]  # Get filename from path (e.g., "something.dat")
                    qpb_filenames.append(filename)

                    # Extract configuration label from CSV
                    csv_filename = filename.replace(
                        ".dat", ".txt"
                    )  # Convert to CSV format
                    matching_row = group_df[group_df["Filename"] == csv_filename]

                    if not matching_row.empty:
                        config_label = str(matching_row["Configuration_label"].iloc[0])
                        config_labels.append(config_label)
                    else:
                        logger.warning(
                            f"Filename {csv_filename} not found in group DataFrame. "
                            f"Using placeholder."
                        )
                        config_labels.append(f"unknown_{len(config_labels)}")

                except Exception as e:
                    logger.warning(f"Failed to load data from {path}: {e}")
                    continue

            if len(g5g5_data) < min_configurations:
                logger.warning(
                    f"Group {group_index} has insufficient configurations "
                    f"({len(g5g5_data)} < {min_configurations}). Skipping."
                )
                skipped_insufficient_configs += 1
                continue

            # Stack into arrays
            g5g5_array = np.array(g5g5_data)
            g4g5g5_array = np.array(g4g5g5_data)

            logger.info(
                f"Loaded data shapes: g5g5={g5g5_array.shape}, "
                f"g4g5g5={g4g5g5_array.shape}"
            )

            # === BUILD METADATA BEFORE PROCESSING === This is needed by
            # the processor
            group_metadata = {
                "configuration_labels": config_labels,
                "qpb_filenames": qpb_filenames,
            }

            # === APPLY JACKKNIFE PROCESSING ===

            logger.debug(f"Applying jackknife processing to group {group_index}")
            try:
                jackknife_results = processor.process_correlator_group(
                    g5g5_data=g5g5_array,
                    g4g5g5_data=g4g5g5_array,
                    group_metadata=group_metadata,
                    min_configurations=min_configurations,
                )

                logger.info(
                    f"Jackknife processing complete for group {group_index}: "
                    f"{jackknife_results['n_gauge_configurations']} configurations"
                )

            except Exception as e:
                logger.error(
                    f"Jackknife processing failed for group {group_index}: {e}"
                )
                continue

            # === STORE RESULTS ===

            # Store complete results for this group
            all_processing_results[temp_group_id] = {
                "jackknife_results": jackknife_results,
                "config_metadata": group_metadata,
                "group_paths": group_paths,  # Keep for reference
            }

            processed_groups += 1
            logger.info(f"✓ Group {group_index} processing complete")

        # === VALIDATE RESULTS ===

        if processed_groups == 0:
            error_msg = "No groups were successfully processed"
            logger.critical(error_msg)
            click.echo(f"ERROR: {error_msg}")
            sys.exit(1)

        logger.info("")
        logger.info(
            f"Successfully processed {processed_groups}/{len(grouped_data)} groups "
            f"(skipped {skipped_insufficient_configs} with insufficient configurations)"
        )

        # === CREATE CUSTOM HDF5 OUTPUT ===

        logger.info(f"Creating output HDF5 file: {output_path}")

        successful_groups, skipped_groups, skipped_filenames = (
            _create_custom_hdf5_output(
                output_path=output_path,
                all_processing_results=all_processing_results,
                input_hdf5_analyzer=input_hdf5_analyzer,
                processed_params_df=processed_df,
                compression=compression,
                compression_level=compression_level,
                logger=logger,
            )
        )

        # === SCRIPT COMPLETION WITH DETAILED SUMMARY ===

        logger.log_script_end(
            f"Processed {successful_groups}/{processed_groups} groups, "
            f"saved to {output_path.name}"
        )

        # === SUCCESS SUMMARY FOR CONSOLE ===

        if verbose or not enable_logging:
            click.echo("")
            click.echo("=" * 70)
            click.echo("  JACKKNIFE ANALYSIS COMPLETED")
            click.echo("=" * 70)
            click.echo(f"✓ CSV-driven grouping: {len(csv_grouped)} parameter groups")
            click.echo(f"✓ Groups with sufficient data: {processed_groups}")

            if skipped_insufficient_configs > 0:
                click.echo(
                    f"✓ Groups skipped (< {min_configurations} configs): {skipped_insufficient_configs}"
                )

            click.echo(
                f"✓ Successfully processed: {successful_groups}/{processed_groups} groups"
            )
            click.echo(
                f"✓ Processed parameters from: {Path(processed_parameters_csv).name}"
            )
            click.echo(f"✓ Results saved to: {output_path}")

            # Warning about skipped groups
            if skipped_groups > 0:
                click.echo("")
                click.echo(
                    f"⚠ Warning: {skipped_groups} groups skipped due to filename mismatches"
                )
                click.echo("  Skipped filenames:")

                # Show first 5 skipped filenames
                for filename in skipped_filenames[:5]:
                    click.echo(f"    - {filename}")

                # Indicate if there are more
                if len(skipped_filenames) > 5:
                    remaining = len(skipped_filenames) - 5
                    click.echo(f"    ... and {remaining} more")

                click.echo("")
                click.echo("  This usually indicates:")
                click.echo("    • Stage 1B and Stage 2A processed different file sets")
                click.echo("    • Incomplete or corrupted Stage 2A output")
                click.echo("")
                click.echo(f"  See log file for details: {log_directory}")

            # Warning about CSV files without HDF5 matches
            if csv_filenames_not_found:
                click.echo("")
                click.echo(
                    f"⚠ Note: {len(csv_filenames_not_found)} CSV files had no matching HDF5 data"
                )
                click.echo(
                    "  This means Stage 2A processed files that Stage 1B did not."
                )
                click.echo("  These files were skipped (see log for details).")

            click.echo("=" * 70)

    except Exception as e:
        logger.log_script_error(e)
        click.echo(f"ERROR: Critical failure during processing: {e}")
        sys.exit(1)

    finally:
        # Clean up resources
        if "input_hdf5_analyzer" in locals():
            input_hdf5_analyzer.close()
        logger.close()


if __name__ == "__main__":
    main()
