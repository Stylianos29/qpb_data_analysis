#!/usr/bin/env python3
"""
PCAC Mass Plateau Extraction Script

This script extracts plateau PCAC mass values from PCAC mass time series
using jackknife analysis and robust plateau detection methods.

The script processes HDF5 files from calculate_PCAC_mass.py, detects
plateau regions, and exports results to CSV.

Usage:
    python extract_plateau_PCAC_mass.py -i pcac_mass_analysis.h5 -o
    output_dir
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import click
import numpy as np
import pandas as pd
import h5py

# Import library components
from library.utils.logging_utilities import create_script_logger
from library.validation.click_validators import (
    hdf5_file,
    directory,
    validate_log_filename,
)
from library.data.hdf5_analyzer import HDF5Analyzer

# Import configuration and core utilities
from src.analysis.plateau_extraction._plateau_extraction_shared_config import (
    PLATEAU_DETECTION_SIGMA_THRESHOLDS,
    MIN_PLATEAU_SIZE,
    METADATA_DATASETS,
    CSV_OUTPUT_CONFIG,
    ERROR_HANDLING,
)
from src.analysis.plateau_extraction._pcac_plateau_config import (
    INPUT_DATASETS,
    TIME_OFFSET,
    APPLY_SYMMETRIZATION,
    SYMMETRIZATION_TRUNCATION,
    PLATEAU_SEARCH_RANGE,
    DEFAULT_OUTPUT_HDF5_FILENAME,
    DEFAULT_OUTPUT_CSV_FILENAME,
    OUTPUT_COLUMN_PREFIX,
    validate_pcac_config,
)
from src.analysis.plateau_extraction._plateau_extraction_core import (
    symmetrize_time_series,
    calculate_jackknife_statistics,
    process_single_group,
)


def _load_configuration_labels(group: h5py.Group) -> List[str]:
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


def _extract_group_metadata(group: h5py.Group) -> Dict:
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


def _apply_preprocessing(
    jackknife_samples: np.ndarray,
    mean_values: np.ndarray,
    error_values: np.ndarray,
    logger,
) -> tuple:
    """Apply symmetrization and truncation if configured."""
    if APPLY_SYMMETRIZATION:
        logger.info("Applying symmetrization to PCAC mass data")
        jackknife_samples = symmetrize_time_series(jackknife_samples)
        mean_values = symmetrize_time_series(mean_values)
        error_values = symmetrize_time_series(error_values)

        if SYMMETRIZATION_TRUNCATION:
            half_length = len(mean_values) // 2
            jackknife_samples = jackknife_samples[:, :half_length]
            mean_values = mean_values[:half_length]
            error_values = error_values[:half_length]
            logger.info(f"Truncated to half length: {half_length} points")

    return jackknife_samples, mean_values, error_values


def _process_analysis_group(
    group: h5py.Group,
    group_name: str,
    logger,
) -> Dict:
    """Process a single analysis group to extract plateau."""
    # Validate required datasets
    jackknife_samples_dataset = group[INPUT_DATASETS["samples"]]
    if not isinstance(jackknife_samples_dataset, h5py.Dataset):
        return {}
    mean_values_dataset = group[INPUT_DATASETS["mean"]]
    if not isinstance(mean_values_dataset, h5py.Dataset):
        return {}
    error_values_dataset = group[INPUT_DATASETS["error"]]
    if not isinstance(error_values_dataset, h5py.Dataset):
        return {}

    # Load data
    jackknife_samples = jackknife_samples_dataset[:]
    mean_values = mean_values_dataset[:]
    error_values = error_values_dataset[:]
    config_labels = _load_configuration_labels(group)

    # Apply preprocessing
    jackknife_samples, mean_values, error_values = _apply_preprocessing(
        jackknife_samples, mean_values, error_values, logger
    )

    # Extract plateau
    result = process_single_group(
        jackknife_samples,
        mean_values,
        error_values,
        config_labels,
        PLATEAU_DETECTION_SIGMA_THRESHOLDS,
        MIN_PLATEAU_SIZE,
        PLATEAU_SEARCH_RANGE,
        logger,
    )

    # Add metadata
    result["group_name"] = group_name
    result["metadata"] = _extract_group_metadata(group)

    return result


def _create_csv_record(result: Dict) -> Dict:
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
        record[f"{OUTPUT_COLUMN_PREFIX}_plateau_mean"] = plateau_value.mean
        record[f"{OUTPUT_COLUMN_PREFIX}_plateau_error"] = plateau_value.sdev
        record[f"{OUTPUT_COLUMN_PREFIX}_plateau_start_time"] = (
            plateau_bounds[0] + TIME_OFFSET
        )
        record[f"{OUTPUT_COLUMN_PREFIX}_plateau_end_time"] = (
            plateau_bounds[1] + TIME_OFFSET
        )
        record[f"{OUTPUT_COLUMN_PREFIX}_plateau_n_points"] = (
            plateau_bounds[1] - plateau_bounds[0]
        )

        # Add statistics
        record["n_successful_samples"] = result["n_samples"]
        record["n_total_samples"] = result["n_samples"]
        record["n_failed_samples"] = 0

        # Add diagnostics if configured
        if CSV_OUTPUT_CONFIG["include_diagnostics"]:
            record["estimation_method"] = result["diagnostics"]["method"]
            record["sigma_threshold_used"] = result["sigma_threshold"]
    else:
        # Failed extraction
        record[f"{OUTPUT_COLUMN_PREFIX}_plateau_mean"] = np.nan
        record[f"{OUTPUT_COLUMN_PREFIX}_plateau_error"] = np.nan
        record[f"{OUTPUT_COLUMN_PREFIX}_plateau_start_time"] = np.nan
        record[f"{OUTPUT_COLUMN_PREFIX}_plateau_end_time"] = np.nan
        record[f"{OUTPUT_COLUMN_PREFIX}_plateau_n_points"] = np.nan
        record["n_successful_samples"] = 0
        record["n_total_samples"] = result.get("n_samples", 0)
        record["n_failed_samples"] = result.get("n_samples", 0)
        record["error_message"] = result.get("error_message", "Unknown error")

    return record


def _process_all_groups(
    input_file: str,
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
                if group_path in hdf5_file:
                    group = hdf5_file[group_path]
                    if not isinstance(group, h5py.Group):
                        logger.error(f"Object {group_path} is not a group")
                        continue
                    if all(dataset in group for dataset in INPUT_DATASETS.values()):
                        valid_groups.append(group_path)

            if not valid_groups:
                logger.warning(f"No groups found with required datasets")
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
                    logger.error(f"Object {group_path} is not a group")
                    continue
                result = _process_analysis_group(group, group_name, logger)
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


def _export_to_csv(
    results: List[Dict],
    output_file: str,
    logger,
) -> None:
    """Export extraction results to CSV file."""
    if not results:
        logger.warning("No results to export")
        return

    # Convert results to records
    records = [_create_csv_record(result) for result in results]

    # Create DataFrame
    df = pd.DataFrame(records)

    # Save to CSV with configured precision
    df.to_csv(
        output_file,
        index=False,
        float_format=f"%.{CSV_OUTPUT_CONFIG['float_precision']}f",
        sep=CSV_OUTPUT_CONFIG["delimiter"],
    )

    logger.info(f"Exported {len(records)} results to {output_file}")


@click.command()
@click.option(
    "-i",
    "--input_hdf5_file",
    required=True,
    callback=hdf5_file.input,
    help="Path to input HDF5 file containing PCAC mass analysis results.",
)
@click.option(
    "-o",
    "--output_directory",
    required=True,
    callback=directory.can_create,
    help="Directory for output CSV file.",
)
@click.option(
    "-out_h5",
    "--output_hdf5_filename",
    default=DEFAULT_OUTPUT_HDF5_FILENAME,
    help=f"Output HDF5 filename. Default: {DEFAULT_OUTPUT_HDF5_FILENAME}",
)
@click.option(
    "-out_csv",
    "--output_csv_filename",
    default=DEFAULT_OUTPUT_CSV_FILENAME,
    help=f"Output CSV filename. Default: {DEFAULT_OUTPUT_CSV_FILENAME}",
)
@click.option(
    "-log_on",
    "--enable_logging",
    is_flag=True,
    default=False,
    help="Enable logging to file.",
)
@click.option(
    "-log_dir",
    "--log_directory",
    default=None,
    callback=directory.can_create,
    help="Directory for log files.",
)
@click.option(
    "-log_name",
    "--log_filename",
    default=None,
    callback=validate_log_filename,
    help="Custom log filename.",
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
    output_directory: str,
    output_hdf5_filename: str,
    output_csv_filename: str,
    enable_logging: bool,
    log_directory: Optional[str],
    log_filename: Optional[str],
    verbose: bool,
) -> None:
    """
    Extract plateau PCAC mass values from PCAC mass time series.

    This script processes PCAC mass jackknife samples, detects plateau
    regions, and exports results to CSV format.
    """
    # Validate configuration
    if not validate_pcac_config():
        click.echo("❌ Invalid configuration detected.", err=True)
        sys.exit(1)

    # Setup logging
    if enable_logging:
        log_dir = log_directory or output_directory
    else:
        log_dir = None

    logger = create_script_logger(
        log_directory=log_dir,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=verbose,
    )

    logger.log_script_start("PCAC mass plateau extraction")

    try:
        # Log parameters
        logger.info(f"Input file: {input_hdf5_file}")
        logger.info(f"Output directory: {output_directory}")
        logger.info(f"Output CSV: {output_csv_filename}")

        # Process all groups
        results = _process_all_groups(input_hdf5_file, logger, verbose)

        if not results:
            logger.warning("No results obtained from processing")
            click.echo("⚠️ No results to export", err=True)
            sys.exit(1)

        # Export to CSV
        output_path = os.path.join(output_directory, output_csv_filename)
        _export_to_csv(results, output_path, logger)

        # Report summary
        n_success = sum(1 for r in results if r["success"])
        n_total = len(results)

        logger.log_script_end(f"Extraction complete: {n_success}/{n_total} successful")
        click.echo(
            f"✅ Plateau extraction complete: {n_success}/{n_total} successful\n"
            f"   Results saved to: {output_path}"
        )

    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.log_script_end("PCAC mass plateau extraction failed")
        click.echo(f"❌ ERROR: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
