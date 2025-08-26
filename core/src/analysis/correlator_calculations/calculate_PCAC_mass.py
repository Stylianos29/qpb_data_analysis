#!/usr/bin/env python3
"""
PCAC mass calculation script for QPB correlator data analysis.

This script calculates PCAC (Partially Conserved Axial Current) mass
from jackknife-analyzed correlator data stored in HDF5 format. The PCAC
mass is calculated using the formula:

    PCAC_mass = 0.5 * g4g5g5_derivative / g5g5_truncated

where g5g5 correlators are truncated to match the length of the
derivative correlators. The script processes each deepest-level group
independently and exports jackknife samples, means, and errors in a new
HDF5 file.

Key features:
    - Configurable truncation parameters
    - Jackknife error propagation
    - Preservation of group hierarchy and metadata
    - Comprehensive validation and error handling
    - Support for alternative dataset naming conventions

Place this file as:
qpb_data_analysis/core/src/analysis/correlator_calculations/calculate_PCAC_mass.py

Usage:
    python calculate_PCAC_mass.py -i jackknife_analysis.h5 -o
    pcac_mass.h5 [options]
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import h5py

# Import from existing library
from library.data.hdf5_analyzer import HDF5Analyzer
from library.validation.click_validators import (
    hdf5_file,
    directory,
    validate_log_filename,
)
from library.utils.logging_utilities import create_script_logger

# Import configuration
from ._pcac_mass_config import (
    TRUNCATE_START,
    TRUNCATE_END,
    EXPECTED_G5G5_LENGTH,
    EXPECTED_DERIVATIVE_LENGTH,
    EXPECTED_PCAC_LENGTH,
    REQUIRED_INPUT_DATASETS,
    ALTERNATIVE_DATASET_NAMES,
    PCAC_MASS_DATASETS,
    METADATA_DATASETS,
    PCAC_MASS_FACTOR,
    VALIDATION_PARAMS,
    ERROR_HANDLING,
    OUTPUT_STRUCTURE,
    LOGGING_CONFIG,
    get_output_filename,
    validate_configuration,
)

# Import from common module
from ._correlator_analysis_core import (
    calculate_jackknife_statistics,
    validate_jackknife_consistency,
    truncate_correlator,
    calculate_pcac_mass,
    validate_correlator_dimensions,
    check_correlator_physicality,
    process_correlator_group,
    copy_parent_attributes,
)


@click.command()
@click.option(
    "-i",
    "--input_hdf5_file",
    required=True,
    callback=hdf5_file.input,
    help="Path to input HDF5 file containing jackknife analysis results.",
)
@click.option(
    "-o",
    "--output_hdf5_file",
    required=True,
    callback=hdf5_file.output,
    help="Path for output HDF5 file with PCAC mass results.",
)
@click.option(
    "-out_dir",
    "--output_directory",
    default=None,
    callback=directory.must_exist,
    help="Directory for output files. If not specified, uses input file directory.",
)
@click.option(
    "--skip_validation",
    is_flag=True,
    default=False,
    help="Skip physical validation checks (use with caution).",
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
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output showing processing progress.",
)
def main(
    input_hdf5_file: str,
    output_hdf5_file: str,
    output_directory: Optional[str],
    skip_validation: bool,
    enable_logging: bool,
    log_directory: Optional[str],
    log_filename: Optional[str],
    verbose: bool,
) -> None:
    """
    Calculate PCAC mass from jackknife-analyzed correlator data.

    This script processes QPB correlator data that has undergone jackknife
    analysis and calculates the PCAC (Partially Conserved Axial Current)
    mass using the formula:

        PCAC_mass = 0.5 * g4g5g5_derivative / g5g5_truncated

    The script preserves the group hierarchy and essential metadata while
    creating a clean output file containing only PCAC mass results.
    """
    # Validate configuration first
    try:
        if not validate_configuration():
            click.echo("ERROR: Invalid configuration detected", err=True)
            sys.exit(1)
    except ValueError as e:
        click.echo(f"ERROR: Configuration validation failed: {e}", err=True)
        sys.exit(1)

    # Setup directories
    if output_directory is None:
        output_directory = os.path.dirname(input_hdf5_file)
    
    if log_directory is None and enable_logging:
        log_directory = output_directory

    # Create full output path
    output_file_path = os.path.join(output_directory, output_hdf5_file)

    # Setup logging
    logger = create_script_logger(
        log_directory=log_directory if enable_logging else None,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=False,
    )

    logger.log_script_start("PCAC mass calculation")

    try:
        # Log configuration
        logger.info(f"Input HDF5 file: {input_hdf5_file}")
        logger.info(f"Output HDF5 file: {output_file_path}")
        logger.info(f"Truncation parameters: start={TRUNCATE_START}, end={TRUNCATE_END}")
        logger.info(f"PCAC mass factor: {PCAC_MASS_FACTOR}")
        logger.info(f"Skip validation: {skip_validation}")

        # Process the file
        successful, failed = _process_pcac_mass_file(
            input_hdf5_file,
            output_file_path,
            skip_validation,
            logger,
            verbose,
        )

        # Report results
        total = successful + failed
        logger.info(f"Processing complete: {successful}/{total} groups successful")

        if failed > 0:
            logger.warning(f"{failed} groups failed processing")
            if not ERROR_HANDLING["skip_invalid_groups"]:
                raise RuntimeError(f"Failed to process {failed} groups")

        logger.log_script_end("PCAC mass calculation completed successfully")
        
        # Console output
        click.echo(f"✓ PCAC mass calculation complete")
        click.echo(f"  Processed: {successful}/{total} groups")
        click.echo(f"  Output: {output_file_path}")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.log_script_end("PCAC mass calculation failed")
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)


def _process_pcac_mass_file(
    input_file_path: str,
    output_file_path: str,
    skip_validation: bool,
    logger,
    verbose: bool,
) -> Tuple[int, int]:
    """
    Process the HDF5 file and calculate PCAC mass for all groups.

    Args:
        input_file_path: Path to input HDF5 file
        output_file_path: Path for output HDF5 file
        skip_validation: Whether to skip validation checks
        logger: Logger instance
        verbose: Whether to show verbose output

    Returns:
        Tuple of (successful_count, failed_count)
    """
    successful = 0
    failed = 0

    # Use HDF5Analyzer to inspect file structure
    with HDF5Analyzer(input_file_path) as analyzer:
        # Find groups with required datasets
        analysis_groups = _find_analysis_groups(analyzer, logger)

        if not analysis_groups:
            raise ValueError(
                "No groups with required PCAC mass datasets found. "
                f"Looking for: {REQUIRED_INPUT_DATASETS}"
            )

        total_groups = len(analysis_groups)
        logger.info(f"Found {total_groups} groups to process")

        # Process with HDF5 files
        with h5py.File(input_file_path, "r") as input_file, h5py.File(
            output_file_path, "w"
        ) as output_file:

            # Add file-level attributes
            output_file.attrs["pcac_mass_factor"] = PCAC_MASS_FACTOR
            output_file.attrs["truncate_start"] = TRUNCATE_START
            output_file.attrs["truncate_end"] = TRUNCATE_END

            # Process each group
            for idx, group_path in enumerate(analysis_groups, 1):
                if verbose and idx % LOGGING_CONFIG["progress_interval"] == 0:
                    click.echo(f"  Processing group {idx}/{total_groups}...")

                try:
                    # Create output group
                    output_group = output_file.create_group(group_path)

                    # Copy parent attributes
                    copy_parent_attributes(input_file, output_file, group_path)

                    # Process the group
                    _process_single_pcac_mass_group(
                        input_file[group_path],
                        output_group,
                        skip_validation,
                        logger,
                    )

                    successful += 1
                    logger.debug(f"Successfully processed: {group_path}")

                except Exception as e:
                    failed += 1
                    logger.error(f"Failed to process {group_path}: {e}")
                    
                    if ERROR_HANDLING["verbose_errors"]:
                        logger.debug(f"Detailed error: {e}", exc_info=True)
                    
                    if not ERROR_HANDLING["skip_invalid_groups"]:
                        raise

    return successful, failed


def _find_analysis_groups(analyzer: HDF5Analyzer, logger) -> List[str]:
    """
    Find all groups containing required datasets for PCAC mass calculation.

    Args:
        analyzer: HDF5Analyzer instance
        logger: Logger instance

    Returns:
        List of group paths containing required datasets
    """
    analysis_groups = []

    for group_path in analyzer.active_groups:
        # Get datasets in this group
        try:
            with h5py.File(analyzer.file_path, "r") as f:
                if group_path in f:
                    group = f[group_path]
                    
                    # Check for both required datasets
                    has_all_required = True
                    found_datasets = {}
                    
                    for required_dataset in REQUIRED_INPUT_DATASETS:
                        found = False
                        
                        # Check primary name
                        if required_dataset in group:
                            found = True
                            found_datasets[required_dataset] = required_dataset
                        else:
                            # Check alternatives
                            for alt_name in ALTERNATIVE_DATASET_NAMES.get(required_dataset, []):
                                if alt_name in group:
                                    found = True
                                    found_datasets[required_dataset] = alt_name
                                    break
                        
                        if not found:
                            has_all_required = False
                            break
                    
                    if has_all_required:
                        analysis_groups.append(group_path)
                        logger.debug(
                            f"Group {group_path} has all required datasets: {found_datasets}"
                        )
                        
        except Exception as e:
            logger.warning(f"Could not check group {group_path}: {e}")
            continue

    logger.debug(f"Found {len(analysis_groups)} groups with all required datasets")
    return analysis_groups


def _process_single_pcac_mass_group(
    input_group: h5py.Group,
    output_group: h5py.Group,
    skip_validation: bool,
    logger,
) -> None:
    """
    Process a single group to calculate PCAC mass.

    Args:
        input_group: Input HDF5 group
        output_group: Output HDF5 group
        skip_validation: Whether to skip validation
        logger: Logger instance
    """
    group_name = input_group.name

    # Read required datasets (with alternative names support)
    g4g5g5_derivative_samples = _read_dataset_with_alternatives(
        input_group,
        REQUIRED_INPUT_DATASETS[0],
        ALTERNATIVE_DATASET_NAMES.get(REQUIRED_INPUT_DATASETS[0], []),
    )
    
    g5g5_samples = _read_dataset_with_alternatives(
        input_group,
        REQUIRED_INPUT_DATASETS[1],
        ALTERNATIVE_DATASET_NAMES.get(REQUIRED_INPUT_DATASETS[1], []),
    )

    # Validate dimensions
    validate_correlator_dimensions(
        g4g5g5_derivative_samples,
        EXPECTED_DERIVATIVE_LENGTH,
        "g4g5g5_derivative",
        group_name,
    )
    
    validate_correlator_dimensions(
        g5g5_samples,
        EXPECTED_G5G5_LENGTH,
        "g5g5",
        group_name,
    )

    # Validate consistency between datasets
    n_samples = validate_jackknife_consistency(
        {
            "g4g5g5_derivative": g4g5g5_derivative_samples,
            "g5g5": g5g5_samples,
        },
        group_name,
    )

    # Check minimum samples requirement
    if n_samples < VALIDATION_PARAMS["min_jackknife_samples"]:
        raise ValueError(
            f"Insufficient jackknife samples in {group_name}: "
            f"got {n_samples}, minimum required {VALIDATION_PARAMS['min_jackknife_samples']}"
        )

    logger.debug(f"Processing {group_name} with {n_samples} jackknife samples")

    # Physical validation of g5g5 (if not skipped)
    if not skip_validation:
        issues = check_correlator_physicality(
            g5g5_samples,
            "g5g5",
            check_positive=VALIDATION_PARAMS["check_g5g5_positive"],
            check_decreasing=VALIDATION_PARAMS["check_g5g5_decreasing"],
            min_value=VALIDATION_PARAMS["min_correlator_value"],
        )
        
        if issues:
            for issue in issues:
                logger.warning(f"{group_name}: {issue}")
            
            if not ERROR_HANDLING["skip_invalid_groups"]:
                raise ValueError(f"Validation failed for {group_name}")

    # Truncate g5g5 to match derivative length
    g5g5_truncated = truncate_correlator(g5g5_samples, TRUNCATE_START, TRUNCATE_END)
    
    # Verify truncation worked correctly
    if g5g5_truncated.shape[-1] != EXPECTED_DERIVATIVE_LENGTH:
        raise ValueError(
            f"Truncation failed: expected length {EXPECTED_DERIVATIVE_LENGTH}, "
            f"got {g5g5_truncated.shape[-1]}"
        )

    # Calculate PCAC mass
    pcac_mass_samples = calculate_pcac_mass(
        g4g5g5_derivative_samples,
        g5g5_truncated,
        PCAC_MASS_FACTOR,
    )

    # Validate PCAC mass results
    if not skip_validation:
        _validate_pcac_mass_results(pcac_mass_samples, group_name, logger)

    # Calculate statistics
    mean_values, error_values = calculate_jackknife_statistics(pcac_mass_samples)

    # Save results
    output_group.create_dataset(
        PCAC_MASS_DATASETS["jackknife_samples"],
        data=pcac_mass_samples,
        compression=OUTPUT_STRUCTURE["compression"],
        compression_opts=OUTPUT_STRUCTURE["compression_level"],
    )
    output_group.create_dataset(
        PCAC_MASS_DATASETS["mean_values"],
        data=mean_values,
    )
    output_group.create_dataset(
        PCAC_MASS_DATASETS["error_values"],
        data=error_values,
    )

    # Copy metadata
    for metadata_name in METADATA_DATASETS:
        if metadata_name in input_group:
            item = input_group[metadata_name]
            if isinstance(item, h5py.Dataset):
                output_group.create_dataset(metadata_name, data=item[:])

    # Copy group attributes
    for attr_name, attr_value in input_group.attrs.items():
        output_group.attrs[attr_name] = attr_value

    # Add processing metadata
    output_group.attrs["pcac_mass_factor"] = PCAC_MASS_FACTOR
    output_group.attrs["truncate_start"] = TRUNCATE_START
    output_group.attrs["truncate_end"] = TRUNCATE_END
    output_group.attrs["n_jackknife_samples"] = n_samples

    logger.debug(f"Successfully processed {group_name}")


def _read_dataset_with_alternatives(
    group: h5py.Group,
    primary_name: str,
    alternative_names: List[str],
) -> np.ndarray:
    """
    Read dataset with support for alternative names.

    Args:
        group: HDF5 group to read from
        primary_name: Primary dataset name
        alternative_names: List of alternative names

    Returns:
        Dataset array

    Raises:
        KeyError: If no valid dataset is found
    """
    # Try primary name first
    if primary_name in group:
        item = group[primary_name]
        if isinstance(item, h5py.Dataset):
            return item[:]

    # Try alternative names
    for alt_name in alternative_names:
        if alt_name in group:
            item = group[alt_name]
            if isinstance(item, h5py.Dataset):
                return item[:]

    # Not found
    available = list(group.keys())
    raise KeyError(
        f"Dataset '{primary_name}' not found in group {group.name}. "
        f"Tried: {primary_name}, {alternative_names}. "
        f"Available datasets: {available}"
    )


def _validate_pcac_mass_results(
    pcac_mass_samples: np.ndarray,
    group_name: str,
    logger,
) -> None:
    """
    Validate PCAC mass calculation results.

    Args:
        pcac_mass_samples: Calculated PCAC mass samples
        group_name: Name of the group for error messages
        logger: Logger instance

    Raises:
        ValueError: If validation fails
    """
    # Check for NaN or inf
    if VALIDATION_PARAMS["check_invalid_values"]:
        n_nan = np.sum(np.isnan(pcac_mass_samples))
        n_inf = np.sum(np.isinf(pcac_mass_samples))
        
        if n_nan > 0:
            logger.warning(f"{group_name}: PCAC mass contains {n_nan} NaN values")
            
            if not ERROR_HANDLING["skip_invalid_groups"]:
                raise ValueError(f"NaN values in PCAC mass for {group_name}")
        
        if n_inf > 0:
            logger.warning(f"{group_name}: PCAC mass contains {n_inf} infinite values")
            
            if not ERROR_HANDLING["skip_invalid_groups"]:
                raise ValueError(f"Infinite values in PCAC mass for {group_name}")

    # Check maximum value
    max_pcac = VALIDATION_PARAMS["max_pcac_mass"]
    if np.any(np.abs(pcac_mass_samples) > max_pcac):
        max_found = np.max(np.abs(pcac_mass_samples))
        logger.warning(
            f"{group_name}: PCAC mass exceeds maximum allowed value "
            f"({max_found:.3f} > {max_pcac})"
        )
        
        if not ERROR_HANDLING["skip_invalid_groups"]:
            raise ValueError(f"PCAC mass exceeds limits for {group_name}")

    # Check expected dimensions
    if pcac_mass_samples.shape[-1] != EXPECTED_PCAC_LENGTH:
        raise ValueError(
            f"Unexpected PCAC mass length: got {pcac_mass_samples.shape[-1]}, "
            f"expected {EXPECTED_PCAC_LENGTH}"
        )


if __name__ == "__main__":
    main()
