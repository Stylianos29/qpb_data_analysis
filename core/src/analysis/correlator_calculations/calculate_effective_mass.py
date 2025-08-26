#!/usr/bin/env python3
"""
Effective mass calculation script for QPB correlator data analysis.

This script calculates effective mass from jackknife-analyzed g5-g5
correlator data stored in HDF5 format. The effective mass is calculated
using the two-state periodic formula by default, with options for other
methods configured in _effective_mass_config.py.

The calculation applies optional symmetrization to g5-g5 correlators and
uses the formula specified in the configuration. The script preserves
the group hierarchy and essential metadata while creating a clean output
file containing only effective mass results.

Key features:
    - Configurable symmetrization and truncation parameters
    - Multiple calculation methods (two-state, single-state, cosh)
    - Jackknife error propagation
    - Comprehensive validation and error handling
    - Preservation of group hierarchy and metadata

Usage:
    python calculate_effective_mass.py -i jackknife_analysis.h5 -o
    effective_mass.h5 [options]
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
from ._effective_mass_config import (
    APPLY_SYMMETRIZATION,
    TRUNCATE_HALF,
    LOWERING_FACTOR,
    EXPECTED_G5G5_LENGTH,
    EXPECTED_EFFECTIVE_MASS_LENGTH,
    REQUIRED_INPUT_DATASETS,
    ALTERNATIVE_DATASET_NAMES,
    EFFECTIVE_MASS_DATASETS,
    PION_EFFECTIVE_MASS_DATASETS,
    METADATA_DATASETS,
    CALCULATION_METHOD,
    VALIDATION_PARAMS,
    ERROR_HANDLING,
    OUTPUT_STRUCTURE,
    LOGGING_CONFIG,
    get_output_filename,
    get_effective_mass_datasets,
    validate_configuration,
)

# Import from common module
from ._correlator_analysis_core import (
    calculate_jackknife_statistics,
    validate_jackknife_consistency,
    symmetrize_correlator,
    calculate_two_state_periodic_effective_mass,
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
    help="Path for output HDF5 file with effective mass results.",
)
@click.option(
    "-out_dir",
    "--output_directory",
    default=None,
    callback=directory.must_exist,
    help="Directory for output files. If not specified, uses input file directory.",
)
@click.option(
    "--use_pion_naming",
    is_flag=True,
    default=False,
    help="Use 'pion_effective_mass' naming convention instead of 'effective_mass'.",
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
    use_pion_naming: bool,
    skip_validation: bool,
    enable_logging: bool,
    log_directory: Optional[str],
    log_filename: Optional[str],
    verbose: bool,
) -> None:
    """
    Calculate effective mass from jackknife-analyzed g5-g5 correlator data.

    This script processes QPB correlator data that has undergone jackknife
    analysis and calculates the effective mass using the method specified
    in the configuration. The default is the two-state periodic effective
    mass formula suitable for periodic boundary conditions.
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

    logger.log_script_start("Effective mass calculation")

    try:
        # Log configuration
        logger.info(f"Input HDF5 file: {input_hdf5_file}")
        logger.info(f"Output HDF5 file: {output_file_path}")
        logger.info(f"Calculation method: {CALCULATION_METHOD['method']}")
        logger.info(f"Apply symmetrization: {APPLY_SYMMETRIZATION}")
        logger.info(f"Truncate half: {TRUNCATE_HALF}")
        logger.info(f"Lowering factor: {LOWERING_FACTOR}")
        logger.info(f"Use pion naming: {use_pion_naming}")
        logger.info(f"Skip validation: {skip_validation}")

        # Get appropriate dataset names
        output_datasets = get_effective_mass_datasets(use_pion_naming)

        # Process the file
        successful, failed = _process_effective_mass_file(
            input_hdf5_file,
            output_file_path,
            output_datasets,
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

        logger.log_script_end("Effective mass calculation completed successfully")
        
        # Console output
        click.echo(f"✓ Effective mass calculation complete")
        click.echo(f"  Processed: {successful}/{total} groups")
        click.echo(f"  Output: {output_file_path}")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.log_script_end("Effective mass calculation failed")
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)


def _process_effective_mass_file(
    input_file_path: str,
    output_file_path: str,
    output_datasets: Dict[str, str],
    skip_validation: bool,
    logger,
    verbose: bool,
) -> Tuple[int, int]:
    """
    Process the HDF5 file and calculate effective mass for all groups.

    Args:
        input_file_path: Path to input HDF5 file
        output_file_path: Path for output HDF5 file
        output_datasets: Dataset name mapping
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
            raise ValueError("No groups with required g5g5 datasets found")

        total_groups = len(analysis_groups)
        logger.info(f"Found {total_groups} groups to process")

        # Process with HDF5 files
        with h5py.File(input_file_path, "r") as input_file, h5py.File(
            output_file_path, "w"
        ) as output_file:

            # Add file-level attributes
            output_file.attrs["calculation_method"] = CALCULATION_METHOD["method"]
            output_file.attrs["symmetrization_applied"] = APPLY_SYMMETRIZATION
            output_file.attrs["truncate_half"] = TRUNCATE_HALF
            output_file.attrs["lowering_factor"] = LOWERING_FACTOR

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
                    _process_single_effective_mass_group(
                        input_file[group_path],
                        output_group,
                        output_datasets,
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
    Find all groups containing required datasets for effective mass calculation.

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
            # Try to get dataset names for this specific group
            # This is a simplified approach - you may need to adjust based on
            # actual HDF5Analyzer API
            with h5py.File(analyzer.file_path, "r") as f:
                if group_path in f:
                    group = f[group_path]
                    
                    # Check for required datasets (with alternatives)
                    has_required = False
                    for required_dataset in REQUIRED_INPUT_DATASETS:
                        if required_dataset in group:
                            has_required = True
                            break
                        
                        # Check alternatives
                        for alt_name in ALTERNATIVE_DATASET_NAMES.get(required_dataset, []):
                            if alt_name in group:
                                has_required = True
                                break
                        
                        if has_required:
                            break
                    
                    if has_required:
                        analysis_groups.append(group_path)
                        
        except Exception as e:
            logger.warning(f"Could not check group {group_path}: {e}")
            continue

    logger.debug(f"Found {len(analysis_groups)} groups with required datasets")
    return analysis_groups


def _process_single_effective_mass_group(
    input_group: h5py.Group,
    output_group: h5py.Group,
    output_datasets: Dict[str, str],
    skip_validation: bool,
    logger,
) -> None:
    """
    Process a single group to calculate effective mass.

    Args:
        input_group: Input HDF5 group
        output_group: Output HDF5 group
        output_datasets: Dataset name mapping
        skip_validation: Whether to skip validation
        logger: Logger instance
    """
    group_name = input_group.name

    # Read g5g5 correlator data (with alternative names support)
    g5g5_samples = _read_g5g5_dataset(input_group)

    # Validate dimensions
    validate_correlator_dimensions(
        g5g5_samples,
        EXPECTED_G5G5_LENGTH,
        "g5g5",
        group_name,
    )

    # Validate consistency
    n_samples = validate_jackknife_consistency(
        {"g5g5": g5g5_samples},
        group_name,
    )

    logger.debug(f"Processing {group_name} with {n_samples} jackknife samples")

    # Physical validation (if not skipped)
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

    # Apply symmetrization if configured
    if APPLY_SYMMETRIZATION:
        g5g5_processed = symmetrize_correlator(g5g5_samples, axis=-1)
        logger.debug(f"Applied symmetrization to {group_name}")
    else:
        g5g5_processed = g5g5_samples

    # Calculate effective mass based on method
    if CALCULATION_METHOD["method"] == "two_state_periodic":
        effective_mass_samples = _calculate_two_state_periodic(g5g5_processed)
    elif CALCULATION_METHOD["method"] == "single_state":
        effective_mass_samples = _calculate_single_state(g5g5_processed)
    else:
        raise NotImplementedError(
            f"Method {CALCULATION_METHOD['method']} not implemented"
        )

    # Validate output dimensions
    if effective_mass_samples.shape[-1] != EXPECTED_EFFECTIVE_MASS_LENGTH:
        raise ValueError(
            f"Unexpected effective mass length: got {effective_mass_samples.shape[-1]}, "
            f"expected {EXPECTED_EFFECTIVE_MASS_LENGTH}"
        )

    # Calculate statistics
    mean_values, error_values = calculate_jackknife_statistics(effective_mass_samples)

    # Save results
    output_group.create_dataset(
        output_datasets["jackknife_samples"],
        data=effective_mass_samples,
        compression=OUTPUT_STRUCTURE["compression"],
        compression_opts=OUTPUT_STRUCTURE["compression_level"],
    )
    output_group.create_dataset(
        output_datasets["mean_values"],
        data=mean_values,
    )
    output_group.create_dataset(
        output_datasets["error_values"],
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
    output_group.attrs["effective_mass_method"] = CALCULATION_METHOD["method"]
    output_group.attrs["symmetrization_applied"] = APPLY_SYMMETRIZATION
    output_group.attrs["n_jackknife_samples"] = n_samples

    logger.debug(f"Successfully processed {group_name}")


def _read_g5g5_dataset(group: h5py.Group) -> np.ndarray:
    """
    Read g5g5 dataset with support for alternative names.

    Args:
        group: HDF5 group to read from

    Returns:
        g5g5 jackknife samples array

    Raises:
        KeyError: If no valid g5g5 dataset is found
    """
    # Try primary name first
    primary_name = REQUIRED_INPUT_DATASETS[0]
    if primary_name in group:
        return group[primary_name][:]

    # Try alternative names
    for alt_name in ALTERNATIVE_DATASET_NAMES.get(primary_name, []):
        if alt_name in group:
            return group[alt_name][:]

    # Not found
    available = list(group.keys())
    raise KeyError(
        f"No g5g5 dataset found in group {group.name}. "
        f"Looked for: {primary_name}, {ALTERNATIVE_DATASET_NAMES.get(primary_name, [])}. "
        f"Available datasets: {available}"
    )


def _calculate_two_state_periodic(g5g5_samples: np.ndarray) -> np.ndarray:
    """
    Calculate two-state periodic effective mass.

    Args:
        g5g5_samples: Input g5g5 correlator samples [n_samples, time]

    Returns:
        Effective mass samples [n_samples, truncated_time]
    """
    return calculate_two_state_periodic_effective_mass(
        g5g5_samples,
        lowering_factor=CALCULATION_METHOD["two_state_params"]["lowering_factor"],
        truncate_half=CALCULATION_METHOD["two_state_params"]["truncate_half"],
    )


def _calculate_single_state(g5g5_samples: np.ndarray) -> np.ndarray:
    """
    Calculate single-state effective mass using log(C(t)/C(t+1)).

    Args:
        g5g5_samples: Input g5g5 correlator samples [n_samples, time]

    Returns:
        Effective mass samples [n_samples, time-1]
    """
    # Simple single-state formula
    shifted = np.roll(g5g5_samples, shift=-1, axis=-1)
    
    # Remove last point (wraparound)
    g5g5_t = g5g5_samples[..., :-1]
    g5g5_t1 = shifted[..., :-1]
    
    # Calculate log ratio with safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        effective_mass = np.log(g5g5_t / g5g5_t1)
        
        # Handle invalid values
        effective_mass = np.where(
            np.isfinite(effective_mass),
            effective_mass,
            ERROR_HANDLING["division_by_zero_replacement"]
        )
    
    return effective_mass


if __name__ == "__main__":
    main()
