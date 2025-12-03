"""
HDF5 output module for jackknife analysis with comprehensive parameter
handling.

This refactored module creates custom HDF5 output files using processed
parameter values from Stage 2A as the single source of truth. Key
improvements:

1. Uses PlotFilenameBuilder for consistent group naming across pipeline
2. Context-aware MPI_geometry handling (attribute vs dataset storage)
3. Robust filename matching with graceful error handling
4. Clear parameter classification using DataFrameAnalyzer
5. Comprehensive logging and user feedback

Design Principles:
    - CSV processed parameters are the authoritative source (never use
      raw HDF5 attributes)
    - Only TUNABLE parameters stored as HDF5 attributes
    - Graceful degradation: skip mismatched groups, continue processing
    - Consistent naming conventions with visualization layer
"""

from pathlib import Path
from typing import Dict, Optional, List, Tuple, Set, Any
import os

import h5py
import numpy as np
import pandas as pd

# Import library components
from library.data.hdf5_analyzer import HDF5Analyzer
from library.data.analyzer import DataFrameAnalyzer
from library.visualization.builders.filename_builder import PlotFilenameBuilder
from library.constants import FILENAME_LABELS_BY_COLUMN_NAME

# Import from auxiliary modules
from src.processing._jackknife_config import get_dataset_description


# ==============================================================================
# DATA STRUCTURES FOR PARAMETER CLASSIFICATION
# ==============================================================================


class ParameterClassification:
    """
    Container for classified parameter information.

    Attributes:
        - constant_tunable: Dict of single-valued tunable parameters
        - multivalued_tunable_names: List of multivalued tunable
          parameter names
        - mpi_geometry_storage: Where to store MPI_geometry
          ("constant_attribute", "group_attribute", or "dataset_list")
        - grouping_params: Set of parameters used for grouping (if
          determinable)
    """

    def __init__(
        self,
        constant_tunable: Dict[str, Any],
        multivalued_tunable_names: List[str],
        mpi_geometry_storage: str,
        grouping_params: Optional[Set[str]] = None,
    ):
        self.constant_tunable = constant_tunable
        self.multivalued_tunable_names = multivalued_tunable_names
        self.mpi_geometry_storage = mpi_geometry_storage
        self.grouping_params = grouping_params or set()


class GroupProcessingResult:
    """
    Result of processing a single jackknife analysis group.

    Attributes:
        - success: Whether processing succeeded
        - group_name: Generated HDF5 group name
        - skipped_reason: Reason for skipping (if applicable)
    """

    def __init__(
        self,
        success: bool,
        group_name: Optional[str] = None,
        skipped_reason: Optional[str] = None,
    ):
        self.success = success
        self.group_name = group_name
        self.skipped_reason = skipped_reason


# ==============================================================================
# FILENAME MATCHING AND PARAMETER LOOKUP
# ==============================================================================


def _create_filename_to_params_lookup(
    processed_csv_df: Optional[pd.DataFrame], logger
) -> Dict[str, Dict]:
    """
    Create dictionary mapping base filename (no extension) to processed
    parameters.

    This function strips file extensions to enable matching between:
        - CSV filenames (typically .txt)
        - HDF5 group names (typically .dat)

    Args:
        - processed_csv_df: DataFrame from
          processed_parameter_values.csv
        - logger: Logger instance

    Returns:
        dict: {base_filename: {param_name: param_value, ...}}

    Example:
        - CSV has: "KL_Brillouin_..._n1.txt"
        - HDF5 has: "KL_Brillouin_..._n1.dat"
        - Lookup key: "KL_Brillouin_..._n1"
    """
    if processed_csv_df is None:
        logger.warning("No processed parameters DataFrame provided")
        return {}

    lookup = {}
    for _, row in processed_csv_df.iterrows():
        # Strip extension from CSV filename to match any extension in
        # HDF5
        filename_with_ext = row["Filename"]
        base_filename = os.path.splitext(filename_with_ext)[0]

        # Store complete row as dictionary
        lookup[base_filename] = row.to_dict()

    logger.info(f"Created filename-to-parameters lookup with {len(lookup)} entries")
    return lookup


def _extract_group_parameters(
    config_metadata: Dict,
    filename_lookup: Dict[str, Dict],
    multivalued_tunable_names: List[str],
    logger,
) -> Optional[Dict[str, Any]]:
    """
    Extract processed parameter values for an HDF5 group from CSV
    lookup.

    This function enforces the principle that ALL parameter values must
    come from the processed CSV (Stage 2A output), not from raw HDF5
    attributes.

    Args:
        - config_metadata: Metadata dict containing qpb_filenames list
        - filename_lookup: Dictionary mapping base filenames to
          parameters
        - multivalued_tunable_names: List of multivalued tunable
          parameter names
        - logger: Logger instance

    Returns:
        dict: {param_name: param_value} for multivalued tunable params,
              or None if filename not found in CSV

    Raises:
        ValueError: If no filenames in config_metadata (indicates data
        corruption)
    """
    # Extract filename from metadata
    filenames = config_metadata.get("qpb_filenames", [])
    if not filenames:
        raise ValueError(
            "No filenames found in config_metadata. "
            "This indicates corrupted jackknife processing results."
        )

    # Use first filename as reference (all configs in group have same
    # parameters)
    reference_filename = filenames[0]
    base_filename = os.path.splitext(reference_filename)[0]

    # Validate filename exists in CSV
    if base_filename not in filename_lookup:
        logger.warning(
            f"Filename '{reference_filename}' not found in processed CSV. "
            f"This group will be skipped. "
            f"Possible causes: "
            f"(1) Different file sets processed in Stage 1B vs 2A, "
            f"(2) Incomplete Stage 2A output."
        )
        return None

    # Extract processed parameters for this file
    processed_params_row = filename_lookup[base_filename]

    # Build parameter dictionary with only multivalued tunable params
    params_dict = {}
    actual_params = [p for p in multivalued_tunable_names if p != "Configuration_label"]

    for param_name in actual_params:
        if param_name not in processed_params_row:
            logger.warning(
                f"Parameter '{param_name}' not found in CSV for {reference_filename}. "
                f"Expected tunable parameter missing from processed CSV."
            )
            continue
        params_dict[param_name] = processed_params_row[param_name]

    logger.debug(
        f"Extracted {len(params_dict)} multivalued tunable parameters "
        f"from CSV for {reference_filename}"
    )

    return params_dict


# ==============================================================================
# PARAMETER CLASSIFICATION
# ==============================================================================


def _classify_parameters(
    processed_csv_analyzer: DataFrameAnalyzer, input_hdf5_analyzer: HDF5Analyzer, logger
) -> ParameterClassification:
    """
    Classify parameters and determine storage strategy, including
    MPI_geometry.

    Classification rules:
        1. Single-valued tunable → constant_attribute (second-to-deepest
           group)
        2. Multivalued tunable:
        - If used for grouping → group_attribute (deepest group)
        - If NOT used for grouping → dataset_list (varies within group)

    Special handling for MPI_geometry:
        - If single-valued → constant_attribute
        - If multivalued:
        * If used for grouping → group_attribute
        * If not used for grouping → dataset_list
          ("mpi_geometry_values")

    Args:
        - processed_csv_analyzer: Analyzer for processed CSV DataFrame
        - input_hdf5_analyzer: Analyzer for input HDF5 file
        - logger: Logger instance

    Returns:
        ParameterClassification instance with all classification info
    """
    logger.info("Classifying parameters using DataFrameAnalyzer")

    # Extract single-valued tunable parameters (constants)
    single_valued_tunable = (
        processed_csv_analyzer.list_of_single_valued_tunable_parameter_names
    )
    constant_tunable = {
        param: processed_csv_analyzer.unique_value_columns_dictionary[param]
        for param in single_valued_tunable
    }

    logger.info(f"Identified {len(constant_tunable)} constant tunable parameters")
    for param, value in constant_tunable.items():
        logger.debug(f"  Constant: {param} = {value}")

    # Extract multivalued tunable parameters
    multivalued_tunable = (
        processed_csv_analyzer.list_of_multivalued_tunable_parameter_names
    )

    # Determine which parameters were used for grouping We infer this
    # from the input HDF5 analyzer's grouping
    grouping_params = set(
        input_hdf5_analyzer.reduced_multivalued_tunable_parameter_names_list
    )
    # Always exclude Configuration_label from grouping
    grouping_params.discard("Configuration_label")

    logger.info(f"Identified {len(multivalued_tunable)} multivalued tunable parameters")
    logger.info(f"Parameters used for grouping: {grouping_params}")

    # Determine MPI_geometry storage strategy
    mpi_storage = _classify_mpi_geometry_storage(
        mpi_in_constant=("MPI_geometry" in single_valued_tunable),
        mpi_in_multivalued=("MPI_geometry" in multivalued_tunable),
        mpi_in_grouping=("MPI_geometry" in grouping_params),
        logger=logger,
    )

    return ParameterClassification(
        constant_tunable=constant_tunable,
        multivalued_tunable_names=list(multivalued_tunable),
        mpi_geometry_storage=mpi_storage,
        grouping_params=grouping_params,
    )


def _classify_mpi_geometry_storage(
    mpi_in_constant: bool, mpi_in_multivalued: bool, mpi_in_grouping: bool, logger
) -> str:
    """
    Determine where to store MPI_geometry based on its variability
    pattern.

    Logic:
        - If single-valued (constant) → "constant_attribute"
          (second-to-deepest)
        - If multivalued:
            * If used for grouping → "group_attribute" (deepest group
              attribute)
            * If NOT used for grouping → "dataset_list"
              (mpi_geometry_values array)

    Args:
        - mpi_in_constant: Is MPI_geometry single-valued?
        - mpi_in_multivalued: Is MPI_geometry multivalued?
        - mpi_in_grouping: Was MPI_geometry used for grouping?
        - logger: Logger instance

    Returns:
        str: One of "constant_attribute", "group_attribute",
        "dataset_list"
    """
    if mpi_in_constant:
        logger.info(
            "MPI_geometry is constant → storing as second-to-deepest group attribute"
        )
        return "constant_attribute"

    elif mpi_in_multivalued:
        if mpi_in_grouping:
            logger.info(
                "MPI_geometry is multivalued and used for grouping → "
                "storing as deepest group attribute"
            )
            return "group_attribute"
        else:
            logger.info(
                "MPI_geometry is multivalued but NOT used for grouping → "
                "storing as 'mpi_geometry_values' dataset"
            )
            return "dataset_list"

    else:
        # MPI_geometry not present in data
        logger.debug("MPI_geometry not found in processed parameters")
        return "not_present"


# ==============================================================================
# GROUP NAME GENERATION
# ==============================================================================


def _generate_group_name(
    processed_params: Dict[str, Any],
    multivalued_tunable_names: List[str],
    filename_builder: PlotFilenameBuilder,
    logger,
) -> str:
    """
    Generate descriptive HDF5 group name using PlotFilenameBuilder.

    Group names follow the pattern:
        jackknife_analysis_[Overlap]_[Kernel]_[param1][value1]_[param2][value2]...

    Examples:
        "jackknife_analysis_KL_Brillouin_m0p01_n2_MPI444_EpsMSCG1e-06"
        "jackknife_analysis_Chebyshev_Wilson_m0p06_n6_MPI444"

    Args:
        - processed_params: Dict of processed parameter values from CSV
        - multivalued_tunable_names: List of multivalued tunable
          parameter names
        - filename_builder: PlotFilenameBuilder instance
        - logger: Logger instance

    Returns:
        str: Descriptive group name
    """
    try:
        # Filter multivalued params (exclude Configuration_label)
        actual_params = [
            p
            for p in multivalued_tunable_names
            if p != "Configuration_label" and p in processed_params
        ]

        # Create a working copy of metadata for the builder IMPORTANT:
        # Include ALL parameters, not just multivalued ones
        # PlotFilenameBuilder needs to see Overlap_operator_method and
        # Kernel_operator_type
        metadata_for_builder = processed_params.copy()

        # Special handling for MPI_geometry: convert "(4, 4, 4)" → "444"
        if "MPI_geometry" in metadata_for_builder:
            mpi_value = metadata_for_builder["MPI_geometry"]
            # Handle string representations of tuples: "(4, 4, 4)" or
            # "(4,4,4)"
            if isinstance(mpi_value, str):
                # Remove parentheses, commas, and spaces to get "444"
                mpi_cleaned = (
                    mpi_value.replace("(", "")
                    .replace(")", "")
                    .replace(",", "")
                    .replace(" ", "")
                )
                metadata_for_builder["MPI_geometry"] = mpi_cleaned
            elif isinstance(mpi_value, tuple):
                # Handle actual tuple: (4, 4, 4) → "444"
                mpi_cleaned = "".join(str(x) for x in mpi_value)
                metadata_for_builder["MPI_geometry"] = mpi_cleaned

        # Use PlotFilenameBuilder for consistent naming Note: We use
        # custom_prefix to put "jackknife_analysis" at the front
        group_name = filename_builder.build(
            metadata_dict=metadata_for_builder,  # Use the working copy with all params
            base_name="",  # Empty base name since prefix contains "jackknife_analysis"
            multivalued_params=actual_params,  # Only list multivalued ones
            custom_prefix="jackknife_analysis_",
        )

        # Clean up any double underscores from empty base_name
        group_name = group_name.replace("__", "_")

        # Remove leading underscore if present
        group_name = group_name.lstrip("_")

        logger.debug(f"Generated group name: {group_name}")
        return group_name

    except Exception as e:
        # Fallback to simple naming if builder fails
        logger.warning(f"PlotFilenameBuilder failed, using fallback naming: {e}")

        # Simple fallback:
        # jackknife_analysis_param1_value1_param2_value2
        parts = ["jackknife_analysis"]

        # Add Overlap_operator_method if present
        if "Overlap_operator_method" in processed_params:
            parts.append(str(processed_params["Overlap_operator_method"]))

        # Add other parameters
        for param in actual_params:
            if param in processed_params:
                label = FILENAME_LABELS_BY_COLUMN_NAME.get(param, param)
                value = str(processed_params[param]).replace(".", "p")
                parts.append(f"{label}{value}")

        return "_".join(parts)


# ==============================================================================
# HDF5 STRUCTURE AND STORAGE
# ==============================================================================


def _get_input_directory_structure(input_hdf5_analyzer: HDF5Analyzer) -> List[str]:
    """
    Extract directory structure from input HDF5 file.

    Args:
        input_hdf5_analyzer: Analyzer for input HDF5 file

    Returns:
        List of group names representing the directory structure
    """
    if input_hdf5_analyzer.active_groups:
        sample_group = list(input_hdf5_analyzer.active_groups)[0]
        # Split path and remove empty strings
        parts = [part for part in sample_group.split("/") if part]
        # Return all but the last part (which is the individual file
        # group)
        return parts[:-1] if len(parts) > 1 else parts
    return []


def _store_constant_parameters(
    parent_group: h5py.Group, constant_params: Dict[str, Any], logger
) -> None:
    """
    Store constant (single-valued) tunable parameters as HDF5
    attributes.

    Args:
        - parent_group: HDF5 group (second-to-deepest level)
        - constant_params: Dictionary of constant parameter values
        - logger: Logger instance
    """
    logger.info(
        f"Storing {len(constant_params)} constant parameters at second-to-deepest level"
    )

    for param_name, param_value in constant_params.items():
        parent_group.attrs[param_name] = param_value
        logger.debug(f"  Constant attribute: {param_name} = {param_value}")


def _store_group_parameters(
    jackknife_group: h5py.Group,
    processed_params: Dict[str, Any],
    multivalued_tunable_names: List[str],
    mpi_geometry_storage: str,
    logger,
) -> None:
    """
    Store multivalued tunable parameters as HDF5 group attributes.

    Stores all multivalued tunable parameters, including MPI_geometry if
    it should be stored as a group attribute.

    Args:
        - jackknife_group: HDF5 group (deepest level)
        - processed_params: Dictionary of processed parameter values
        - multivalued_tunable_names: List of multivalued tunable
          parameter names
        - mpi_geometry_storage: Where to store MPI_geometry
        - logger: Logger instance
    """
    # Filter out Configuration_label (handled separately as dataset)
    actual_params = [p for p in multivalued_tunable_names if p != "Configuration_label"]

    # Handle MPI_geometry separately based on storage strategy
    if mpi_geometry_storage == "dataset_list":
        # MPI_geometry will be stored as a dataset, not an attribute
        actual_params = [p for p in actual_params if p != "MPI_geometry"]
        logger.debug("MPI_geometry will be stored as dataset, not attribute")

    # Store all qualifying parameters as attributes
    stored_count = 0
    for param_name in actual_params:
        if param_name not in processed_params:
            logger.warning(
                f"Parameter '{param_name}' expected but not found in processed params"
            )
            continue

        jackknife_group.attrs[param_name] = processed_params[param_name]
        logger.debug(
            f"  Group attribute: {param_name} = {processed_params[param_name]}"
        )
        stored_count += 1

    logger.info(
        f"Stored {stored_count} multivalued tunable parameters as group attributes"
    )


def _store_jackknife_datasets(
    group: h5py.Group,
    jackknife_results: Dict,
    compression: Optional[str],
    compression_opts: Optional[int],
    logger,
) -> None:
    """
    Store jackknife analysis results as HDF5 datasets.

    Creates datasets for:
        - Jackknife samples (all resampled values)
        - Mean values (central estimates)
        - Error values (standard errors from jackknife)

    Args:
        - group: HDF5 group to store datasets in
        - jackknife_results: Dictionary containing jackknife analysis
          results
        - compression: HDF5 compression method
        - compression_opts: Compression level
        - logger: Logger instance
    """
    # Define expected dataset keys and their descriptions
    dataset_keys = {
        "g5g5_jackknife_samples": "g5-g5 correlator jackknife samples",
        "g5g5_mean_values": "g5-g5 correlator mean values",
        "g5g5_error_values": "g5-g5 correlator standard errors",
        "g4g5g5_jackknife_samples": "g4g5-g5 correlator jackknife samples",
        "g4g5g5_mean_values": "g4g5-g5 correlator mean values",
        "g4g5g5_error_values": "g4g5-g5 correlator standard errors",
        "g4g5g5_derivative_jackknife_samples": "g4g5-g5 derivative jackknife samples",
        "g4g5g5_derivative_mean_values": "g4g5-g5 derivative mean values",
        "g4g5g5_derivative_error_values": "g4g5-g5 derivative standard errors",
    }

    stored_count = 0
    for key, description in dataset_keys.items():
        if key in jackknife_results:
            data = jackknife_results[key]

            # Create dataset with compression
            group.create_dataset(
                key,
                data=data,
                compression=compression,
                compression_opts=compression_opts,
            )

            # Add description as attribute
            dataset_description = get_dataset_description(key)
            if dataset_description:
                group[key].attrs["description"] = dataset_description

            logger.debug(f"  Created dataset: {key} {data.shape}")
            stored_count += 1

    logger.info(f"Stored {stored_count} jackknife analysis datasets")


def _store_metadata_arrays(
    group: h5py.Group,
    config_metadata: Dict,
    mpi_geometry_storage: str,
    compression: Optional[str],
    compression_opts: Optional[int],
    logger,
) -> None:
    """
    Store configuration metadata as HDF5 datasets.

    Always stores:
        - gauge_configuration_labels: List of configuration labels
        - qpb_log_filenames: List of log filenames

    Conditionally stores:
        - mpi_geometry_values: Only if mpi_geometry_storage ==
          "dataset_list"

    Args:
        - group: HDF5 group to store datasets in
        - config_metadata: Dictionary containing configuration metadata
        - mpi_geometry_storage: Where MPI_geometry should be stored
        - compression: HDF5 compression method
        - compression_opts: Compression level
        - logger: Logger instance
    """
    # Always store these metadata arrays
    required_metadata = {
        "gauge_configuration_labels": "configuration_labels",
        "qpb_log_filenames": "qpb_filenames",
    }

    stored_count = 0
    for dataset_name, metadata_key in required_metadata.items():
        if metadata_key in config_metadata:
            data = config_metadata[metadata_key]

            # Convert to numpy array of strings
            if isinstance(data, list):
                data = np.array(data, dtype=h5py.string_dtype(encoding="utf-8"))

            group.create_dataset(
                dataset_name,
                data=data,
                compression=compression,
                compression_opts=compression_opts,
            )

            logger.debug(f"  Created metadata dataset: {dataset_name}")
            stored_count += 1
        else:
            logger.warning(f"Expected metadata key '{metadata_key}' not found")

        # Conditionally store MPI_geometry values
        if mpi_geometry_storage == "dataset_list":
            if "mpi_geometries" in config_metadata:
                mpi_data = config_metadata["mpi_geometries"]

                # Convert to appropriate format
                if isinstance(mpi_data, list):
                    mpi_data = np.array(
                        mpi_data, dtype=h5py.string_dtype(encoding="utf-8")
                    )

                group.create_dataset(
                    "mpi_geometry_values",
                    data=mpi_data,
                    compression=compression,
                    compression_opts=compression_opts,
                )

                logger.info("  Created MPI_geometry dataset (varies within group)")
                stored_count += 1
            else:
                # Classification suggested dataset storage, but
                # MPI_geometry is actually constant within each group,
                # so it's correctly stored as an attribute. No
                # mpi_geometry_values dataset will be created.
                logger.info(
                    "MPI_geometry stored as group attribute (constant within each group). "
                    "No 'mpi_geometry_values' dataset created."
                )

    logger.info(f"Stored {stored_count} metadata arrays")


# ==============================================================================
# MAIN OUTPUT CREATION FUNCTION
# ==============================================================================


def _create_custom_hdf5_output(
    output_path: Path,
    all_processing_results: Dict,
    input_hdf5_analyzer: HDF5Analyzer,
    processed_params_df: Optional[pd.DataFrame],
    compression: str,
    compression_level: int,
    logger,
) -> Tuple[int, int, List[str]]:
    """
    Create custom HDF5 output file with processed parameters and robust
    error handling.

    This function orchestrates the complete HDF5 output creation
    process:
        1. Classify parameters using DataFrameAnalyzer
        2. Create filename lookup for parameter matching
        3. Build HDF5 structure with constant parameters
        4. Process each jackknife group with error handling
        5. Return detailed statistics for user feedback

    Args:
        - output_path: Path for output HDF5 file
        - all_processing_results: Dictionary with all jackknife
          processing results
        - input_hdf5_analyzer: Analyzer for input HDF5 file
        - processed_params_df: DataFrame from
          processed_parameter_values.csv
        - compression: Compression method ('gzip', 'lzf', or 'none')
        - compression_level: Compression level (1-9 for gzip)
        - logger: Logger instance

    Returns:
        Tuple of (successful_groups, skipped_groups, skipped_filenames)
    """
    # Validate inputs
    if processed_params_df is None:
        raise ValueError(
            "Processed parameters DataFrame is required. "
            "Stage 2B cannot run without Stage 2A output."
        )

    logger.info("=" * 80)
    logger.info("CREATING CUSTOM HDF5 OUTPUT WITH PROCESSED PARAMETERS")
    logger.info("=" * 80)

    # Prepare compression settings
    compression_opts = None if compression == "none" else compression_level
    final_compression = None if compression == "none" else compression

    # Phase 1: Setup analyzers and builders
    logger.info("Phase 1: Setting up analyzers and builders")

    processed_csv_analyzer = DataFrameAnalyzer(processed_params_df)
    logger.info(
        f"Created DataFrameAnalyzer for processed CSV: {len(processed_params_df)} rows"
    )

    filename_builder = PlotFilenameBuilder(FILENAME_LABELS_BY_COLUMN_NAME)
    logger.info("Created PlotFilenameBuilder for group naming")

    filename_lookup = _create_filename_to_params_lookup(processed_params_df, logger)

    # Phase 2: Classify parameters
    logger.info("\nPhase 2: Classifying parameters")

    param_classification = _classify_parameters(
        processed_csv_analyzer=processed_csv_analyzer,
        input_hdf5_analyzer=input_hdf5_analyzer,
        logger=logger,
    )

    # Phase 3: Create HDF5 structure
    logger.info("\nPhase 3: Creating HDF5 file structure")

    try:
        with h5py.File(output_path, "w") as output_file:
            # Recreate directory structure from input
            input_structure = _get_input_directory_structure(input_hdf5_analyzer)
            logger.info(f"Recreating directory structure: {'/'.join(input_structure)}")

            # Create group hierarchy
            parent_group = output_file
            for level_name in input_structure:
                parent_group = parent_group.create_group(level_name)

            # Store constant parameters at second-to-deepest level
            _store_constant_parameters(
                parent_group=parent_group,
                constant_params=param_classification.constant_tunable,
                logger=logger,
            )

            # Phase 4: Process each jackknife analysis group
            logger.info("\nPhase 4: Processing jackknife analysis groups")
            logger.info(f"Total groups to process: {len(all_processing_results)}")

            successful_groups = 0
            skipped_groups = 0
            skipped_filenames = []

            for group_index, (original_group_name, results) in enumerate(
                all_processing_results.items(), 1
            ):
                logger.info(
                    f"\n--- Processing group {group_index}/{len(all_processing_results)} ---"
                )

                # Extract parameters from CSV using filename matching
                config_metadata = results["config_metadata"]

                processed_params = _extract_group_parameters(
                    config_metadata=config_metadata,
                    filename_lookup=filename_lookup,
                    multivalued_tunable_names=param_classification.multivalued_tunable_names,
                    logger=logger,
                )

                # Check if parameter extraction failed (filename not in
                # CSV)
                if processed_params is None:
                    skipped_groups += 1
                    skipped_filenames.append(
                        config_metadata.get("qpb_filenames", ["unknown"])[0]
                    )
                    logger.warning(
                        f"Skipping group {group_index}: filename not found in processed CSV"
                    )
                    continue

                # Generate descriptive group name Combine constant and
                # multivalued parameters for complete metadata
                complete_metadata = {
                    **param_classification.constant_tunable,  # Add constant params
                    **processed_params  # Add multivalued params
                }

                group_name = _generate_group_name(
                    processed_params=complete_metadata,  # Pass combined metadata
                    multivalued_tunable_names=param_classification.multivalued_tunable_names,
                    filename_builder=filename_builder,
                    logger=logger
                )

                # Create the jackknife analysis group
                jackknife_group = parent_group.create_group(group_name)
                logger.info(f"Created group: {group_name}")

                # Store multivalued tunable parameters as attributes
                _store_group_parameters(
                    jackknife_group=jackknife_group,
                    processed_params=processed_params,
                    multivalued_tunable_names=param_classification.multivalued_tunable_names,
                    mpi_geometry_storage=param_classification.mpi_geometry_storage,
                    logger=logger,
                )

                # Store number of gauge configurations
                jackknife_results = results["jackknife_results"]
                n_configs = jackknife_results.get("n_gauge_configurations")
                if n_configs:
                    jackknife_group.attrs["Number_of_gauge_configurations"] = n_configs
                    logger.debug(f"  Stored n_gauge_configurations: {n_configs}")

                # Store jackknife analysis results
                _store_jackknife_datasets(
                    group=jackknife_group,
                    jackknife_results=jackknife_results,
                    compression=final_compression,
                    compression_opts=compression_opts,
                    logger=logger,
                )

                # Store configuration metadata
                _store_metadata_arrays(
                    group=jackknife_group,
                    config_metadata=config_metadata,
                    mpi_geometry_storage=param_classification.mpi_geometry_storage,
                    compression=final_compression,
                    compression_opts=compression_opts,
                    logger=logger,
                )

                successful_groups += 1
                logger.info(f"✓ Group {group_index} processing complete")

            # Phase 5: Summary
            logger.info("\n" + "=" * 80)
            logger.info("HDF5 OUTPUT CREATION COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Output file: {output_path}")
            logger.info(
                f"Successfully processed: {successful_groups}/{len(all_processing_results)} groups"
            )

            if skipped_groups > 0:
                logger.warning(f"Skipped groups: {skipped_groups}")
                logger.warning("Skipped filenames:")
                for filename in skipped_filenames:
                    logger.warning(f"  - {filename}")

            return successful_groups, skipped_groups, skipped_filenames

    except Exception as e:
        logger.error(f"Critical error creating HDF5 output: {e}")
        raise
