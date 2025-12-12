"""
Configuration file for jackknife analysis preprocessing.

This module contains all configuration constants, dataset naming
patterns, and processing parameters used in the jackknife analysis
script.
"""

from typing import Dict
from enum import Enum


class DerivativeMethod(Enum):
    """Available finite difference methods for derivative
    calculation."""

    SECOND_ORDER = "second_order"
    FOURTH_ORDER = "fourth_order"


# === PROCESSING CONFIGURATION ===

# Default derivative calculation method
DEFAULT_DERIVATIVE_METHOD = DerivativeMethod.FOURTH_ORDER

# Parameters to exclude from grouping (these don't affect physics
# results)
GROUPING_PARAMETERS = [
    "Configuration_label",  # Used as fundamental sampling unit - never group by individual configs
    "MPI_geometry",  # Optional: affects computation speed but not physics
]

# Note: Remove "MPI_geometry" from this list if you want to separate
# (4,4,4) and (6,6,6) groups (as you did by commenting it out)

# Minimum number of gauge configurations required for jackknife analysis
MIN_GAUGE_CONFIGURATIONS = 2

# HDF5 output compression settings
DEFAULT_COMPRESSION = "gzip"  # Options: "gzip", "lzf", "szip", None
DEFAULT_COMPRESSION_LEVEL = 4  # 1-9 for gzip, ignored for other methods


# === DATASET NAMING PATTERNS ===

# Input correlator dataset names (what we expect to find in input HDF5)
INPUT_CORRELATOR_DATASETS = {
    "g5g5": "g5-g5",
    "g4g5g5": "g4g5-g5",
}

# Output dataset naming patterns (clean, consistent names)
OUTPUT_DATASET_PATTERNS = {
    # Jackknife samples (2D arrays: samples × time)
    "samples": {
        "g5g5": "g5g5_jackknife_samples",
        "g4g5g5": "g4g5g5_jackknife_samples",
        "g4g5g5_derivative": "g4g5g5_derivative_jackknife_samples",
    },
    # Mean values (gvar pairs)
    "mean": {
        "g5g5": "g5g5_mean_values",
        "g4g5g5": "g4g5g5_mean_values",
        "g4g5g5_derivative": "g4g5g5_derivative_mean_values",
    },
    # Error values (gvar pairs)
    "error": {
        "g5g5": "g5g5_error_values",
        "g4g5g5": "g4g5g5_error_values",
        "g4g5g5_derivative": "g4g5g5_derivative_error_values",
    },
}

# Metadata dataset names
METADATA_DATASETS = {
    "gauge_configs": "gauge_configuration_labels",
    "qpb_filenames": "qpb_log_filenames",
}


# === FINITE DIFFERENCE COEFFICIENTS ===

# Finite difference stencils for derivative calculation
FINITE_DIFFERENCE_STENCILS = {
    DerivativeMethod.SECOND_ORDER: {
        "coefficients": [-1.0, 0.0, 1.0],
        "denominator": 2.0,
        "offset": 1,  # Number of points to skip at each boundary
    },
    DerivativeMethod.FOURTH_ORDER: {
        "coefficients": [1.0, -8.0, 0.0, 8.0, -1.0],
        "denominator": 12.0,
        "offset": 2,  # Number of points to skip at each boundary
    },
}


# === DATASET DESCRIPTIONS ===

# Comprehensive descriptions for HDF5 dataset attributes
DATASET_DESCRIPTIONS = {
    # Jackknife samples
    "g5g5_jackknife_samples": (
        "2D array containing jackknife replicas of g5-g5 correlator values. "
        "Shape: (N_jackknife_samples, N_time_points). Each row represents "
        "correlator values computed from N-1 gauge configurations with one "
        "configuration systematically excluded."
    ),
    "g4g5g5_jackknife_samples": (
        "2D array containing jackknife replicas of g4g5-g5 correlator values. "
        "Shape: (N_jackknife_samples, N_time_points). Each row represents "
        "correlator values computed from N-1 gauge configurations with one "
        "configuration systematically excluded."
    ),
    "g4g5g5_derivative_jackknife_samples": (
        "2D array containing jackknife replicas of g4g5-g5 derivative correlator "
        "values computed using fourth-order centered finite difference. "
        "Shape: (N_jackknife_samples, N_time_points-4). Boundary points are "
        "excluded due to finite difference stencil requirements."
    ),
    # Mean values
    "g5g5_mean_values": (
        "Mean values of g5-g5 correlator jackknife samples. "
        "These represent the central estimates of the correlator at each time point."
    ),
    "g4g5g5_mean_values": (
        "Mean values of g4g5-g5 correlator jackknife samples. "
        "These represent the central estimates of the correlator at each time point."
    ),
    "g4g5g5_derivative_mean_values": (
        "Mean values of g4g5-g5 derivative correlator jackknife samples. "
        "Computed using fourth-order centered finite difference approximation."
    ),
    # Error values
    "g5g5_error_values": (
        "Statistical uncertainties of g5-g5 correlator jackknife samples. "
        "Computed using jackknife error estimation method."
    ),
    "g4g5g5_error_values": (
        "Statistical uncertainties of g4g5-g5 correlator jackknife samples. "
        "Computed using jackknife error estimation method."
    ),
    "g4g5g5_derivative_error_values": (
        "Statistical uncertainties of g4g5-g5 derivative correlator jackknife samples. "
        "Errors propagated through finite difference calculation."
    ),
    # Metadata
    "gauge_configuration_labels": (
        "Array of gauge configuration labels corresponding to jackknife samples. "
        "IMPORTANT: gauge_configuration_labels[i] identifies the configuration that was "
        "EXCLUDED when computing the i-th jackknife sample (row i of *_jackknife_samples "
        "datasets). This enables identification of problematic configurations by examining "
        "which exclusion produces anomalous results."
    ),
    "qpb_log_filenames": (
        "Array of QPB log filenames corresponding to each gauge configuration. "
        "These trace back to the original simulation log files."
    ),
}


# === VALIDATION PATTERNS ===

# Required input datasets for successful processing
REQUIRED_INPUT_DATASETS = ["g5-g5", "g4g5-g5"]

# Expected file extensions
VALID_HDF5_EXTENSIONS = [".h5", ".hdf5"]

# Valid compression options for HDF5 output
VALID_COMPRESSION_OPTIONS = ["gzip", "lzf", "szip", None]


# === LOGGING CONFIGURATION ===

# Log message templates
LOG_MESSAGES = {
    "analysis_start": "Starting jackknife analysis for group: {}",
    "analysis_complete": "Completed jackknife analysis for group: {}",
    "insufficient_configs": (
        "Skipping group {} - insufficient gauge configurations "
        "(found {}, minimum required: {})"
    ),
    "missing_dataset": (
        "Warning: Missing dataset '{}' in group {}. Skipping this group."
    ),
    "dataset_created": "Created dataset: {} with shape {}",
    "group_processed": "Successfully processed {} groups",
}


# === UTILITY FUNCTIONS ===


def get_finite_difference_config(method: DerivativeMethod) -> Dict:
    """
    Get finite difference configuration for the specified method.

    Args:
        method: The derivative calculation method

    Returns:
        Dictionary containing coefficients, denominator, and offset
    """
    return FINITE_DIFFERENCE_STENCILS[method]


def get_output_dataset_name(correlator_type: str, data_type: str) -> str:
    """
    Get standardized output dataset name.

    Args:
        correlator_type: Type of correlator ('g5g5', 'g4g5g5',
        'g4g5g5_derivative') data_type: Type of data ('samples', 'mean',
        'error')

    Returns:
        Standardized dataset name

    Raises:
        ValueError: If correlator_type or data_type is invalid
    """
    if data_type not in OUTPUT_DATASET_PATTERNS:
        raise ValueError(f"Invalid data_type: {data_type}")

    if correlator_type not in OUTPUT_DATASET_PATTERNS[data_type]:
        raise ValueError(f"Invalid correlator_type: {correlator_type}")

    return OUTPUT_DATASET_PATTERNS[data_type][correlator_type]


def get_dataset_description(dataset_name: str) -> str:
    """
    Get description for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Descriptive text for the dataset
    """
    return DATASET_DESCRIPTIONS.get(dataset_name, f"Dataset: {dataset_name}")


def validate_derivative_method(method_str: str) -> DerivativeMethod:
    """
    Validate and convert string to DerivativeMethod enum.

    Args:
        method_str: String representation of method

    Returns:
        DerivativeMethod enum value

    Raises:
        ValueError: If method string is invalid
    """
    try:
        return DerivativeMethod(method_str.lower())
    except ValueError:
        valid_methods = [m.value for m in DerivativeMethod]
        raise ValueError(
            f"Invalid derivative method: {method_str}. "
            f"Valid options: {valid_methods}"
        )
