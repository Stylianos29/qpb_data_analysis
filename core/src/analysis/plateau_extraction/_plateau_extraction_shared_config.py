#!/usr/bin/env python3
"""
Shared configuration for plateau extraction scripts.

This module contains configuration parameters common to both PCAC mass
and pion effective mass plateau extraction.
"""

from typing import List


# =============================================================================
# CONSTANTS
# =============================================================================

# Plateau detection parameters
PLATEAU_DETECTION_SIGMA_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
MIN_PLATEAU_SIZE = 5
PLATEAU_DETECTION_METHOD = (
    "weighted_range"  # Options: 'weighted_range', 'chi_squared', 'range_based'
)

# Estimation methods
PLATEAU_ESTIMATION_METHOD = "inverse_variance_weighted"
USE_INVERSE_VARIANCE_WEIGHTING = True

# Metadata datasets to read from HDF5
METADATA_DATASETS = [
    "gauge_configuration_labels",
    "mpi_geometry_values",
    "qpb_log_filenames",
    "Number_of_gauge_configurations",
]

# CSV output configuration
CSV_OUTPUT_CONFIG = {
    "float_precision": 6,
    "include_diagnostics": True,
    "delimiter": ",",
    "columns_order": [  # Order of columns in output CSV
        # Group metadata columns (will be filled from HDF5 attributes)
        "bare_mass",
        "kappa",
        "clover_coefficient",
        "kernel_operator_type",
        "solver_type",
        # Extraction results
        "plateau_mean",
        "plateau_error",
        "plateau_start_time",
        "plateau_end_time",
        "plateau_n_points",
        # Sample statistics
        "n_successful_samples",
        "n_total_samples",
        "n_failed_samples",
        # Optional diagnostics
        "estimation_method",
        "sigma_threshold_used",
    ],
}

# Error handling configuration
ERROR_HANDLING = {
    "min_successful_fraction": 0.5,  # Minimum fraction of successful samples to proceed
    "failed_sample_action": "exclude",  # Options: 'exclude', 'skip_group'
    "log_failures": True,
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_shared_config() -> bool:
    """Validate shared configuration parameters."""
    # Check sigma thresholds
    if not PLATEAU_DETECTION_SIGMA_THRESHOLDS:
        raise ValueError("PLATEAU_DETECTION_SIGMA_THRESHOLDS cannot be empty")

    if any(s <= 0 for s in PLATEAU_DETECTION_SIGMA_THRESHOLDS):
        raise ValueError("All sigma thresholds must be positive")

    if max(PLATEAU_DETECTION_SIGMA_THRESHOLDS) > 5.0:
        raise ValueError("Maximum sigma threshold should not exceed 5.0")

    # Check plateau size
    if MIN_PLATEAU_SIZE < 3:
        raise ValueError("MIN_PLATEAU_SIZE must be at least 3")

    # Check detection method
    valid_methods = {"weighted_range", "chi_squared", "range_based"}
    if PLATEAU_DETECTION_METHOD not in valid_methods:
        raise ValueError(f"PLATEAU_DETECTION_METHOD must be one of {valid_methods}")

    # Check error handling
    if not 0 < ERROR_HANDLING["min_successful_fraction"] <= 1:
        raise ValueError("min_successful_fraction must be between 0 and 1")

    return True
