#!/usr/bin/env python3
"""
Shared configuration for critical mass extrapolation.

This module contains configuration parameters common to both PCAC mass
and pion effective mass critical mass calculations.
"""

from typing import List, Dict, Any


# =============================================================================
# CONSTANTS
# =============================================================================

# Parameters used for grouping data points (lattice configurations)
GROUPING_PARAMETERS = [
    "KL_diagonal_order",
    "Kernel_operator_type",
    "Number_of_gauge_configurations",
    "CG_epsilon",
    "Clover_coefficient",
    "KL_scaling_factor",
    "MSCG_epsilon",
    "Overlap_operator_method",
    "Rho_value",
]

# Data filtering parameters
FILTERING_PARAMETERS = {
    "max_bare_mass": 0.20,  # Upper cut on bare mass values
    "min_data_points_per_group": 3,  # Minimum points for linear fit
    "require_extraction_success": True,  # Only use successful plateau extractions
}

# Linear fit quality thresholds
FIT_QUALITY_THRESHOLDS = {
    "min_r_squared": 0.80,  # Minimum R² for acceptable fit
    "min_q_value": 0.01,  # Minimum fit probability (Q-value)
    "max_chi2_reduced": 10.0,  # Maximum reduced chi-squared
    "min_slope_significance": 2.0,  # Minimum |slope|/slope_error
}

# Physical validation ranges
PHYSICAL_VALIDATION = {
    "min_critical_bare_mass": -1.0,  # Minimum reasonable critical mass
    "max_critical_bare_mass": 0.5,  # Maximum reasonable critical mass
    "max_critical_mass_error_ratio": 0.5,  # Max uncertainty/value ratio
    "require_negative_slope": False,  # Whether slope must be negative
}

# CSV output configuration
CSV_OUTPUT_CONFIG = {
    "float_precision": 6,
    "include_fit_diagnostics": True,
    "delimiter": ",",
}

# Error handling configuration
ERROR_HANDLING = {
    "skip_failed_fits": True,
    "log_fit_failures": True,
    "min_successful_groups": 1,  # Minimum successful groups to proceed
}


# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================


def get_grouping_parameters():
    """Get list of parameters for data grouping."""
    return GROUPING_PARAMETERS.copy()


def get_filtering_config():
    """Get data filtering configuration."""
    return FILTERING_PARAMETERS.copy()


def get_fit_quality_config():
    """Get fit quality threshold configuration."""
    return FIT_QUALITY_THRESHOLDS.copy()


def get_physical_validation_config():
    """Get physical validation configuration."""
    return PHYSICAL_VALIDATION.copy()


def get_csv_output_config():
    """Get CSV output configuration."""
    return CSV_OUTPUT_CONFIG.copy()


def get_error_handling_config():
    """Get error handling configuration."""
    return ERROR_HANDLING.copy()


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_shared_config():
    """Validate shared configuration parameters."""
    # Validate grouping parameters
    if not isinstance(GROUPING_PARAMETERS, list):
        raise ValueError("GROUPING_PARAMETERS must be a list")

    if not GROUPING_PARAMETERS:
        raise ValueError("GROUPING_PARAMETERS cannot be empty")

    # Validate filtering parameters
    if FILTERING_PARAMETERS["max_bare_mass"] <= 0:
        raise ValueError("max_bare_mass must be positive")

    if FILTERING_PARAMETERS["min_data_points_per_group"] < 2:
        raise ValueError("min_data_points_per_group must be at least 2")

    # Validate fit quality thresholds
    if not (0 <= FIT_QUALITY_THRESHOLDS["min_r_squared"] <= 1):
        raise ValueError("min_r_squared must be between 0 and 1")

    if not (0 <= FIT_QUALITY_THRESHOLDS["min_q_value"] <= 1):
        raise ValueError("min_q_value must be between 0 and 1")

    if FIT_QUALITY_THRESHOLDS["max_chi2_reduced"] <= 0:
        raise ValueError("max_chi2_reduced must be positive")

    if FIT_QUALITY_THRESHOLDS["min_slope_significance"] <= 0:
        raise ValueError("min_slope_significance must be positive")

    # Validate physical ranges
    min_critical = PHYSICAL_VALIDATION["min_critical_bare_mass"]
    max_critical = PHYSICAL_VALIDATION["max_critical_bare_mass"]

    if min_critical >= max_critical:
        raise ValueError(
            "min_critical_bare_mass must be less than max_critical_bare_mass"
        )

    if not (0 < PHYSICAL_VALIDATION["max_critical_mass_error_ratio"] <= 1):
        raise ValueError("max_critical_mass_error_ratio must be between 0 and 1")

    # Validate CSV config
    if CSV_OUTPUT_CONFIG["float_precision"] < 1:
        raise ValueError("float_precision must be at least 1")

    # Validate error handling
    if ERROR_HANDLING["min_successful_groups"] < 1:
        raise ValueError("min_successful_groups must be at least 1")
