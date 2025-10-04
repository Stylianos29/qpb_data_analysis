"""
Shared configuration for computational cost extrapolation.

Contains configuration parameters common to both PCAC and pion
mass-based cost extrapolation methods. This includes grouping
parameters, data validation thresholds, averaging settings, and CSV
output formatting.
"""

from typing import List, Dict, Any


# =============================================================================
# GROUPING PARAMETERS
# =============================================================================

# Parameters used to group data for separate analysis Each unique
# combination of these parameters gets its own fit
GROUPING_PARAMETERS = [
    "beta",
    "volume_label",
    "kappa_critical",
    "Kernel_operator_type",
    "Overlap_operator_method",
    "KL_diagonal_order",
]

# Parameters excluded from grouping (averaged over instead)
GROUPING_EXCLUDED_PARAMETERS = [
    "Configuration_label",  # Always average across configurations
    "Bare_mass",  # This is our independent variable
]


# =============================================================================
# DATA FILTERING AND VALIDATION
# =============================================================================

# Data filtering criteria
FILTERING_PARAMETERS = {
    "max_bare_mass": 0.1,  # Maximum bare mass to include in analysis
    "min_data_points_per_group": 3,  # Minimum points needed per group for fitting
    "min_configurations_for_averaging": 2,  # Minimum configs to average
}

# Data validation thresholds
DATA_VALIDATION = {
    "min_data_points_for_mass_fitting": 3,  # Mass vs bare mass fit
    "min_data_points_for_cost_fitting": 4,  # Cost vs bare mass fit
    "require_positive_costs": True,  # Cost must be positive
}


# =============================================================================
# FIT QUALITY THRESHOLDS
# =============================================================================

FIT_QUALITY_THRESHOLDS = {
    # R-squared thresholds
    "min_r_squared": 0.7,  # Minimum R² for acceptable fit
    "warn_r_squared": 0.85,  # R² below this triggers warning
    # Chi-squared thresholds
    "max_chi2_reduced": 5.0,  # Maximum reduced χ² for acceptable fit
    "warn_chi2_reduced": 2.0,  # Reduced χ² above this triggers warning
    # Q-value (fit probability) thresholds
    "min_q_value": 0.01,  # Minimum Q-value for acceptable fit
    "warn_q_value": 0.05,  # Q-value below this triggers warning
    # Parameter significance
    "min_slope_significance": 2.0,  # Minimum |slope/error| ratio
}


# =============================================================================
# PHYSICAL VALIDATION
# =============================================================================

PHYSICAL_VALIDATION = {
    # Cost validation
    "min_cost_value": 0.1,  # Minimum physically reasonable cost (core-hours)
    "max_cost_value": 1e6,  # Maximum physically reasonable cost (core-hours)
    "max_cost_error_ratio": 0.5,  # Maximum error/value ratio for cost
    # Bare mass validation
    "min_bare_mass": -0.01,  # Minimum physically reasonable bare mass
    "max_bare_mass": 0.1,  # Maximum physically reasonable bare mass
    "require_negative_critical_mass": False,  # Critical mass can be positive
}


# =============================================================================
# COST FITTING CONFIGURATION
# =============================================================================

# Required columns in cost data CSV
COST_DATA_COLUMNS = [
    "Bare_mass",
    "Configuration_label",
    "Average_core_hours_per_spinor",
]

COST_FIT_CONFIG = {
    "fit_function": "shifted_power_law",  # Function type for cost vs bare mass
    "initial_guess": None,  # Let DataPlotter auto-determine
    "fit_method": "leastsq",  # Optimization method
}


# =============================================================================
# CSV OUTPUT CONFIGURATION
# =============================================================================

CSV_OUTPUT_CONFIG = {
    "float_precision": 6,  # Decimal places for float values
    "include_fit_diagnostics": True,  # Include R², χ², Q-value
    "include_data_counts": True,  # Include number of points used
    "delimiter": ",",  # CSV delimiter
    "index": False,  # Don't write row indices
}

# Column names for output CSV (standard naming)
OUTPUT_COLUMN_MAPPING = {
    # Grouping parameters (will be taken from input data) Derived bare
    # mass (from mass fit inversion)
    "derived_bare_mass_mean": "derived_bare_mass_value",
    "derived_bare_mass_error": "derived_bare_mass_error",
    # Mass fit parameters (slope and intercept from mass vs bare mass
    # fit)
    "mass_fit_slope_mean": "mass_fit_slope_value",
    "mass_fit_slope_error": "mass_fit_slope_error",
    "mass_fit_intercept_mean": "mass_fit_intercept_value",
    "mass_fit_intercept_error": "mass_fit_intercept_error",
    # Mass fit quality
    "mass_fit_r_squared": "mass_fit_r_squared",
    "mass_fit_chi2_reduced": "mass_fit_chi2_reduced",
    "mass_fit_quality": "mass_fit_q_value",
    # Cost fit parameters (shifted power law: a/(x-b)+c)
    "cost_fit_param_a_mean": "cost_fit_amplitude_value",
    "cost_fit_param_a_error": "cost_fit_amplitude_error",
    "cost_fit_param_b_mean": "cost_fit_shift_value",
    "cost_fit_param_b_error": "cost_fit_shift_error",
    "cost_fit_param_c_mean": "cost_fit_offset_value",
    "cost_fit_param_c_error": "cost_fit_offset_error",
    # Cost fit quality
    "cost_fit_r_squared": "cost_fit_r_squared",
    "cost_fit_chi2_reduced": "cost_fit_chi2_reduced",
    "cost_fit_quality": "cost_fit_q_value",
    # Extrapolated cost at reference bare mass
    "extrapolated_cost_mean": "extrapolated_cost_value",
    "extrapolated_cost_error": "extrapolated_cost_error",
    # Data counts
    "n_bare_mass_points": "number_of_bare_mass_points",
    "n_configurations": "number_of_configurations_averaged",
}


# =============================================================================
# ERROR HANDLING CONFIGURATION
# =============================================================================

ERROR_HANDLING = {
    "skip_failed_fits": True,  # Continue if a group fails to fit
    "log_fit_failures": True,  # Log details of failed fits
    "min_successful_groups": 1,  # Minimum successful groups to proceed
    "raise_on_no_results": True,  # Raise exception if all groups fail
}


# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================


def get_grouping_parameters() -> List[str]:
    """Get list of parameters for data grouping."""
    return GROUPING_PARAMETERS.copy()


def get_grouping_excluded_parameters() -> List[str]:
    """Get list of parameters excluded from grouping."""
    return GROUPING_EXCLUDED_PARAMETERS.copy()


def get_filtering_config() -> Dict[str, Any]:
    """Get data filtering configuration."""
    return FILTERING_PARAMETERS.copy()


def get_validation_config() -> Dict[str, Any]:
    """Get data validation configuration."""
    return DATA_VALIDATION.copy()


def get_fit_quality_config() -> Dict[str, Any]:
    """Get fit quality threshold configuration."""
    return FIT_QUALITY_THRESHOLDS.copy()


def get_physical_validation_config() -> Dict[str, Any]:
    """Get physical validation configuration."""
    return PHYSICAL_VALIDATION.copy()


def get_cost_data_columns() -> List[str]:
    """Get required cost data columns."""
    return COST_DATA_COLUMNS.copy()


def get_cost_fit_config() -> Dict[str, Any]:
    """Get cost fitting configuration."""
    return COST_FIT_CONFIG.copy()


def get_csv_output_config() -> Dict[str, Any]:
    """Get CSV output configuration."""
    return CSV_OUTPUT_CONFIG.copy()


def get_output_column_mapping() -> Dict[str, str]:
    """Get output column name mapping."""
    return OUTPUT_COLUMN_MAPPING.copy()


def get_error_handling_config() -> Dict[str, Any]:
    """Get error handling configuration."""
    return ERROR_HANDLING.copy()


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_shared_cost_config():
    """
    Validate shared configuration parameters.

    Raises:
        ValueError: If configuration is invalid
    """
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

    if FILTERING_PARAMETERS["min_configurations_for_averaging"] < 1:
        raise ValueError("min_configurations_for_averaging must be at least 1")

    # Validate data validation thresholds
    if DATA_VALIDATION["min_data_points_for_mass_fitting"] < 2:
        raise ValueError("min_data_points_for_mass_fitting must be at least 2")

    if DATA_VALIDATION["min_data_points_for_cost_fitting"] < 3:
        raise ValueError("min_data_points_for_cost_fitting must be at least 3")

    # Validate fit quality thresholds
    if not (0 <= FIT_QUALITY_THRESHOLDS["min_r_squared"] <= 1):
        raise ValueError("min_r_squared must be between 0 and 1")

    if not (0 <= FIT_QUALITY_THRESHOLDS["min_q_value"] <= 1):
        raise ValueError("min_q_value must be between 0 and 1")

    if FIT_QUALITY_THRESHOLDS["max_chi2_reduced"] <= 0:
        raise ValueError("max_chi2_reduced must be positive")

    # Validate physical ranges
    if PHYSICAL_VALIDATION["min_cost_value"] <= 0:
        raise ValueError("min_cost_value must be positive")

    if PHYSICAL_VALIDATION["min_cost_value"] >= PHYSICAL_VALIDATION["max_cost_value"]:
        raise ValueError("min_cost_value must be less than max_cost_value")

    if not (0 < PHYSICAL_VALIDATION["max_cost_error_ratio"] <= 1):
        raise ValueError("max_cost_error_ratio must be between 0 and 1")

    # Validate CSV config
    if CSV_OUTPUT_CONFIG["float_precision"] < 1:
        raise ValueError("float_precision must be at least 1")

    # Validate error handling
    if ERROR_HANDLING["min_successful_groups"] < 1:
        raise ValueError("min_successful_groups must be at least 1")
