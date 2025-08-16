"""
Configuration for critical bare mass calculation script.

This module contains all configuration parameters for calculating critical
bare mass values from plateau PCAC mass estimates using linear extrapolation
to the chiral limit (PCAC mass = 0).

Place this file as:
qpb_data_analysis/core/src/analysis/_critical_bare_mass_config.py
"""

from typing import List, Dict, Any, Optional

# =============================================================================
# INPUT/OUTPUT CONFIGURATION
# =============================================================================

# Input CSV column names
INPUT_CSV_COLUMNS = {
    "bare_mass": "Bare_mass",
    "pcac_mass_mean": "plateau_PCAC_mass_mean",
    "pcac_mass_error": "plateau_PCAC_mass_error",
    "extraction_success": "extraction_success",
}

# Parameter columns used for grouping data points
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

# Parameters to exclude from filenames (typically single-valued or not useful for identification)
FILENAME_EXCLUDED_PARAMETERS = [
    "Number_of_gauge_configurations",  # Not useful for filename identification
    "CG_epsilon",  # Typically single-valued
    "MSCG_epsilon",  # Typically single-valued
    "Clover_coefficient",  # Typically single-valued
    "KL_scaling_factor",  # Typically single-valued
    "Rho_value",  # Typically single-valued
]

# Additional metadata columns to preserve in output
METADATA_COLUMNS = [
    "plateau_start_time",
    "plateau_end_time",
    "plateau_n_points",
    "sample_size",
    "n_total_samples",
    "n_failed_samples",
    "estimation_method",
    "use_inverse_variance",
    "avg_correlation",
    "failed_config_labels",
]

# Output CSV configuration
OUTPUT_CSV_CONFIG = {
    "default_filename": "critical_bare_mass_values.csv",
    "float_precision": 6,  # Decimal places for floating point values
    "include_fit_diagnostics": True,  # Include linear fit quality metrics
    "include_metadata": True,  # Include plateau extraction metadata
}

# =============================================================================
# LINEAR FITTING CONFIGURATION
# =============================================================================

# Data filtering and validation
DATA_FILTERING = {
    "upper_bare_mass_cut": 0.15,  # Exclude data points above this bare mass
    "require_extraction_success": True,  # Only use successfully extracted plateaus
    "min_data_points": 3,  # Minimum number of points required for fitting
    "max_data_points": None,  # Maximum number of points (None = no limit)
}

# Linear fitting parameters
LINEAR_FIT_CONFIG = {
    "method": "lsqfit",  # Options: 'lsqfit', 'scipy_odr', 'numpy_polyfit'
    "use_error_weighting": True,  # Weight data points by inverse variance
    "fit_function": "linear",  # Currently only 'linear' supported
    "initial_guess_method": "two_point",  # 'two_point', 'least_squares', 'manual'
    "convergence_tolerance": 1e-8,  # Fitting convergence criterion
    "max_iterations": 1000,  # Maximum fitting iterations
}

# Fit quality validation criteria
FIT_QUALITY_THRESHOLDS = {
    "max_chi2_per_dof": 5.0,  # Maximum acceptable reduced chi-squared
    "min_slope_significance": 2.0,  # Minimum slope/slope_error ratio
    "max_critical_mass_error_ratio": 0.5,  # Max error/value ratio for critical mass
    "min_r_squared": 0.5,  # Minimum R-squared value for fit quality
}

# Error handling for failed fits
ERROR_HANDLING = {
    "failed_fit_action": "skip",  # Options: 'skip', 'include_nan', 'raise_error'
    "insufficient_data_action": "skip",  # Action when not enough data points
    "poor_fit_action": "include_with_warning",  # Action for poor quality fits
    "log_failed_groups": True,  # Log details of failed parameter groups
}

# =============================================================================
# PHYSICAL VALIDATION
# =============================================================================

# Physical reasonableness checks
PHYSICAL_VALIDATION = {
    "max_critical_bare_mass": 0.5,  # Maximum physically reasonable critical mass
    "min_critical_bare_mass": -0.5,  # Minimum physically reasonable critical mass
    "require_negative_slope": False,  # Require slope < 0 (PCAC decreases with bare mass)
    "check_extrapolation_distance": True,  # Warn if extrapolation is far from data
    "max_extrapolation_ratio": 2.0,  # Max extrapolation distance / data range
}

# =============================================================================
# PLOTTING CONFIGURATION
# =============================================================================

# Main plotting settings
PLOTTING_CONFIG = {
    "default_figure_size": (10, 8),
    "default_dpi": 300,
    "font_size": 12,
    "line_width": 2,
    "marker_size": 8,
    "capsize": 5,  # Error bar cap size
    "alpha_fill": 0.2,  # Transparency for fit uncertainty bands
}

# Plot styling
PLOT_STYLE = {
    "data_points": {
        "marker": "o",
        "color": "blue",
        "alpha": 0.7,
        "label": "PCAC mass data",
    },
    "linear_fit": {
        "color": "red",
        "linestyle": "--",
        "alpha": 0.8,
        "label": "Linear fit",
    },
    "fit_uncertainty": {
        "color": "red",
        "alpha": 0.2,
    },
    "critical_mass_line": {
        "color": "green",
        "linestyle": ":",
        "alpha": 0.8,
        "label": "Critical bare mass",
    },
    "zero_lines": {
        "color": "black",
        "linestyle": "-",
        "alpha": 0.5,
        "linewidth": 1,
    },
}

# Plot content configuration
PLOT_CONTENT = {
    "show_fit_equation": True,  # Display fit equation on plot
    "show_chi2": True,  # Show chi-squared statistics
    "show_critical_mass_value": True,  # Annotate critical mass value
    "show_data_range": True,  # Show fitted data range
    "show_extrapolation_region": True,  # Highlight extrapolation region
    "include_grid": True,  # Add grid to plots
    "title_max_width": 100,  # Maximum characters in plot title
}

# Plot output settings
PLOT_OUTPUT = {
    "base_directory": "Critical_bare_mass_calculation",
    "subdirectory": "Critical_bare_mass_from_PCAC_mass",
    "base_filename": "Critical_bare_mass",
    "format": "png",  # Options: 'png', 'pdf', 'svg'
    "clear_existing": True,  # Clear existing plots before creating new ones
}

# =============================================================================
# ADVANCED CONFIGURATION
# =============================================================================

# Extrapolation configuration
EXTRAPOLATION_CONFIG = {
    "extend_fit_line": True,  # Extend fit line beyond data range
    "extension_factor": 0.2,  # Fraction of data range to extend on each side
    "highlight_extrapolation": True,  # Visually distinguish extrapolated region
    "include_extrapolation_uncertainty": True,  # Show uncertainty in extrapolated region
}

# Diagnostic output configuration
DIAGNOSTICS_CONFIG = {
    "export_fit_parameters": True,  # Include slope and intercept in output
    "export_correlation_matrix": False,  # Include parameter correlations
    "export_residuals": False,  # Include fit residuals in output
    "export_prediction_bands": False,  # Include confidence/prediction intervals
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_config() -> bool:
    """
    Validate the configuration parameters for consistency and reasonableness.

    Returns:
        bool: True if configuration is valid, False otherwise
    """
    # Check that required columns are specified
    required_columns = ["bare_mass", "pcac_mass_mean", "pcac_mass_error"]
    for col in required_columns:
        if col not in INPUT_CSV_COLUMNS:
            print(f"ERROR: Missing required column specification: {col}")
            return False

    # Check minimum data points requirement
    if DATA_FILTERING["min_data_points"] < 2:
        print("ERROR: Minimum data points must be at least 2 for linear fitting")
        return False

    # Check physical validation bounds
    phys_val = PHYSICAL_VALIDATION
    if phys_val["max_critical_bare_mass"] <= phys_val["min_critical_bare_mass"]:
        print("ERROR: Maximum critical bare mass must be greater than minimum")
        return False

    # Check fit quality thresholds
    fit_qual = FIT_QUALITY_THRESHOLDS
    if fit_qual["max_chi2_per_dof"] <= 0:
        print("ERROR: Maximum chi-squared per DOF must be positive")
        return False

    if fit_qual["min_slope_significance"] <= 0:
        print("ERROR: Minimum slope significance must be positive")
        return False

    # Validate extrapolation configuration
    extrap_config = EXTRAPOLATION_CONFIG
    if extrap_config["extension_factor"] < 0:
        print("ERROR: Extension factor must be non-negative")
        return False

    return True


def get_linear_fit_config() -> Dict[str, Any]:
    """Get the linear fitting configuration."""
    return LINEAR_FIT_CONFIG.copy()


def get_plotting_config() -> Dict[str, Any]:
    """Get the plotting configuration."""
    return {
        "style": PLOT_STYLE.copy(),
        "content": PLOT_CONTENT.copy(),
        "output": PLOT_OUTPUT.copy(),
        "main": PLOTTING_CONFIG.copy(),
    }


def get_error_handling_config() -> Dict[str, Any]:
    """Get the error handling configuration."""
    return ERROR_HANDLING.copy()


def get_validation_config() -> Dict[str, Any]:
    """Get the physical validation configuration."""
    return PHYSICAL_VALIDATION.copy()
