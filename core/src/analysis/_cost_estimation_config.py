"""
Configuration for computational cost estimation using DataPlotter integration.

This module contains all configuration parameters for analyzing computational
costs (core-hours per spinor per configuration) using the DataPlotter class
for automatic grouping, fitting, and visualization.

Place this file as:
qpb_data_analysis/core/src/analysis/_cost_estimation_config.py
"""

from typing import List, Dict, Any, Optional

# =============================================================================
# INPUT/OUTPUT CONFIGURATION
# =============================================================================

# Processed parameter values CSV column names (minimal set)
PROCESSED_PARAMS_CSV_COLUMNS = {
    "bare_mass": "Bare_mass",
    "configuration_label": "Configuration_label",
    "core_hours_per_spinor": "Average_core_hours_per_spinor",
    "mpi_geometry": "MPI_geometry",
    "kernel_operator_type": "Kernel_operator_type",
}

# Output CSV configuration
OUTPUT_CSV_CONFIG = {
    "default_filename": "computational_cost_analysis.csv",
    "float_precision": 6,
    "include_fit_results": True,
    "include_group_metadata": True,
}

# Reference PCAC mass for target bare mass extraction
REFERENCE_CONFIG = {
    "default_reference_pcac_mass": 0.05,  # Reference PCAC mass value
    "reference_pcac_mass_range": (0.001, 0.2),  # Valid range for reference values
}

# =============================================================================
# DATAPLOTTER CONFIGURATION
# =============================================================================

# DataPlotter plotting configuration
DATAPLOTTER_CONFIG = {
    # Plot variables
    "x_variable": "Bare_mass",
    "y_variable": "Average_core_hours_per_spinor_per_configuration_with_errors",  # With error bars
    # Figure settings
    "figure_size": (12, 8),
    "font_size": 12,
    # Plot titles
    "include_plot_title": True,
    # "custom_plot_title": None,  # Let DataPlotter auto-generate from parameters
    "title_from_columns": None,  # Auto-detect from grouping parameters
    "title_size": 14,
    "bold_title": True,
    "title_wrapping_length": 80,
    # Curve fitting settings
    "fit_function": "shifted_power_law",  # a/(x-b) + c - perfect for cost analysis
    "show_fit_parameters_on_plot": True,
    "fit_label_location": "top right",
    "fit_label_format": ".3f",
    "fit_curve_range": None,  # Auto-determine from data
    # Styling
    "marker_size": 8,
    "capsize": 3,
    "empty_markers": False,
    "include_legend": True,
    "legend_location": "upper left",
    # Axis settings
    "xaxis_log_scale": False,
    "yaxis_log_scale": False,
    # File output
    "save_figure": True,
    "file_format": "png",
    "dpi": 300,
}

# Data filtering configuration
DATA_FILTERING_CONFIG = {
    "min_data_points_for_fitting": 4,  # Minimum points needed for reliable fitting
    "max_bare_mass_for_analysis": 0.15,  # Upper cutoff for bare mass
    "min_bare_mass_for_analysis": 0.005,  # Lower cutoff for bare mass
    "remove_outliers": False,  # Whether to apply outlier detection
    "outlier_threshold": 3.0,  # Standard deviations for outlier detection
}

# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

# Statistical analysis configuration
ANALYSIS_CONFIG = {
    "calculate_group_statistics": True,
    "include_correlation_analysis": False,
    "bootstrap_samples": 1000,  # For uncertainty estimation
    "confidence_interval": 0.95,
}

# Fit quality validation
FIT_VALIDATION_CONFIG = {
    "min_r_squared": 0.7,  # Minimum acceptable R²
    "max_parameter_uncertainty": 0.5,  # Maximum relative parameter uncertainty
    "check_physical_reasonableness": True,
    "max_reasonable_cost": 1000.0,  # Maximum reasonable core-hours
    "min_reasonable_cost": 0.1,  # Minimum reasonable core-hours
}

# =============================================================================
# RESULT COMPILATION CONFIGURATION
# =============================================================================

# What to include in output CSV
RESULT_COMPILATION_CONFIG = {
    "include_fit_parameters": True,
    "include_fit_quality_metrics": True,
    "include_group_statistics": True,
    "include_data_point_counts": True,
    "include_cost_predictions": True,
    "cost_prediction_points": [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
    ],  # Bare mass values for prediction
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_config() -> bool:
    """Validate configuration parameters."""
    try:
        # Validate data filtering
        if DATA_FILTERING_CONFIG["min_data_points_for_fitting"] < 3:
            raise ValueError("Need at least 3 points for fitting")

        # Validate bare mass ranges
        if (
            DATA_FILTERING_CONFIG["min_bare_mass_for_analysis"]
            >= DATA_FILTERING_CONFIG["max_bare_mass_for_analysis"]
        ):
            raise ValueError("Invalid bare mass range")

        # Validate fit quality thresholds
        if not (0 <= FIT_VALIDATION_CONFIG["min_r_squared"] <= 1):
            raise ValueError("R² threshold must be between 0 and 1")

        # Validate cost prediction points
        prediction_points = RESULT_COMPILATION_CONFIG["cost_prediction_points"]
        min_mass = DATA_FILTERING_CONFIG["min_bare_mass_for_analysis"]
        max_mass = DATA_FILTERING_CONFIG["max_bare_mass_for_analysis"]

        for point in prediction_points:
            if not (min_mass <= point <= max_mass):
                raise ValueError(
                    f"Prediction point {point} outside valid range [{min_mass}, {max_mass}]"
                )

        return True

    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


def get_dataplotter_config() -> Dict[str, Any]:
    """Get DataPlotter configuration."""
    return DATAPLOTTER_CONFIG.copy()


def get_data_filtering_config() -> Dict[str, Any]:
    """Get data filtering configuration."""
    return DATA_FILTERING_CONFIG.copy()


def get_analysis_config() -> Dict[str, Any]:
    """Get analysis configuration."""
    return ANALYSIS_CONFIG.copy()


def get_fit_validation_config() -> Dict[str, Any]:
    """Get fit validation configuration."""
    return FIT_VALIDATION_CONFIG.copy()


def get_result_compilation_config() -> Dict[str, Any]:
    """Get result compilation configuration."""
    return RESULT_COMPILATION_CONFIG.copy()
