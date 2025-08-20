"""
Configuration for computational cost extrapolation using DataPlotter
integration.

This module contains all configuration parameters for extrapolating
computational costs (core-hours per spinor per configuration) using the
DataPlotter class for automatic grouping, fitting, and visualization.
"""

from typing import List, Dict, Any, Optional


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

CONFIG = {
    # Input CSV column mappings
    "input_columns": {
        "bare_mass": "Bare_mass",
        "configuration_label": "Configuration_label",
        "core_hours_per_spinor": "Average_core_hours_per_spinor",
        "kernel_operator_type": "Kernel_operator_type",
    },
    # Output settings
    "output": {
        "csv_filename": "computational_cost_extrapolation.csv",
        "float_precision": 6,
        "include_fit_results": True,
        "include_metadata": True,
    },
    # Data validation
    "data_validation": {
        "min_data_points_for_fitting": 4,  # Minimum bare mass values to average
        "min_data_points_for_averaging": 3,  # Minimum configurations to average
    },
    # Configuration averaging settings
    "averaging": {
        "filter_out_parameters": [
            "Configuration_label",  # Always exclude - we average across configs
            # "MPI_geometry",         # Usually computational detail, not physics
        ],
    },
    # DataPlotter settings
    "plotting": {
        # Directory organization
        "base_subdirectory": "Computational_cost_extrapolation",
        # Plot variables
        "x_variable": "Bare_mass",
        "y_variable": "Average_core_hours_per_spinor_per_configuration",
        # Figure settings
        "figure_size": (7, 5),
        # Curve fitting
        "fit_function": "shifted_power_law",  # a/(x-b)+c
        "show_fit_parameters": True,
        "fit_label_location": "top right",
        # Styling
        "marker_size": 8,
        "capsize": 5,
        "include_legend": True,
        "legend_location": "upper right",
        # Error bars
        "left_margin_adjustment": 0.13,
        "top_margin_adjustment": 0.88,
    },
    # Extrapolation settings
    "extrapolation": {
        "group_by_multivalued_params": True,
        "average_across_configurations": True,
        "include_single_valued_params": True,
        "validate_results": True,
        "min_success_rate": 0.5,  # Require 50% of groups to pass validation
        "target_bare_mass": 0.005,
        # Fitting range constraints (None = no constraint)
        "fit_range_min_bare_mass": 0.0,
        "fit_range_max_bare_mass": None,  # No upper limit (yet)
    },
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_config() -> bool:
    """
    Validate the configuration parameters.

    Returns
    -------
    bool
        True if configuration is valid, False otherwise
    """
    try:
        # Check required sections exist
        required_sections = [
            "input_columns",
            "output",
            "data_validation",
            "averaging",
            "plotting",
            "extrapolation",
        ]
        for section in required_sections:
            if section not in CONFIG:
                print(f"Missing required configuration section: {section}")
                return False

        # Check required input columns
        required_columns = ["bare_mass", "configuration_label", "core_hours_per_spinor"]
        for col in required_columns:
            if col not in CONFIG["input_columns"]:
                print(f"Missing required input column mapping: {col}")
                return False

        # Check numeric ranges
        if CONFIG["data_validation"]["min_data_points_for_fitting"] < 2:
            print("min_data_points_for_fitting must be at least 2")
            return False

        # Check plot variables are defined
        if not CONFIG["plotting"]["x_variable"] or not CONFIG["plotting"]["y_variable"]:
            print("Plot variables must be defined")
            return False

        return True

    except Exception as e:
        print(f"Configuration validation error: {e}")
        return False


def get_input_column(key: str) -> str:
    """Get input column name by key."""
    return CONFIG["input_columns"][key]


def get_output_config() -> Dict[str, Any]:
    """Get output configuration."""
    return CONFIG["output"].copy()


def get_averaging_config() -> Dict[str, Any]:
    """Get configuration averaging configuration."""
    return CONFIG["averaging"].copy()


def get_validation_config() -> Dict[str, Any]:
    """Get data validation configuration."""
    return CONFIG["data_validation"].copy()


def get_plotting_config() -> Dict[str, Any]:
    """Get plotting configuration."""
    return CONFIG["plotting"].copy()


def get_extrapolation_config() -> Dict[str, Any]:
    """Get extrapolation configuration."""
    return CONFIG["extrapolation"].copy()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_cost_column_names() -> Dict[str, str]:
    """
    Get the standard column names for cost extrapolation.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping purpose to column name
    """
    return {
        "mean": "Average_core_hours_per_spinor_per_configuration_mean",
        "error": "Average_core_hours_per_spinor_per_configuration_error",
        "std": "Average_core_hours_per_spinor_per_configuration_std",
        "count": "Number_of_configurations",
        "tuple": "Average_core_hours_per_spinor_per_configuration",  # (value, error) tuples
    }


def get_validation_thresholds() -> Dict[str, float]:
    """
    Get validation thresholds for results.

    Returns
    -------
    Dict[str, float]
        Dictionary of validation thresholds
    """
    return {
        "min_data_points": CONFIG["data_validation"]["min_data_points_for_fitting"],
        # "min_data_points_for_averaging": CONFIG["data_validation"][
        # "min_data_points_for_averaging"
        # ],
        # "min_success_rate": CONFIG["extrapolation"]["min_success_rate"],
    }
