"""
Configuration for computational cost extrapolation using DataPlotter
integration.

This module contains all configuration parameters for extrapolating
computational costs using both fixed bare mass and fixed PCAC mass
methods. Uses hierarchical configuration structure for clean separation
of concerns.
"""

from typing import Dict, Any


# =============================================================================
# HIERARCHICAL CONFIGURATION
# =============================================================================
CONFIG = {
    # SHARED SETTINGS (Used by both PCAC and cost analysis workflows)
    "shared": {
        # Base directory organization
        "base_subdirectory": "Computational_cost_extrapolation",
        # Output settings
        "output": {
            "csv_filename": "computational_cost_extrapolation.csv",
            "float_precision": 6,
        },
        # Data validation thresholds
        "data_validation": {
            "min_data_points_for_fitting": 4,  # Minimum bare masses to fit
            "min_data_points_for_averaging": 3,  # Minimum configurations to average
        },
        # Configuration averaging settings
        "averaging": {
            "filter_out_parameters": [
                "Configuration_label",  # Always exclude - we average across configs
                # "MPI_geometry",         # Usually computational
                # detail, not physics
            ],
        },
        # Global extrapolation method and targets
        "extrapolation": {
            "target_bare_mass": 0.005,  # Reference bare mass value
            "reference_pcac_mass": 0.005,  # Reference PCAC mass value
            # Fitting range constraints (None = no constraint)
            "fit_range_min_bare_mass": None,
            "fit_range_max_bare_mass": None,
        },
        # Extrapolation line styling (used by both PCAC and cost
        # analyses)
        "extrapolation_lines": {
            "vertical_line_style": {
                "color": "green",
                "linestyle": "--",
                "alpha": 0.7,
                "linewidth": 1.5,
            },
            "horizontal_line_style": {
                "color": "red",
                "linestyle": "--",
                "alpha": 0.7,
                "linewidth": 1.5,
            },
            # Uncertainty band styling
            "uncertainty_band_style": {
                "alpha": 0.2,
                "color": None,  # None to inherit from horizontal_line_style
            },
        },
    },
    # PCAC MASS ANALYSIS CONFIGURATION
    "pcac_analysis": {
        # Input CSV column mappings for PCAC data
        "input_columns": {
            "bare_mass": "Bare_mass",
            "pcac_mass_mean": "plateau_PCAC_mass_mean",
            "pcac_mass_error": "plateau_PCAC_mass_error",
            # Future: "pcac_mass": "plateau_PCAC_mass"  # for tuple
            # format
        },
        # DataPlotter settings for PCAC analysis
        "plotting": {
            # Plot variables
            "x_variable": "Bare_mass",
            "y_variable": "PCAC_mass_estimate",  # Constructed from mean+error
            # Figure settings
            "figure_size": (7, 5),
            "top_margin_adjustment": 0.88,
            "left_margin_adjustment": 0.13,
            # Curve fitting for PCAC vs bare mass relationship
            "fit_function": "linear",  # PCAC_mass = a * bare_mass + b
            "show_fit_parameters": True,
            "fit_label_location": "top left",
            "fit_curve_range": (0, 0.08),  # None or tuple
            # Styling
            "marker_size": 8,
            "capsize": 5,
            "include_legend": True,
            "legend_location": "upper left",
            # Output settings
            "verbose": False,
            "file_format": "png",
        },
        # PCAC-specific validation settings
        "validation": {
            "min_data_points_for_pcac_fit": 3,  # Minimum bare mass values for PCAC fit
            "require_positive_slope": True,  # PCAC mass should increase with bare mass
        },
        # PCAC plot generation settings
        "plot_generation": {
            "create_pcac_plots": True,  # Generate PCAC vs bare mass plots
        },
        # PCAC extrapolation line labels
        "extrapolation_labels": {
            "vertical_line_label": r"a$m^{\text{target}}$",  # Derived bare mass
            "horizontal_line_label": r"a$m^{\text{ref.}}_{\text{PCAC}}$",  # Reference PCAC mass
        },
    },
    # COMPUTATIONAL COST ANALYSIS CONFIGURATION
    "cost_analysis": {
        # Input CSV column mappings for processed cost data
        "input_columns": {
            "bare_mass": "Bare_mass",
            "configuration_label": "Configuration_label",
            "core_hours_per_spinor": "Average_core_hours_per_spinor",
            "kernel_operator_type": "Kernel_operator_type",
        },
        # DataPlotter settings for cost analysis
        "plotting": {
            # Plot variables
            "x_variable": "Bare_mass",
            "y_variable": "Average_core_hours_per_spinor_per_configuration",
            # Figure settings
            "figure_size": (7, 5),
            "top_margin_adjustment": 0.88,
            "left_margin_adjustment": 0.13,
            # Curve fitting for cost vs bare mass relationship
            "fit_function": "shifted_power_law",  # a/(x-b)+c
            "show_fit_parameters": True,
            "fit_label_location": "top right",
            # Styling
            "marker_size": 8,
            "capsize": 5,
            "include_legend": True,
            "legend_location": "upper right",
            # Output settings
            "verbose": False,
            "file_format": "png",
        },
        # Cost extrapolation line labels
        "extrapolation_labels": {
            "vertical_line_label": r"a$m^{\text{ref.}}$",  # Target bare mass
            "horizontal_line_label": r"$t^{\text{target}}$",  # Extrapolated cost
        },
    },
}


# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================


def get_shared_config() -> Dict[str, Any]:
    """Get shared configuration used by both analysis workflows."""
    return CONFIG["shared"]


def get_pcac_config() -> Dict[str, Any]:
    """Get PCAC mass analysis configuration."""
    return CONFIG["pcac_analysis"]


def get_cost_config() -> Dict[str, Any]:
    """Get computational cost analysis configuration."""
    return CONFIG["cost_analysis"]


def get_reference_pcac_mass() -> float:
    """Get reference PCAC mass for extrapolation."""
    return CONFIG["shared"]["extrapolation"]["reference_pcac_mass"]


def get_reference_bare_mass() -> float:
    """Get target bare mass for extrapolation."""
    return CONFIG["shared"]["extrapolation"]["target_bare_mass"]


def get_base_subdirectory() -> str:
    """Get base subdirectory for all plots."""
    return CONFIG["shared"]["base_subdirectory"]


# Backward compatibility functions
def get_averaging_config() -> Dict[str, Any]:
    """Get averaging configuration (backward compatibility)."""
    return CONFIG["shared"]["averaging"]


def get_validation_config() -> Dict[str, Any]:
    """Get validation configuration (backward compatibility)."""
    return CONFIG["shared"]["data_validation"]


def get_output_config() -> Dict[str, Any]:
    """Get output configuration (backward compatibility)."""
    return CONFIG["shared"]["output"]


def get_extrapolation_config() -> Dict[str, Any]:
    """Get extrapolation configuration (backward compatibility)."""
    return CONFIG["shared"]["extrapolation"]


# Legacy accessor (backward compatibility)
def get_plotting_config() -> Dict[str, Any]:
    """Get cost analysis plotting configuration (backward
    compatibility)."""
    return CONFIG["cost_analysis"]["plotting"]


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_config() -> bool:
    """
    Validate the hierarchical configuration parameters.

    Returns
    -------
    bool
        True if configuration is valid

    Raises
    ------
    ValueError
        If configuration is invalid
    """
    try:
        # Check required top-level sections exist
        required_sections = ["shared", "pcac_analysis", "cost_analysis"]
        for section in required_sections:
            if section not in CONFIG:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate shared configuration
        _validate_shared_config()

        # Validate PCAC analysis configuration
        _validate_pcac_config()

        # Validate cost analysis configuration
        _validate_cost_config()

        # Validate method-specific requirements
        _validate_method_specific_config()

        return True

    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


def _validate_shared_config():
    """Validate shared configuration section."""
    shared = CONFIG["shared"]

    # Check required shared subsections
    required_subsections = [
        "base_subdirectory",
        "output",
        "data_validation",
        "averaging",
        "extrapolation",
    ]
    for subsection in required_subsections:
        if subsection not in shared:
            raise ValueError(f"Missing required shared subsection: {subsection}")


def _validate_pcac_config():
    """Validate PCAC analysis configuration section."""
    pcac = CONFIG["pcac_analysis"]

    # Check required PCAC subsections
    required_subsections = [
        "input_columns",
        "plotting",
        "validation",
        "plot_generation",
    ]
    for subsection in required_subsections:
        if subsection not in pcac:
            raise ValueError(f"Missing required pcac_analysis subsection: {subsection}")

    # Check required input columns
    required_columns = ["bare_mass", "pcac_mass_mean", "pcac_mass_error"]
    for column in required_columns:
        if column not in pcac["input_columns"]:
            raise ValueError(f"Missing required PCAC input column mapping: {column}")


def _validate_cost_config():
    """Validate cost analysis configuration section."""
    cost = CONFIG["cost_analysis"]

    # Check required cost subsections
    required_subsections = [
        "input_columns",
        "plotting",
        "extrapolation_labels",
    ]  # REMOVED "extrapolation_lines"
    for subsection in required_subsections:
        if subsection not in cost:
            raise ValueError(f"Missing required cost_analysis subsection: {subsection}")

    # Check required input columns
    required_columns = [
        "bare_mass",
        "configuration_label",
        "core_hours_per_spinor",
        "kernel_operator_type",
    ]
    for column in required_columns:
        if column not in cost["input_columns"]:
            raise ValueError(f"Missing required cost input column mapping: {column}")


def _validate_method_specific_config():
    """Validate method-specific configuration requirements."""
    extrapolation = CONFIG["shared"]["extrapolation"]

    # Validate reference_bare_massis present and numeric
    if "target_bare_mass" not in extrapolation:
        raise ValueError("reference_bare_massmust be present in configuration")

    target = extrapolation["target_bare_mass"]
    if not isinstance(target, (int, float)):
        raise ValueError("reference_bare_massmust be a numerical value")

    # Validate reference_pcac_mass is present and numeric
    if "reference_pcac_mass" not in extrapolation:
        raise ValueError("reference_pcac_mass must be present in configuration")

    reference = extrapolation["reference_pcac_mass"]
    if not isinstance(reference, (int, float)):
        raise ValueError("reference_pcac_mass must be a numerical value")
