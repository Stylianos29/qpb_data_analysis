#!/usr/bin/env python3
"""
Configuration for critical mass extrapolation visualization.

This module contains styling, layout, and configuration parameters for
creating critical mass extrapolation plots.
"""

from typing import Dict

from library.constants.labels import AXES_LABELS_BY_COLUMN_NAME

from src.analysis.critical_mass_extrapolation._critical_mass_shared_config import (
    OUTPUT_COLUMN_NAMES,
)


# =============================================================================
# CONSTANTS
# =============================================================================


# Plot styling configuration
PLOT_STYLING = {
    # Figure settings
    "figure_size": (10, 8),
    "output_dpi": 300,
    "bbox_inches": "tight",
    "facecolor": "white",
    # Data points styling
    "data_color": "#2E86C1",
    "data_marker_style": "o",
    "data_marker_size": 8,
    "error_bar_cap_size": 4,
    # Fit line styling
    "fit_line_color": "#E74C3C",
    "fit_line_width": 2.5,
    "fit_line_style": "-",
    # Critical mass line styling
    "critical_mass_line_color": "#1F4788",  # Dark blue
    "critical_mass_line_style": "--",
    "critical_mass_line_width": 2.0,
    "critical_mass_line_alpha": 0.6,
    # Zero line styling
    "zero_line_color": "#717D7E",
    "zero_line_style": "-",
    "zero_line_width": 1.0,
    "zero_line_alpha": 0.6,
    # Grid styling
    "grid_alpha": 0.3,
    "grid_style": "--",
    # Text and annotations
    "title_font_size": 14,
    "title_pad": 10,
    "title_width": 80,
    "axis_label_font_size": 12,
    "legend_font_size": 10,
    "legend_location": "upper left",
    "legend_frame_alpha": 1,
    "annotation_font_size": 10,
    "annotation_bbox": {
        "boxstyle": "round,pad=0.3",
        "facecolor": "wheat",
        "alpha": 0.8,
        "edgecolor": "black",
        "linewidth": 1.0,
    },
}

# Layout configuration
LAYOUT_CONFIG = {
    "subplot_spacing": {
        "wspace": 0.3,
        "hspace": 0.4,
    },
    "margin_config": {
        "left": 0.10,
        "right": 0.95,
        "top": 0.90,
        "bottom": 0.15,
    },
}

# Visualization general configuration
VISUALIZATION_CONFIG = {
    "max_plots_per_directory": 50,
    "plot_file_format": "png",
    "clear_existing_plots": False,
    "enable_plot_validation": True,
}

# =============================================================================
# Y-AXIS LABEL MAPPING
# =============================================================================
# Maps analysis types to canonical column names in the centralized
# AXES_LABELS_BY_COLUMN_NAME dictionary from library.constants.labels
#
# This approach maintains a single source of truth for all axis labels while
# allowing this module to specify which label to use for each analysis type.

Y_AXIS_LABEL_COLUMN_MAPPING = {
    "pcac": "PCAC_plateau_mean",
    "pion": "pion_mass_squared",
}

ANALYSIS_CONFIGS = {
    "pcac": {
        "plot_subdirectory": "Critical_bare_mass_extrapolation_pcac",
    },
    "pion": {
        "plot_subdirectory": "Critical_bare_mass_extrapolation_pion",
    },
}

# Parameters to exclude from plot titles (too verbose or not relevant)
TITLE_EXCLUDED_PARAMETERS = [
    "Main_program_type",
    "Maximum_Lanczos_iterations",
    "MPI_geometry",
    "Threads_per_process",
    "Number_of_spinors",
    "Number_of_vectors",
    "APE_alpha",
    # "APE_iterations",
    # "CG_epsilon",
    # "Lanczos_epsilon",
    # "Clover_coefficient",
    # "Delta_Max",
    # "Delta_Min",
    # "QCD_beta_value",
    # "Rho_value",
    # "Number_of_gauge_configurations",
]
# That's it! No output quantities needed.

# Filename configuration
FILENAME_CONFIG = {
    "base_name_pcac": "",
    "base_name_pion": "",
    "custom_prefix_pcac": "",
    "custom_prefix_pion": "",
}

PLOT_DIRECTORY_CONFIG = {
    "parent_directory_name": "Critical_bare_mass_extrapolation",
    "use_parent_directory": True,
    "subdirectory_name_pcac": "from_PCAC_mass",
    "subdirectory_name_pion": "from_Pion_mass",
}

# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================


def get_plot_styling():
    """Get plot styling configuration."""
    return PLOT_STYLING.copy()


def get_layout_config():
    """Get layout configuration."""
    return LAYOUT_CONFIG.copy()


def get_default_clear_existing():
    """Get default value for clearing existing plots."""
    return VISUALIZATION_CONFIG["clear_existing_plots"]


def get_visualization_config():
    """Get general visualization configuration."""
    return VISUALIZATION_CONFIG.copy()


def get_analysis_config(analysis_type):
    """
    Get analysis-specific configuration. Dynamically adds y-axis label
    from centralized dictionary.

    Args:
        analysis_type: "pcac" or "pion"

    Returns:
        Dictionary with analysis-specific settings including dynamically
        retrieved y-axis label from centralized
        AXES_LABELS_BY_COLUMN_NAME
    """
    if analysis_type not in ANALYSIS_CONFIGS:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    # Get base config
    config = ANALYSIS_CONFIGS[analysis_type].copy()

    # Add y-axis label dynamically from centralized dictionary
    column_name = Y_AXIS_LABEL_COLUMN_MAPPING[analysis_type]
    config["default_y_label"] = AXES_LABELS_BY_COLUMN_NAME.get(column_name, column_name)

    return config


def get_results_column_mapping() -> Dict[str, str]:
    """
    Get results column mapping from shared config.

    Returns:
        Dictionary mapping standard names to CSV column names
    """
    return OUTPUT_COLUMN_NAMES.copy()


def get_plateau_column_mapping(analysis_type: str) -> Dict[str, str]:
    """
    Get plateau column mapping from analysis-specific config.

    Args:
        - analysis_type: Type of analysis ("pcac" or "pion")

    Returns:
        Dictionary mapping standard names to CSV column names
    """
    if analysis_type == "pcac":
        from src.analysis.critical_mass_extrapolation._pcac_critical_mass_config import (
            COLUMN_MAPPING,
        )

        return COLUMN_MAPPING.copy()
    elif analysis_type == "pion":
        from src.analysis.critical_mass_extrapolation._pion_critical_mass_config import (
            COLUMN_MAPPING,
        )

        return COLUMN_MAPPING.copy()
    else:
        raise ValueError(f"Unsupported analysis_type: {analysis_type}")


def get_plateau_mass_power(analysis_type: str) -> int:
    """
    Get plateau mass power from analysis-specific config.

    Args:
        - analysis_type: Type of analysis ("pcac" or "pion")

    Returns:
        Power to raise plateau mass values (1 for PCAC, 2 for pion)
    """
    if analysis_type == "pcac":
        from src.analysis.critical_mass_extrapolation._pcac_critical_mass_config import (
            PLATEAU_MASS_POWER,
        )

        return PLATEAU_MASS_POWER
    elif analysis_type == "pion":
        from src.analysis.critical_mass_extrapolation._pion_critical_mass_config import (
            PLATEAU_MASS_POWER,
        )

        return PLATEAU_MASS_POWER
    else:
        raise ValueError(f"Unsupported analysis_type: {analysis_type}")


def get_analysis_prefix(analysis_type: str) -> str:
    """Get analysis prefix from analysis-specific config."""
    if analysis_type == "pcac":
        from src.analysis.critical_mass_extrapolation._pcac_critical_mass_config import (
            OUTPUT_COLUMN_PREFIX,
        )

        return OUTPUT_COLUMN_PREFIX
    elif analysis_type == "pion":
        from src.analysis.critical_mass_extrapolation._pion_critical_mass_config import (
            OUTPUT_COLUMN_PREFIX,
        )

        return OUTPUT_COLUMN_PREFIX
    else:
        raise ValueError(f"Unsupported analysis_type: {analysis_type}")


def get_title_excluded_parameters():
    return TITLE_EXCLUDED_PARAMETERS.copy()


def get_filename_base_name(analysis_type: str) -> str:
    return FILENAME_CONFIG.get(f"base_name_{analysis_type}", "")


def get_filename_custom_prefix(analysis_type: str) -> str:
    """Get the custom prefix for filenames."""
    key = f"custom_prefix_{analysis_type}"
    return FILENAME_CONFIG.get(key, "")


def get_plot_subdirectory_name(analysis_type: str) -> tuple:
    """
    Get subdirectory name(s) for plots.

    Returns:
        - Tuple of (parent_name, subdir_name)
        If use_parent_directory is False, parent_name will be None
    """
    config = PLOT_DIRECTORY_CONFIG

    if config["use_parent_directory"]:
        parent = config["parent_directory_name"]
        subdir = config[f"subdirectory_name_{analysis_type}"]
        return (parent, subdir)
    else:
        # Backward compatibility: flat structure
        parent = None
        subdir = f"{config['parent_directory_name']}_{analysis_type}"
        return (parent, subdir)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_plot_styling():
    """Validate plot styling configuration."""
    # Check figure size
    if len(PLOT_STYLING["figure_size"]) != 2:
        raise ValueError("figure_size must be a tuple of 2 values")

    if any(x <= 0 for x in PLOT_STYLING["figure_size"]):
        raise ValueError("figure_size values must be positive")

    # Check DPI
    if PLOT_STYLING["output_dpi"] <= 0:
        raise ValueError("output_dpi must be positive")

    # Check font sizes
    font_size_keys = [
        "title_font_size",
        "axis_label_font_size",
        "legend_font_size",
        "annotation_font_size",
    ]
    for key in font_size_keys:
        if PLOT_STYLING[key] <= 0:
            raise ValueError(f"{key} must be positive")

    # Check marker size
    if PLOT_STYLING["data_marker_size"] <= 0:
        raise ValueError("data_marker_size must be positive")

    # Check line widths
    line_width_keys = ["fit_line_width", "critical_mass_line_width", "zero_line_width"]
    for key in line_width_keys:
        if PLOT_STYLING[key] <= 0:
            raise ValueError(f"{key} must be positive")

    # Check alpha values
    alpha_keys = ["critical_mass_line_alpha", "zero_line_alpha", "grid_alpha"]
    for key in alpha_keys:
        if not (0 <= PLOT_STYLING[key] <= 1):
            raise ValueError(f"{key} must be between 0 and 1")


def validate_layout_config():
    """Validate layout configuration."""
    # Check subplot spacing
    spacing = LAYOUT_CONFIG["subplot_spacing"]
    for key, value in spacing.items():
        if not (0 <= value <= 1):
            raise ValueError(f"subplot_spacing.{key} must be between 0 and 1")

    # Check margins
    margins = LAYOUT_CONFIG["margin_config"]
    for key, value in margins.items():
        if not (0 <= value <= 1):
            raise ValueError(f"margin_config.{key} must be between 0 and 1")

    # Check margin consistency
    if margins["left"] >= margins["right"]:
        raise ValueError("left margin must be less than right margin")

    if margins["bottom"] >= margins["top"]:
        raise ValueError("bottom margin must be less than top margin")


def validate_visualization_config():
    """
    Validate all visualization configuration.
    """
    validate_plot_styling()
    validate_layout_config()

    # Check general config
    if VISUALIZATION_CONFIG["max_plots_per_directory"] <= 0:
        raise ValueError("max_plots_per_directory must be positive")

    if not VISUALIZATION_CONFIG["plot_file_format"]:
        raise ValueError("plot_file_format cannot be empty")

    # Check analysis configs
    for analysis_type, config in ANALYSIS_CONFIGS.items():
        if not config["plot_subdirectory"]:
            raise ValueError(f"plot_subdirectory for {analysis_type} cannot be empty")

    # NEW: Validate that all mapped column names exist in centralized dictionary
    for analysis_type, column_name in Y_AXIS_LABEL_COLUMN_MAPPING.items():
        if column_name not in AXES_LABELS_BY_COLUMN_NAME:
            raise ValueError(
                f"Y-axis label mapping error: Column name '{column_name}' "
                f"(for analysis type '{analysis_type}') not found in "
                f"AXES_LABELS_BY_COLUMN_NAME. Please add it to "
                f"library/constants/labels.py"
            )
