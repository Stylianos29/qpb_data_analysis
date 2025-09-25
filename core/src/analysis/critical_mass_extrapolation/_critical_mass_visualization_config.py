#!/usr/bin/env python3
"""
Configuration for critical mass extrapolation visualization.

This module contains styling, layout, and configuration parameters for
creating critical mass extrapolation plots.
"""

from typing import Dict, Any


# =============================================================================
# CONSTANTS
# =============================================================================

# Results CSV column mapping (standard across all analysis types)
RESULTS_COLUMN_MAPPING = {
    "critical_mass_mean": "critical_mass_mean",
    "critical_mass_error": "critical_mass_error",
    "slope_mean": "slope_mean",
    "intercept_mean": "intercept_mean",
    "r_squared": "r_squared",
}


def get_results_column_mapping():
    """Get results CSV column mapping."""
    return RESULTS_COLUMN_MAPPING.copy()


# Analysis-specific plateau column mappings
PLATEAU_COLUMN_MAPPINGS = {
    "pcac": {
        "bare_mass": "Bare_mass",
        "plateau_mean": "PCAC_plateau_mean",
        "plateau_error": "PCAC_plateau_error",
    },
    "pion": {
        "bare_mass": "Bare_mass",
        "plateau_mean": "pion_plateau_mean",
        "plateau_error": "pion_plateau_error",
    },
}


def get_plateau_column_mapping(analysis_type: str):
    """Get plateau column mapping for specified analysis type."""
    return PLATEAU_COLUMN_MAPPINGS[analysis_type].copy()


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
    "critical_mass_line_color": "#28B463",
    "critical_mass_line_style": "--",
    "critical_mass_line_width": 2.0,
    "critical_mass_line_alpha": 0.8,
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
    "title_pad": 20,
    "title_width": 80,
    "axis_label_font_size": 12,
    "legend_font_size": 10,
    "legend_location": "upper right",
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
    "clear_existing_plots": True,
    "enable_plot_validation": True,
}

# Analysis-specific configurations
ANALYSIS_CONFIGS = {
    "pcac": {
        "plot_subdirectory": "critical_mass_extrapolation_pcac",
        "default_y_label": "PCAC Mass",
        "data_label_prefix": "PCAC",
    },
    "pion": {
        "plot_subdirectory": "critical_mass_extrapolation_pion",
        "default_y_label": "Pion Effective Mass",
        "data_label_prefix": "Pion",
    },
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


def get_visualization_config():
    """Get general visualization configuration."""
    return VISUALIZATION_CONFIG.copy()


def get_analysis_config(analysis_type):
    """Get analysis-specific configuration."""
    if analysis_type not in ANALYSIS_CONFIGS:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    return ANALYSIS_CONFIGS[analysis_type].copy()


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
    """Validate all visualization configuration."""
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

        if not config["default_y_label"]:
            raise ValueError(f"default_y_label for {analysis_type} cannot be empty")

        if not config["data_label_prefix"]:
            raise ValueError(f"data_label_prefix for {analysis_type} cannot be empty")
