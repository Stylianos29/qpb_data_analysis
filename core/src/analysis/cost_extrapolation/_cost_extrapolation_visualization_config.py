"""
Visualization configuration for computational cost extrapolation.

Defines plotting parameters, styling, and column mappings for creating
high-quality visualizations of cost extrapolation results.
"""

from typing import Dict, Any, List


# =============================================================================
# PLOT TYPES AND SUBDIRECTORIES
# =============================================================================

PLOT_SUBDIRECTORIES = {
    "pcac": "cost_extrapolation_pcac",
    "pion": "cost_extrapolation_pion",
}

# Plot type sub-subdirectories
PLOT_TYPE_SUBDIRECTORIES = {
    "mass_fit": "plateau_mass_vs_bare_mass",
    "cost_fit": "cost_vs_bare_mass",
}

# =============================================================================
# FIGURE CONFIGURATION
# =============================================================================

FIGURE_CONFIG = {
    # Figure dimensions
    "figure_size": (10, 6),  # Width x Height in inches
    "dpi": 300,  # Resolution for saved figures
    # Margins and spacing
    "left_margin": 0.10,
    "right_margin": 0.95,
    "bottom_margin": 0.10,
    "top_margin": 0.92,
    # Grid
    "grid": True,
    "grid_alpha": 0.3,
    "grid_linestyle": "--",
}

# =============================================================================
# DATA POINT STYLING
# =============================================================================

DATA_POINT_STYLE = {
    # Markers
    "marker": "o",
    "marker_size": 4,
    "marker_edge_width": 1.5,
    # Error bars
    "capsize": 5,
    "capthick": 1.5,
    "error_linewidth": 1.5,
    # Colors (will be overridden by analysis type)
    "color": "blue",
    "edge_color": "darkblue",
}

# =============================================================================
# FIT LINE STYLING
# =============================================================================

FIT_LINE_STYLE = {
    "linewidth": 2.0,
    "linestyle": "-",
    "alpha": 0.8,
    "color": "red",
}

# =============================================================================
# EXTRAPOLATION MARKER STYLING
# =============================================================================

EXTRAPOLATION_MARKER_STYLE = {
    # Marker for extrapolated point
    "marker": "D",  # Diamond
    "marker_size": 12,
    "color": "green",
    "edge_color": "darkgreen",
    "edge_width": 2,
    "alpha": 0.8,
    "zorder": 10,  # Draw on top
}

# Vertical/horizontal lines for extrapolation
EXTRAPOLATION_LINES_STYLE = {
    "linewidth": 1.5,
    "linestyle": "--",
    "color": "green",
    "alpha": 0.6,
}

# =============================================================================
# LEGEND CONFIGURATION
# =============================================================================

LEGEND_CONFIG = {
    "loc": "best",  # "lower right",
    "frameon": True,
    "framealpha": 1.0,
    "edgecolor": "black",
    "fancybox": False,
    "fontsize": 10,
}

# =============================================================================
# AXIS LABELS
# =============================================================================

AXIS_LABELS = {
    "bare_mass": {
        "label": r"$am$",
        "fontsize": 12,
    },
    "pcac_mass": {
        "label": r"$am_{\mathrm{PCAC}}$",
        "fontsize": 12,
    },
    "pion_mass_squared": {
        "label": r"$a^2m^2_{\pi}$",
        "fontsize": 12,
    },
    "cost": {
        "label": "Computational cost (core-hours/spinor/config)",
        "fontsize": 12,
    },
}

# =============================================================================
# TITLE STYLING
# =============================================================================

TITLE_STYLING = {
    "fontsize": 14,
    "pad": 10,  # Vertical space between title and plot
}

# =============================================================================
# ANNOTATION STYLING
# =============================================================================

# Annotation styling
ANNOTATION_STYLE = {
    "fontsize": 10,
    "ha": "center",
    "bbox_style": "round,pad=0.3",
    "bbox_facecolor": "white",
    "bbox_edgecolor": "gray",
    "bbox_linewidth": 0.5,
    "arrow_color": "gray",
    "arrow_linewidth": 0.5,
    "offset_x": 15,
    "offset_y": -15,
}

# Reference axes styling
REFERENCE_AXES_STYLE = {
    "color": "black",
    "linestyle": "-",
    "linewidth": 1.2,
    "alpha": 0.8,
    "zorder": 1,
}

# Sample count column names by analysis type
SAMPLE_COUNT_COLUMNS = {
    "pcac": "PCAC_n_successful_samples",
    "pion": "pion_n_successful_samples",
}

# Fit line extension configuration
FIT_LINE_EXTENSION = {
    "mass_fit_y_min": -0.001,  # Extend to slightly below y=0
    "mass_fit_x_max_factor": 1.05,  # 5% beyond max data on right
    "cost_fit_x_min_factor": 0.95,  # 5% before min data
    "cost_fit_x_max_factor": 1.05,  # 5% beyond max data
}

# =============================================================================
# ANALYSIS-SPECIFIC CONFIGURATION
# =============================================================================

ANALYSIS_CONFIG = {
    "pcac": {
        "mass_column_mean": "PCAC_plateau_mean",
        "mass_column_error": "PCAC_plateau_error",
        "mass_power": 1,  # PCAC mass to power 1
        "mass_label": "PCAC Mass",
        "plot_title_prefix": "PCAC Cost Extrapolation",
        "data_color": "blue",
        "fit_color": "red",
    },
    "pion": {
        "mass_column_mean": "pion_plateau_mean",
        "mass_column_error": "pion_plateau_error",
        "mass_power": 2,  # Pion mass squared
        "mass_label": r"$m_\pi^2$",
        "plot_title_prefix": "Pion Cost Extrapolation",
        "data_color": "purple",
        "fit_color": "orange",
    },
}

# =============================================================================
# COLUMN MAPPINGS FOR RESULTS CSV
# =============================================================================

RESULTS_COLUMN_MAPPING = {
    # Grouping parameters
    "grouping_params": [
        "Kernel_operator_type",
        "Overlap_operator_method",
        "KL_diagonal_order",
    ],
    # Mass fit results
    "mass_fit": {
        "slope_mean": "mass_fit_slope_mean",
        "slope_error": "mass_fit_slope_error",
        "intercept_mean": "mass_fit_intercept_mean",
        "intercept_error": "mass_fit_intercept_error",
        "r_squared": "mass_fit_r_squared",
        "chi2_reduced": "mass_fit_chi2_reduced",
        "q_value": "mass_fit_q_value",
    },
    # Cost fit results
    "cost_fit": {
        "param_a_mean": "cost_fit_param_a_mean",
        "param_a_error": "cost_fit_param_a_error",
        "param_b_mean": "cost_fit_param_b_mean",
        "param_b_error": "cost_fit_param_b_error",
        "param_c_mean": "cost_fit_param_c_mean",
        "param_c_error": "cost_fit_param_c_error",
        "r_squared": "cost_fit_r_squared",
        "chi2_reduced": "cost_fit_chi2_reduced",
        "q_value": "cost_fit_q_value",
    },
    # Derived quantities
    "derived_bare_mass_mean": "derived_bare_mass_mean",
    "derived_bare_mass_error": "derived_bare_mass_error",
    "extrapolated_cost_mean": "extrapolated_cost_mean",
    "extrapolated_cost_error": "extrapolated_cost_error",
}

# Column mappings for mass data CSV (plateau estimates)
MASS_DATA_COLUMN_MAPPING = {
    "bare_mass": "Bare_mass",
    # Analysis-specific columns set dynamically
}

# Column mappings for cost data CSV
COST_DATA_COLUMN_MAPPING = {
    "bare_mass": "Bare_mass",
    "cost_mean": "Average_core_hours_per_spinor",
}

# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================


def get_figure_config() -> Dict[str, Any]:
    """Get figure configuration."""
    return FIGURE_CONFIG.copy()


def get_data_point_style() -> Dict[str, Any]:
    """Get data point styling."""
    return DATA_POINT_STYLE.copy()


def get_fit_line_style() -> Dict[str, Any]:
    """Get fit line styling."""
    return FIT_LINE_STYLE.copy()


def get_extrapolation_marker_style() -> Dict[str, Any]:
    """Get extrapolation marker styling."""
    return EXTRAPOLATION_MARKER_STYLE.copy()


def get_extrapolation_lines_style() -> Dict[str, Any]:
    """Get extrapolation lines styling."""
    return EXTRAPOLATION_LINES_STYLE.copy()


def get_legend_config() -> Dict[str, Any]:
    """Get legend configuration."""
    return LEGEND_CONFIG.copy()


def get_axis_labels() -> Dict[str, Dict[str, Any]]:
    """Get axis labels."""
    return AXIS_LABELS.copy()


def get_title_styling() -> Dict[str, Any]:
    """Get title styling configuration."""
    return TITLE_STYLING.copy()


def get_analysis_config(analysis_type: str) -> Dict[str, Any]:
    """
    Get analysis-specific configuration.

    Args:
        analysis_type: "pcac" or "pion"

    Returns:
        Dictionary with analysis-specific settings
    """
    if analysis_type not in ANALYSIS_CONFIG:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    return ANALYSIS_CONFIG[analysis_type].copy()


def get_plot_subdirectory(analysis_type: str) -> str:
    """Get plot subdirectory name for analysis type."""
    if analysis_type not in PLOT_SUBDIRECTORIES:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    return PLOT_SUBDIRECTORIES[analysis_type]


def get_plot_type_subdirectories() -> Dict[str, str]:
    """Get plot type subdirectory names."""
    return PLOT_TYPE_SUBDIRECTORIES.copy()


def get_results_column_mapping() -> Dict[str, Any]:
    """Get column mapping for results CSV."""
    return RESULTS_COLUMN_MAPPING.copy()


def get_mass_data_column_mapping(analysis_type: str) -> Dict[str, str]:
    """
    Get column mapping for mass data CSV.

    Args:
        analysis_type: "pcac" or "pion"

    Returns:
        Dictionary mapping standard names to CSV column names
    """
    analysis_cfg = get_analysis_config(analysis_type)

    mapping = MASS_DATA_COLUMN_MAPPING.copy()
    mapping["mass_mean"] = analysis_cfg["mass_column_mean"]
    mapping["mass_error"] = analysis_cfg["mass_column_error"]

    return mapping


def get_cost_data_column_mapping() -> Dict[str, str]:
    """Get column mapping for cost data CSV."""
    return COST_DATA_COLUMN_MAPPING.copy()


def get_annotation_style() -> Dict[str, Any]:
    """Get annotation styling configuration."""
    return ANNOTATION_STYLE.copy()


def get_reference_axes_style() -> Dict[str, Any]:
    """Get reference axes styling."""
    return REFERENCE_AXES_STYLE.copy()


def get_sample_count_columns() -> Dict[str, str]:
    """Get sample count column names by analysis type."""
    return SAMPLE_COUNT_COLUMNS.copy()


def get_fit_line_extension() -> Dict[str, Any]:
    """Get fit line extension configuration."""
    return FIT_LINE_EXTENSION.copy()


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_visualization_config():
    """
    Validate visualization configuration.

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate figure config
    if FIGURE_CONFIG["figure_size"][0] <= 0 or FIGURE_CONFIG["figure_size"][1] <= 0:
        raise ValueError("Figure size dimensions must be positive")

    if FIGURE_CONFIG["dpi"] <= 0:
        raise ValueError("DPI must be positive")

    # Validate analysis configs
    for analysis_type in ["pcac", "pion"]:
        if analysis_type not in ANALYSIS_CONFIG:
            raise ValueError(f"Missing analysis config for: {analysis_type}")

        cfg = ANALYSIS_CONFIG[analysis_type]
        required_keys = [
            "mass_column_mean",
            "mass_column_error",
            "mass_power",
            "mass_label",
        ]
        for key in required_keys:
            if key not in cfg:
                raise ValueError(
                    f"Missing required key '{key}' in {analysis_type} config"
                )
