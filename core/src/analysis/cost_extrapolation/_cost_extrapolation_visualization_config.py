"""
Visualization configuration for computational cost extrapolation.

Defines plotting parameters, styling, and column mappings for creating
high-quality visualizations of cost extrapolation results.
"""

from typing import Dict, Any

from library.constants.labels import AXES_LABELS_BY_COLUMN_NAME


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
    "marker_size": 6,
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
    "mass_fit": {
        "loc": "lower right",
        "fontsize": 10,
        "framealpha": 0.9,
        "edgecolor": "gray",
        "fancybox": True,
    },
    "cost_fit": {
        "loc": "upper right",
        "fontsize": 10,
        "framealpha": 0.9,
        "edgecolor": "gray",
        "fancybox": True,
    },
}

# =============================================================================
# AXIS LABEL MAPPING AND STYLING
# =============================================================================
# Maps local analysis-specific keys to canonical column names in the
# centralized AXES_LABELS_BY_COLUMN_NAME dictionary from library.constants.labels

AXIS_LABEL_COLUMN_MAPPING = {
    "bare_mass": "Bare_mass",
    "pcac_mass": "PCAC_plateau_mean",
    "pion_mass_squared": "pion_mass_squared",
    "cost": "Average_core_hours_per_spinor_per_configuration",
}

# Axis styling configuration (separate from label content)
AXIS_STYLING = {
    "fontsize": 12,  # Default fontsize for all axis labels
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

# Parameters to exclude from plot titles
TITLE_EXCLUDED_PARAMETERS = [
    "APE_alpha",
    "Main_program_type",
    "Maximum_Lanczos_iterations",
    "Number_of_spinors",
    "Number_of_vectors",
    "CG_max_iterations",
    "MSCG_max_iterations",
    "MPI_geometry",
    "Threads_per_process",
    "Delta_Max",
    "Delta_Min",
]

# =============================================================================
# PLOT DIRECTORY CONFIGURATION
# =============================================================================

PLOT_DIRECTORY_CONFIG = {
    "parent_directory_name": "Cost_extrapolation",
    "use_parent_directory": True,  # Toggle between hierarchical/flat structure
    "subdirectory_name_pcac": "from_PCAC_mass",
    "subdirectory_name_pion": "from_Pion_mass",
}

# Plot subdirectories (DEPRECATED - kept for backward compatibility) Use
# get_plot_subdirectory_name() instead
PLOT_SUBDIRECTORIES = {
    "pcac": "Cost_extrapolation_pcac",
    "pion": "Cost_extrapolation_pion",
}

# Plot type sub-subdirectories (these remain within the analysis
# subdirectory)
PLOT_TYPE_SUBDIRECTORIES = {
    "mass_fit": "Plateau_mass_vs_Bare_mass",
    "cost_fit": "Computational_cost_vs_Bare_mass",
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


def get_legend_config(plot_type: str = "mass_fit") -> Dict[str, Any]:
    """
    Get legend configuration for specific plot type.

    Args:
        plot_type: "mass_fit" or "cost_fit"

    Returns:
        Dictionary with legend configuration
    """
    if plot_type not in LEGEND_CONFIG:
        raise ValueError(f"Unknown plot type: {plot_type}")
    return LEGEND_CONFIG[plot_type].copy()


def get_axis_labels() -> Dict[str, Dict[str, Any]]:
    """
    Get axis labels with styling information.

    Pulls label text from the centralized AXES_LABELS_BY_COLUMN_NAME and
    combines it with local styling configuration.

    Returns:
        Dictionary with structure: {
            "bare_mass": {"label": "$a m$", "fontsize": 12},
            "pcac_mass": {"label": "$am_{\\mathrm{PCAC}}$", "fontsize":
            12}, ...
        }
    """
    result = {}

    for local_key, column_name in AXIS_LABEL_COLUMN_MAPPING.items():
        # Get label from centralized dictionary
        label_text = AXES_LABELS_BY_COLUMN_NAME.get(column_name, column_name)

        # Combine with styling
        result[local_key] = {
            "label": label_text,
            "fontsize": AXIS_STYLING["fontsize"],
        }

    return result


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
    """
    Get plot subdirectory name for analysis type.

    DEPRECATED: Use get_plot_subdirectory_name() for hierarchical
    support. Kept for backward compatibility.
    """
    if analysis_type not in PLOT_SUBDIRECTORIES:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    return PLOT_SUBDIRECTORIES[analysis_type]


def get_plot_subdirectory_name(analysis_type: str) -> tuple:
    """
    Get subdirectory name(s) for plots.

    Args:
        analysis_type: Type of analysis ("pcac" or "pion")

    Returns:
        Tuple of (parent_name, subdir_name) If use_parent_directory is
        False, parent_name will be None

    Examples:
        Hierarchical mode (use_parent_directory=True):
            ("Cost_extrapolation", "from_PCAC_mass")

        Flat mode (use_parent_directory=False):
            (None, "cost_extrapolation_pcac")
    """
    if analysis_type not in ["pcac", "pion"]:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    config = PLOT_DIRECTORY_CONFIG

    if config["use_parent_directory"]:
        parent = config["parent_directory_name"]
        subdir = config[f"subdirectory_name_{analysis_type}"]
        return (parent, subdir)
    else:
        # Backward compatibility: flat structure
        parent = None
        subdir = PLOT_SUBDIRECTORIES[analysis_type]
        return (parent, subdir)


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
