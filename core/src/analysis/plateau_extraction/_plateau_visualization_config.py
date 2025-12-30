#!/usr/bin/env python3
"""
Configuration for plateau extraction visualization.

This module provides comprehensive configuration for visualizing plateau 
extraction results from both PCAC mass and pion effective mass analyses.
Designed to match the high-quality output of the original 
extract_plateau_PCAC_mass.py visualization.
"""

from typing import Dict, Any, List

import numpy as np

from library.constants.labels import AXES_LABELS_BY_COLUMN_NAME

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_FONT_SIZE = 14
DEFAULT_DPI = 300

# =============================================================================
# Y-AXIS LABEL MAPPING
# =============================================================================
# Maps analysis types to canonical column names in the centralized
# AXES_LABELS_BY_COLUMN_NAME dictionary from library.constants.labels
#
# This approach maintains a single source of truth for all axis labels while
# allowing this module to specify which label to use for each analysis type.

Y_AXIS_LABEL_COLUMN_MAPPING = {
    "pcac_mass": "PCAC_plateau_mean",
    "pion_mass": "pion_effective_mass",
}

# =============================================================================
# ANALYSIS-SPECIFIC CONFIGURATIONS
# =============================================================================

ANALYSIS_CONFIGS = {
    "pcac_mass": {
        "input_datasets": {
            "time_series": "PCAC_time_series_samples",
            "plateau_estimates": "PCAC_plateau_estimates",
            "sigma_thresholds": "PCAC_individual_sigma_thresholds",
            "config_labels": "gauge_configuration_labels",
        },
        "time_offset": 2,  # PCAC mass starts at t=2
        "title_prefix": "PCAC Mass Plateau Extraction",
        "plot_subdirectory": "Plateau_extraction_pcac",
        "column_prefix": "PCAC",
        "description": "PCAC mass",
        # PCAC-mass-specific trimming settings
        "trimming": {
            "apply_trimming": True,
            "trim_start_points": 1,
            "trim_end_points": 2,
        },
    },
    "pion_mass": {
        "input_datasets": {
            "time_series": "pion_time_series_samples",
            "plateau_estimates": "pion_plateau_estimates",
            "sigma_thresholds": "pion_individual_sigma_thresholds",
            "config_labels": "gauge_configuration_labels",
        },
        "time_offset": 1,  # Effective mass starts at t=1
        "title_prefix": "Pion Effective Mass Plateau Extraction",
        "plot_subdirectory": "Plateau_extraction_pion",
        "column_prefix": "pion",
        "description": "pion effective mass",
        # Pion-mass-specific trimming settings
        "trimming": {
            "apply_trimming": True,
            "trim_start_points": 3,
            "trim_end_points": 1,
        },
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
]

# =============================================================================
# PLOT DIRECTORY CONFIGURATION
# =============================================================================

PLOT_DIRECTORY_CONFIG = {
    "parent_directory_name": "Plateau_extraction",
    "use_parent_directory": True,  # Toggle between hierarchical/flat structure
    "subdirectory_name_pcac_mass": "PCAC_mass",
    "subdirectory_name_pion_mass": "Pion_mass",
}

# =============================================================================
# PLOT LAYOUT CONFIGURATION
# =============================================================================

LAYOUT_CONFIG = {
    "samples_per_figure": 5,  # Maximum panels per figure
    "figure_size": (15, 12),  # Figure dimensions (width, height)
    "subplot_spacing": {
        "hspace": 0.4,  # Horizontal spacing between rows
        "wspace": 0.3,  # Vertical spacing between columns
        "top": 0.92,  # Top margin (leave room for title)
        "bottom": 0.08,  # Bottom margin
        "left": 0.08,  # Left margin
        "right": 0.92,  # Right margin
    },
    "panel_arrangement": "single_column",  # "single_column", "auto", or "grid"
    "create_group_subdirectories": True,  # Create subdirectory for each group
}

# =============================================================================
# PLOT STYLING CONFIGURATION
# =============================================================================

PLOT_STYLING = {
    # Time series data points
    "time_series": {
        "marker": "o",
        "markersize": 6,
        "linestyle": "none",  # No connecting lines
        "color": "blue",
        "alpha": 0.8,
        "markerfacecolor": "blue",
        "markeredgecolor": "blue",
        "markeredgewidth": 0.5,
    },
    # Plateau fitting region (green shaded area)
    "plateau_region": {
        "facecolor": "lightgreen",
        "alpha": 0.3,
        "edgecolor": "green",
        "linewidth": 1,
        "linestyle": "-",
    },
    # Plateau fit line (red dashed horizontal line)
    "plateau_fit": {
        "color": "red",
        "linestyle": "--",
        "linewidth": 2,
        "alpha": 0.9,
        "label": "Plateau fit",
    },
    # Axes styling
    "axes": {
        "grid": True,
        "grid_alpha": 0.3,
        "grid_color": "gray",
        "grid_linestyle": "-",
        "grid_linewidth": 0.5,
        "spines_color": "black",
        "spines_linewidth": 1.0,
    },
    # Font configurations
    "fonts": {
        "title_size": DEFAULT_FONT_SIZE + 2,
        "axis_label_size": DEFAULT_FONT_SIZE,
        "tick_label_size": DEFAULT_FONT_SIZE - 2,
        "legend_size": DEFAULT_FONT_SIZE - 2,
        "annotation_size": DEFAULT_FONT_SIZE - 1,
        "family": "serif",  # or "sans-serif"
    },
}

# =============================================================================
# TEXT ANNOTATIONS CONFIGURATION
# =============================================================================

ANNOTATION_CONFIG = {
    # Plateau information text box
    "plateau_info_box": {
        "template": (
            r"$m_{{{mass_type}}}^{{\mathrm{{plateau}}}}$ = {plateau_mean:.{precision}f} ± {plateau_error:.{precision}f}"
            "\n$\\sigma$ threshold = {sigma_threshold:.1f}"
            "\nFit points: {n_fit_points}"
        ),
        "position": {
            "x": 0.02,  # Relative position (0-1)
            "y": 0.10,  # Relative position from bottom
            "transform": "axes",  # Use axes coordinates
        },
        "bbox_props": {
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "gray",
            "linewidth": 0.5,
        },
        "font_props": {
            "size": PLOT_STYLING["fonts"]["annotation_size"],
            "family": PLOT_STYLING["fonts"]["family"],
        },
        "precision": 4,  # Decimal places for plateau value
    },
    # Configuration label (if needed)
    "config_label": {
        "template": "Sample: {config_label}",
        "position": {
            "x": 0.98,
            "y": 0.95,
            "transform": "axes",
        },
        "font_props": {
            "size": PLOT_STYLING["fonts"]["annotation_size"],
            "horizontalalignment": "right",
            "verticalalignment": "top",
        },
        "show": True,  # Whether to show config labels
    },
}

# =============================================================================
# AXES AND LABELING CONFIGURATION
# =============================================================================

AXES_CONFIG = {
    # X-axis configuration
    "x_axis": {
        "label_key": "time_index",  # Key for AXES_LABELS_BY_COLUMN_NAME
        "show_label_on": "bottom_only",  # "all", "bottom_only", "none"
        "tick_format": "integer",  # Force integer ticks
        "limits": "auto",  # "auto", "tight", or (min, max) tuple
        "margin": 0.02,  # Margin as fraction of range
    },
    # Y-axis configuration
    "y_axis": {
        "show_label_on": "all",  # "all", "left_only", "none"
        "tick_format": "auto",
        "limits": "auto",
        "margin": 0.05,  # Margin as fraction of range
        "scientific_notation": False,
    },
}

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

OUTPUT_CONFIG = {
    "file_format": "png",
    "dpi": DEFAULT_DPI,
    "bbox_inches": "tight",
    "pad_inches": 0.1,
    "facecolor": "white",
    "transparent": False,
    # Filename pattern: groupname_startidx_endidx.png (3-digit zero-padded indices)
    "filename_template": "{group_name}_{start_idx:03d}_{end_idx:03d}.{format}",
    # File organization
    "create_subdirectories": True,
    "clear_existing": False,  # Whether to clear existing plots
}

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================

DATA_CONFIG = {
    # How many samples to show per figure
    "max_samples_per_figure": LAYOUT_CONFIG["samples_per_figure"],
    # Data filtering/display options
    "show_all_time_points": True,  # Show full time range
    "hide_edge_points": False,  # Option to hide first/last points
    "time_range_restriction": None,  # (start, end) or None for full range
    # Sample selection
    "max_samples_total": None,  # Limit total samples (None for all)
    "sample_selection_strategy": "all",  # "all", "random", "first_n"
}

# =============================================================================
# ERROR HANDLING CONFIGURATION
# =============================================================================

ERROR_HANDLING_CONFIG = {
    "skip_missing_data": True,
    "log_skipped_samples": True,
    "continue_on_plot_error": True,
    "min_samples_per_figure": 1,
    "handle_infinite_values": "skip",  # "skip", "replace", "error"
    "handle_nan_values": "skip",
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_analysis_type(analysis_type: str) -> None:
    """Validate that analysis type is supported."""
    if analysis_type not in ANALYSIS_CONFIGS:
        valid_types = list(ANALYSIS_CONFIGS.keys())
        raise ValueError(
            f"Invalid analysis_type '{analysis_type}'. Must be one of: {valid_types}"
        )


def validate_visualization_config() -> None:
    """Validate all configuration parameters."""
    # Check layout config
    if LAYOUT_CONFIG["samples_per_figure"] < 1:
        raise ValueError("samples_per_figure must be at least 1")

    if len(LAYOUT_CONFIG["figure_size"]) != 2:
        raise ValueError("figure_size must be (width, height)")

    # Check analysis configs
    for analysis_type, config in ANALYSIS_CONFIGS.items():
        required_keys = ["input_datasets", "time_offset", "y_label", "title_prefix"]
        for key in required_keys:
            if key not in config:
                raise ValueError(
                    f"Missing required key '{key}' in {analysis_type} config"
                )

    # Check output config
    valid_formats = ["png", "pdf", "svg", "jpg", "tiff"]
    if OUTPUT_CONFIG["file_format"] not in valid_formats:
        raise ValueError(f"file_format must be one of: {valid_formats}")

    if OUTPUT_CONFIG["dpi"] < 50:
        raise ValueError("dpi must be at least 50")

    # Validate that all mapped column names exist in centralized
    # dictionary
    for analysis_type, column_name in Y_AXIS_LABEL_COLUMN_MAPPING.items():
        if column_name not in AXES_LABELS_BY_COLUMN_NAME:
            raise ValueError(
                f"Y-axis label mapping error: Column name '{column_name}' "
                f"(for analysis type '{analysis_type}') not found in "
                f"AXES_LABELS_BY_COLUMN_NAME. Please add it to "
                f"library/constants/labels.py"
            )


# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================


def get_analysis_config(analysis_type: str) -> Dict[str, Any]:
    """
    Get configuration for specific analysis type. Dynamically adds
    y-axis label from centralized dictionary.

    Args:
        analysis_type: "pcac_mass" or "pion_mass"

    Returns:
        Dictionary with analysis-specific settings including dynamically
        retrieved y-axis label from centralized
        AXES_LABELS_BY_COLUMN_NAME
    """
    validate_analysis_type(analysis_type)
    config = ANALYSIS_CONFIGS[analysis_type].copy()

    # Add y-axis label dynamically from centralized dictionary
    column_name = Y_AXIS_LABEL_COLUMN_MAPPING[analysis_type]
    config["y_label"] = AXES_LABELS_BY_COLUMN_NAME.get(column_name, column_name)

    return config


def get_layout_config() -> Dict[str, Any]:
    """Get plot layout configuration."""
    return LAYOUT_CONFIG.copy()


def get_plot_styling() -> Dict[str, Any]:
    """Get plot styling configuration."""
    return PLOT_STYLING.copy()


def get_annotation_config() -> Dict[str, Any]:
    """Get text annotation configuration."""
    return ANNOTATION_CONFIG.copy()


def get_axes_config() -> Dict[str, Any]:
    """Get axes configuration."""
    return AXES_CONFIG.copy()


def get_output_config() -> Dict[str, Any]:
    """Get output configuration."""
    return OUTPUT_CONFIG.copy()


def get_data_config() -> Dict[str, Any]:
    """Get data processing configuration."""
    return DATA_CONFIG.copy()


def get_error_handling_config() -> Dict[str, Any]:
    """Get error handling configuration."""
    return ERROR_HANDLING_CONFIG.copy()


def get_plot_subdirectory_name(analysis_type: str) -> tuple:
    """
    Get subdirectory name(s) for plots.

    Args:
        analysis_type: Type of analysis ("pcac_mass" or "pion_mass")

    Returns:
        Tuple of (parent_name, subdir_name). If use_parent_directory is
        False, parent_name will be None

    Examples:
        Hierarchical mode (use_parent_directory=True):
            ("Plateau_extraction", "from_PCAC_mass")

        Flat mode (use_parent_directory=False):
            (None, "Plateau_extraction_pcac")
    """
    validate_analysis_type(analysis_type)
    config = PLOT_DIRECTORY_CONFIG

    if config["use_parent_directory"]:
        parent = config["parent_directory_name"]
        subdir = config[f"subdirectory_name_{analysis_type}"]
        return (parent, subdir)
    else:
        # Backward compatibility: flat structure
        parent = None
        # Convert pcac_mass -> pcac, pion_mass -> pion for backward compatibility
        short_type = analysis_type.replace("_mass", "")
        subdir = f"{config['parent_directory_name']}_{short_type}"
        return (parent, subdir)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_mass_type_for_template(analysis_type: str) -> str:
    """Get mass type string for annotation templates."""
    if analysis_type == "pcac_mass":
        return "PCAC"
    elif analysis_type == "pion_mass":
        return "\\pi"
    else:
        return "mass"


def calculate_panel_layout(n_samples: int, max_per_figure: int) -> List[int]:
    """Calculate how to distribute samples across figures."""
    n_figures = (n_samples + max_per_figure - 1) // max_per_figure
    samples_per_figure = []

    remaining = n_samples
    for i in range(n_figures):
        samples_this_figure = min(max_per_figure, remaining)
        samples_per_figure.append(samples_this_figure)
        remaining -= samples_this_figure

    return samples_per_figure


def get_subplot_grid_size(n_panels: int, arrangement: str = "single_column") -> tuple:
    """Calculate optimal subplot grid (rows, cols) for n_panels."""
    if arrangement == "single_column":
        return (n_panels, 1)  # Always single column
    elif arrangement == "grid":
        # Original grid logic for backward compatibility
        if n_panels <= 4:
            return (n_panels, 1)  # Single column for few panels
        elif n_panels <= 6:
            return (3, 2)  # 3x2 grid
        elif n_panels <= 9:
            return (3, 3)  # 3x3 grid
        elif n_panels <= 12:
            return (4, 3)  # 4x3 grid
        else:
            # For more panels, use ceiling of sqrt
            cols = int(np.ceil(np.sqrt(n_panels)))
            rows = int(np.ceil(n_panels / cols))
            return (rows, cols)
    else:  # "auto" - intelligent choice
        if n_panels <= 6:
            return (n_panels, 1)  # Single column for reasonable numbers
        else:
            # Use grid for many panels to avoid very tall figures
            cols = min(3, int(np.ceil(np.sqrt(n_panels))))
            rows = int(np.ceil(n_panels / cols))
            return (rows, cols)
