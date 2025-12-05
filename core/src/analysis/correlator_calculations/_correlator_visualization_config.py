#!/usr/bin/env python3
"""
Unified configuration for correlator analysis visualization script.

This module contains configuration parameters for visualizing both PCAC
mass and effective mass jackknife samples, including plot styling,
dataset specifications, and analysis-specific parameters.
"""

from typing import Dict

# Import analysis-specific configurations
from src.analysis.correlator_calculations._pcac_mass_config import (
    TRUNCATE_START as PCAC_TRUNCATE_START,
    OUTPUT_DATASETS as PCAC_OUTPUT_DATASETS,
)
from src.analysis.correlator_calculations._effective_mass_config import (
    OUTPUT_DATASETS as EFFECTIVE_OUTPUT_DATASETS,
)

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_FONT_SIZE = 14

# Plot styling configuration
PLOT_STYLING = {
    "samples": {
        "marker_size": 10,
        "alpha": 0.7,
        "linestyle": "none",
        "zorder": 10,  # Higher z-order puts samples on top
    },
    "average": {
        "legend_label": "Jackknife Average",
        "marker_size": 8,
        "alpha": 1.0,
        "color": "red",
        "marker": "s",
        "capsize": 8,
        "capthick": 2,
        "elinewidth": 2,
        "zorder": 5,  # Lower z-order puts it behind samples
    },
    "title": {
        "wrapping_length": 120,
        "font_size_offset": 2,  # +2 from DEFAULT_FONT_SIZE
        "leading_substring": "",
    },
    "legend": {
        "title": "Samples {sample_range[0]} to {sample_range[1]} out of {total_samples}",
        "location": "best",  # "best", "upper right", "center left", etc.
        "font_size_offset": -1,  # -1 from DEFAULT_FONT_SIZE
    },
    "grid": {
        "enabled": True,
        "alpha": 0.8,
    },
    "layout": {
        "default_figure_size": (12, 8),
        "title_wrap_length": 100,
    },
    "output": {
        "dpi": 300,
        "bbox_inches": "tight",
        "format": "png",
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

# Analysis-specific configurations
ANALYSIS_CONFIGS = {
    "pcac_mass": {
        "dataset_pattern": "PCAC_mass_{suffix}",
        "samples_dataset": PCAC_OUTPUT_DATASETS["samples"],
        "mean_dataset": PCAC_OUTPUT_DATASETS["mean"],
        "error_dataset": PCAC_OUTPUT_DATASETS["error"],
        "plot_base_directory": "PCAC_mass_visualization",
        "time_offset": PCAC_TRUNCATE_START,  # PCAC starts at t=2 due to truncation
        "time_range": {
            "min": 5,  # Positive integer, or None
            "max": -2,  # If negative: offset from max, if positive: absolute value
        },
        "plot_config": {
            "y_scale": "linear",
            "x_label": "Time slice (t/a)",
            "y_label": "PCAC Mass",
            "x_start_index": 0,  # Start from first array element
            "x_end_offset": 0,  # Include all time points up to the end
            "show_zero_line": False,
            "show_full_time_range": False,
        },
    },
    "effective_mass": {
        "dataset_pattern": "pion_effective_mass_{suffix}",
        "samples_dataset": EFFECTIVE_OUTPUT_DATASETS["samples"],
        "mean_dataset": EFFECTIVE_OUTPUT_DATASETS["mean"],
        "error_dataset": EFFECTIVE_OUTPUT_DATASETS["error"],
        "plot_base_directory": "Effective_mass_visualization",
        "time_offset": 1,  # Effective mass starts at t=1
        "time_range": {
            "min": 6,  # Positive integer, or None
            "max": None,  # If negative: offset from max, if positive: absolute value
        },
        "plot_config": {
            "y_scale": "linear",
            "x_label": "Time slice (t/a)",
            "y_label": "Effective Mass",
            "x_start_index": 0,  # Start from first array element
            "x_end_offset": 0,  # Include all time points up to the end
            "show_zero_line": False,
            "show_full_time_range": False,
        },
    },
}

# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================


def get_analysis_config(analysis_type):
    """Get configuration for specified analysis type."""
    if analysis_type not in ANALYSIS_CONFIGS:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    return ANALYSIS_CONFIGS[analysis_type]


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def _validate_time_range(time_range_config: Dict) -> None:
    """Validate time_range configuration values."""
    if not time_range_config:
        return

    t_min = time_range_config.get("min")
    t_max = time_range_config.get("max")

    # Check min is integer or None
    if t_min is not None and not isinstance(t_min, int):
        raise ValueError(
            f"time_range.min must be integer or None, got {type(t_min).__name__}"
        )

    # Check max is integer or None
    if t_max is not None and not isinstance(t_max, int):
        raise ValueError(
            f"time_range.max must be integer or None, got {type(t_max).__name__}"
        )

    # Check min is not negative
    if t_min is not None and t_min < 0:
        raise ValueError(f"time_range.min cannot be negative, got {t_min}")


def validate_visualization_config():
    """Validate visualization configuration."""
    # Check that all analysis types have required keys
    required_keys = [
        "dataset_pattern",
        "mean_dataset",
        "error_dataset",
        "samples_dataset",
        "plot_base_directory",
        "time_offset",
        "plot_config",
    ]

    for analysis_type, config in ANALYSIS_CONFIGS.items():
        for key in required_keys:
            if key not in config:
                raise ValueError(
                    f"Missing key '{key}' in {analysis_type} configuration"
                )

    # Validate time_range for each analysis type
    for analysis_type, analysis_config in ANALYSIS_CONFIGS.items():
        if "time_range" in analysis_config:
            _validate_time_range(analysis_config["time_range"])
