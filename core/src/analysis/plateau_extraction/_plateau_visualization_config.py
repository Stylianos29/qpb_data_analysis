#!/usr/bin/env python3
"""
Configuration for plateau extraction visualization.

This module provides configuration for visualizing plateau extraction
results from both PCAC mass and pion effective mass analyses.
"""

from typing import Dict, Any


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_FONT_SIZE = 14

# Plot styling configuration
PLOT_STYLING = {
    "figure": {
        "size": (15, 12),  # Multi-panel figure size
        "dpi": 100,
        "subplot_spacing": 0.4,
        "max_panels": 12,  # Maximum panels per figure
    },
    "time_series": {
        "marker": "o",
        "markersize": 6,
        "linestyle": "-",
        "linewidth": 1,
        "alpha": 0.7,
        "color": "blue",
    },
    "plateau_fit": {
        "color": "red",
        "linestyle": "--",
        "linewidth": 2,
        "alpha": 0.8,
        "fill_alpha": 0.2,  # For uncertainty band
    },
    "axes": {
        "grid": True,
        "grid_alpha": 0.3,
        "share_x": True,
    },
    "title": {
        "main_fontsize_offset": 2,  # Added to DEFAULT_FONT_SIZE
        "panel_fontsize_offset": 0,
        "wrapping_length": 80,
    },
    "legend": {
        "location": "best",
        "fontsize_offset": -2,
        "template": "Config: {config_label}",
    },
    "output": {
        "dpi": 300,
        "bbox_inches": "tight",
        "format": "png",
    },
}

# Analysis-specific configurations
ANALYSIS_CONFIGS = {
    "pcac_mass": {
        "input_datasets": {
            "samples": "PCAC_mass_jackknife_samples",
            "mean": "PCAC_mass_mean_values",
            "error": "PCAC_mass_error_values",
        },
        "time_offset": 2,  # PCAC mass starts at t=2
        "y_label": r"$am_{\mathrm{PCAC}}$",
        "x_label": "Time slice",
        "plot_subdirectory": "plateau_extraction_pcac",
        "title_prefix": "PCAC Mass Plateau Extraction",
    },
    "pion_mass": {
        "input_datasets": {
            "samples": "pion_effective_mass_jackknife_samples",
            "mean": "pion_effective_mass_mean_values",
            "error": "pion_effective_mass_error_values",
        },
        "time_offset": 1,  # Effective mass starts at t=1
        "y_label": r"$am_{\pi}^{\mathrm{eff}}$",
        "x_label": "Time slice",
        "plot_subdirectory": "plateau_extraction_pion",
        "title_prefix": "Pion Effective Mass Plateau Extraction",
    },
}

# Data processing configuration
DATA_PROCESSING = {
    "max_samples_per_figure": 12,  # Split into multiple figures if needed
    "apply_trimming": False,  # Whether to trim edge points for clarity
    "trim_start_points": 0,
    "trim_end_points": 0,
}


# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================


def get_analysis_config(analysis_type: str) -> Dict[str, Any]:
    """Get configuration for specific analysis type."""
    if analysis_type not in ANALYSIS_CONFIGS:
        raise ValueError(
            f"Unknown analysis type: {analysis_type}. "
            f"Must be one of {list(ANALYSIS_CONFIGS.keys())}"
        )
    return ANALYSIS_CONFIGS[analysis_type]


def get_plot_styling() -> Dict[str, Any]:
    """Get plot styling configuration."""
    return PLOT_STYLING


def get_data_processing_config() -> Dict[str, Any]:
    """Get data processing configuration."""
    return DATA_PROCESSING


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_visualization_config() -> bool:
    """Validate visualization configuration consistency."""
    # Check that all analysis configs have required keys
    required_keys = {
        "input_datasets",
        "time_offset",
        "y_label",
        "x_label",
        "plot_subdirectory",
        "title_prefix",
    }

    for analysis_type, config in ANALYSIS_CONFIGS.items():
        missing_keys = required_keys - set(config.keys())
        if missing_keys:
            raise ValueError(
                f"Analysis config '{analysis_type}' missing keys: {missing_keys}"
            )

    # Check that input datasets have required sub-keys
    dataset_keys = {"samples", "mean", "error"}
    for analysis_type, config in ANALYSIS_CONFIGS.items():
        missing_dataset_keys = dataset_keys - set(config["input_datasets"].keys())
        if missing_dataset_keys:
            raise ValueError(
                f"Analysis '{analysis_type}' missing dataset keys: {missing_dataset_keys}"
            )

    # Validate numeric ranges
    if DATA_PROCESSING["max_samples_per_figure"] < 1:
        raise ValueError("max_samples_per_figure must be at least 1")

    if PLOT_STYLING["figure"]["max_panels"] < 1:
        raise ValueError("max_panels must be at least 1")

    return True
