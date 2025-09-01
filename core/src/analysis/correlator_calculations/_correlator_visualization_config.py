#!/usr/bin/env python3
"""
Unified configuration for correlator analysis visualization script.

This module contains configuration parameters for visualizing both PCAC
mass and effective mass jackknife samples, including plot styling,
dataset specifications, and analysis-specific parameters.
"""

# Import analysis-specific configurations for time offsets and dataset
# info
from src.analysis.correlator_calculations._pcac_mass_config import (
    TRUNCATE_START as PCAC_TRUNCATE_START,
)
from src.analysis.correlator_calculations._effective_mass_config import (
    APPLY_SYMMETRIZATION,
    TRUNCATE_HALF,
)

# Common plot styling configuration
SAMPLE_PLOT_STYLE = {
    "marker": "o",
    "markersize": 10,  # Larger markers for better visibility
    "alpha": 0.8,
    "linestyle": "none",
    "label_suffix": " (Sample)",
}

AVERAGE_PLOT_STYLE = {
    "marker": "s",
    "markersize": 8,
    "alpha": 1.0,
    "capsize": 12,  # Error bar cap size
    "capthick": 2,  # Error bar cap thickness
    "elinewidth": 2,  # Error bar line width
    "label": "Jackknife average",
    "color": "red",  # Distinctive color for average
}

# Default plot appearance
DEFAULT_FIGURE_SIZE = (10, 7)
DEFAULT_FONT_SIZE = 14

# Multi-sample plotting configuration
SAMPLES_PER_PLOT = 8  # Number of jackknife samples to include in each plot

# Plot quality settings
PLOT_QUALITY = {
    "dpi": 300,
    "bbox_inches": "tight",
    "format": "png",
}

# Analysis-specific configurations
ANALYSIS_CONFIGS = {
    "pcac_mass": {
        "dataset_pattern": "PCAC_mass_{suffix}",
        "mean_dataset": "PCAC_mass_mean_values",
        "error_dataset": "PCAC_mass_error_values",
        "samples_dataset": "PCAC_mass_jackknife_samples",
        "plot_base_directory": "PCAC_mass_visualization",
        "time_offset": PCAC_TRUNCATE_START,  # PCAC starts at t=2 due to truncation
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
        "mean_dataset": "pion_effective_mass_mean_values",
        "error_dataset": "pion_effective_mass_error_values",
        "samples_dataset": "pion_effective_mass_jackknife_samples",
        "plot_base_directory": "effective_mass_visualization",
        "time_offset": 1,  # Effective mass starts at t=1
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

# Color cycling for sample plots
SAMPLE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def get_analysis_config(analysis_type):
    """Get configuration for specified analysis type."""
    if analysis_type not in ANALYSIS_CONFIGS:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    return ANALYSIS_CONFIGS[analysis_type]


def get_sample_color(sample_index):
    """Get consistent color for sample based on index."""
    return SAMPLE_COLORS[sample_index % len(SAMPLE_COLORS)]


def apply_dataset_slicing(
    time_index, samples_data, mean_values, error_values, plot_config
):
    """Apply dataset-specific slicing based on plot configuration."""
    start_idx = plot_config.get("x_start_index", 0)
    end_offset = plot_config.get("x_end_offset", 0)

    if end_offset > 0:
        end_idx = len(time_index) - end_offset
    else:
        end_idx = len(time_index)

    # Apply slicing
    sliced_time = time_index[start_idx:end_idx]
    sliced_samples = (
        samples_data[:, start_idx:end_idx]
        if samples_data.ndim > 1
        else samples_data[start_idx:end_idx]
    )
    sliced_mean = (
        mean_values[start_idx:end_idx] if mean_values.ndim > 0 else mean_values
    )
    sliced_error = (
        error_values[start_idx:end_idx] if error_values.ndim > 0 else error_values
    )

    return sliced_time, sliced_samples, sliced_mean, sliced_error


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

    # Check plot styling parameters
    if not isinstance(SAMPLES_PER_PLOT, int) or SAMPLES_PER_PLOT <= 0:
        raise ValueError(
            f"SAMPLES_PER_PLOT must be positive integer, got {SAMPLES_PER_PLOT}"
        )

    if len(SAMPLE_COLORS) < SAMPLES_PER_PLOT:
        raise ValueError(
            f"Need at least {SAMPLES_PER_PLOT} sample colors, got {len(SAMPLE_COLORS)}"
        )
