"""
Configuration for PCAC mass visualization script.

This module contains configuration parameters for visualizing PCAC mass
jackknife samples, including plot styling, dataset specifications, and
display parameters.

Place this file as:
qpb_data_analysis/core/src/analysis/_pcac_mass_visualization_config.py
"""

# PCAC mass dataset to visualize
PCAC_MASS_DATASET_TO_PLOT = "PCAC_mass_jackknife_samples"

# Base directory name for PCAC mass visualization plots
PCAC_MASS_PLOTS_BASE_DIRECTORY = "PCAC_mass_visualization"

# Plot styling configuration
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

# PCAC mass specific plotting configuration


PCAC_MASS_PLOT_CONFIG = {
    "y_scale": "linear",  # Use linear y-axis for PCAC mass
    "x_start_index": 2,  # Start from first PCAC mass element (index 0)
    "x_end_offset": 2,  # Include all time points up to the end
    "time_offset": 2,  # PCAC mass array index 0 corresponds to t/a=2
    # Time offset explanation: PCAC mass is calculated from
    # g4g5g5_derivative/g5g5_truncated where:
    #     - Original g5g5 correlator has 48 time points (t/a = 0, 1, 2, ..., 47)
    #     - g5g5 is truncated by removing first 2 and last 2 points → 44 points
    #     - So PCAC mass array index 0 corresponds to t/a=2, index 1 to t/a=3, etc.
    #     - time_offset=2 ensures proper time axis labeling
    #     - x_start_index and x_end_offset can further truncate for analysis
    "show_full_time_range": True,  # Always show x-axis from 0 for perspective
    "x_label": r"$t/a$",  # LaTeX x-axis label
    "y_label": r"a$m_{\text{PCAC}}$",  # LaTeX y-axis label for PCAC mass
    "description": "PCAC mass with linear scale, time offset +2, and full time range display",
    "title_prefix": "PCAC Mass",
    "grid_alpha": 0.4,  # Grid transparency
    "legend_location": "best",  # Legend position
    "show_zero_line": False,  # Show horizontal line at y=0
    "zero_line_style": {
        "color": "black",
        "linestyle": "--",
        "alpha": 0.5,
        "linewidth": 1,
    },
}

# Color palette for multiple samples (cycling through these colors)
SAMPLE_COLOR_PALETTE = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]

# Time range optimization settings
TIME_RANGE_SETTINGS = {
    "exclude_problematic_points": False,  # Set to True if certain time points are problematic
    "problematic_start": 0,  # Start index to exclude (if needed)
    "problematic_end": 0,  # End offset to exclude (if needed)
}

# Plot quality settings
PLOT_QUALITY = {
    "dpi": 150,  # Plot resolution
    "bbox_inches": "tight",  # Tight bounding box
    "facecolor": "white",  # Background color
    "edgecolor": "none",  # Edge color
}


def get_pcac_mass_plot_config() -> dict:
    """
    Get the plot configuration for PCAC mass dataset.

    Returns:
        Dictionary with plotting configuration parameters
    """
    return PCAC_MASS_PLOT_CONFIG.copy()


def get_sample_color(sample_index: int) -> str:
    """
    Get color for a specific sample index using cycling palette.

    Args:
        sample_index: Index of the sample

    Returns:
        Color string for the sample
    """
    return SAMPLE_COLOR_PALETTE[sample_index % len(SAMPLE_COLOR_PALETTE)]
