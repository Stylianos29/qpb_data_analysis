"""
Private configuration file for jackknife visualization script.

This module contains the list of 2D jackknife datasets that should be
visualized by the jackknife visualization script.
"""

# List of 2D jackknife datasets to visualize
# These correspond to the exact dataset names in the HDF5 file
JACKKNIFE_DATASETS_TO_PLOT = [
    "g4g5g5_derivative_jackknife_samples",
    "g4g5g5_jackknife_samples",
    "g5g5_jackknife_samples",
]

# Base directory name for jackknife visualization plots
JACKKNIFE_PLOTS_BASE_DIRECTORY = "Jackknife_samples_visualization"

# Plot styling configuration
SAMPLE_PLOT_STYLE = {
    "marker": "o",
    "markersize": 8,  # Increase from 4 to 8
    "alpha": 1.0,  # Increase from 0.7 to 1.0 (fully opaque)
    "linestyle": "none",
    "label_suffix": " (Sample)",
}

AVERAGE_PLOT_STYLE = {
    "marker": "s",
    "markersize": 6,
    "alpha": 0.9,
    "capsize": 10,  # Increase from 3 to 10
    "capthick": 2,  # Add thickness
    "label": "Jackknife average",
}

# Default plot appearance
# DEFAULT_FIGURE_SIZE = (10, 6)
DEFAULT_FIGURE_SIZE = (8, 6)
DEFAULT_FONT_SIZE = 12

# Multi-sample plotting configuration
SAMPLES_PER_PLOT = 10  # Number of jackknife samples to include in each plot

# Dataset-specific plotting configurations
DATASET_PLOT_CONFIGS = {
    "g5g5_jackknife_samples": {
        "y_scale": "log",  # Use logarithmic y-axis
        "x_start_index": 1,  # Start from time index 1 (skip t=0)
        "x_end_offset": 0,  # Include all points up to the end
        "x_label": r"$t/a$",  # LaTeX x-axis label
        "y_label": r"$C_{\gamma_5\text{-}\gamma_5}(t)$",  # LaTeX y-axis label
        "description": "g5-g5 correlator with log scale starting from t=1",
    },
    "g4g5g5_jackknife_samples": {
        "y_scale": "linear",  # Use linear y-axis
        "x_start_index": 2,  # Start from time index 2
        "x_end_offset": 2,  # Exclude last 2 time points
        "x_label": r"$t/a$",  # LaTeX x-axis label
        "y_label": r"$C_{\gamma_4\gamma_5\text{-}\gamma_5}(t)$",  # LaTeX y-axis label
        "description": "g4g5-g5 correlator with reduced time range",
    },
    "g4g5g5_derivative_jackknife_samples": {
        "y_scale": "linear",  # Use linear y-axis
        "x_start_index": 2,  # Start from time index 2
        "x_end_offset": 2,  # Exclude last 2 time points
        "x_label": r"$t/a$",  # LaTeX x-axis label
        "y_label": r"$\partial_t C_{\gamma_4\gamma_5\text{-}\gamma_5}(t)$",  # LaTeX y-axis label
        "description": "g4g5-g5 derivative correlator with reduced time range",
    },
}

# Default configuration for datasets not explicitly specified
DEFAULT_DATASET_PLOT_CONFIG = {
    "y_scale": "linear",
    "x_start_index": 0,
    "x_end_offset": 0,
    "x_label": r"$t/a$",  # Default LaTeX x-axis label
    "y_label": "Correlator Value",  # Default y-axis label
    "description": "Default linear scale with full time range",
}


def get_dataset_plot_config(dataset_name: str) -> dict:
    """
    Get the plot configuration for a specific dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dictionary with plotting configuration parameters
    """
    return DATASET_PLOT_CONFIGS.get(dataset_name, DEFAULT_DATASET_PLOT_CONFIG)
