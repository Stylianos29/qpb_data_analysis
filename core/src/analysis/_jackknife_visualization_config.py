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
    "markersize": 4,
    "alpha": 0.7,
    "linestyle": "none",
    "label_suffix": " (Sample)",
}

AVERAGE_PLOT_STYLE = {
    "marker": "s",
    "markersize": 6,
    "alpha": 0.9,
    "capsize": 3,
    "label_suffix": " (Average ± Error)",
}

# Default plot appearance
DEFAULT_FIGURE_SIZE = (10, 6)
DEFAULT_FONT_SIZE = 12
