"""
Configuration for PCAC mass plateau extraction script.

This module contains all configuration parameters for extracting plateau PCAC mass
values from PCAC mass time series using jackknife analysis methods.

Place this file as: qpb_data_analysis/core/src/analysis/_plateau_extraction_config.py
"""

from typing import List, Dict, Any, Optional

# =============================================================================
# INPUT/OUTPUT CONFIGURATION
# =============================================================================

# Input dataset names from calculate_PCAC_mass.py output
INPUT_DATASETS = {
    "jackknife_samples": "PCAC_mass_jackknife_samples",
    "mean_values": "PCAC_mass_mean_values",
    "error_values": "PCAC_mass_error_values",
}

# Metadata datasets to read and include in output
METADATA_DATASETS = [
    "gauge_configuration_labels",
    "mpi_geometry_values",
    "qpb_log_filenames",
]

# Output CSV configuration
OUTPUT_CSV_CONFIG = {
    "default_filename": "plateau_PCAC_mass_estimates.csv",
    "float_precision": 6,  # Decimal places for floating point values
    "include_diagnostics": True,  # Include fit diagnostics in CSV
}

# =============================================================================
# PLATEAU DETECTION CONFIGURATION
# =============================================================================

# Sigma thresholds to try in sequence (max 5.0 as discussed)
PLATEAU_DETECTION_SIGMA_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# Minimum plateau size (number of consecutive points)
MIN_PLATEAU_SIZE = 5

# Plateau detection test method
# Options: 'weighted_range', 'chi_squared', 'range_based'
PLATEAU_DETECTION_METHOD = "weighted_range"

# Time range constraints for plateau search
PLATEAU_SEARCH_CONFIG = {
    "min_start_time": 2,  # Don't search for plateaus before t=2
    "max_end_time": -3,  # Don't include last 3 time points
    "prefer_central_region": True,  # Prefer plateaus in central time region
}

# =============================================================================
# PLATEAU ESTIMATION CONFIGURATION
# =============================================================================

# Plateau estimation method
# Options: 'simple', 'median', 'covariance_quadrature'
PLATEAU_ESTIMATION_METHOD = "covariance_quadrature"

# Weighting scheme for plateau points
USE_INVERSE_VARIANCE_WEIGHTING = True

# Alternative estimation methods to try if primary fails
FALLBACK_ESTIMATION_METHODS = [
    {"method": "simple", "use_inverse_variance": True},
    {"method": "median", "use_inverse_variance": False},
    {"method": "simple", "use_inverse_variance": False},
]

# =============================================================================
# ERROR HANDLING CONFIGURATION
# =============================================================================

# Minimum number of successful samples required per group for reliable jackknife
MIN_SAMPLE_SIZE = 3

# Maximum number of failed samples allowed per group (fraction of total)
MAX_FAILED_SAMPLE_FRACTION = 0.5  # Allow up to 50% failures

# Action when group fails completely
# Options: 'exclude' (don't include in CSV), 'include_nan' (include with NaN values)
FAILED_GROUP_ACTION = "exclude"

# Logging configuration for failed extractions
FAILURE_LOGGING = {
    "log_failed_samples": True,
    "log_failed_groups": True,
    "detailed_error_messages": True,
}

# =============================================================================
# PLOTTING CONFIGURATION
# =============================================================================

# Multi-panel plotting setup
PLOTTING_CONFIG = {
    "enabled": True,
    "samples_per_plot": 5,  # Number of samples in each multi-panel plot
    "figure_size": (12, 14),  # Width, height for 5-panel plot
    "subplot_spacing": 0.15,  # Vertical spacing between subplots
    "share_x_axis": True,  # Share x-axis across panels
    "show_individual_titles": False,  # Don't show titles on individual panels
}

# Data trimming for better visibility (like visualize_PCAC_mass.py)
PLOTTING_DATA_RANGE = {
    "trim_start_points": 1,  # Remove first N points from display
    "trim_end_points": 1,  # Remove last N points from display
    "apply_trimming": True,  # Enable/disable data trimming
}

# Individual subplot styling
SUBPLOT_STYLE = {
    "grid": True,
    "grid_alpha": 0.4,
    "font_size": 11,
    "marker_size": 8,
    "line_width": 2,
}

# Time series plotting style
TIME_SERIES_STYLE = {
    "marker": "o",
    "linestyle": "none",
    "color": "blue",
    "alpha": 0.8,
    "capsize": 4,  # Correct parameter name for matplotlib errorbar
}

# Plateau fit visualization style
PLATEAU_FIT_STYLE = {
    "line_color": "red",
    "line_style": "--",
    "line_width": 3,
    "band_color": "red",
    "band_alpha": 0.3,
    "extend_fraction": 0.1,  # Extend line beyond fit range by this fraction
}

# Plot labeling and formatting
PLOT_LABELS = {
    "x_label": r"$t/a$",
    "y_label": r"$am_{\text{PCAC}}$",  # Corrected: a times m_PCAC
    "legend_title_template": "Sample: {config_label}",
    "fit_info_template": r"$m_{{\text{{PCAC}}}}^{{\text{{plateau}}}} = {value:.4f} \pm {error:.4f}$",
}

# Plot output configuration
PLOT_OUTPUT = {
    "base_directory": "plateau_extraction_plots",
    "file_format": "png",
    "dpi": 150,
    "bbox_inches": "tight",
    "clear_existing": False,  # Whether to clear existing plots
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

# =============================================================================
# ADVANCED CONFIGURATION
# =============================================================================

# Quality control parameters
QUALITY_CONTROL = {
    "check_plateau_stability": True,  # Verify plateau doesn't drift
    "max_plateau_drift": 2.0,  # Max allowed drift in sigma units
    "require_positive_errors": True,  # Require positive error estimates
    "min_relative_error": 1e-6,  # Minimum relative error threshold
    "max_relative_error": 1.0,  # Maximum relative error threshold
}

# Performance optimization
PERFORMANCE_CONFIG = {
    "parallel_processing": False,  # Enable parallel group processing
    "chunk_size": 10,  # Groups to process per chunk
    "memory_efficient": True,  # Use memory-efficient algorithms
}

# Debugging and validation
DEBUG_CONFIG = {
    "save_intermediate_results": False,  # Save plateau detection intermediates
    "validate_input_data": True,  # Perform input data validation
    "verbose_fitting": False,  # Detailed fitting information
    "plot_failed_extractions": False,  # Create plots for failed extractions
}

# =============================================================================
# CONFIGURATION VALIDATION AND UTILITIES
# =============================================================================


def validate_config() -> bool:
    """
    Validate configuration parameters for consistency and correctness.

    Returns:
        True if configuration is valid, False otherwise
    """
    # Check sigma thresholds
    if not all(0 < sigma <= 5.0 for sigma in PLATEAU_DETECTION_SIGMA_THRESHOLDS):
        return False

    # Check minimum sample size
    if MIN_SAMPLE_SIZE < 2:
        return False

    # Check failed sample fraction
    if not 0 <= MAX_FAILED_SAMPLE_FRACTION <= 1:
        return False

    # Check plotting configuration
    if PLOTTING_CONFIG["samples_per_plot"] < 1:
        return False

    return True


def get_plateau_detection_config() -> Dict[str, Any]:
    """
    Get plateau detection configuration dictionary.

    Returns:
        Dictionary with plateau detection parameters
    """
    return {
        "sigma_thresholds": PLATEAU_DETECTION_SIGMA_THRESHOLDS,
        "min_plateau_size": MIN_PLATEAU_SIZE,
        "test_method": PLATEAU_DETECTION_METHOD,
        "search_config": PLATEAU_SEARCH_CONFIG,
    }


def get_plateau_estimation_config() -> Dict[str, Any]:
    """
    Get plateau estimation configuration dictionary.

    Returns:
        Dictionary with plateau estimation parameters
    """
    return {
        "method": PLATEAU_ESTIMATION_METHOD,
        "use_inverse_variance": USE_INVERSE_VARIANCE_WEIGHTING,
        "fallback_methods": FALLBACK_ESTIMATION_METHODS,
    }


def get_error_handling_config() -> Dict[str, Any]:
    """
    Get error handling configuration dictionary.

    Returns:
        Dictionary with error handling parameters
    """
    return {
        "min_sample_size": MIN_SAMPLE_SIZE,
        "max_failed_fraction": MAX_FAILED_SAMPLE_FRACTION,
        "failed_group_action": FAILED_GROUP_ACTION,
        "logging": FAILURE_LOGGING,
    }


def get_plotting_config() -> Dict[str, Any]:
    """
    Get complete plotting configuration dictionary.

    Returns:
        Dictionary with all plotting parameters
    """
    return {
        "config": PLOTTING_CONFIG,
        "data_range": PLOTTING_DATA_RANGE,
        "subplot_style": SUBPLOT_STYLE,
        "time_series_style": TIME_SERIES_STYLE,
        "plateau_fit_style": PLATEAU_FIT_STYLE,
        "labels": PLOT_LABELS,
        "output": PLOT_OUTPUT,
    }


def get_sample_color(sample_index: int) -> str:
    """
    Get color for a specific sample index using cycling palette.

    Args:
        sample_index: Index of the sample

    Returns:
        Color string for the sample
    """
    return SAMPLE_COLOR_PALETTE[sample_index % len(SAMPLE_COLOR_PALETTE)]


# Validate configuration on import
if not validate_config():
    raise ValueError("Invalid configuration parameters detected!")
