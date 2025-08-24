"""
Configuration constants for effective mass calculation.

This module defines the parameters used for calculating effective mass
values from jackknife-analyzed g5-g5 correlator data.

Place this file as:
qpb_data_analysis/core/src/analysis/_effective_mass_config.py
"""

# Symmetrization and truncation parameters
APPLY_SYMMETRIZATION = (
    True  # Apply symmetrization to g5-g5 correlators before calculation
)
TRUNCATE_HALF = True  # Truncate to half length (standard for periodic effective mass)
LOWERING_FACTOR = 0.99  # Factor for middle value in effective mass calculation

# Expected dataset lengths for validation
EXPECTED_G5G5_LENGTH = 48  # Original g5-g5 correlator length
EXPECTED_EFFECTIVE_MASS_LENGTH = (
    23  # Expected effective mass length after calculation (T/2 - 1)
)

# Required input datasets for effective mass calculation
REQUIRED_INPUT_DATASETS = [
    "g5g5_jackknife_samples",  # Only g5-g5 correlators needed
]

# Alternative input dataset names (for compatibility with different
# naming conventions)
ALTERNATIVE_INPUT_DATASETS = [
    "Jackknife_samples_of_g5_g5_correlator_2D_array",
    "g5_g5_jackknife_samples",
    "g5g5_correlator_jackknife_samples",
]

# Output dataset names
EFFECTIVE_MASS_DATASETS = {
    "jackknife_samples": "effective_mass_jackknife_samples",
    "mean_values": "effective_mass_mean_values",
    "error_values": "effective_mass_error_values",
}

# Metadata datasets to copy from input
METADATA_DATASETS = [
    "gauge_configuration_labels",
    "mpi_geometry_values",
    "qpb_log_filenames",
]

# Mathematical parameters for effective mass calculation
EFFECTIVE_MASS_CALCULATION_PARAMS = {
    "lowering_factor": LOWERING_FACTOR,
    "truncate_half": TRUNCATE_HALF,
    "remove_extreme_points": True,  # Remove first and last points before calculation
    "use_periodic_formula": True,  # Use two-state periodic effective mass formula
}

# Validation parameters
VALIDATION_PARAMS = {
    "check_non_zero": True,  # Verify correlators are non-zero
    "check_decreasing": True,  # Verify correlators decrease (for physical behavior)
    "min_correlation_value": 1e-15,  # Minimum allowed correlator value
    "max_correlation_ratio": 1e10,  # Maximum allowed ratio between consecutive points
}

# Error handling configuration
ERROR_HANDLING = {
    "handle_division_by_zero": True,  # Replace inf/nan with configurable values
    "handle_negative_sqrt": True,  # Handle negative values under square root
    "invalid_replacement_value": 0.0,  # Value to use for invalid calculations
    "log_invalid_calculations": True,  # Log when invalid calculations occur
}

# Plotting configuration (if plotting is enabled)
PLOTTING_CONFIG = {
    "default_plots_enabled": False,  # Enable plotting by default
    "plot_original_correlators": False,  # Plot original g5-g5 correlators
    "plot_effective_mass": True,  # Plot calculated effective mass
    "symmetrize_for_plotting": True,  # Apply symmetrization for plots
    "exclude_first_point_from_plots": True,  # Skip t=0 in plots for better visualization
}
