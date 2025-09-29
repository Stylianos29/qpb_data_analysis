#!/usr/bin/env python3
"""
Configuration for Pion critical mass calculation.

This module contains Pion-specific parameters for critical mass
calculation from Pion plateau mass estimates.
"""

from src.analysis.critical_mass_extrapolation._critical_mass_shared_config import (
    validate_shared_config,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Required input CSV columns
REQUIRED_COLUMNS = [
    "Bare_mass",
    "pion_plateau_mean",
    "pion_plateau_error",
]

# Column mapping for flexibility
COLUMN_MAPPING = {
    "bare_mass": "Bare_mass",
    "plateau_mean": "pion_plateau_mean",
    "plateau_error": "pion_plateau_error",
}

PLATEAU_MASS_POWER = 2  # Fit pion_mass^2 vs bare_mass

# Quadratic fitting configuration
QUADRATIC_FIT_CONFIG = {
    "enable_quadratic_fit": True,  # Enable quadratic fit for validation
    "quadratic_coefficient_scale": 0.1,  # Scale factor for initial guess: |slope|/range * scale
}

# Fitting range configuration
FIT_RANGE_CONFIG = {
    "linear": {
        "bare_mass_min": 0,  # None = use actual data minimum
        "bare_mass_max": None,  # None = use actual data maximum
    },
    "quadratic": {
        "bare_mass_min": 0,  # None = use actual data minimum
        "bare_mass_max": None,  # None = use actual data maximum
    },
}

# Output file configuration
DEFAULT_OUTPUT_FILENAME = "critical_bare_mass_from_pion.csv"
OUTPUT_COLUMN_PREFIX = "pion"

# Pion-specific filtering (if any)
PION_SPECIFIC_FILTERS = {
    "min_plateau_points": 3,  # Minimum plateau size
    "max_relative_error": 0.5,  # Maximum relative error for plateau estimates
    "require_positive_mass": True,  # Pion effective mass should be positive
}


# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================


def get_required_columns():
    """Get list of required CSV columns for Pion analysis."""
    return REQUIRED_COLUMNS.copy()


def get_pion_filters():
    """Get Pion-specific filtering parameters."""
    return PION_SPECIFIC_FILTERS.copy()


def get_quadratic_fit_config():
    """Get quadratic fitting configuration for pion analysis."""
    return QUADRATIC_FIT_CONFIG.copy()


def get_fit_range_config():
    """Get fitting range configuration for pion analysis."""
    return FIT_RANGE_CONFIG.copy()


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_pion_critical_config():
    """Validate Pion-specific critical mass configuration."""
    # First validate shared config
    validate_shared_config()

    # Check required columns
    if not isinstance(REQUIRED_COLUMNS, list):
        raise ValueError("REQUIRED_COLUMNS must be a list")

    if len(REQUIRED_COLUMNS) < 3:
        raise ValueError("REQUIRED_COLUMNS must contain at least 3 columns")

    # Check for essential columns
    has_mass_col = any("mass" in col.lower() for col in REQUIRED_COLUMNS)
    has_mean_col = any("mean" in col.lower() for col in REQUIRED_COLUMNS)
    has_error_col = any("error" in col.lower() for col in REQUIRED_COLUMNS)

    if not (has_mass_col and has_mean_col and has_error_col):
        raise ValueError("REQUIRED_COLUMNS must include mass, mean, and error columns")

    # Validate quadratic fit config
    if not (0 < QUADRATIC_FIT_CONFIG["quadratic_coefficient_scale"] <= 1.0):
        raise ValueError("quadratic_coefficient_scale must be between 0 and 1")

    if not isinstance(QUADRATIC_FIT_CONFIG["enable_quadratic_fit"], bool):
        raise ValueError("enable_quadratic_fit must be boolean")

    # Validate fit range config
    for fit_type in ["linear", "quadratic"]:
        range_min = FIT_RANGE_CONFIG[fit_type]["bare_mass_min"]
        range_max = FIT_RANGE_CONFIG[fit_type]["bare_mass_max"]

        if range_min is not None and range_max is not None:
            if range_min >= range_max:
                raise ValueError(
                    f"{fit_type} fit: bare_mass_min ({range_min}) must be "
                    f"less than bare_mass_max ({range_max})"
                )

    # Check output filename
    if not DEFAULT_OUTPUT_FILENAME or not DEFAULT_OUTPUT_FILENAME.endswith(".csv"):
        raise ValueError("DEFAULT_OUTPUT_FILENAME must be a non-empty CSV filename")

    # Check Pion-specific filters
    if PION_SPECIFIC_FILTERS["min_plateau_points"] < 2:
        raise ValueError("min_plateau_points must be at least 2")

    if not (0 < PION_SPECIFIC_FILTERS["max_relative_error"] <= 1):
        raise ValueError("max_relative_error must be between 0 and 1")

    if not isinstance(PION_SPECIFIC_FILTERS["require_positive_mass"], bool):
        raise ValueError("require_positive_mass must be boolean")
