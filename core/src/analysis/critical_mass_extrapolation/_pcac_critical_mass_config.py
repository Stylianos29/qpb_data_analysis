#!/usr/bin/env python3
"""
Configuration for PCAC critical mass calculation.

This module contains PCAC-specific parameters for critical mass
calculation from PCAC plateau mass estimates.
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
    "PCAC_plateau_mean",
    "PCAC_plateau_error",
]

# Column mapping for flexibility
COLUMN_MAPPING = {
    "bare_mass": "Bare_mass",
    "plateau_mean": "PCAC_plateau_mean",
    "plateau_error": "PCAC_plateau_error",
}

PLATEAU_MASS_POWER = 1  # Fit PCAC_mass^1 vs bare_mass

# Quadratic fitting configuration
QUADRATIC_FIT_CONFIG = {
    "enable_quadratic_fit": True,  # Enable quadratic fit for validation
    "quadratic_coefficient_scale": 0.1,  # Scale factor for initial guess: |slope|/range * scale
}

# Output file configuration
DEFAULT_OUTPUT_FILENAME = "critical_bare_mass_from_pcac.csv"
OUTPUT_COLUMN_PREFIX = "pcac"

# PCAC-specific filtering (if any)
PCAC_SPECIFIC_FILTERS = {
    "min_plateau_points": 3,  # Minimum plateau size
    "max_relative_error": 0.5,  # Maximum relative error for plateau estimates
}


# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================


def get_required_columns():
    """Get list of required CSV columns for PCAC analysis."""
    return REQUIRED_COLUMNS.copy()


def get_pcac_filters():
    """Get PCAC-specific filtering parameters."""
    return PCAC_SPECIFIC_FILTERS.copy()


def get_quadratic_fit_config():
    """Get quadratic fitting configuration for PCAC analysis."""
    return QUADRATIC_FIT_CONFIG.copy()


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_pcac_critical_config():
    """Validate PCAC-specific critical mass configuration."""
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

    # Check output filename
    if not DEFAULT_OUTPUT_FILENAME or not DEFAULT_OUTPUT_FILENAME.endswith(".csv"):
        raise ValueError("DEFAULT_OUTPUT_FILENAME must be a non-empty CSV filename")

    # Check PCAC-specific filters
    if PCAC_SPECIFIC_FILTERS["min_plateau_points"] < 2:
        raise ValueError("min_plateau_points must be at least 2")

    if not (0 < PCAC_SPECIFIC_FILTERS["max_relative_error"] <= 1):
        raise ValueError("max_relative_error must be between 0 and 1")
