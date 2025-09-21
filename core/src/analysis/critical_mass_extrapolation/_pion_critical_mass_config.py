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

# Optional columns for metadata preservation
METADATA_COLUMNS = [
    "pion_plateau_start_time",
    "pion_plateau_end_time",
    "pion_n_successful_samples",
    "pion_n_total_samples",
    "pion_n_failed_samples",
    "pion_sigma_threshold_used",
]

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


def get_metadata_columns():
    """Get list of metadata columns to preserve."""
    return METADATA_COLUMNS.copy()


def get_pion_filters():
    """Get Pion-specific filtering parameters."""
    return PION_SPECIFIC_FILTERS.copy()


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

    # Check output filename
    if not DEFAULT_OUTPUT_FILENAME or not DEFAULT_OUTPUT_FILENAME.endswith(".csv"):
        raise ValueError("DEFAULT_OUTPUT_FILENAME must be a non-empty CSV filename")

    # Check Pion-specific filters
    if PION_SPECIFIC_FILTERS["min_plateau_points"] < 2:
        raise ValueError("min_plateau_points must be at least 2")

    if not (0 < PION_SPECIFIC_FILTERS["max_relative_error"] <= 1):
        raise ValueError("max_relative_error must be between 0 and 1")
