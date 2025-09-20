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

# Optional columns for metadata preservation
METADATA_COLUMNS = [
    "PCAC_plateau_start_time",
    "PCAC_plateau_end_time",
    "PCAC_plateau_n_points",
    "PCAC_n_successful_samples",
    "PCAC_n_total_samples",
    "PCAC_n_failed_samples",
    "PCAC_estimation_method",
    "PCAC_sigma_threshold_used",
]

# Output file configuration
OUTPUT_FILENAME = "critical_bare_mass_from_pcac.csv"
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


def get_metadata_columns():
    """Get list of metadata columns to preserve."""
    return METADATA_COLUMNS.copy()


def get_output_filename():
    """Get default output filename for PCAC results."""
    return OUTPUT_FILENAME


def get_pcac_filters():
    """Get PCAC-specific filtering parameters."""
    return PCAC_SPECIFIC_FILTERS.copy()


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

    # Check output filename
    if not OUTPUT_FILENAME or not OUTPUT_FILENAME.endswith(".csv"):
        raise ValueError("OUTPUT_FILENAME must be a non-empty CSV filename")

    # Check PCAC-specific filters
    if PCAC_SPECIFIC_FILTERS["min_plateau_points"] < 2:
        raise ValueError("min_plateau_points must be at least 2")

    if not (0 < PCAC_SPECIFIC_FILTERS["max_relative_error"] <= 1):
        raise ValueError("max_relative_error must be between 0 and 1")
