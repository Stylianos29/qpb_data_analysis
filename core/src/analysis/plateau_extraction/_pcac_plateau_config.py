#!/usr/bin/env python3
"""
Configuration for PCAC mass plateau extraction.

This module contains PCAC-specific parameters for plateau extraction
from PCAC mass time series.
"""

from src.analysis.plateau_extraction._plateau_extraction_shared_config import (
    validate_shared_config,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Input dataset names from calculate_PCAC_mass.py
INPUT_DATASETS = {
    "samples": "PCAC_mass_jackknife_samples",
    "mean": "PCAC_mass_mean_values",
    "error": "PCAC_mass_error_values",
}

# Time configuration
TIME_OFFSET = 2  # PCAC mass starts at t=2
APPLY_SYMMETRIZATION = True  # Whether to symmetrize before plateau extraction
SYMMETRIZATION_TRUNCATION = True  # Truncate to T/2 after symmetrization

# Plateau search range (in array indices, not time values)
PLATEAU_SEARCH_RANGE = {
    "min_start": 2,  # Don't search before index 2
    "max_end": -2,  # Don't include last 2 points
    "prefer_central": True,  # Prefer plateaus in central region
}

# Output file configuration
DEFAULT_OUTPUT_HDF5_FILENAME = "plateau_PCAC_mass_extraction.h5"
DEFAULT_OUTPUT_CSV_FILENAME = "plateau_PCAC_mass_estimates.csv"
OUTPUT_COLUMN_PREFIX = "PCAC"  # For column names like "PCAC_plateau_mean"


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_pcac_config():
    """Validate PCAC-specific configuration."""
    # First validate shared config
    validate_shared_config()

    # Check time offset
    if TIME_OFFSET < 0:
        raise ValueError("TIME_OFFSET must be non-negative")

    # Check plateau search range
    if PLATEAU_SEARCH_RANGE["min_start"] < 0:
        raise ValueError("min_start must be non-negative")

    if not OUTPUT_COLUMN_PREFIX:
        raise ValueError("OUTPUT_COLUMN_PREFIX cannot be empty")
