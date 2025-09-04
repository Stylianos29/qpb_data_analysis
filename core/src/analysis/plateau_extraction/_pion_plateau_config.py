#!/usr/bin/env python3
"""
Configuration for pion effective mass plateau extraction.

This module contains pion-specific parameters for plateau extraction
from pion effective mass time series.
"""

from src.analysis.plateau_extraction._plateau_extraction_shared_config import (
    validate_shared_config,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Input dataset names from calculate_effective_mass.py
INPUT_DATASETS = {
    "samples": "pion_effective_mass_jackknife_samples",
    "mean": "pion_effective_mass_mean_values",
    "error": "pion_effective_mass_error_values",
}

# Time configuration
TIME_OFFSET = 1  # Effective mass starts at t=1
APPLY_SYMMETRIZATION = False  # Already symmetrized in calculate_effective_mass
SYMMETRIZATION_TRUNCATION = False

# Plateau search range (in array indices, not time values)
PLATEAU_SEARCH_RANGE = {
    "min_start": 3,  # Don't search before index 3
    "max_end": -1,  # Don't include last point
    "prefer_central": True,  # Prefer plateaus in central region
}

# Output file configuration
DEFAULT_OUTPUT_HDF5_FILENAME = "plateau_pion_mass_extraction.h5"
DEFAULT_OUTPUT_CSV_FILENAME = "plateau_pion_mass_estimates.csv"
OUTPUT_COLUMN_PREFIX = "pion"  # For column names like "pion_plateau_mean"


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_pion_config() -> bool:
    """Validate pion-specific configuration."""
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

    return True
