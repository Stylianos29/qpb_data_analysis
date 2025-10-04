"""
Pion-specific configuration for computational cost extrapolation.

Configuration for extrapolating computational costs using pion effective
mass as the intermediate physics observable. Typically uses quadratic
fit for m_π² vs bare mass relationship.
"""

from typing import List, Dict, Any

from src.analysis.cost_extrapolation._cost_extrapolation_shared_config import (
    validate_shared_cost_config,
)


# =============================================================================
# PION-SPECIFIC SETTINGS
# =============================================================================

# Analysis type identifier
ANALYSIS_TYPE = "pion"

# Default output filename
DEFAULT_OUTPUT_FILENAME = "computational_cost_extrapolation_from_pion.csv"

# Reference pion mass for extrapolation
REFERENCE_PION_MASS = 0.135  # Target pion mass value (in lattice units)

# Pion mass power for fitting (1 for m_π, 2 for m_π²)
PION_MASS_POWER = 2  # Fit m_π² vs bare mass (more linear relationship)

# Pion mass fitting configuration
PION_FIT_CONFIG = {
    "fit_function": "linear",  # m_π² = a * bare_mass + b (after squaring)
    "fit_method": "leastsq",
    "require_positive_mass": True,  # Pion effective mass must be positive
}


# =============================================================================
# INPUT COLUMN MAPPINGS
# =============================================================================

# Column name mapping (standard name -> CSV column name) All columns
# listed here are considered required
COLUMN_MAPPING = {
    "bare_mass": "Bare_mass",
    "mass_mean": "pion_plateau_mean",
    "mass_error": "pion_plateau_error",
}

# Columns required in cost data CSV
COST_DATA_COLUMNS = [
    "Bare_mass",
    "Configuration_label",
    "Average_core_hours_per_spinor",
]


# =============================================================================
# PION-SPECIFIC VALIDATION
# =============================================================================

PION_VALIDATION = {
    "require_positive_mass": True,  # Pion effective mass must be positive
    "min_pion_mass": 0.01,  # Minimum reasonable pion mass
    "max_pion_mass": 2.0,  # Maximum reasonable pion mass
}


# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================


def get_analysis_type() -> str:
    """Get analysis type identifier."""
    return ANALYSIS_TYPE


def get_default_output_filename() -> str:
    """Get default output CSV filename."""
    return DEFAULT_OUTPUT_FILENAME


def get_reference_pion_mass() -> float:
    """Get reference pion mass for extrapolation."""
    return REFERENCE_PION_MASS


def get_pion_mass_power() -> int:
    """Get power for pion mass (1 for m_π, 2 for m_π²)."""
    return PION_MASS_POWER


def get_pion_fit_config() -> Dict[str, Any]:
    """Get pion fitting configuration."""
    return PION_FIT_CONFIG.copy()


def get_required_columns() -> List[str]:
    """Get list of required input columns (derived from
    COLUMN_MAPPING)."""
    return list(COLUMN_MAPPING.values())


def get_column_mapping() -> Dict[str, str]:
    """Get column name mapping."""
    return COLUMN_MAPPING.copy()


def get_cost_data_columns() -> List[str]:
    """Get required cost data columns."""
    return COST_DATA_COLUMNS.copy()


def get_pion_validation_config() -> Dict[str, Any]:
    """Get pion-specific validation configuration."""
    return PION_VALIDATION.copy()


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_pion_cost_config():
    """
    Validate pion-specific configuration.

    Raises:
        ValueError: If configuration is invalid
    """
    # First validate shared configuration
    validate_shared_cost_config()

    # Validate reference pion mass
    if not isinstance(REFERENCE_PION_MASS, (int, float)):
        raise ValueError("REFERENCE_PION_MASS must be numeric")

    if REFERENCE_PION_MASS <= 0:
        raise ValueError("REFERENCE_PION_MASS must be positive")

    # Validate pion mass power
    if PION_MASS_POWER not in [1, 2]:
        raise ValueError("PION_MASS_POWER must be 1 (m_π) or 2 (m_π²)")

    # Validate column mapping
    required_mappings = ["bare_mass", "mass_mean", "mass_error"]
    for key in required_mappings:
        if key not in COLUMN_MAPPING:
            raise ValueError(f"Missing required column mapping: {key}")

    # Validate pion fit configuration
    if PION_FIT_CONFIG["fit_function"] != "linear":
        raise ValueError("Pion fit function must be 'linear'")

    # Validate pion validation settings
    if PION_VALIDATION["min_pion_mass"] <= 0:
        raise ValueError("min_pion_mass must be positive")

    if PION_VALIDATION["min_pion_mass"] >= PION_VALIDATION["max_pion_mass"]:
        raise ValueError("min_pion_mass must be less than max_pion_mass")
