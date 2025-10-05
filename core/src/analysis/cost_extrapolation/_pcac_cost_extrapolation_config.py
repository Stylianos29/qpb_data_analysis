"""
PCAC-specific configuration for computational cost extrapolation.

Configuration for extrapolating computational costs using PCAC mass as
the intermediate physics observable. Uses linear fit for PCAC mass vs
bare mass relationship.
"""

from typing import List, Dict, Any, Optional

from src.analysis.cost_extrapolation._cost_extrapolation_shared_config import (
    validate_shared_cost_config,
)


# =============================================================================
# PCAC-SPECIFIC SETTINGS
# =============================================================================

# Analysis type identifier
ANALYSIS_TYPE = "pcac"

# Default output filename
DEFAULT_OUTPUT_FILENAME = "computational_cost_extrapolation_from_pcac.csv"

# Reference PCAC mass for extrapolation
REFERENCE_PCAC_MASS = 0.005  # Target PCAC mass value

# PCAC mass fitting configuration
PCAC_FIT_CONFIG = {
    "fit_function": "linear",  # PCAC_mass = a * bare_mass + b
    "fit_method": "leastsq",
    "require_positive_mass": False,  # PCAC mass can be negative near critical point
}


# =============================================================================
# INPUT COLUMN MAPPINGS
# =============================================================================

# Column name mapping (standard name -> CSV column name) All columns
# listed here are considered required
COLUMN_MAPPING = {
    "bare_mass": "Bare_mass",
    "mass_mean": "PCAC_plateau_mean",
    "mass_error": "PCAC_plateau_error",
}

# Columns required in cost data CSV
COST_DATA_COLUMNS = [
    "Bare_mass",
    "Configuration_label",
    "Average_core_hours_per_spinor",
]

# =============================================================================
# FITTING RANGE CONFIGURATION
# =============================================================================

# Fitting range configuration for mass and cost fits
FIT_RANGE_CONFIG = {
    "mass_fit": {
        "bare_mass_min": 0.005,  # None = use actual data minimum
        "bare_mass_max": 0.055,  # None = use actual data maximum
    },
    "cost_fit": {
        "bare_mass_min": 0.005,  # None = use actual data minimum
        "bare_mass_max": 0.055,  # None = use actual data maximum
    },
}

# =============================================================================
# PCAC-SPECIFIC VALIDATION
# =============================================================================

PCAC_VALIDATION = {
    "allow_negative_pcac": True,  # PCAC mass can be negative
    "min_pcac_magnitude": 0.0,  # No minimum magnitude requirement
    "max_pcac_magnitude": 1.0,  # Maximum reasonable PCAC mass
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


def get_reference_pcac_mass() -> float:
    """Get reference PCAC mass for extrapolation."""
    return REFERENCE_PCAC_MASS


def get_pcac_fit_config() -> Dict[str, Any]:
    """Get PCAC fitting configuration."""
    return PCAC_FIT_CONFIG.copy()


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


def get_fit_range_config() -> Dict[str, Dict[str, float | None]]:
    """Get fitting range configuration."""
    return {
        "mass_fit": FIT_RANGE_CONFIG["mass_fit"].copy(),
        "cost_fit": FIT_RANGE_CONFIG["cost_fit"].copy(),
    }


def get_pcac_validation_config() -> Dict[str, Any]:
    """Get PCAC-specific validation configuration."""
    return PCAC_VALIDATION.copy()


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_pcac_cost_config():
    """
    Validate PCAC-specific configuration.

    Raises:
        ValueError: If configuration is invalid
    """
    # First validate shared configuration
    validate_shared_cost_config()

    # Validate reference PCAC mass
    if not isinstance(REFERENCE_PCAC_MASS, (int, float)):
        raise ValueError("REFERENCE_PCAC_MASS must be numeric")

    if REFERENCE_PCAC_MASS <= 0:
        raise ValueError("REFERENCE_PCAC_MASS must be positive")

    # Validate column mapping
    required_mappings = ["bare_mass", "mass_mean", "mass_error"]
    for key in required_mappings:
        if key not in COLUMN_MAPPING:
            raise ValueError(f"Missing required column mapping: {key}")

    # Validate PCAC fit configuration
    if PCAC_FIT_CONFIG["fit_function"] != "linear":
        raise ValueError("PCAC fit function must be 'linear'")

    # Validate fitting range configuration
    for fit_type in ["mass_fit", "cost_fit"]:
        range_min = FIT_RANGE_CONFIG[fit_type]["bare_mass_min"]
        range_max = FIT_RANGE_CONFIG[fit_type]["bare_mass_max"]

        if range_min is not None and range_max is not None:
            if range_min >= range_max:
                raise ValueError(
                    f"{fit_type}: bare_mass_min ({range_min}) must be "
                    f"less than bare_mass_max ({range_max})"
                )
