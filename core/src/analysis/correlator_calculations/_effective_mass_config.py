#!/usr/bin/env python3
"""
Effective mass calculation configuration.
"""

from src.analysis.correlator_calculations._correlator_analysis_shared_config import (
    validate_shared_config,
)


# Effective mass calculation parameters
APPLY_SYMMETRIZATION = True
TRUNCATE_HALF = True
LOWERING_FACTOR = 0.99

# Required datasets
REQUIRED_DATASETS = [
    "g5g5_jackknife_samples",
]

# Output dataset names
OUTPUT_DATASETS = {
    "samples": "pion_effective_mass_jackknife_samples",
    "mean": "pion_effective_mass_mean_values",
    "error": "pion_effective_mass_error_values",
}

# Analysis documentation attributes
ANALYSIS_DOCUMENTATION = {
    "analysis_type": "Effective mass calculation from jackknife g5g5 correlator data",
    "effective_mass_formula": (
        "Two-state periodic: 0.5 * ln((C(t-1) + sqrt(...)) / (C(t+1) + sqrt(...)))"
    ),
    "symmetrization_applied": (
        "Correlator symmetrization: "
        f"{'enabled' if APPLY_SYMMETRIZATION else 'disabled'}"
    ),
    "truncation_applied": (
        "Half-length truncation: "
        f"{'enabled' if TRUNCATE_HALF else 'disabled'} "
        "(periodic boundary conditions)"
    ),
    "time_range_note": (
        "Effective mass calculated for t=1 to "
        f"{'T//2' if TRUNCATE_HALF else 'T-2'} "
        "(boundary points excluded for finite differences)"
    ),
}


# Validation
def validate_effective_config():
    """Validate effective mass configuration."""
    # First validate shared dependencies
    validate_shared_config()

    # Check boolean flags
    if not isinstance(APPLY_SYMMETRIZATION, bool):
        raise ValueError(
            f"APPLY_SYMMETRIZATION must be boolean, "
            f"got {type(APPLY_SYMMETRIZATION)}"
        )
    if not isinstance(TRUNCATE_HALF, bool):
        raise ValueError(f"TRUNCATE_HALF must be boolean, got {type(TRUNCATE_HALF)}")

    # Check lowering factor
    if not isinstance(LOWERING_FACTOR, (int, float)) or not (0 < LOWERING_FACTOR < 1):
        raise ValueError(
            f"LOWERING_FACTOR must be between 0 and 1, got {LOWERING_FACTOR}"
        )

    # Check required datasets
    if len(REQUIRED_DATASETS) != 1:
        raise ValueError(
            f"REQUIRED_DATASETS must have exactly 1 element, "
            f"got {len(REQUIRED_DATASETS)}"
        )
    if "g5g5" not in REQUIRED_DATASETS[0]:
        raise ValueError("REQUIRED_DATASETS must contain a dataset with 'g5g5'")

    # Check output datasets structure (same as PCAC)
    expected_keys = {"samples", "mean", "error"}
    if len(OUTPUT_DATASETS) != 3:
        raise ValueError(
            f"OUTPUT_DATASETS must have exactly 3 elements, "
            f"got {len(OUTPUT_DATASETS)}"
        )
    if set(OUTPUT_DATASETS.keys()) != expected_keys:
        raise ValueError(
            f"OUTPUT_DATASETS keys must be {expected_keys}, "
            f"got {set(OUTPUT_DATASETS.keys())}"
        )

    for key, value in OUTPUT_DATASETS.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"OUTPUT_DATASETS['{key}'] must be non-empty string, "
                f"got {repr(value)}"
            )
