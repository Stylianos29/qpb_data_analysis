#!/usr/bin/env python3
"""
PCAC mass calculation configuration.
"""

from src.analysis.correlator_calculations._correlator_analysis_shared_config import (
    validate_shared_config,
)

# PCAC calculation parameters
TRUNCATE_START = 2
TRUNCATE_END = 2

# Required datasets
REQUIRED_DATASETS = [  # Order is important
    "g4g5g5_derivative_jackknife_samples",  # g4g5g5 derivative dataset
    "g5g5_jackknife_samples",  # g5g5 dataset
]

# Output dataset names
OUTPUT_DATASETS = {
    "samples": "PCAC_mass_jackknife_samples",
    "mean": "PCAC_mass_mean_values",
    "error": "PCAC_mass_error_values",
}

# Analysis documentation attributes
ANALYSIS_DOCUMENTATION = {
    "analysis_type": "PCAC mass calculation from jackknife correlator data",
    "pcac_formula": "PCAC_mass = 0.5 * g4g5g5_derivative / g5g5_truncated",
    "truncation_description": (
        f"g5g5 correlator truncated by removing {TRUNCATE_START} initial and "
        f"{TRUNCATE_END} final time points to match g4g5g5_derivative length"
    ),
    "time_range_note": (
        f"PCAC mass calculated for t={TRUNCATE_START} to t=T-{TRUNCATE_END+1} "
        "(after g5g5 truncation to match derivative length)"
    ),
}


# Validation
def validate_pcac_config():
    """Validate PCAC configuration."""
    # First validate shared dependencies
    validate_shared_config()

    # Check truncation parameters
    if not isinstance(TRUNCATE_START, int) or TRUNCATE_START < 0:
        raise ValueError(
            f"TRUNCATE_START must be non-negative integer, got {TRUNCATE_START}"
        )
    if not isinstance(TRUNCATE_END, int) or TRUNCATE_END < 0:
        raise ValueError(
            f"TRUNCATE_END must be non-negative integer, got {TRUNCATE_END}"
        )
    if TRUNCATE_START >= 8:
        raise ValueError(f"TRUNCATE_START too large: {TRUNCATE_START} >= 8")
    if TRUNCATE_END >= 8:
        raise ValueError(f"TRUNCATE_END too large: {TRUNCATE_END} >= 8")

    # Check required datasets
    if len(REQUIRED_DATASETS) != 2:
        raise ValueError(
            f"REQUIRED_DATASETS must have exactly 2 elements, "
            f"got {len(REQUIRED_DATASETS)}"
        )

    # Check for distinct dataset types
    has_g4g5g5_only = False
    has_g5g5_only = False

    for dataset in REQUIRED_DATASETS:
        if "g4g5g5" in dataset and "derivative" in dataset:
            has_g4g5g5_only = True
        elif "g5g5" in dataset and "g4g5g5" not in dataset:
            has_g5g5_only = True

    if not has_g4g5g5_only:
        raise ValueError("REQUIRED_DATASETS must contain a g4g5g5 derivative dataset")
    if not has_g5g5_only:
        raise ValueError("REQUIRED_DATASETS must contain a g5g5 dataset (not g4g5g5)")

    # Check output datasets structure
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

    # Check all output values are non-empty strings
    for key, value in OUTPUT_DATASETS.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"OUTPUT_DATASETS['{key}'] must be non-empty string, "
                f"got {repr(value)}"
            )
