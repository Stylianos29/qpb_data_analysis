#!/usr/bin/env python3
"""
Effective mass calculation configuration.
"""

# Effective mass calculation parameters
APPLY_SYMMETRIZATION = True
TRUNCATE_HALF = True
LOWERING_FACTOR = 0.99

# Expected dimensions
G5G5_LENGTH = 48
EXPECTED_OUTPUT_LENGTH = 22  # (48/2) - 2 for two-state periodic

# Required datasets
REQUIRED_DATASETS = [
    "g5g5_jackknife_samples",
]

# Output dataset names (standard)
OUTPUT_DATASETS = {
    "samples": "effective_mass_jackknife_samples",
    "mean": "effective_mass_mean_values",
    "error": "effective_mass_error_values",
}

# Output dataset names (pion naming)
PION_OUTPUT_DATASETS = {
    "samples": "pion_effective_mass_jackknife_samples",
    "mean": "pion_effective_mass_mean_values",
    "error": "pion_effective_mass_error_values",
}


# Validation
def validate_effective_config():
    """Validate effective mass configuration."""
    if not 0 < LOWERING_FACTOR < 1:
        raise ValueError(f"Lowering factor must be in (0,1), got {LOWERING_FACTOR}")
    return True
