#!/usr/bin/env python3
"""
PCAC mass calculation configuration.
"""

# PCAC calculation parameters
TRUNCATE_START = 2
TRUNCATE_END = 2

# Required datasets
REQUIRED_DATASETS = [
    "g4g5g5_derivative_jackknife_samples",
    "g5g5_jackknife_samples",
]

# Output dataset names
OUTPUT_DATASETS = {
    "samples": "PCAC_mass_jackknife_samples",
    "mean": "PCAC_mass_mean_values",
    "error": "PCAC_mass_error_values",
}


# Validation
def validate_pcac_config():
    """Validate PCAC configuration."""
    pass


#     expected_truncated = G5G5_LENGTH - TRUNCATE_START - TRUNCATE_END
#     if expected_truncated != DERIVATIVE_LENGTH:
#         raise ValueError(
#             f"Truncation error: {expected_truncated} != {DERIVATIVE_LENGTH}"
#         )
#     return True
