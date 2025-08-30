#!/usr/bin/env python3
"""
Minimal shared configuration for correlator analysis.
"""

# Shared constants
MIN_JACKKNIFE_SAMPLES = 10

# Metadata datasets to copy
METADATA_DATASETS = [
    "gauge_configuration_labels",
    "mpi_geometry_values",
    "qpb_log_filenames",
    "Number_of_gauge_configurations",
]


# Validation settings
def check_correlator_basic(correlator, name="correlator"):
    """Basic correlator validation."""
    issues = []
    if correlator.ndim != 2:
        issues.append(f"{name} must be 2D array")
    if (correlator <= 0).any():
        issues.append(f"{name} contains non-positive values")
    return issues
