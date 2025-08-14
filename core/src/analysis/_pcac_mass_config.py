"""
Configuration constants for PCAC mass calculation.

This module defines the parameters used for truncating correlator data
and calculating PCAC mass values from jackknife analysis results.

Place this file as:
qpb_data_analysis/core/src/analysis/_pcac_mass_config.py
"""

# Truncation parameters for PCAC mass calculation
TRUNCATE_START = 2  # Remove first 2 elements from g5g5 correlators
TRUNCATE_END = 2  # Remove last 2 elements from g5g5 correlators

# Expected dataset lengths for validation
EXPECTED_G5G5_LENGTH = 48  # Original g5g5 correlator length
EXPECTED_DERIVATIVE_LENGTH = 44  # g4g5g5_derivative length (already truncated)
EXPECTED_PCAC_LENGTH = 44  # Expected PCAC mass length after calculation

# Required input datasets for PCAC mass calculation
REQUIRED_INPUT_DATASETS = [
    "g4g5g5_derivative_jackknife_samples",
    "g5g5_jackknife_samples",
]

# Output dataset names
PCAC_MASS_DATASETS = {
    "jackknife_samples": "PCAC_mass_jackknife_samples",
    "mean_values": "PCAC_mass_mean_values",
    "error_values": "PCAC_mass_error_values",
}

# Metadata datasets to copy from input
METADATA_DATASETS = [
    "gauge_configuration_labels",
    "mpi_geometry_values",
    "qpb_log_filenames",
]

# PCAC mass calculation constant
PCAC_MASS_FACTOR = 0.5
