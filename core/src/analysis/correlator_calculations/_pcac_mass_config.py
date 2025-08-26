"""
Configuration constants for PCAC mass calculation.

This module defines all parameters specific to PCAC mass calculation
from jackknife-analyzed correlator data. These settings control the
truncation, validation, and output structure of the PCAC mass analysis.
"""

# =============================================================================
# TRUNCATION PARAMETERS
# =============================================================================

# Truncation for g5g5 correlators to match derivative length
TRUNCATE_START = 2  # Remove first 2 elements from g5g5 correlators
TRUNCATE_END = 2    # Remove last 2 elements from g5g5 correlators

# =============================================================================
# EXPECTED DATA DIMENSIONS
# =============================================================================

# Expected temporal lengths for validation
EXPECTED_G5G5_LENGTH = 48         # Original g5g5 correlator temporal length
EXPECTED_DERIVATIVE_LENGTH = 44   # g4g5g5_derivative length (pre-truncated)
EXPECTED_PCAC_LENGTH = 44         # Final PCAC mass temporal length

# =============================================================================
# DATASET NAMES
# =============================================================================

# Required input datasets (must exist in input HDF5)
REQUIRED_INPUT_DATASETS = [
    "g4g5g5_derivative_jackknife_samples",
    "g5g5_jackknife_samples",
]

# Alternative dataset names for backward compatibility
ALTERNATIVE_DATASET_NAMES = {
    "g4g5g5_derivative_jackknife_samples": [
        "g4g5_g5_derivative_jackknife_samples",
        "derivative_g4g5_g5_jackknife_samples",
    ],
    "g5g5_jackknife_samples": [
        "g5_g5_jackknife_samples",
        "Jackknife_samples_of_g5_g5_correlator_2D_array",
    ],
}

# Output dataset names
PCAC_MASS_DATASETS = {
    "jackknife_samples": "PCAC_mass_jackknife_samples",
    "mean_values": "PCAC_mass_mean_values",
    "error_values": "PCAC_mass_error_values",
}

# Metadata datasets to preserve from input
METADATA_DATASETS = [
    "gauge_configuration_labels",
    "mpi_geometry_values",
    "qpb_log_filenames",
    "Number_of_gauge_configurations",
]

# =============================================================================
# PHYSICS PARAMETERS
# =============================================================================

# PCAC mass calculation factor
PCAC_MASS_FACTOR = 0.5

# =============================================================================
# VALIDATION PARAMETERS
# =============================================================================

# Physical validation settings
VALIDATION_PARAMS = {
    # Check that g5g5 correlators are positive
    "check_g5g5_positive": True,
    
    # Check that g5g5 correlators decrease with time
    "check_g5g5_decreasing": True,
    
    # Minimum allowed correlator value (to avoid numerical issues)
    "min_correlator_value": 1e-15,
    
    # Check for NaN/inf in results
    "check_invalid_values": True,
    
    # Maximum allowed PCAC mass value (sanity check)
    "max_pcac_mass": 10.0,
    
    # Minimum number of jackknife samples required
    "min_jackknife_samples": 10,
}

# =============================================================================
# ERROR HANDLING
# =============================================================================

ERROR_HANDLING = {
    # How to handle division by zero in PCAC mass calculation
    "division_by_zero_replacement": 0.0,
    
    # Whether to skip groups with validation failures
    "skip_invalid_groups": False,
    
    # Whether to log detailed error information
    "verbose_errors": True,
}

# =============================================================================
# OUTPUT FILE STRUCTURE
# =============================================================================

OUTPUT_STRUCTURE = {
    # Whether to preserve full group hierarchy
    "preserve_hierarchy": True,
    
    # Whether to copy parent group attributes
    "copy_parent_attributes": True,
    
    # HDF5 compression settings
    "compression": "gzip",
    "compression_level": 4,
    
    # Chunk size for large datasets (None = auto)
    "chunk_size": None,
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    # Default log level for this module
    "default_log_level": "INFO",
    
    # Log progress every N groups
    "progress_interval": 10,
    
    # Log detailed timing information
    "log_timing": False,
}

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_output_filename(input_filename: str) -> str:
    """Generate default output filename based on input."""
    from pathlib import Path
    stem = Path(input_filename).stem
    return f"{stem}_PCAC_mass.h5"


def validate_configuration() -> bool:
    """
    Validate configuration consistency.
    
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is inconsistent
    """
    # Check truncation consistency
    expected_truncated_length = EXPECTED_G5G5_LENGTH - TRUNCATE_START - TRUNCATE_END
    if expected_truncated_length != EXPECTED_DERIVATIVE_LENGTH:
        raise ValueError(
            f"Truncation parameters inconsistent: "
            f"G5G5 length {EXPECTED_G5G5_LENGTH} - {TRUNCATE_START} - {TRUNCATE_END} "
            f"= {expected_truncated_length}, expected {EXPECTED_DERIVATIVE_LENGTH}"
        )
    
    # Check PCAC mass factor is positive
    if PCAC_MASS_FACTOR <= 0:
        raise ValueError(f"PCAC_MASS_FACTOR must be positive, got {PCAC_MASS_FACTOR}")
    
    # Check validation parameters
    if VALIDATION_PARAMS["min_correlator_value"] <= 0:
        raise ValueError("min_correlator_value must be positive")
    
    if VALIDATION_PARAMS["max_pcac_mass"] <= 0:
        raise ValueError("max_pcac_mass must be positive")
    
    return True
