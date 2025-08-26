"""
Configuration constants for effective mass calculation.

This module defines all parameters specific to effective mass
calculation from jackknife-analyzed g5-g5 correlator data. These
settings control the symmetrization, truncation, and calculation methods
for the effective mass analysis.
"""

from typing import Dict

# =============================================================================
# SYMMETRIZATION AND TRUNCATION PARAMETERS
# =============================================================================

# Whether to apply symmetrization to g5-g5 correlators before
# calculation
APPLY_SYMMETRIZATION = True

# Whether to truncate to half the temporal extent (standard for periodic
# effective mass)
TRUNCATE_HALF = True

# Factor for middle value in two-state effective mass calculation
# (slightly less than 1.0 to ensure numerical stability)
LOWERING_FACTOR = 0.99

# =============================================================================
# EXPECTED DATA DIMENSIONS
# =============================================================================

# Expected temporal lengths for validation
EXPECTED_G5G5_LENGTH = 48         # Original g5-g5 correlator temporal length

# Expected effective mass length after processing For periodic boundary
# conditions with truncate_half=True: (T-2)/2 = (48-2)/2 = 23
EXPECTED_EFFECTIVE_MASS_LENGTH = 23

# =============================================================================
# DATASET NAMES
# =============================================================================

# Required input datasets (must exist in input HDF5)
REQUIRED_INPUT_DATASETS = [
    "g5g5_jackknife_samples",  # Only g5-g5 correlators needed
]

# Alternative dataset names for backward compatibility
ALTERNATIVE_DATASET_NAMES = {
    "g5g5_jackknife_samples": [
        "g5_g5_jackknife_samples",
        "Jackknife_samples_of_g5_g5_correlator_2D_array",
        "g5g5_correlator_jackknife_samples",
    ],
}

# Output dataset names
EFFECTIVE_MASS_DATASETS = {
    "jackknife_samples": "effective_mass_jackknife_samples", 
    "mean_values": "effective_mass_mean_values",
    "error_values": "effective_mass_error_values",
}

# Alternative output names for pion effective mass (if preferred)
PION_EFFECTIVE_MASS_DATASETS = {
    "jackknife_samples": "pion_effective_mass_jackknife_samples",
    "mean_values": "pion_effective_mass_mean_values",
    "error_values": "pion_effective_mass_error_values",
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

# Effective mass calculation method
CALCULATION_METHOD = {
    # Which formula to use
    "method": "two_state_periodic",  # Options: "two_state_periodic", "single_state", "cosh"
    
    # Parameters for two-state periodic effective mass
    "two_state_params": {
        "lowering_factor": LOWERING_FACTOR,
        "truncate_half": TRUNCATE_HALF,
    },
    
    # Parameters for single-state effective mass (log(C(t)/C(t+1)))
    "single_state_params": {
        "use_symmetrized": True,
    },
    
    # Parameters for cosh effective mass
    "cosh_params": {
        "temporal_extent": EXPECTED_G5G5_LENGTH,
    },
}

# =============================================================================
# VALIDATION PARAMETERS
# =============================================================================

# Physical validation settings
VALIDATION_PARAMS = {
    # Check that g5g5 correlators are positive
    "check_g5g5_positive": True,
    
    # Check that g5g5 correlators decrease with time (before T/2)
    "check_g5g5_decreasing": True,
    
    # Check that g5g5 correlators are symmetric (if periodic BC)
    "check_g5g5_symmetric": True,
    "symmetry_tolerance": 1e-10,
    
    # Minimum allowed correlator value (to avoid numerical issues)
    "min_correlator_value": 1e-15,
    
    # Check for NaN/inf in results
    "check_invalid_values": True,
    
    # Maximum allowed effective mass value (sanity check)
    "max_effective_mass": 10.0,
    
    # Check effective mass plateau behavior
    "check_plateau": False,  # Optional advanced check
    "plateau_tolerance": 0.01,  # Relative tolerance for plateau
    
    # Minimum number of jackknife samples required
    "min_jackknife_samples": 10,
}

# =============================================================================
# ERROR HANDLING
# =============================================================================

ERROR_HANDLING = {
    # How to handle negative values under square root
    "negative_sqrt_replacement": 0.0,
    
    # How to handle division by zero
    "division_by_zero_replacement": 0.0,
    
    # Whether to skip groups with validation failures
    "skip_invalid_groups": False,
    
    # Whether to log detailed error information
    "verbose_errors": True,
    
    # Whether to save problematic correlators for debugging
    "save_debug_data": False,
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
    
    # Whether to include symmetrized g5g5 in output
    "include_symmetrized_g5g5": False,
}

# =============================================================================
# PLOTTING CONFIGURATION (if integrated with visualization)
# =============================================================================

PLOTTING_CONFIG = {
    # Whether to generate plots during calculation
    "generate_plots": False,
    
    # Plot settings
    "plot_original_correlators": False,
    "plot_symmetrized_correlators": False,
    "plot_effective_mass": True,
    
    # Exclude first point (t=0) for better visualization
    "exclude_first_point": True,
    
    # Y-axis limits for effective mass plots (None = auto)
    "y_limits": None,
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
    
    # Log calculation method details
    "log_method_details": True,
}

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_output_filename(input_filename: str, use_pion_naming: bool = False) -> str:
    """
    Generate default output filename based on input.
    
    Args:
        input_filename: Input HDF5 filename use_pion_naming: Whether to
        use "pion_effective_mass" in filename
        
    Returns:
        Generated output filename
    """
    from pathlib import Path
    stem = Path(input_filename).stem
    suffix = "pion_effective_mass" if use_pion_naming else "effective_mass"
    return f"{stem}_{suffix}.h5"


def get_effective_mass_datasets(use_pion_naming: bool = False) -> Dict[str, str]:
    """
    Get the appropriate dataset names based on naming preference.
    
    Args:
        use_pion_naming: Whether to use pion-specific naming
        
    Returns:
        Dictionary of dataset names
    """
    return PION_EFFECTIVE_MASS_DATASETS if use_pion_naming else EFFECTIVE_MASS_DATASETS


def calculate_expected_output_length() -> int:
    """
    Calculate expected effective mass array length based on settings.
    
    Returns:
        Expected length of effective mass array
    """
    if CALCULATION_METHOD["method"] == "two_state_periodic":
        if TRUNCATE_HALF:
            return (EXPECTED_G5G5_LENGTH - 2) // 2
        else:
            return EXPECTED_G5G5_LENGTH - 2
    elif CALCULATION_METHOD["method"] == "single_state":
        return EXPECTED_G5G5_LENGTH - 1
    else:
        return EXPECTED_G5G5_LENGTH


def validate_configuration() -> bool:
    """
    Validate configuration consistency.
    
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is inconsistent
    """
    # Check lowering factor is in valid range
    if not 0 < LOWERING_FACTOR < 1:
        raise ValueError(f"LOWERING_FACTOR must be in (0,1), got {LOWERING_FACTOR}")
    
    # Check expected dimensions consistency
    calculated_length = calculate_expected_output_length()
    if calculated_length != EXPECTED_EFFECTIVE_MASS_LENGTH:
        raise ValueError(
            f"Expected effective mass length mismatch: "
            f"calculated {calculated_length}, expected {EXPECTED_EFFECTIVE_MASS_LENGTH}"
        )
    
    # Check validation parameters
    if VALIDATION_PARAMS["min_correlator_value"] <= 0:
        raise ValueError("min_correlator_value must be positive")
    
    if VALIDATION_PARAMS["max_effective_mass"] <= 0:
        raise ValueError("max_effective_mass must be positive")
    
    # Check calculation method is valid
    valid_methods = {"two_state_periodic", "single_state", "cosh"}
    if CALCULATION_METHOD["method"] not in valid_methods:
        raise ValueError(
            f"Invalid calculation method: {CALCULATION_METHOD['method']}. "
            f"Must be one of {valid_methods}"
        )
    
    return True
