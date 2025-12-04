"""
Enhanced Parameter Transformation Configuration with Solver Resolution.

This updated configuration adds systematic solver parameter resolution
that disambiguates Inner/Outer/Generic solver parameters into canonical
CG_epsilon/MSCG_epsilon names based on Overlap_operator_method and
program type.
"""

import ast

import numpy as np


# =============================================================================
# RATIONAL ORDER RESOLUTION CONFIGURATION
# =============================================================================
"""
Resolve generic Rational_order (from filename) to method-specific names
based on Overlap_operator_method.
"""

RATIONAL_ORDER_RESOLUTION_MAPPING = {
    "KL": "KL_diagonal_order",
    "Neuberger": "Neuberger_order",
    "Zolotarev": "Zolotarev_order",
}

# =============================================================================
# SOLVER PARAMETER RESOLUTION CONFIGURATION
# =============================================================================
"""
This configuration handles the resolution of ambiguous solver parameters
parsed from log files into canonical names based on context.

The resolution logic depends on two factors:
    1. Overlap_operator_method: Bare, Chebyshev, KL, Neuberger,
       Zolotarev
    2. Program type: invert vs non-invert (detected from
       Main_program_type)

Resolution Rules Summary:
-------------------------
Canonical Names (output):
  - MSCG_epsilon / MSCG_max_iterations: Sign function (inner) solver
  - CG_epsilon / CG_max_iterations: Full overlap (outer) solver

Raw Names (input from parsing):
  - Inner_solver_*: Always means MSCG (sign function)
  - Outer_solver_*: Always means CG (full overlap)
  - Generic_solver_*: Context-dependent (see mapping below)
"""

SOLVER_PARAMETER_RESOLUTION_RULES = {
    # Define resolution mapping for each (method, program_type)
    # combination Format: (overlap_method, is_invert): {raw_param:
    # canonical_param}
    # =========================================================================
    # BARE METHOD
    # =========================================================================
    ("Bare", False): {
        # Non-invert Bare: No solver parameters
    },
    ("Bare", True): {
        # Invert Bare: Generic → CG (outer solver only)
        "Generic_solver_epsilon": "CG_epsilon",
        "Generic_solver_max_iterations": "CG_max_iterations",
    },
    # =========================================================================
    # CHEBYSHEV METHOD
    # =========================================================================
    ("Chebyshev", False): {
        # Non-invert Chebyshev: No solver parameters (only Lanczos)
    },
    ("Chebyshev", True): {
        # Invert Chebyshev: Generic → CG (outer solver only)
        "Generic_solver_epsilon": "CG_epsilon",
        "Generic_solver_max_iterations": "CG_max_iterations",
    },
    # =========================================================================
    # KL METHOD
    # =========================================================================
    ("KL", False): {
        # Non-invert KL: Generic → MSCG (inner/sign function)
        "Generic_solver_epsilon": "MSCG_epsilon",
        "Generic_solver_max_iterations": "MSCG_max_iterations",
    },
    ("KL", True): {
        # Invert KL: Inner → MSCG, Outer → CG
        "Inner_solver_epsilon": "MSCG_epsilon",
        "Inner_solver_max_iterations": "MSCG_max_iterations",
        "Outer_solver_epsilon": "CG_epsilon",
        "Outer_solver_max_iterations": "CG_max_iterations",
    },
    # =========================================================================
    # NEUBERGER METHOD
    # =========================================================================
    ("Neuberger", False): {
        # Non-invert Neuberger: Generic → MSCG (inner/sign function)
        "Generic_solver_epsilon": "MSCG_epsilon",
        "Generic_solver_max_iterations": "MSCG_max_iterations",
    },
    ("Neuberger", True): {
        # Invert Neuberger: Inner → MSCG, Outer → CG
        "Inner_solver_epsilon": "MSCG_epsilon",
        "Inner_solver_max_iterations": "MSCG_max_iterations",
        "Outer_solver_epsilon": "CG_epsilon",
        "Outer_solver_max_iterations": "CG_max_iterations",
    },
    # =========================================================================
    # ZOLOTAREV METHOD
    # =========================================================================
    ("Zolotarev", False): {
        # Non-invert Zolotarev: Generic → MSCG (inner/sign function)
        "Generic_solver_epsilon": "MSCG_epsilon",
        "Generic_solver_max_iterations": "MSCG_max_iterations",
    },
    ("Zolotarev", True): {
        # Invert Zolotarev: Inner → MSCG, Outer → CG
        "Inner_solver_epsilon": "MSCG_epsilon",
        "Inner_solver_max_iterations": "MSCG_max_iterations",
        "Outer_solver_epsilon": "CG_epsilon",
        "Outer_solver_max_iterations": "CG_max_iterations",
    },
}

# List of all raw parameter names that will be resolved
RAW_SOLVER_PARAMETER_NAMES = [
    "Inner_solver_epsilon",
    "Inner_solver_max_iterations",
    "Outer_solver_epsilon",
    "Outer_solver_max_iterations",
    "Generic_solver_epsilon",
    "Generic_solver_max_iterations",
]

# List of canonical parameter names (output)
CANONICAL_SOLVER_PARAMETER_NAMES = [
    "MSCG_epsilon",
    "MSCG_max_iterations",
    "CG_epsilon",
    "CG_max_iterations",
]


# =============================================================================
# STRING TRANSFORMATION CONFIGURATIONS
# =============================================================================

STRING_TRANSFORMATIONS = {
    # Value replacement rules
    "Kernel_operator_type": {"replacements": {"Standard": "Wilson"}},
    # Formatting rules
    "Configuration_label": {"formatter": "zero_pad", "length": 7},
    "QCD_beta_value": {"formatter": "decimal_places", "places": 2},
}


# =============================================================================
# MATHEMATICAL TRANSFORMATION CONFIGURATIONS
# =============================================================================

MATH_TRANSFORMATIONS = {
    # Parameters that need square root applied
    "sqrt_parameters": ["CG_epsilon", "MSCG_epsilon"],
    # Ratio calculations: result_column: (numerator, denominator)
    "ratio_calculations": {
        "Condition_number": ("Maximum_eigenvalue_squared", "Minimum_eigenvalue_squared")
    },
    # Type conversions
    "type_conversions": {"Clover_coefficient": int},
}


# =============================================================================
# COMPOUND PARAMETER EXTRACTION CONFIGURATIONS
# =============================================================================

EXTRACTION_RULES = {
    "Lattice_geometry": {
        "extract_to": {
            "Temporal_lattice_size": {"index": 0, "type": int},
            "Spatial_lattice_size": {"index": 1, "type": int},
        },
        "parser": "ast_literal_eval",
        "remove_source": True,
    },
    "MPI_geometry": {
        "transform": "remove_first_element_and_stringify",
        "parser": "ast_literal_eval",
    },
}


# =============================================================================
# COLUMN OPERATION CONFIGURATIONS
# =============================================================================

COLUMN_OPERATIONS = {
    # Column additions: result_column: [source_columns]
    "additions": {
        "APE_iterations": ["APE_iterations", "Initial_APE_iterations"],
    },
    # Columns to remove after processing
    "columns_to_remove": [
        "Initial_APE_iterations",  # After adding to APE_iterations
        "Lattice_geometry",  # After extracting components
        "Rational_order",  # After resolving to specific order names
        # Raw solver parameters (removed after resolution to canonical
        # names)
        "Inner_solver_epsilon",
        "Inner_solver_max_iterations",
        "Outer_solver_epsilon",
        "Outer_solver_max_iterations",
        "Generic_solver_epsilon",
        "Generic_solver_max_iterations",
    ],
    # Default value assignments for missing columns
    "default_values": {"Number_of_vectors": 1},
    # Complex derived calculations
    "derived_calculations": {
        "Number_of_cores": {
            "formula": "mpi_geometry_product_times_threads",
            "inputs": ["MPI_geometry", "Threads_per_process"],
        }
    },
}


# =============================================================================
# HDF5 PROCESSING CONFIGURATIONS
# =============================================================================

HDF5_PROCESSING_RULES = {
    "Calculation_result_per_vector": {
        "output_pattern": "Average_{main_program_type}_values",
        "aggregation_method": "mean_with_error",
        "condition": "all_lengths_greater_than_one",
        "fallback": {
            "output_pattern": "{main_program_type}_value_with_no_error",
            "aggregation_method": "single_value",
            "condition": "all_lengths_equal_to_one",
        },
    },
    "Total_number_of_MSCG_iterations": {
        "output_pattern": "Average_number_of_MSCG_iterations_per_{unit}",
        "aggregation_method": "mean_with_error",
        "condition": "all_lengths_greater_than_one",
        "unit_mapping": {
            "has_Number_of_spinors": {"unit": "spinor"},
            "default": {"unit": "vector"},
        },
    },
    "CG_total_calculation_time_per_spinor": {
        "output_pattern": "Average_CG_calculation_time_per_spinor",
        "aggregation_method": "mean_with_error",
        "condition": "all_lengths_greater_than_one",
    },
    "Total_number_of_CG_iterations_per_spinor": {
        "output_pattern": "Average_number_of_CG_iterations_per_spinor",
        "aggregation_method": "mean_with_error",
        "condition": "all_lengths_greater_than_one",
    },
}


# =============================================================================
# ANALYSIS CASE CONFIGURATIONS
# =============================================================================

ANALYSIS_CASES = {
    "inversions": {
        "mv_multiplication_formula": "calculate_mv_from_cg_iterations",
        "time_cost_input": "Average_wall_clock_time_per_spinor",
    },
    "forward_operator_applications": {
        "mv_multiplication_formula": "calculate_mv_from_mscg_data",
        "time_cost_input": "Average_wall_clock_time_per_vector",
    },
}


# =============================================================================
# TIME AND COST CALCULATION CONFIGURATIONS
# =============================================================================

TIME_COST_CALCULATIONS = {
    "Average_wall_clock_time_per_spinor": {
        "formula": "mean_from_hdf5_dataset",
        "hdf5_dataset": "CG_total_calculation_time_per_spinor",
    },
    "Average_wall_clock_time_per_vector": {
        "formula": "total_time_divided_by_vectors",
        "inputs": ["Total_calculation_time", "Number_of_vectors"],
    },
    "Average_core_hours_per_spinor": {
        "formula": "cores_times_time_divided_by_3600",
        "output_column": "Average_core_hours_per_spinor",
    },
    "Average_core_hours_per_vector": {
        "formula": "cores_times_time_divided_by_3600",
        "output_column": "Average_core_hours_per_vector",
    },
}


# =============================================================================
# HELPER FUNCTION REGISTRIES
# =============================================================================

FORMATTER_FUNCTIONS = {
    "zero_pad": lambda value, length: str(value).zfill(length),
    "decimal_places": lambda value, places: f"{float(value):.{places}f}",
}

PARSER_FUNCTIONS = {
    "ast_literal_eval": ast.literal_eval,
}

CALCULATION_FUNCTIONS = {
    "mpi_geometry_product_times_threads": lambda mpi_geo, threads: (
        int(np.prod(ast.literal_eval(mpi_geo))) * int(threads)
    ),
    "total_time_divided_by_vectors": lambda total_time, n_vectors: (
        float(total_time) / int(n_vectors)
    ),
}
