"""
Parameter transformation configuration for QPB log file processing.

This module defines declarative configurations for all parameter
transformations, replacing hardcoded logic with systematic, maintainable
transformation rules.
"""

import ast
from typing import List, Union, Any

import numpy as np


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
    "sqrt_parameters": ["Solver_epsilon", "CG_epsilon", "MSCG_epsilon"],
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
        "Rational_order": ["KL_diagonal_order", "Zolotarev_order", "Neuberger_order"],
    },
    # Columns to remove after processing
    "columns_to_remove": [
        "Initial_APE_iterations",  # After adding to APE_iterations
        "Lattice_geometry",  # After extracting components
        "Solver_epsilon",  # Conditional removal based on MSCG_epsilon presence
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
        "aggregation_method": "sum_then_divide",
        "unit_mapping": {
            "has_Number_of_spinors": {"unit": "spinor", "divisor": "Number_of_spinors"},
            "default": {"unit": "vector", "divisor": "Number_of_vectors"},
        },
    },
    "Number_of_kernel_applications_per_MSCG": {
        "output_pattern": "Average_number_of_MV_multiplications_per_{unit}",
        "aggregation_method": "sum_plus_length_then_divide",
        "unit_mapping": {
            "has_Number_of_spinors": {"unit": "spinor", "divisor": "Number_of_spinors"},
            "default": {"unit": "vector", "divisor": "Number_of_vectors"},
        },
    },
    "MS_expansion_shifts": {
        "output_pattern": "MS_expansion_shifts",
        "aggregation_method": "unique_values_as_list",
        "data_type": "float",
    },
    "Total_number_of_CG_iterations_per_spinor": {
        "output_pattern": "Average_number_of_CG_iterations_per_spinor",
        "aggregation_method": "mean",
        "data_type": "float",
    },
}


# =============================================================================
# ANALYSIS CASE CONFIGURATIONS
# =============================================================================

ANALYSIS_CASES = {
    "forward_operator_applications": {
        "identifier": "not_has_Number_of_spinors",
        "mv_multiplication_rules": {
            "Chebyshev": {
                "formula": "2 * Total_number_of_Lanczos_iterations + 1 + 2 * Number_of_Chebyshev_terms - 1"
            }
            # KL case would be added here when uncommented in original
        },
    },
    "inversions": {
        "identifier": "has_Number_of_spinors",
        "mv_multiplication_rules": {
            "Chebyshev": {
                "formula": "(2 * Total_number_of_Lanczos_iterations + 1) + (2 * Average_number_of_CG_iterations_per_spinor + 1) * (2 * Number_of_Chebyshev_terms - 1)"
            }
            # KL case would be added here when uncommented in original
        },
    },
}


# =============================================================================
# TIME AND COST CALCULATION CONFIGURATIONS
# =============================================================================

TIME_COST_CALCULATIONS = {
    "wall_clock_time": {
        "forward_case": {
            "base_formula": "Total_calculation_time / Number_of_vectors",
            "with_overhead": "(Total_calculation_time - Total_overhead_time) / Number_of_vectors + Total_overhead_time",
            "output_column": "Average_wall_clock_time_per_vector",
        },
        "inversion_case": {
            "base_formula": "Total_calculation_time / Number_of_spinors",
            "with_overhead": "(Total_calculation_time - Total_overhead_time) / Number_of_spinors + Total_overhead_time",
            "output_column": "Average_wall_clock_time_per_spinor",
        },
    },
    "core_hours": {
        "forward_case": {
            "formula": "Number_of_cores * Average_wall_clock_time_per_vector / 3600",
            "output_column": "Average_core_hours_per_vector",
        },
        "inversion_case": {
            "formula": "Number_of_cores * Average_wall_clock_time_per_spinor / 3600",
            "output_column": "Average_core_hours_per_spinor",
        },
    },
}


# =============================================================================
# HELPER FUNCTION MAPPINGS
# =============================================================================


def zero_pad_formatter(value: str, length: int) -> str:
    """Zero-pad a string to specified length."""
    return str(value).zfill(length)


def decimal_places_formatter(value: Union[str, float], places: int) -> str:
    """Format number to specified decimal places."""
    return f"{float(value):.{places}f}"


def ast_literal_eval_parser(value: str) -> Any:
    """Safely evaluate string as Python literal."""
    return ast.literal_eval(value)


def remove_first_element_and_stringify(parsed_list: List) -> str:
    """Remove first element from list and convert back to string."""
    return str(parsed_list[1:])


def mpi_geometry_product_times_threads(mpi_geometry: str, threads: int) -> int:
    """Calculate total cores from MPI geometry and threads."""
    return np.prod(ast.literal_eval(mpi_geometry)) * threads


# Function registry for dynamic lookup
FORMATTER_FUNCTIONS = {
    "zero_pad": zero_pad_formatter,
    "decimal_places": decimal_places_formatter,
}

PARSER_FUNCTIONS = {"ast_literal_eval": ast_literal_eval_parser}

CALCULATION_FUNCTIONS = {
    "mpi_geometry_product_times_threads": mpi_geometry_product_times_threads,
    "remove_first_element_and_stringify": remove_first_element_and_stringify,
}
