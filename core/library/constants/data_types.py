"""Data type mappings and conversion functions."""

import ast
import pandas as pd

DTYPE_MAPPING = {
    "Configuration_label": str,
    "APE_iterations": int,
    "Number_of_spinors": int,
    "Number_of_vectors": int,
    "MSCG_max_iterations": int,
    "CG_max_iterations": int,
    "Threads_per_process": int,
    "Number_of_Chebyshev_terms": int,
    "KL_diagonal_order": int,
    "Rational_order": int,
}


def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


# Handle empty strings gracefully
CONVERTERS_MAPPING = {
    # TUNABLE PARAMETERS
    # Format floats to 2 decimal places
    "QCD_beta_value": lambda x: f"{float(x):.2f}" if x.strip() else x,
    "Delta_Min": lambda x: f"{float(x):.2f}" if x.strip() else x,
    "Delta_Max": lambda x: f"{float(x):.2f}" if x.strip() else x,
    # Smart converter for Clover_coefficient
    "Clover_coefficient": lambda x: (
        (int(float(x)) if x.strip() and float(x).is_integer() else float(x))
        if x.strip()
        else x
    ),
    # Additional_text (stores unmatched filename segments as tuples)
    "Additional_text": lambda x: safe_literal_eval(x) if x.strip() else x,
    # OUTPUT QUANTITIES
    # TODO: Add the rest of the 'Average_calculation_result' variations
    "PCAC_mass_estimate": lambda x: safe_literal_eval(x) if x.strip() else x,
    "Pion_effective_mass_estimate": lambda x: ast.literal_eval(x) if x.strip() else x,
    "Critical_bare_mass": lambda x: ast.literal_eval(x) if x.strip() else x,
    "Average_calculation_result": lambda x: safe_literal_eval(x) if x.strip() else x,
    "Average_sign_squared_values": lambda x: ast.literal_eval(x) if x.strip() else x,
    "Average_sign_squared_violation_values": lambda x: (
        ast.literal_eval(x) if x.strip() else x
    ),
    "Average_ginsparg_wilson_relation_values": lambda x: (
        ast.literal_eval(x) if x.strip() else x
    ),
    "Average_CG_calculation_time_per_spinor": lambda x: (
        ast.literal_eval(x) if x.strip() else x
    ),
    "Average_number_of_CG_iterations_per_spinor": lambda x: (
        ast.literal_eval(x) if x.strip() else x
    ),
    "Average_number_of_MSCG_iterations_per_spinor": lambda x: (
        ast.literal_eval(x) if x.strip() else x
    ),
    "Linear_fit_slope": lambda x: ast.literal_eval(x) if x.strip() else x,
    # TODO: Revisit whether Kernel_operator_type needs a converter
    # "Kernel_operator_type": lambda x: pd.Categorical( x,
    #         categories=["Wilson", "Brillouin"], ordered=True )
}

PARAMETERS_WITH_EXPONENTIAL_FORMAT = [
    "Lanczos_epsilon",
    "Outer_solver_epsilon",
    "CG_epsilon",
    "Solver_epsilon",
    "Inner_solver_epsilon",
    "MSCG_epsilon",
]

PARAMETERS_OF_INTEGER_VALUE = [
    "KL_diagonal_order",
    "Zolotarev_order",
    "Neuberger_order",
    "Rational_order",
    "Number_of_Chebyshev_terms",
]
