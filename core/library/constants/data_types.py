"""Data type mappings and conversion functions."""

import ast
import pandas as pd

DTYPE_MAPPING = {
    "Clover_coefficient": int,
    "Configuration_label": str,
}


def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


# Handle empty strings gracefully
CONVERTERS_MAPPING = {
    # Format floats to 2 decimal places
    "QCD_beta_value": lambda x: f"{float(x):.2f}" if x.strip() else x,
    # Format floats to 2 decimal places
    "Delta_Min": lambda x: f"{float(x):.2f}" if x.strip() else x,
    # Format floats to 2 decimal places
    "Delta_Max": lambda x: f"{float(x):.2f}" if x.strip() else x,
    # Format floats to 2 decimal places
    "PCAC_mass_estimate": lambda x: safe_literal_eval(x) if x.strip() else x,
    #
    "Pion_effective_mass_estimate": lambda x: ast.literal_eval(x) if x.strip() else x,
    #
    "Critical_bare_mass": lambda x: ast.literal_eval(x) if x.strip() else x,
    # TODO: Add the rest of the 'Average_calculation_result' variations
    "Average_calculation_result": lambda x: safe_literal_eval(x) if x.strip() else x,
    "Average_sign_squared_values": lambda x: ast.literal_eval(x) if x.strip() else x,
    "Average_sign_squared_violation_values": lambda x: (
        ast.literal_eval(x) if x.strip() else x
    ),
    "Average_ginsparg_wilson_relation_values": lambda x: (
        ast.literal_eval(x) if x.strip() else x
    ),
    # Format floats in exponential notation
    "Solver_epsilon": lambda x: f"{float(x):.0e}" if x.strip() else x,
    "Linear_fit_slope": lambda x: ast.literal_eval(x) if x.strip() else x,
    # "Kernel_operator_type": lambda x: pd.Categorical(
    #         x, categories=["Wilson", "Brillouin"], ordered=True
    #     )
}

PARAMETERS_WITH_EXPONENTIAL_FORMAT = [
    "CG_epsilon",
    "Lanczos_epsilon",
    "MSCG_epsilon",
    "Solver_epsilon",
]

PARAMETERS_OF_INTEGER_VALUE = [
    "KL_diagonal_order",
    "Number_of_Chebyshev_terms",
]
