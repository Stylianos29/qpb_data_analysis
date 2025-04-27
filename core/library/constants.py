from pathlib import Path
import ast
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd


# Define the root directory of the project
ROOT = Path(__file__).resolve().parents[2]
# TODO: Revisit its usefulness
RAW_DATA_FILES_DIRECTORY = "/nvme/h/cy22sg1/scratch/raw_qpb_data_files/"
PROCESSED_DATA_FILES_DIRECTORY = "../data_files/processed/"
# TODO: Validate provided constant directories. Relative paths do not make sense

# Log files filenames might contain specific labels corresponding to
# parameters next to their values. Listed below are the parameters and their
# identifying labels
FILENAME_SINGLE_VALUE_PATTERNS_DICTIONARY = {
    # General parameters
    "Overlap_operator_method": {
        "pattern": r"(?P<Overlap_operator_method>Chebyshev|KL|Bare)",
        "type": str,
    },
    "Kernel_operator_type": {
        "pattern": r"(?P<Kernel_operator_type>Standard|Brillouin)",
        "type": str,
    },
    "QCD_beta_value": {"pattern": r"beta(?P<QCD_beta_value>\d+p?\d*)", "type": float},
    "Lattice_geometry": {
        "pattern": r"_T(?P<TemporalDimension>\d+)L(?P<SpatialDimension>\d+)_",
        "type": int,
    },
    "Configuration_label": {
        "pattern": r"config(?P<Configuration_label>\d+)",
        "type": str,
    },
    "MPI_geometry": {
        "pattern": r"_cores(?P<MPI_geometry>\d{3})",
        "type": str,
    },
    "Number_of_vectors": {"pattern": r"NVecs(?P<Number_of_vectors>\d+)", "type": int},
    "APE_iterations": {"pattern": r"APEiters(?P<APE_iterations>\d+)", "type": int},
    "Rho_value": {"pattern": r"rho(?P<Rho_value>\d+p?\d*)", "type": float},
    "Bare_mass": {"pattern": r"m(?P<Bare_mass>-?\d+p?\d*)", "type": float},
    "Kappa_value": {"pattern": r"kappa(?P<Kappa_value>-?\d+p?\d*)", "type": float},
    "Clover_coefficient": {
        "pattern": r"cSW(?P<Clover_coefficient>\d+p?\d*)",
        "type": float,
    },
    # Chebyshev-specific parameters
    "Delta_Min": {"pattern": r"dMin(?P<Delta_Min>\d+p?\d*)", "type": float},
    "Delta_Max": {"pattern": r"dMax(?P<Delta_Max>\d+p?\d*)", "type": float},
    "Number_of_Chebyshev_terms": {
        "pattern": r"N(?P<Number_of_Chebyshev_terms>\d+)",
        "type": int,
    },
    "Lanczos_epsilon": {
        "pattern": r"EpsLanczos(?P<Lanczos_epsilon>\d+\.\d+e[+-]\d+|\d+e[+-]\d+)",
        "type": float,
    },
    # KL-specific parameters
    "CG_epsilon": {
        "pattern": r"_EpsCG(?P<CG_epsilon>\d*[\.p]?\d+e[+-]\d+|\d+e[+-]\d+)",
        "type": float,
    },
    "MSCG_epsilon": {
        "pattern": r"_EpsMSCG(?P<MSCG_epsilon>\d*[\.p]?\d+e[+-]\d+|\d+e[+-]\d+)",
        "type": float,
    },
    "KL_diagonal_order": {"pattern": r"_n(?P<KL_diagonal_order>\d+)", "type": int},
    "KL_scaling_factor": {
        "pattern": r"mu(?P<KL_scaling_factor>\d+p?\d*)",
        "type": float,
    },
}


# Most of the information will be extracted from the contents of the
# log files. Listed below are the line identifiers for locating the
# line containing the parameter value, along with the regex type and the
# value type
FILE_CONTENTS_SINGLE_VALUE_PATTERNS_DICTIONARY = {
    # General parameters
    "Cluster_partition": {
        "line_identifier": "Partition:",
        "regex_pattern": r"Partition:\s*(.+)",
        "type": str,
    },
    "Kernel_operator_type": {
        "line_identifier": "Dslash operator is",
        "regex_pattern": r"Dslash operator is (.+)",
        "type": str,
    },
    "Lattice_geometry": {
        "line_identifier": "(Lt, Lz, Ly, Lx)",
        "regex_pattern": r"\(Lt, Lz, Ly, Lx\) =\s*(.*)",
        "type": str,
    },
    "MPI_geometry": {
        "line_identifier": "Processes =",
        "regex_pattern": r"Processes =\s*(.*)",
        "type": str,
    },
    "Threads_per_process": {
        "line_identifier": "Threads per process = ",
        "regex_pattern": r"(\d+)",
        "type": int,
    },
    "Configuration_label": {
        "line_identifier": "Gauge field (raw_32) = ",
        "regex_pattern": r"\.(\d+)$",
        "type": str,
    },
    "QCD_beta_value": {
        "line_identifier": "Gauge field (raw_32) = ",
        "regex_pattern": r"Nf0_b(.*?)_L",
        "type": float,
    },
    "Initial_APE_iterations": {
        "line_identifier": "Gauge field (raw_32) = ",
        "regex_pattern": r"_apeN(\d+)a",
        "type": int,
    },
    "APE_alpha": {
        "line_identifier": "APE alpha =",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": float,
    },
    "APE_iterations": {
        "line_identifier": "APE iterations =",
        "regex_pattern": r"(\d+)",
        "type": int,
    },
    "Rho_value": {
        "line_identifier": "rho =",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": float,
    },
    "Bare_mass": {
        "line_identifier": "mass = ",
        "regex_pattern": r"(-?\d+(\.\d+)?)",
        "type": float,
    },
    "Number_of_vectors": {
        "line_identifier": "Done,",
        "regex_pattern": r"Done,\s*(\d+)\s*vectors",
        "type": int,
    },
    "Kappa_value": {
        "line_identifier": "kappa = ",
        "regex_pattern": r"(-?\d+(\.\d+)?)",
        "type": float,
    },
    # NOTE: The Clover coefficient is an integer number, 0 or 1.  But it is
    # given a float type for flexibility
    "Clover_coefficient": {
        "line_identifier": "Clover param = ",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": float,
    },
    "Plaquette": {
        "line_identifier": "Plaquette =",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": float,
    },
    # Chebyshev-specific parameters
    "Number_of_Chebyshev_terms": {
        "line_identifier": "Number of Chebyshev terms = ",
        "regex_pattern": r"(\d+)",
        "type": int,
    },
    "Lanczos_epsilon": {
        "line_identifier": "Lanczos epsilon =",
        "regex_pattern": r"(\d+\.\d+e[+-]\d+)",
        "type": float,
    },
    "Maximum_Lanczos_iterations": {
        "line_identifier": "Max Lanczos iters = ",
        "regex_pattern": r"(\d+)",
        "type": int,
    },
    "Delta_Min": {
        "line_identifier": "Min eigenvalue squared modification",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": float,
    },
    "Delta_Max": {
        "line_identifier": "Max eigenvalue squared modification",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": float,
    },
    "Minimum_eigenvalue_squared": {
        "line_identifier": "Min eigenvalue squared =",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": float,
    },
    "Maximum_eigenvalue_squared": {
        "line_identifier": "Max eigenvalue squared =",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": float,
    },
    "Total_overhead_time": {
        "line_identifier": "Total overhead time",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": float,
    },
    "Total_number_of_Lanczos_iterations": {
        "line_identifier": "Total number of Lanczos algorithm iterations = ",
        "regex_pattern": r"(\d+)",
        "type": int,
    },
    "Number_of_spinors": {
        "line_identifier": "CG done,",
        "regex_pattern": r"CG done, (\d+) vectors",
        "type": int,
    },
    # TODO: Very problematic. Revisit!
    # "Solver_epsilon": {
    #     "line_identifier": "Solver epsilon =",
    #     "regex_pattern": r"(\d+\.\d+e[+-]\d+)",
    #     "type": float,
    # },
    # "Maximum_solver_iterations": {
    #     "line_identifier": "Max solver iters = ",
    #     "regex_pattern": r"(\d+)",
    #     "type": int,
    # },
    # KL-specific parameters
    "KL_diagonal_order": {
        "line_identifier": "KL iters = ",
        "regex_pattern": r"(\d+)",
        "type": int,
    },
    "KL_scaling_factor": {
        "line_identifier": "Mu =",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": float,
    },
    # This one can attributed to both Chebyshev and KL cases with different
    # meaning though
    # TODO: I need to find a way to anticipate all the invert cases
    "Outer_solver_epsilon": {
        # "line_identifier": "Solver epsilon =",
        "line_identifier": "Outer solver epsilon =",
        "regex_pattern": r"(\d+\.\d+e[+-]\d+)",
        "type": float,
    },
    # "Maximum_solver_iterations": {
    #     "line_identifier": "Max solver iters = ",
    #     "regex_pattern": r"(\d+)",
    #     "type": int,
    # },
    "Total_calculation_time": {
        "line_identifier": "vectors in t",
        "regex_pattern": r"(\d+\.\d+)",
        "type": float,
    },
}


FILE_CONTENTS_MULTIVALUED_PATTERNS_DICTIONARY = {
    "Calculation_result_per_vector": {
        "line_identifier": "Done vector",
        "regex_pattern": r"= (\d+\.\d+e[+-]\d+)",
        "type": float,
    },
    "MSCG_iterations_running_count": {
        "line_identifier": "iters = ",
        "regex_pattern": r"(\d+) , res = ",
        "type": int,
    },
    "Total_number_of_MSCG_iterations": {
        "line_identifier": "msCG converged, t",
        "regex_pattern": r"After (\d+) iterations msCG converged,",
        "type": int,
    },
    "MSCG_total_calculation_time": {
        "line_identifier": "msCG converged, t",
        "regex_pattern": r", t = (\d+\.\d+)",
        "type": float,
    },
    "Total_number_of_CG_iterations_per_spinor": {
        "line_identifier": ", CG converged,",
        "regex_pattern": r"After (\d+) iters,",
        "type": int,
    },
    "Number_of_kernel_applications_per_MSCG": {
        "line_identifier": " Total number of dslash applications",
        "regex_pattern": r"dslash applications (\d+)",
        "type": int,
    },
    "CG_total_calculation_time_per_spinor": {
        "line_identifier": ", CG converged,",
        "regex_pattern": r", t = (\d+\.\d+) sec",
        "type": float,
    },
    "Lanczos_iterations_running_count": {
        "line_identifier": "iter =",
        "regex_pattern": r"(\d+), CN",
        "type": int,
    },
    "Solver_iterations_running_count": {
        "line_identifier": " iters =",
        "regex_pattern": r"(\d+), res",
        "type": int,
    },
    "Running_squared_relative_residual": {
        "line_identifier": " iters =",
        "regex_pattern": r", res = (\d+\.\d+e[+-]\d+)",
        "type": float,
    },
    "MS_expansion_shifts": {
        "line_identifier": "Shift = ",
        "regex_pattern": r"(\d+\.\d+),",
        "type": float,
    },
    "Final_squared_residual": {
        "line_identifier": "Shift = ",
        "regex_pattern": r", residual = (\d+\.\d+e[+-]\d+)",
        "type": float,
    },
    "Final_squared_relative_residual": {
        "line_identifier": "Shift = ",
        "regex_pattern": r", relative = (\d+\.\d+e[+-]\d+)",
        "type": float,
    },
}

MAIN_PROGRAM_TYPE_MAPPING = {
    "|| Sign^2(X) - 1 ||^2": "sign_squared_violation",
    "|| Sign^2(X) ||^2": "sign_squared_values",
    "||[D^+D, DD^+]||": "normality",
    "GW diff": "ginsparg_wilson_relation",
    "CG done": "invert",
}

CORRELATOR_IDENTIFIERS_LIST = [
    "1-1",
    "g5-g5",
    "g5-g4g5",
    "g4g5-g5",
    "g4g5-g4g5",
    "g1-g1",
    "g2-g2",
    "g3-g3",
]

DTYPE_MAPPING = {
    "Clover_coefficient": int,
    "Configuration_label": str,
}


def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


CONVERTERS_MAPPING = {
    # Format floats to 2 decimal places
    "QCD_beta_value": lambda x: f"{float(x):.2f}",
    # Format floats to 2 decimal places
    "Delta_Min": lambda x: f"{float(x):.2f}",
    # Format floats to 2 decimal places
    "Delta_Max": lambda x: f"{float(x):.2f}",
    # Format floats to 2 decimal places
    "PCAC_mass_estimate": lambda x: safe_literal_eval(x),
    #
    "Pion_effective_mass_estimate": lambda x: ast.literal_eval(x),
    #
    "Critical_bare_mass": lambda x: ast.literal_eval(x),
    # TODO: Add the rest of the 'Average_calculation_result' variations
    "Average_calculation_result": lambda x: safe_literal_eval(x),
    "Average_sign_squared_values": lambda x: ast.literal_eval(x),
    "Average_sign_squared_violation_values": lambda x: ast.literal_eval(x),
    "Average_ginsparg_wilson_relation_values": lambda x: ast.literal_eval(x),
    # Format floats in exponential notation
    "Solver_epsilon": lambda x: f"{float(x):.0e}",
    # "Kernel_operator_type": lambda x: pd.Categorical(
    #         x, categories=["Wilson", "Brillouin"], ordered=True
    #     )
}


TITLE_LABELS_DICTIONARY = {
    # Tunable parameters
    "APE_iterations": "APE iters",
    "APE_alpha": "$\\alpha_{APE}$",
    "QCD_beta_value": "$\\beta$",
    "Configuration_label": "config label",
    "Rho_value": "$\\rho$",
    "Bare_mass": "a$m_{bare}$",
    "Clover_coefficient": "$c_{SW}$",
    "Delta_Min": "$\\delta\\lambda_{min}^2$",
    "Delta_Max": "$\\delta\\lambda_{max}^2$",
    "Number_of_Chebyshev_terms": "N",
    "Lanczos_epsilon": "$\\epsilon_{Lanczos}$",
    "Solver_epsilon": "$\\epsilon_{CG}$",
    "CG_epsilon": "$\\epsilon_{CG}$",
    "MSCG_epsilon": "$\\epsilon_{MSCG}$",
    "KL_diagonal_order": "n",
    "KL_scaling_factor": "$\\mu$",
    "Number_of_vectors": "# of random vecs",
    "Kappa_value": "$\\kappa$",
    # Output quantity
    "Condition_number": "$\\kappa$",
    "Minimum_eigenvalue_squared": "$\\lambda_{min}^2$",
    "Maximum_eigenvalue_squared": "$\\lambda_{max}^2$",
    "Number_of_gauge_configurations": "# of configs",
}


AXES_LABELS_DICTIONARY = {
    # Output quantity
    "Average_number_of_MV_multiplications_per_spinor_per_configuration": "Average # of MV multiplications [per spinor per config]",
    "Average_number_of_MV_multiplications_per_vector": "Average # of MV multiplications per vector",
    "Number_of_MV_multiplications_for_constant_PCAC_mass": "# of MV multiplications [per spinor per config]",
    "Number_of_MV_multiplications_for_constant_bare_mass": "# of MV multiplications [per spinor per config]",
    "Average_calculation_time_per_spinor_per_configuration": "Average wall-clock time [per spinor per config] (s)",
    "Average_core_hours_per_vector": "Average cost of calculation per vector (core-hours)",
    "Average_core_hours_per_spinor": "Average cost of calculation per spinor (core-hours)",
    "Average_core_hours_per_spinor_per_configuration": "Average cost [per spinor per config] (core-hours)",
    "Adjusted_average_core_hours_per_spinor_per_configuration": "Average cost [per spinor per config] (core-hours)",
    "Core_hours_for_constant_PCAC_mass": "Average cost [per spinor per config] (core-hours)",
    "Total_overhead_time": "Overhead (s)",
    "Total_calculation_time": "Total wall-clock time (s)",
    "Average_wall_clock_time_per_vector": "Average wall-clock time per vector (s)",
    "Average_wall_clock_time_per_spinor": "Average wall-clock time per spinor (s)",
    "Total_number_of_Lanczos_iterations": "Total # of Lanczos algorithm iterations",
    # "Average_calculation_result": "||sgn$^2$(X) - I||$^2$",
    "PCAC_mass_estimate": "a$m_{PCAC}$",
    "Pion_effective_mass_estimate": "a$m_{eff.}$",
    "Critical_bare_mass": "a$m^{{critical}}_{{bare}}$",
    "Number_of_cores": "# of cores",
}

AXES_LABELS_DICTIONARY.update(TITLE_LABELS_DICTIONARY)

# TITLE_LABELS_DICTIONARY.update(
#     {
#         "Overlap_operator_method": "",
#         "Kernel_operator_type": "",
#     }
# )

MAIN_PROGRAM_TYPE_AXES_LABEL = {
    "Average_sign_squared_violation_values": "||(sgn$^2$(X) - I)$\\eta$||$^2$ / ||$\\eta$||$^2$",
    "Sign_squared_violation_with_no_error": "||(sgn$^2$(X) - I)$\\eta$||$^2$ / ||$\\eta$||$^2$",
    "Average_sign_squared_values": "||sgn$^2$(X)||$^2$",
    "Sign_squared_value_with_no_error": "||sgn$^2$(X)||$^2$",
    "Average_normality_values": "||$aD_{ov.}^{\\dagger} aD_{ov.} - aD_{ov.} aD_{ov.}^{\\dagger}$||$^2$",
    "Normality_with_no_error": "||$aD_{ov.}^{\\dagger} aD_{ov.} - aD_{ov.} aD_{ov.}^{\\dagger}$||$^2$",
    "Average_ginsparg_wilson_relation_values": "||$\\gamma_5 aD_{ov.} + aD_{ov.} \\gamma_5 - aD_{ov.} \\gamma_5 aD_{ov.}$||$^2$",
    "Ginsparg_wilson_relation_with_no_error": "||$\\gamma_5 aD_{ov.} + aD_{ov.} \\gamma_5 - aD_{ov.} \\gamma_5 aD_{ov.}$||$^2$",
}

AXES_LABELS_DICTIONARY.update(MAIN_PROGRAM_TYPE_AXES_LABEL)

# TODO: Maybe I need separate lists for qpb input parameters and qpb output
# values

# TODO: I need a list for output values labels for plot axes


PARAMETERS_PRINTED_LABELS_DICTIONARY = {
    "Overlap_operator_method": "method",
    "Kernel_operator_type": "kernel",
    "QCD_beta_value": "beta",
    "Configuration_label": "config",
    "APE_iterations": "APEiters",
    "Rho_value": "rho",
    "Bare_mass": "m",
    "Kappa_value": "kappa",
    "Clover_coefficient": "cSW",
    "Delta_Min": "dMin",
    "Delta_Max": "dMax",
    "Number_of_Chebyshev_terms": "N",
    "Lanczos_epsilon": "EpsLanczos",
    "CG_epsilon": "EpsCG",
    "MSCG_epsilon": "EpsMSCG",
    "KL_diagonal_order": "n",
    "KL_scaling_factor": "mu",
    "MPI_geometry": "cores",
}


TUNABLE_PARAMETER_NAMES_LIST = [
    "APE_alpha",
    "APE_iterations",
    "Bare_mass",
    "CG_epsilon",
    "Clover_coefficient",
    "Configuration_label",
    "Delta_Max",
    "Delta_Min",
    "Kappa_value",
    "Kernel_operator_type",
    "KL_diagonal_order",
    "KL_scaling_factor",
    "Lanczos_epsilon",
    "Main_program_type",
    "Maximum_Lanczos_iterations",
    "Maximum_solver_iterations",
    "MPI_geometry",
    "MSCG_epsilon",
    "Number_of_Chebyshev_terms",
    "Number_of_spinors",
    "Number_of_vectors",
    "Overlap_operator_method",
    "QCD_beta_value",
    "Rho_value",
    "Solver_epsilon",
    "Threads_per_process",
]


OUTPUT_QUANTITY_NAMES_LIST = [
    "Adjusted_average_core_hours_per_spinor",
    "Average_calculation_result",
    "Average_calculation_time_per_spinor_per_configuration",
    "Average_CG_calculation_time_per_spinor",
    "Average_core_hours_per_spinor",
    "Average_number_of_CG_iterations_per_spinor",
    "Average_number_of_MSCG_iterations_per_spinor",
    "Average_number_of_MSCG_iterations_per_vector",
    "Average_number_of_MV_multiplications_per_spinor",
    "Average_number_of_MV_multiplications_per_spinor_per_configuration",
    "Average_wall_clock_time_per_spinor",
    "Calculation_result_per_vector",
    "Calculation_result_with_no_error",
    "Condition_number",
    "Filename",
    "Final_squared_relative_residual",
    "Final_squared_residual",
    "Maximum_eigenvalue",
    "Maximum_eigenvalue_squared",
    "Minimum_eigenvalue",
    "Minimum_eigenvalue_squared",
    "MS_expansion_shifts",
    "MSCG_Elapsed_time",
    "Number_of_cores",
    "Number_of_gauge_configurations",
    "Number_of_MSCG_iterations",
    "PCAC_mass_estimate",
    "Plaquette",
    "Running_squared_relative_residual",
    "Solver_iterations_running_count",
    "Spatial_lattice_size",
    "Temporal_lattice_size",
    "Total_calculation_time",
    "Total_number_of_Lanczos_iterations",
    "Total_overhead_time",
]

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

MARKER_STYLES = [
    "o",  # Circle
    "^",  # Triangle Up
    "v",  # Triangle Down
    "<",  # Triangle Left
    ">",  # Triangle Right
    "p",  # Pentagon
    # "8",  # Octagon
    "D",  # Diamond
    "s",  # Square
    "d",  # Thin Diamond
    "h",  # Hexagon2
    "*",  # Star
    "X",  # X-shaped marker
    "P",  # Plus (filled)
    "x",  # X (cross)
    "H",  # Hexagon1
    # "|",  # Vertical Line
    # "_",  # Horizontal Line
    # ".",  # Point
    # ",",  # Pixel (small point)
    # "+",  # Plus
    # "h",  # Alternate Hexagon
    # "H",  # Alternate Hexagon (filled)
    # "p",  # Alternate Pentagon
    # "*",  # Alternate Star
    # "X",  # Alternate X
    # "1",  # Tri-down (Alternative)
    # "2",  # Tri-up (Alternative)
    # "3",  # Tri-left (Alternative)
    # "4",  # Tri-right (Alternative)
]


# MARKER_COLORS = list(plt.cm.tab40.colors)
# Get the default color cycle
#
# DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# DEFAULT_COLORS = plt.cm.tab20.colors

# num_colors = 20
# colormap = cm.get_cmap("turbo", num_colors)  # 'viridis' has good contrast #turbo
# DEFAULT_COLORS = [colormap(i) for i in range(num_colors)]

DEFAULT_COLORS = [
    "blue",
    "red",
    "magenta",
    "darkgoldenrod",  # Dark yellow option
    "green",
    "black",
    "orange",
    "purple",
    "brown",
    "darkslategray",
    "indigo",
    "gray",
    # "cyan",
    # "pink",
    # "lime",
    # "teal",
    # "olive",
    # "maroon",
    # "gold",
    # "navy",
    # "darkred",        # Added
    # "darkviolet"      # Added
    # "darkorange",     # Added
    "darkgreen",
    "violet",
]

FILENAME_LABELS_BY_COLUMN_NAME = {
    "QCD_beta_value": "beta",
    "Configuration_label": "config",
    "APE_iterations": "APEiters",
    "Rho_value": "rho",
    "Bare_mass": "m",
    "Kappa_value": "kappa",
    "Clover_coefficient": "cSW",
    "Delta_Min": "dMin",
    "Delta_Max": "dMax",
    "Number_of_Chebyshev_terms": "N",
    "Lanczos_epsilon": "EpsLanczos",
    "CG_epsilon": "EpsCG",
    "MSCG_epsilon": "EpsMSCG",
    "KL_diagonal_order": "n",
    "KL_scaling_factor": "mu",
    "MPI_geometry": "MPI",
    "Threads_per_process": "OMP",
}

AXES_LABELS_BY_COLUMN_NAME = {
    "Average_sign_squared_violation_values": "||(sgn$^2$(X) - I)$\\eta$||$^2$ / ||$\\eta$||$^2$",
    "KL_diagonal_order": "n",
    "Average_number_of_MV_multiplications_per_spinor_per_configuration": "Average # of MV multiplications [per spinor per config]",
    "Average_number_of_MV_multiplications_per_vector": "Average # of MV multiplications per vector",
    "Number_of_MV_multiplications_for_constant_PCAC_mass": "# of MV multiplications [per spinor per config]",
    "Number_of_MV_multiplications_for_constant_bare_mass": "# of MV multiplications [per spinor per config]",
    "Average_calculation_time_per_spinor_per_configuration": "Average wall-clock time [per spinor per config] (s)",
    "Average_core_hours_per_vector": "Average cost of calculation per vector (core-hours)",
    "Average_core_hours_per_spinor": "Average cost of calculation per spinor (core-hours)",
    "Average_core_hours_per_spinor_per_configuration": "Average cost [per spinor per config] (core-hours)",
    "Adjusted_average_core_hours_per_spinor_per_configuration": "Average cost [per spinor per config] (core-hours)",
    "Core_hours_for_constant_PCAC_mass": "Average cost [per spinor per config] (core-hours)",
    "Total_overhead_time": "Overhead (s)",
    "Total_calculation_time": "Total wall-clock time (s)",
    "Average_wall_clock_time_per_vector": "Average wall-clock time per vector (s)",
    "Average_wall_clock_time_per_spinor": "Average wall-clock time per spinor (s)",
    "Total_number_of_Lanczos_iterations": "Total # of Lanczos algorithm iterations",
    "PCAC_mass_estimate": "a$m_{PCAC}$",
    "Pion_effective_mass_estimate": "a$m_{eff.}$",
    "Critical_bare_mass": "a$m^{{critical}}_{{bare}}$",
    "Number_of_cores": "# of cores",
    "time_index": "$t/a$",
    "Jackknife_average_of_PCAC_mass_correlator": "a$m_{PCAC}$(t)",
}

LEGEND_LABELS_BY_COLUMN_NAME = {
    'Condition_number': "$\kappa = \lambda_{\max}^2 / \lambda_{\min}^2$:",
    'Configuration_label': 'Config. labels:',
    'KL_diagonal_order': 'diagonal KL order:',
    'Kernel_operator_type': 'Kernel:',
}

TITLE_LABELS_BY_COLUMN_NAME = TITLE_LABELS_DICTIONARY.copy()
