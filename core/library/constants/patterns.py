"""Regex patterns for parsing log files and extracting parameters."""

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
