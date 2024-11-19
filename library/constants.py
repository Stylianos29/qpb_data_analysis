# Log files filenames might contain specific labels corresponding to
# parameters next to their values. Listed below are the parameters and their
# identifying labels
FILENAME_SINGLE_VALUE_PATTERNS_DICTIONARY = {
    # General parameters
    "Overlap_operator_method": {
        "pattern": r"(?P<Overlap_operator_method>Chebyshev|KL|Bare)",
        "type": str
    },
    "Kernel_operator_type": {
        "pattern": r"(?P<Kernel_operator_type>Standard|Brillouin)",
        "type": str
    },
    "QCD_beta_value": {
        "pattern": r"beta(?P<QCD_beta_value>\d+p?\d*)",
        "type": float
    },
    # TODO: Add a Lattice_geometry pattern
    "Configuration_label": {
        "pattern": r"config(?P<Configuration_label>\d+)",
        "type": str
    },
    "APE_iterations": {
        "pattern": r"APEiters(?P<APE_iterations>\d+)",
        "type": int
    },
    "Rho_value": {
        "pattern": r"rho(?P<Rho_value>\d+p?\d*)",
        "type": float
    },
    "Bare_mass": {
        "pattern": r"m(?P<Bare_mass>\d+p?\d*)",
        "type": float
    },
    "Kappa_value": {
        "pattern": r"kappa(?P<Kappa_value>\d+p?\d*)",
        "type": float
    },
    "Clover_coefficient": {
        "pattern": r"cSW(?P<Clover_coefficient>\d+p?\d*)",
        "type": float
    },
    # Chebyshev-specific parameters
    "Delta_Min": {
        "pattern": r"dMin(?P<Delta_Min>\d+p?\d*)",
        "type": float
    },
    "Delta_Max": {
        "pattern": r"dMax(?P<Delta_Max>\d+p?\d*)",
        "type": float
    },
    "Number_of_Chebyshev_terms": {
        "pattern": r"N(?P<Number_of_Chebyshev_terms>\d+)",
        "type": int
    },
    "Lanczos_Epsilon": {
        "pattern": r"EpsLanczos(?P<Lanczos_Epsilon>\d+\.\d+e[+-]\d+|\d+e[+-]\d+)",
        "type": float
    },
    # KL-specific parameters
    "CG_epsilon": {
        "pattern": r"_EpsCG(?P<CG_epsilon>\d*[\.p]?\d+e[+-]\d+|\d+e[+-]\d+)",
        "type": float
    },
    "MSCG_epsilon": {
        "pattern": r"_EpsMSCG(?P<MSCG_epsilon>\d*[\.p]?\d+e[+-]\d+|\d+e[+-]\d+)",
        "type": float
    },
    "KL_diagonal_order": {
        "pattern": r"_n(?P<KL_diagonal_order>\d+)",
        "type": int
    },
    "KL_scaling_factor": {
        "pattern": r"mu(?P<KL_scaling_factor>\d+p?\d*)",
        "type": float
    },
}


# Most of the information will be extracted from the contents of the
# log files. Listed below are the line identifiers for locating the
# line containing the parameter value, along with the regex type and the
# value type
FILE_CONTENTS_SINGLE_VALUE_PATTERNS_DICTIONARY = {
    # General parameters
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
    # TODO: Investigate whether identifier: "Mass = " needs to be added as well
    "Bare_mass": {
        "line_identifier": "mass = ",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": float,
    },
    "Kappa_value": {
        "line_identifier": "kappa = ",
        "regex_pattern": r"(\d+(\.\d+)?)",
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
        "line_identifier": "Number of Chebyshev polynomial terms = ",
        "regex_pattern": r"(\d+)",
        "type": int,
    },
    "Minimum_eigenvalue": {
        "line_identifier": ", beta =",
        "regex_pattern": r"alpha = (\d+(\.\d+)?)",
        "type": float,
    },
    "Maximum_eigenvalue": {
        "line_identifier": ", beta =",
        "regex_pattern": r", beta = (\d+(\.\d+)?)",
        "type": float,
    },
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
    "Solver_epsilon": {
        # "line_identifier": "Solver epsilon =",
        "line_identifier": "Outer solver epsilon =",
        "regex_pattern": r"(\d+\.\d+e-\d+)",
        "type": float,
    },
    "Maximum_solver_iterations": {
        "line_identifier": "Max solver iters = ",
        "regex_pattern": r"(\d+)",
        "type": int,
    },
}


FILE_CONTENTS_MULTIVALUED_PATTERNS_DICTIONARY = {
    "MSCG_Elapsed_time": {
        # "line_identifier": "sec",
        "line_identifier": "msCG converged, t = ",
        "regex_pattern": r"(\d+\.\d+)",
        "type": float,
    },
    "Number_of_MSCG_iterations": {
        "line_identifier": "msCG converged, t = ",
        "regex_pattern": r"After (\d+) iterations msCG converged,",
        "type": int,
    },
}
    # TODO: I need to decide what to do with this option
    # Results
    # "Calculation_result": {
    #     "line_identifier": "Done vector =",
    #     "regex_pattern": r"(\d+\.\d+e[+-]\d+)",
    #     "type": float,
    # }, 