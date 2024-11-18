# Log files filenames might contain specific labels corresponding to
# parameters next to their values. Listed below are the parameters and their
# identifying labels
FILENAME_REGEX_PATTERNS_DICTIONARY = {
    # General parameters
    # TODO: What about about "Bare"?
    "Overlap_operator_method": r"(?P<Operator_method>Chebyshev|KL|Bare)",
    "Kernel_operator_type": r"(?P<Operator_type>Standard|Brillouin)",
    "QCD_beta_value": r"beta(?P<QCD_beta_value>\d+p?\d*)",
    "Configuration_label": r"config(?P<Configuration_label>\d+)",
    "APE_iterations": r"APEiters(?P<APE_iterations>\d+)",
    "Rho_value": r"rho(?P<Rho_value>\d+p?\d*)",
    "Bare_mass": r"m(?P<Bare_mass>\d+p?\d*)",
    "Clover_coefficient": r"cSW(?P<Clover_coefficient>\d+p?\d*)",
    # Chebyshev-specific parameters
    "Delta_Min": r"dMin(?P<Delta_Min>\d+p?\d*)",
    "Delta_Max": r"dMax(?P<Delta_Max>\d+p?\d*)",
    "Number_of_Chebyshev_terms": r"N(?P<Number_of_Chebyshev_terms>\d+)",
    "Lanczos_Epsilon":\
            r"EpsLanczos(?P<Lanczos_Epsilon>\d+\.\d+e[+-]\d+|\d+e[+-]\d+)",
    # KL-specific parameters
    "CG_epsilon": r"_EpsCG(?P<CG_epsilon>\d*[\.p]?\d+e[+-]\d+|\d+e[+-]\d+)",
    "MSCG_epsilon": r"_EpsMSCG(?P<MSCG_epsilon>\d*[\.p]?\d+e[+-]\d+|\d+e[+-]\d+)",
    "KL_iterations": r"_n(?P<KL_iterations>\d+)",
    "KL_scaling_factor": r"mu(?P<KL_scaling_factor>\d+p?\d*)",
}

# Most of the information will be extracted from the contents of the
# log files. Listed below are the line identifiers for locating the
# line containing the parameter value, along with the regex type and the
# value type
PARAMETER_ATTRIBUTES_DICTIONARY = {
    # General parameters
    "Kernel_operator_type": {
        "line_identifier": "Dslash operator is",
        "regex_pattern": r"Dslash operator is (.+)",
        "type": "str",
    },
    "Lattice_geometry": {
        "line_identifier": "(Lt, Lz, Ly, Lx)",
        "regex_pattern": r"\(Lt, Lz, Ly, Lx\) =\s*(.*)",
        "type": "str",
    },
    "Configuration_label": {
        "line_identifier": "Gauge field (raw_32) = ",
        "regex_pattern": r"\.(\d+)$",
        "type": "str",
    },
    "QCD_beta_value": {
        "line_identifier": "Gauge field (raw_32) = ",
        "regex_pattern": r"Nf0_b(.*?)_L",
        "type": "str",
    },
    "Initial_APE_iterations": {
        "line_identifier": "Gauge field (raw_32) = ",
        "regex_pattern": r"T\d+(_[^.]*)?\.",
        "type": "str",
    },
    "APE_alpha": {
        "line_identifier": "APE alpha =",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": "float",
    },
    "APE_iterations": {
        "line_identifier": "APE iterations =",
        "regex_pattern": r"(\d+)",
        "type": "int",
    },
    "Rho_value": {
        "line_identifier": "rho =",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": "float",
    },
    "Bare_mass": {
        "line_identifier": "mass = ",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": "float",
    },
    # NOTE: The Clover coefficient is an integer number, 0 or 1.  But it is
    # given a float type for flexibility
    "Clover_coefficient": {
        "line_identifier": "Clover param = ",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": "float",
    },
    "Maximum_solver_iterations": {
        "line_identifier": "Max solver iters = ",
        "regex_pattern": r"(\d+)",
        "type": "int",
    },
    "Plaquette": {
        "line_identifier": "Plaquette =",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": "float",
    },
    # TODO: I need to decide what to do with this option
    # Results
    # "Calculation_result": {
    #     "line_identifier": "Done vector =",
    #     "regex_pattern": r"(\d+\.\d+e[+-]\d+)",
    #     "type": "float",
    # },
    "Elapsed_time": {
        # "line_identifier": "sec",
        "line_identifier": "converged, t = ",
        "regex_pattern": r"(\d+\.\d+)",
        "type": "float",
    },
    # Chebyshev-specific parameters
    "Number_of_Chebyshev_terms": {
        "line_identifier": "Number of Chebyshev polynomial terms = ",
        "regex_pattern": r"(\d+)",
        "type": "int",
    },
    "Minimum_eigenvalue": {
        "line_identifier": ", beta =",
        "regex_pattern": r"alpha = (\d+(\.\d+)?)",
        "type": "float",
    },
    "Maximum_eigenvalue": {
        "line_identifier": ", beta =",
        "regex_pattern": r", beta = (\d+(\.\d+)?)",
        "type": "float",
    },
    # KL-specific parameters
    "KL_iterations": {
        "line_identifier": "KL iters = ",
        "regex_pattern": r"(\d+)",
        "type": "int",
    },
    "KL_scaling_factor": {
        "line_identifier": "Mu =",
        "regex_pattern": r"(\d+(\.\d+)?)",
        "type": "float",
    },
    "Number_of_CG_iterations": {
        "line_identifier": "After",
        "regex_pattern": r"After (\d+)",
        "type": "int",
    },
    # This one can attributed to both Chebyshev and KL cases with different
    # meaning though
    "Solver_epsilon": {
        # "line_identifier": "Solver epsilon =",
        "line_identifier": "Outer solver epsilon =",
        "regex_pattern": r"(\d+\.\d+e-\d+)",
        "type": "float",
    },
}
