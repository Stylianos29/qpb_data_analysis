"""Display labels and formatting for plots and output."""

TITLE_LABELS_DICTIONARY = {
    # Tunable parameters
    "APE_iterations": "APE iters",
    "APE_alpha": "$\\alpha_{APE}$",
    "QCD_beta_value": "$\\beta$",
    "Configuration_label": "config",
    "Rho_value": "$\\rho$",
    "Bare_mass": "am",
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
    "Threads_per_process": "$n_{\\text{OMP}}$"
}

AXES_LABELS_DICTIONARY = {
    # Output quantity
    "Average_number_of_MV_multiplications_per_spinor_per_configuration": "Average # of MV multiplications (per spinor per config)",
    "Average_number_of_MV_multiplications_per_vector": "Average # of MV multiplications per vector",
    "Number_of_MV_multiplications_for_constant_PCAC_mass": "# of MV multiplications (per spinor per config)",
    "Number_of_MV_multiplications_for_constant_bare_mass": "# of MV multiplications (per spinor per config)",
    "Average_calculation_time_per_spinor_per_configuration": "Average wall-clock time (per spinor per config) (s)",
    "Average_core_hours_per_vector": "Average cost of calculation per vector (core-hours)",
    "Average_core_hours_per_spinor": "Average cost of calculation per spinor (core-hours)",
    "Average_core_hours_per_spinor_per_configuration": "Average cost (per spinor per config) (core-hours)",
    "Adjusted_average_core_hours_per_spinor_per_configuration": "Average cost (per spinor per config) (core-hours)",
    "Core_hours_for_constant_PCAC_mass": "Average cost (per spinor per config) (core-hours)",
    "Total_overhead_time": "Overhead (s)",
    "Total_calculation_time": "Total wall-clock time (s)",
    "Average_wall_clock_time_per_vector": "Average wall-clock time per vector (s)",
    "Average_wall_clock_time_per_spinor": "Average wall-clock time per spinor (s)",
    "Total_number_of_Lanczos_iterations": "Total Lanczos iterations",
    # "Average_calculation_result": "||sgn$^2$(X) - I||$^2$",
    "PCAC_mass_estimate": "a$m_{PCAC}$",
    "Pion_effective_mass_estimate": "a$m_{eff.}$",
    "Critical_bare_mass": "a$m^{{critical}}_{{bare}}$",
    "Number_of_cores": "Number of cores",
}


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
    "Absolute_critical_bare_mass": "|a$m^{\\text{critical}}_{\\text{bare}}$|",
    "Adjusted_average_core_hours_per_spinor_per_configuration": "Computational cost (core-hours/spinor)",
    "Average_calculation_time_per_spinor_per_configuration": "Average wall-clock time (per spinor per config) (s)",
    "Average_core_hours_per_spinor": "Computational cost (core-hours/spinor)",
    "Average_core_hours_per_spinor_per_configuration": "Computational cost (core-hours/spinor/config)",
    "Average_core_hours_per_vector": "Computational cost (core-hours/vector)",
    "Average_number_of_MV_multiplications_per_spinor_per_configuration": "Average number of MV multiplications (per spinor per config)",
    "Average_number_of_MV_multiplications_per_spinor": "Average number of MV multiplications (per spinor)",
    "Average_number_of_MV_multiplications_per_vector": "Average number of MV muls per vector",
    "Average_sign_squared_violation_values": "||(sgn$^2$(X) - I)$\\eta$||$^2$ / ||$\\eta$||$^2$",
    "Average_wall_clock_time_per_spinor": "Average wall-clock time per spinor (s)",
    "Average_wall_clock_time_per_vector": "Average wall-clock time per vector (s)",
    "Bare_mass": "a$m$",
    "Core_hours_for_constant_PCAC_mass": "Computational cost (core-hours/spinor/config)",
    "Critical_bare_mass": "a$m^{\\text{critical}}_{\\text{bare}}$",
    "Jackknife_average_of_PCAC_mass_correlator": "a$m_{\\text{PCAC}}$(t)",
    "Jackknife_average_of_PCAC_mass_correlator_mean_values": "a$m_{\\text{PCAC}}$(t)",
    "Jackknife_average_of_pion_effective_mass_correlator": "a$m_{\\text{eff.}}(t)$",
    "KL_diagonal_order": "n",
    "Number_of_Chebyshev_terms": "N",
    "Number_of_cores": "Number of cores",
    "Number_of_MV_multiplications_for_constant_bare_mass": "Number of MV muls (per spinor per config)",
    "Number_of_MV_multiplications_for_constant_PCAC_mass": "Number of MV muls (per spinor per config)",
    "PCAC_mass_estimate": "a$m_{PCAC}$",
    "Pion_effective_mass_estimate": "a$m_{\\pi}$",
    "Pion_effective_mass_estimate_squared": "$a^2 m^2_{\\pi}$",
    "time_index": "$t/a$",
    "Total_calculation_time": "Total wall-clock time (s)",
    "Total_number_of_Lanczos_iterations": "Total Lanczos iterations",
    "Total_overhead_time": "Overhead time (s)",
}

LEGEND_LABELS_BY_COLUMN_NAME = {
    "Condition_number": "$\\kappa_{\\mathbb{X}^2}=$",
    "Configuration_label": "Config. labels:",
    "KL_diagonal_order": "n=",
    "Kernel_operator_type": "Kernel:",
    "Bare_mass": "a$m_{\\text{bare}}=$",
    "Threads_per_process": "$n_{\\text{OMP}}=$",
}

TITLE_LABELS_BY_COLUMN_NAME = TITLE_LABELS_DICTIONARY.copy()


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

FIT_LABEL_POSITIONS = {
    "top left": ((0.05, 0.95), ("left", "top")),
    "top right": ((0.95, 0.95), ("right", "top")),
    "bottom left": ((0.05, 0.05), ("left", "bottom")),
    "bottom right": ((0.95, 0.05), ("right", "bottom")),
    "center": ((0.5, 0.5), ("center", "center")),
}


# Combine with title labels
AXES_LABELS_DICTIONARY.update(TITLE_LABELS_DICTIONARY)


AXES_LABELS_DICTIONARY.update(MAIN_PROGRAM_TYPE_AXES_LABEL)

# TODO: Maybe I need separate lists for qpb input parameters and qpb output
# values

# TODO: I need a list for output values labels for plot axes


# TITLE_LABELS_DICTIONARY.update(
#     {
#         "Overlap_operator_method": "",
#         "Kernel_operator_type": "",
#     }
# )
