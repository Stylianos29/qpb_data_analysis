"""
Display labels and formatting for plots and output.

This module provides standardized label mappings for visualization and
file generation throughout the QPB data analysis library. All
dictionaries follow a consistent naming convention: *_BY_COLUMN_NAME.

Dictionaries:
    - TITLE_LABELS_BY_COLUMN_NAME: Labels for plot titles (short,
      descriptive)
    - FILENAME_LABELS_BY_COLUMN_NAME: Abbreviated labels for filenames
    - AXES_LABELS_BY_COLUMN_NAME: LaTeX-formatted labels for plot axes
    - LEGEND_LABELS_BY_COLUMN_NAME: Labels for legend entries and titles
"""

# =============================================================================
# TITLE LABELS
# =============================================================================
# Short, human-readable labels for plot titles.
# Used by PlotTitleBuilder for constructing informative plot titles.

TITLE_LABELS_BY_COLUMN_NAME = {
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
    "Zolotarev_order": "n",
    "Neuberger_order": "n",
    "Rational_order": "n",
    "KL_scaling_factor": "$\\mu$",
    "Number_of_vectors": "# of random vecs",
    "Kappa_value": "$\\kappa$",
    # Output quantities
    "Condition_number": "$\\kappa$",
    "Minimum_eigenvalue_squared": "$\\lambda_{min}^2$",
    "Maximum_eigenvalue_squared": "$\\lambda_{max}^2$",
    "Number_of_gauge_configurations": "# of configs",
    "Threads_per_process": "$n_{\\text{OMP}}$",
}


# =============================================================================
# FILENAME LABELS
# =============================================================================
# Short abbreviations for constructing filenames.
# Used by PlotFilenameBuilder for generating plot file names.

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
    "Zolotarev_order": "n",
    "Neuberger_order": "n",
    "Rational_order": "n",
    "KL_scaling_factor": "mu",
    "MPI_geometry": "MPI",
    "Threads_per_process": "OMP",
}


# =============================================================================
# AXES LABELS
# =============================================================================
# LaTeX-formatted labels for plot axes.
# Used by PlotLayoutManager and PlotStyleManager for axis labeling.

AXES_LABELS_BY_COLUMN_NAME = {
    # ---------------------------------------------------------------------------
    # Critical mass and bare mass
    # ---------------------------------------------------------------------------
    "Absolute_critical_bare_mass": "|a$m^{\\text{critical}}_{\\text{bare}}$|",
    "Critical_bare_mass": "a$m^{\\text{critical}}_{\\text{bare}}$",
    "Bare_mass": "a$m$",
    # ---------------------------------------------------------------------------
    # Computational cost metrics
    # ---------------------------------------------------------------------------
    "Average_calculation_time_per_spinor_per_configuration": (
        "Average wall-clock time (per spinor per config) (s)"
    ),
    "Average_core_hours_per_spinor": "Computational cost (core-hours/spinor)",
    "Average_core_hours_per_spinor_per_configuration": (
        "Computational cost (core-hours/spinor/config)"
    ),
    "Average_core_hours_per_vector": "Computational cost (core-hours/vector)",
    "Core_hours_for_constant_PCAC_mass": (
        "Computational cost (core-hours/spinor/config)"
    ),
    "Average_wall_clock_time_per_spinor": "Average wall-clock time per spinor (s)",
    "Average_wall_clock_time_per_vector": "Average wall-clock time per vector (s)",
    "Total_calculation_time": "Total wall-clock time (s)",
    "Total_overhead_time": "Overhead time (s)",
    # ---------------------------------------------------------------------------
    # Matrix-vector multiplication counts
    # ---------------------------------------------------------------------------
    "Average_number_of_MV_multiplications_per_spinor_per_configuration": (
        "Average number of MV multiplications (per spinor per config)"
    ),
    "Average_number_of_MV_multiplications_per_spinor": (
        "Average number of MV multiplications (per spinor)"
    ),
    "Average_number_of_MV_multiplications_per_vector": (
        "Average number of MV muls per vector"
    ),
    "Number_of_MV_multiplications_for_constant_bare_mass": (
        "Number of MV muls (per spinor per config)"
    ),
    "Number_of_MV_multiplications_for_constant_PCAC_mass": (
        "Number of MV muls (per spinor per config)"
    ),
    # ---------------------------------------------------------------------------
    # Iteration counts
    # ---------------------------------------------------------------------------
    "Average_number_of_CG_iterations_per_spinor": (
        "Average # of CG iterations (per spinor)"
    ),
    "Total_number_of_Lanczos_iterations": "Total Lanczos iterations",
    # ---------------------------------------------------------------------------
    # PCAC mass observables
    # ---------------------------------------------------------------------------
    "PCAC_mass_estimate": "a$m_{PCAC}$",
    "Plateau_PCAC_mass": "a$m_{PCAC}$",
    "Jackknife_average_of_PCAC_mass_correlator": "a$m_{\\text{PCAC}}$(t)",
    "Jackknife_average_of_PCAC_mass_correlator_mean_values": "a$m_{\\text{PCAC}}$(t)",
    "PCAC_mass_jackknife_samples": "a$m_{\\text{PCAC}}$(t)",
    # ---------------------------------------------------------------------------
    # Pion effective mass observables
    # ---------------------------------------------------------------------------
    "Pion_effective_mass_estimate": "a$m_{\\pi}$",
    "Pion_effective_mass_estimate_squared": "$a^2 m^2_{\\pi}$",
    "Jackknife_average_of_pion_effective_mass_correlator": "a$m_{\\text{eff.}}(t)$",
    "pion_effective_mass_jackknife_samples": "a$m_{\\text{eff.}}(t)$",
    # ---------------------------------------------------------------------------
    # Correlator observables
    # ---------------------------------------------------------------------------
    "Jackknife_average_of_g5_g5_correlator": ("$C_{\\gamma_5\\text{-}\\gamma_5}$(t)"),
    "Jackknife_average_of_g5_g5_correlator_mean_values": (
        "$C_{\\gamma_5\\text{-}\\gamma_5}$(t)"
    ),
    "Jackknife_average_of_g4g5_g5_correlator": (
        "$C_{\\gamma_4\\gamma_5\\text{-}\\gamma_5}$(t)"
    ),
    "Jackknife_average_of_g4g5_g5_correlator_mean_values": (
        "$C_{\\gamma_4\\gamma_5\\text{-}\\gamma_5}$(t)"
    ),
    "Jackknife_average_of_g4g5_g5_derivative_correlator": (
        "$\\partial_t C_{\\gamma_4\\gamma_5\\text{-}\\gamma_5}'(t)$"
    ),
    "Jackknife_average_of_g4g5_g5_derivative_correlator_mean_values": (
        "$\\partial_t C_{\\gamma_4\\gamma_5\\text{-}\\gamma_5}'(t)$"
    ),
    # ---------------------------------------------------------------------------
    # Overlap operator quality metrics
    # ---------------------------------------------------------------------------
    "Average_sign_squared_violation_values": (
        "||(sgn$^2$(X) - I)$\\eta$||$^2$ / ||$\\eta$||$^2$"
    ),
    "Sign_squared_violation_with_no_error": (
        "||(sgn$^2$(X) - I)$\\eta$||$^2$ / ||$\\eta$||$^2$"
    ),
    "Average_sign_squared_values": "||sgn$^2$(X)||$^2$",
    "Sign_squared_value_with_no_error": "||sgn$^2$(X)||$^2$",
    "Average_normality_values": (
        "||$aD_{ov.}^{\\dagger} aD_{ov.} - aD_{ov.} aD_{ov.}^{\\dagger}$||$^2$"
    ),
    "Normality_with_no_error": (
        "||$aD_{ov.}^{\\dagger} aD_{ov.} - aD_{ov.} aD_{ov.}^{\\dagger}$||$^2$"
    ),
    "Average_ginsparg_wilson_relation_values": (
        "||$\\gamma_5 aD_{ov.} + aD_{ov.} \\gamma_5 "
        "- aD_{ov.} \\gamma_5 aD_{ov.}$||$^2$"
    ),
    "Ginsparg_wilson_relation_with_no_error": (
        "||$\\gamma_5 aD_{ov.} + aD_{ov.} \\gamma_5 "
        "- aD_{ov.} \\gamma_5 aD_{ov.}$||$^2$"
    ),
    # ---------------------------------------------------------------------------
    # Time and order parameters
    # ---------------------------------------------------------------------------
    "time_index": "$t/a$",
    "KL_diagonal_order": "n",
    "Zolotarev_order": "n",
    "Neuberger_order": "n",
    "Rational_order": "n",
    "Number_of_Chebyshev_terms": "N",
    "Number_of_cores": "Number of cores",
}


# =============================================================================
# LEGEND LABELS
# =============================================================================
# Labels for legend entries and titles.
# Used by PlotStyleManager for legend configuration.

LEGEND_LABELS_BY_COLUMN_NAME = {
    "Condition_number": "$\\kappa_{\\mathbb{X}^2}=$",
    "Configuration_label": "Config. labels:",
    "KL_diagonal_order": "n=",
    "Kernel_operator_type": "Kernel:",
    "Bare_mass": "a$m_{\\text{bare}}=$",
    "Threads_per_process": "$n_{\\text{OMP}}=$",
}
