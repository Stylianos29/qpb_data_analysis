"""Domain-specific constants for QPB data analysis."""

TUNABLE_PARAMETER_NAMES_LIST = [
    # =========================================================================
    # Gauge Configuration Parameters
    # =========================================================================
    # Parameters related to gauge field configuration and smearing
    "APE_alpha",  # APE smearing parameter alpha
    "APE_iterations",  # Number of APE smearing iterations
    "Configuration_label",  # Gauge field configuration identifier
    "QCD_beta_value",  # QCD coupling parameter beta
    # =========================================================================
    # Overlap Operator Method Parameters
    # =========================================================================
    # Parameters defining the overlap operator and its kernel
    "Kappa_value",  # Hopping parameter kappa
    "Clover_coefficient",  # Clover term coefficient (c_SW)
    "Rho_value",  # Rho parameter for overlap operators
    "Kernel_operator_type",  # Type of Dirac kernel (Wilson, Brillouin, etc.)
    "Overlap_operator_method",  # Overlap operator approximation method
    "Bare_mass",  # Bare quark mass
    # =========================================================================
    # Rational Approximation Methods
    # =========================================================================
    # Order parameters for different rational approximation methods
    "Rational_order",  # Generic rational approximation order
    "KL_diagonal_order",  # Order for KL method diagonal
    "Zolotarev_order",  # Order for Zolotarev rational approximation
    # Scaling factor for rational approximations
    "KL_scaling_factor",  # Scaling factor mu for KL/Neuberger/Zolotarev
    # =========================================================================
    # Chebyshev Approximation Method Parameters
    # =========================================================================
    # Parameters specific to Chebyshev polynomial approximation
    "Number_of_Chebyshev_terms",  # Number of terms in Chebyshev expansion
    "Lanczos_epsilon",  # Lanczos convergence tolerance
    "Delta_Max",  # Maximum eigenvalue modification delta
    "Delta_Min",  # Minimum eigenvalue modification delta
    # =========================================================================
    # Solver Parameters
    # =========================================================================
    # Convergence tolerances and iteration limits for various solvers
    "CG_epsilon",  # Conjugate gradient solver tolerance
    "Solver_epsilon",  # Generic/outer solver tolerance
    "MSCG_epsilon",  # Multi-shift CG solver tolerance (inner)
    "Outer_solver_epsilon",  # Outer solver tolerance (for nested solvers)
    "Inner_solver_epsilon",  # Inner solver tolerance (for nested solvers)
    # NOTE: Maximum iterations for solvers are absent due to variability
    # that does not affect results
    # =========================================================================
    # Stochastic Estimation Parameters
    # =========================================================================
    # Parameters for stochastic trace estimation methods
    "Number_of_spinors",  # Number of spinor fields
    "Number_of_vectors",  # Number of random vectors for estimation
    # =========================================================================
    # Computational/Technical Parameters
    # =========================================================================
    # System and execution configuration parameters
    "MPI_geometry",  # MPI process geometry/topology
    "Threads_per_process",  # Number of OpenMP threads per MPI process
    # =========================================================================
    # Metadata and Identification
    # =========================================================================
    # Program metadata and additional information
    "Main_program_type",  # Main program type (Bare, Chebyshev, KL, etc.)
    "Additional_text",  # Additional text/metadata field
]


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
