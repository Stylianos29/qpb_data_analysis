"""
Operator type constants for QPB data analysis.

This module defines the valid values for overlap operator methods and
kernel operator types used throughout the analysis pipeline.
"""

# Valid matrix sign function methods used in overlap operators
OVERLAP_OPERATOR_METHODS = frozenset(
    ["Bare", "Chebyshev", "KL", "Neuberger", "Zolotarev"]
)

# Valid Dirac operator types used as overlap provcedure kernels
KERNEL_OPERATOR_TYPES = frozenset(
    [
        "Brillouin",
        "Wilson",  # Note: "Standard" appears in QPB log files instead of "Wilson"
    ]
)
