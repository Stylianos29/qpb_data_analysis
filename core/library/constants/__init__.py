"""
Constants Package
================

Central repository for all constants used throughout the QPB data
analysis library.

This package organizes constants into thematic modules for better
maintainability:
    - **paths**: File system paths and directory structures
    - **patterns**: Regular expressions for data extraction
    - **data_types**: Type mappings and converters
    - **labels**: UI labels and display strings
    - **visualization**: Plot styling constants
    - **domain**: QPB-specific parameter definitions

All constants are exposed at the package level for backward
compatibility, allowing both:
    >>> from library.constants import ROOT
    >>> from library.constants.paths import ROOT
"""

# Import all constants from submodules
from .paths import *
from .patterns import *
from .data_types import *
from .labels import *
from .visualization import *
from .domain import *

# Define explicit exports for clarity
__all__ = [
    # Paths
    "ROOT",
    "RAW_DATA_FILES_DIRECTORY",
    "PROCESSED_DATA_FILES_DIRECTORY",
    # Patterns
    "FILENAME_SCALAR_PATTERNS_DICTIONARY",
    "FILE_CONTENTS_SCALAR_PATTERNS_DICTIONARY",
    "FILE_CONTENTS_ARRAY_PATTERNS_DICTIONARY",
    "MAIN_PROGRAM_TYPE_MAPPING",
    # Data types
    "DTYPE_MAPPING",
    "CONVERTERS_MAPPING",
    "safe_literal_eval",
    "PARAMETERS_WITH_EXPONENTIAL_FORMAT",
    "PARAMETERS_OF_INTEGER_VALUE",
    # Labels
    "TITLE_LABELS_DICTIONARY",
    "AXES_LABELS_DICTIONARY",
    "PARAMETERS_PRINTED_LABELS_DICTIONARY",
    "FILENAME_LABELS_BY_COLUMN_NAME",
    "AXES_LABELS_BY_COLUMN_NAME",
    "LEGEND_LABELS_BY_COLUMN_NAME",
    "TITLE_LABELS_BY_COLUMN_NAME",
    "MAIN_PROGRAM_TYPE_AXES_LABEL",
    "FIT_LABEL_POSITIONS",
    "PARAMETER_LABELS",
    # Visualization
    "MARKER_STYLES",
    "DEFAULT_COLORS",
    # Domain
    "TUNABLE_PARAMETER_NAMES_LIST",
    "OUTPUT_QUANTITY_NAMES_LIST",
    "CORRELATOR_IDENTIFIERS_LIST",
]
