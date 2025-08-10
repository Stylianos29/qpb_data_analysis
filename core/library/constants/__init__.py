"""
Constants package for the QPB data analysis library.

This package organizes constants into thematic modules while maintaining
backward compatibility by exposing all constants at the package level.
"""

# Import all constants from submodules
from .paths import *
from .patterns import *
from .data_types import *
from .labels import *
from .visualization import *
from .domain import *

# Define what gets exported when someone does "from library.constants import *"
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
