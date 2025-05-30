"""
QPB Data Analysis Library
========================

A comprehensive toolkit for processing, analyzing, and visualizing qpb data

Main Components:
---------------
- Data Analysis: Tools for manipulating and analyzing pandas DataFrames
- Visualization: Plotting utilities for data visualization
- I/O: Utilities for handling various file formats and filesystem operations
TODO: Update docustring

For more information, see the documentation at: docs/api
"""

# Import from data module
from .data import DataFrameAnalyzer, TableGenerator, load_csv

# Import from visualization module
from .visualization import DataPlotter

# Import constants
from .constants import ROOT, RAW_DATA_FILES_DIRECTORY, PROCESSED_DATA_FILES_DIRECTORY

# Import input validation functions
from .filesystem_utilities import (
    validate_file,
    # validate_output_directory,
    # validate_input_directory,
    validate_input_script_log_filename,
    # is_valid_file,
    # is_valid_directory
)

# Import
from .data_files_checks import get_yes_or_no_user_response

# Import
from .specialized import generate_config_labels

from .validation import (
    validate_input_directory,
    validate_output_directory,
    validate_input_file,
    validate_output_file,
)

# Import 
from .validation.filesystem import is_valid_directory, is_valid_file

from .filesystem_utilities import LoggingWrapper


# Define public API
__all__ = [
    # Data components
    "DataFrameAnalyzer",
    # "HDF5Analyzer",
    "TableGenerator",
    "load_csv",
    # Visualization components
    "DataPlotter",
    # "HDF5Plotter",
    # Important constants
    "ROOT",
    "RAW_DATA_FILES_DIRECTORY",
    "PROCESSED_DATA_FILES_DIRECTORY",
    # Input validation functions
    "validate_input_directory",
    "validate_file",
    "validate_output_directory",
    "validate_input_script_log_filename",
    # Filesystem validation utilities
    "is_valid_file",
    "is_valid_directory",
    # User input
    "get_yes_or_no_user_response",
    # Specialized functions
    "generate_config_labels",
    # Click Validators
    "validate_input_directory",
    "validate_output_directory",
    "validate_input_file",
    "validate_output_file",
    # Logging system
    "LoggingWrapper",
]
