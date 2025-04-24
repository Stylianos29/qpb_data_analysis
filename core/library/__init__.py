"""
Core library package for qpb data analysis.

This package provides utilities for data processing, file system operations,
and data validation.
"""

from .constants import ROOT, RAW_DATA_FILES_DIRECTORY, PROCESSED_DATA_FILES_DIRECTORY
from .data_processing import load_csv, DataFrameAnalyzer, TableGenerator
from .data_files_checks import get_yes_or_no_user_response
from .filesystem_utilities import (
    validate_input_directory,
    validate_input_script_log_filename,
)

__all__ = [
    'ROOT',
    'RAW_DATA_FILES_DIRECTORY',
    'PROCESSED_DATA_FILES_DIRECTORY',
    'load_csv',
    'DataFrameAnalyzer',
    'TableGenerator',
    'get_yes_or_no_user_response',
    'validate_input_directory',
    'validate_input_script_log_filename',
]
