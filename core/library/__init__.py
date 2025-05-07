"""
Core library package for qpb data analysis.

This package provides utilities for data processing, file system operations,
and data validation.
"""

from .constants import ROOT, RAW_DATA_FILES_DIRECTORY, PROCESSED_DATA_FILES_DIRECTORY
from .data_processing import load_csv, DataFrameAnalyzer, TableGenerator
from .data_files_checks import get_yes_or_no_user_response
from .filesystem_utilities import (
    is_valid_file,
    is_valid_directory,
    validate_file,
    validate_input_directory,
    validate_output_directory,
    validate_input_script_log_filename,
)
from .plotting import DataPlotter, HDF5Plotter, EnhancedHDF5Analyzer
from .specialized import generate_config_labels

__all__ = [
    'ROOT',
    'RAW_DATA_FILES_DIRECTORY',
    'PROCESSED_DATA_FILES_DIRECTORY',
    'is_valid_file',
    'is_valid_directory',
    'load_csv',
    'generate_config_labels',
    'DataFrameAnalyzer',
    'TableGenerator',
    'get_yes_or_no_user_response',
    'validate_file',
    'validate_input_directory',
    'validate_output_directory',
    'validate_input_script_log_filename',
    'DataPlotter',
    'HDF5Plotter',
    'EnhancedHDF5Analyzer',
]
