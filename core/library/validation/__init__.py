"""
Validation Module
================

Input validation and verification utilities for robust data processing.

This module provides comprehensive validation capabilities:
    - Click-based CLI validators for command-line interfaces
    - Filesystem path validation
    - Data format verification
    - Parameter range checking

Components
----------
Click Validators:
    - File and directory validators with format checking
    - Customizable validation rules
    - User-friendly error messages

Filesystem Validators:
    - Path existence and accessibility checks
    - File type verification
    - Permission validation
"""

# Click validators for CLI
from .click_validators import (
    validate_input_directory,
    validate_output_directory,
    validate_input_file,
    validate_output_file,
    validate_log_filename,
    # Pre-configured validators
    csv_file,
    hdf5_file,
    directory,
    data_directory,
)

# Filesystem validation utilities
from .filesystem import (
    is_valid_directory,
    is_valid_file,
    validate_path,
)

__all__ = [
    # Click validators
    "validate_input_directory",
    "validate_output_directory",
    "validate_input_file",
    "validate_output_file",
    "validate_log_filename",
    # Pre-configured validators
    "csv_file",
    "hdf5_file",
    "directory",
    "data_directory",
    # Filesystem validators
    "is_valid_directory",
    "is_valid_file",
    "validate_path",
]
