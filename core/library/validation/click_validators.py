"""
Flexible Click validation callbacks for qpb_data_analysis

This module provides a clean, minimal API for Click option validation
with both convenience functions and configurable validators.

Philosophy:
    - Only 4 core validator functions (input/output x file/directory)
    - Convenience layer for common use cases
    - Highly configurable for custom needs
    - Clear separation of concerns

Usage:
    # 1. Use convenience validators (recommended for most cases)
    from library.validation.click_validators import csv_file, hdf5_file
        @click.option("--input", callback=csv_file.input)
        @click.option("--output", callback=csv_file.output)
    
    # 2. Use configurable validators for custom needs
    from library.validation.click_validators import validate_input_file
    from functools import partial
    custom_validator = partial(validate_input_file, extensions=['.custom'])
"""

import os
import sys
from functools import partial

import click


# ============================================================================
# CORE VALIDATION FUNCTIONS (4 total - minimal API surface)
# ============================================================================


def validate_input_file(
    ctx, param, value, extensions=None, format_checker=None, readable=True
):
    """
    Validate input file path with optional extension and format
    checking.

    Args:
        - extensions: List of allowed extensions (e.g., ['.csv', '.h5'])
        - format_checker: Function to validate file format
        - readable: Whether file must be readable
    """
    if value is None:
        return None

    if not isinstance(value, str):
        raise click.BadParameter(f"Expected string, got {type(value).__name__}")

    # Check if it's a full path or just filename
    is_full_path = os.sep in value or ("\\" in value and os.name == "nt")

    if is_full_path:
        # Full path validation
        if not os.path.exists(value):
            raise click.BadParameter(f"File '{value}' does not exist")
        if not os.path.isfile(value):
            raise click.BadParameter(f"Path '{value}' is not a file")
        if readable and not os.access(value, os.R_OK):
            raise click.BadParameter(f"File '{value}' is not readable")

        validated_path = os.path.abspath(value)

        # Format validation for existing files
        if format_checker:
            try:
                format_checker(validated_path)
            except Exception as e:
                file_type = "file"
                if extensions:
                    file_type = f"{extensions[0].replace('.', '').upper()} file"
                raise click.BadParameter(f"Invalid {file_type}: {e}")
    else:
        # Filename only - just validate structure
        if not value.strip():
            raise click.BadParameter("Filename cannot be empty")
        validated_path = value.strip()

    # Extension validation
    if extensions:
        if not any(validated_path.lower().endswith(ext.lower()) for ext in extensions):
            ext_list = ", ".join(extensions)
            filename = (
                os.path.basename(validated_path) if is_full_path else validated_path
            )
            raise click.BadParameter(
                f"File '{filename}' must have extension: {ext_list}"
            )

    return validated_path


def validate_output_file(ctx, param, value, extensions=None, check_parent_exists=True):
    """
    Validate output file path (doesn't need to exist yet).

    Args:
        - extensions: List of allowed extensions
        - check_parent_exists: Whether parent directory must exist
    """
    if value is None:
        return None

    if not isinstance(value, str):
        raise click.BadParameter(f"Expected string, got {type(value).__name__}")

    is_full_path = os.sep in value or ("\\" in value and os.name == "nt")

    if is_full_path:
        # Check if path exists and is not a file
        if os.path.exists(value) and not os.path.isfile(value):
            raise click.BadParameter(f"Path '{value}' exists but is not a file")

        # Check parent directory
        if check_parent_exists:
            parent_dir = os.path.dirname(value)
            if parent_dir and not os.path.exists(parent_dir):
                raise click.BadParameter(
                    f"Parent directory '{parent_dir}' does not exist. "
                    f"Create it with: mkdir -p '{parent_dir}'"
                )
        validated_path = os.path.abspath(value)
    else:
        # Filename only
        if not value.strip():
            raise click.BadParameter("Filename cannot be empty")
        validated_path = value.strip()

    # Extension validation
    if extensions:
        if not any(validated_path.lower().endswith(ext.lower()) for ext in extensions):
            ext_list = ", ".join(extensions)
            filename = (
                os.path.basename(validated_path) if is_full_path else validated_path
            )
            raise click.BadParameter(
                f"File '{filename}' must have extension: {ext_list}"
            )

    return validated_path


def validate_input_directory(
    ctx, param, value, must_exist=True, readable=True, not_empty=False
):
    """
    Validate input directory path.

    Args:
        - must_exist: Whether directory must exist
        - readable: Whether directory must be readable
        - not_empty: Whether directory must contain files
    """
    if value is None:
        return None

    if not isinstance(value, str):
        raise click.BadParameter(f"Expected string, got {type(value).__name__}")

    if must_exist:
        if not os.path.exists(value):
            raise click.BadParameter(f"Directory '{value}' does not exist")
        if not os.path.isdir(value):
            raise click.BadParameter(f"Path '{value}' is not a directory")
        if readable and not os.access(value, os.R_OK):
            raise click.BadParameter(f"Directory '{value}' is not readable")
        if not_empty and not os.listdir(value):
            raise click.BadParameter(f"Directory '{value}' is empty")

    return os.path.abspath(value)


def validate_output_directory(ctx, param, value, check_parent_exists=True):
    """
    Validate output directory path (doesn't need to exist yet).

    Args:
        check_parent_exists: Whether parent directory must exist
    """
    if value is None:
        return None

    if not isinstance(value, str):
        raise click.BadParameter(f"Expected string, got {type(value).__name__}")

    # Check if path exists and is not a directory
    if os.path.exists(value) and not os.path.isdir(value):
        raise click.BadParameter(f"Path '{value}' exists but is not a directory")

    # Check parent directory
    if check_parent_exists:
        parent_dir = os.path.dirname(value)
        if parent_dir and not os.path.exists(parent_dir):
            raise click.BadParameter(
                f"Parent directory '{parent_dir}' does not exist. "
                f"Create it with: mkdir -p '{parent_dir}'"
            )

    return os.path.abspath(value)


# ============================================================================
# FORMAT CHECKERS (reusable validation functions)
# ============================================================================


def check_hdf5_format(filepath):
    """Validate HDF5 file format"""
    try:
        import h5py

        with h5py.File(filepath, "r") as f:
            pass  # Just test if it opens
    except Exception as e:
        raise ValueError(f"Invalid HDF5 format: {e}")


def check_csv_format(filepath):
    """Validate CSV file format (basic check)"""
    try:
        import pandas as pd

        pd.read_csv(filepath, nrows=1)  # Just read first row
    except Exception as e:
        raise ValueError(f"Invalid CSV format: {e}")


# ============================================================================
# CONVENIENCE LAYER (90% of use cases)
# ============================================================================


class FileValidator:
    """
    Factory for creating file validators with consistent behavior

    Example:
        csv = FileValidator(['.csv'], check_csv_format)
        @click.option("--file", callback=csv.input)
    """

    def __init__(self, extensions, format_checker=None):
        self.extensions = extensions
        self.format_checker = format_checker

    @property
    def input(self):
        """Input file validator"""
        return partial(
            validate_input_file,
            extensions=self.extensions,
            format_checker=self.format_checker,
        )

    @property
    def output(self):
        """Output file validator"""
        return partial(validate_output_file, extensions=self.extensions)


class DirectoryValidator:
    """Factory for creating directory validators with clear semantics"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    def must_exist(self):
        """Directory must already exist (for input directories)"""
        return partial(validate_input_directory, **self.kwargs)

    @property
    def can_create(self):
        """Directory can be created if it doesn't exist (for output directories)"""
        return partial(validate_output_directory)


# ============================================================================
# PRE-CONFIGURED VALIDATORS (for your specific use cases)
# ============================================================================

# File validators
csv_file = FileValidator([".csv"], check_csv_format)
hdf5_file = FileValidator([".hdf5", ".h5"], check_hdf5_format)

# Directory validators
data_directory = DirectoryValidator(not_empty=True)
directory = DirectoryValidator()  # General purpose


# ============================================================================
# Specialized validators for specific use cases
# ============================================================================


def validate_log_filename(ctx, param, value):
    """Generate default log filename or validate provided one"""
    # If no log filename is provided, generate a default name
    if value is None:
        # Get the name of the script being executed (entry point)
        script_name = os.path.basename(sys.argv[0])
        return script_name.replace(".py", "_python_script.log")

    # Validate the provided log filename (e.g., ensure it's a valid
    # string)
    if not value.strip():  # Ensure it's not empty or just spaces
        raise click.BadParameter("Log filename cannot be an empty string.")

    if not value.lower().endswith(".log"):
        raise click.BadParameter(
            f"The file name '{value}' is invalid. Current script's log file "
            "names must end with '.log'."
        )

    return value
