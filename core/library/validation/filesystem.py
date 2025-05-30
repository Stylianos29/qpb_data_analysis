"""
Filesystem validation utilities.

This module provides functions for validating filesystem paths and ensuring they
meet specific requirements (file vs directory, existence, accessibility). These
validation functions raise descriptive ValueError exceptions when paths are
invalid, making them suitable for input validation and error handling in data
processing pipelines.

Functions:
    - is_valid_directory: Validates that a path exists and is a directory
    - is_valid_file: Validates that a path exists and is a file

Example:
    >>> from core.library.validation.filesystem import is_valid_directory
    >>> is_valid_directory("/path/to/data")  # Raises ValueError if invalid
    True

Note:
    All functions use pathlib.Path internally for cross-platform compatibility
    and raise ValueError with descriptive messages for different failure modes
    (non-existent paths, wrong type, permission errors).
"""

from pathlib import Path


def is_valid_directory(directory_path):
    """Check if a given path is a valid directory, raise error if not."""
    try:
        path = Path(directory_path)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        if not path.is_dir():
            raise ValueError(f"Path exists but is not a directory: {directory_path}")
        return True
    except (OSError, PermissionError) as e:
        raise ValueError(f"Cannot access directory path: {directory_path} ({e})")


def is_valid_file(file_path):
    """Check if a given path is a valid file, raise error if not."""
    try:
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        if not path.is_file():
            raise ValueError(f"Path exists but is not a file: {file_path}")
        return True
    except (OSError, PermissionError) as e:
        raise ValueError(f"Cannot access file path: {file_path} ({e})")
