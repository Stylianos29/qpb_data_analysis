"""
Filesystem validation utilities.

This module provides functions for validating filesystem paths and
ensuring they meet specific requirements (file vs directory, existence,
accessibility). These validation functions raise descriptive ValueError
exceptions when paths are invalid, making them suitable for input
validation and error handling in data processing pipelines.

Functions:
    - is_valid_directory: Validates that a path exists and is a
      directory
    - is_valid_file: Validates that a path exists and is a file

Example:
    >>> from core.library.validation.filesystem import is_valid_directory
    >>> is_valid_directory("/path/to/data")  # Raises ValueError if invalid
    True

Note:
    All functions use pathlib.Path internally for cross-platform
    compatibility and raise ValueError with descriptive messages for
    different failure modes (non-existent paths, wrong type, permission
    errors).
"""

import os
from pathlib import Path


def is_valid_directory(directory_path):
    """Check if a given path is a valid directory, raise error if
    not."""
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


def validate_path(*path_parts, must_be_file=None, must_be_dir=None):
    """
    Validate that a path exists and return it, raising an error if
    invalid.

    This is a convenience function that automatically detects whether
    the path is a file or directory and validates accordingly. It's
    designed to make Jupyter notebook code cleaner by combining path
    joining and validation in a single call.

    Args:
        - *path_parts: Either a single full path string, or multiple
          path components that will be joined with os.path.join()
        - must_be_file (bool, optional):
            - If True, raises error if path is not a file.
            - If False, raises error if path IS a file.
            - If None (default), no file type checking.
        - must_be_dir (bool, optional):
            - If True, raises error if path is not a directory.
            - If False, raises error if path IS a directory.
            - If None (default), no directory type checking.

    Returns:
        str: The validated absolute path

    Raises:
        ValueError: If the path doesn't exist or doesn't match the
        specified type

    Examples:
        >>> # Basic usage with root + relative path
        >>> csv_file = validate_path(ROOT, "data_files/processed/data.csv")

        >>> # Using full path
        >>> csv_file = validate_path("/home/user/project/data.csv")

        >>> # Enforce file type checking
        >>> csv_file = validate_path(ROOT, "data.csv", must_be_file=True)

        >>> # Enforce directory type checking
        >>> output_dir = validate_path(ROOT, "output/tables", must_be_dir=True)

        >>> # Multiple path components
        >>> data_file = validate_path(ROOT, "data_files", "processed", "data.csv")

    Note:
        - If both must_be_file and must_be_dir are None, the function
          only checks existence
        - Setting both must_be_file=True and must_be_dir=True will raise
          ValueError
        - The function returns the absolute path for consistency
    """
    # Validate conflicting parameters
    if must_be_file is True and must_be_dir is True:
        raise ValueError("Cannot specify both must_be_file=True and must_be_dir=True")

    # Check for no arguments
    if len(path_parts) == 0:
        raise ValueError("At least one path component must be provided")

    # Join all path parts
    if len(path_parts) == 1:
        full_path = path_parts[0]
    else:
        full_path = os.path.join(*path_parts)

    # Check for empty or whitespace-only strings
    if not full_path or not str(full_path).strip():
        raise ValueError("Path cannot be empty or whitespace-only")

    # Convert to Path object for easier handling
    path = Path(full_path)

    try:
        # Check if path exists
        if not path.exists():
            raise ValueError(f"Path does not exist: {full_path}")

        # Check file type constraints if specified
        if must_be_file is True and not path.is_file():
            raise ValueError(f"Path exists but is not a file: {full_path}")

        if must_be_file is False and path.is_file():
            raise ValueError(
                f"Path exists but is a file (expected directory): {full_path}"
            )

        if must_be_dir is True and not path.is_dir():
            raise ValueError(f"Path exists but is not a directory: {full_path}")

        if must_be_dir is False and path.is_dir():
            raise ValueError(
                f"Path exists but is a directory (expected file): {full_path}"
            )

        # Return the absolute path as a string
        return str(path.absolute())

    except (OSError, PermissionError) as e:
        raise ValueError(f"Cannot access path: {full_path} ({e})")
