"""
Private shared utilities for parameter parsing operations.

This module contains common functionality used by both filename_parser.py and
file_contents_parser.py. It is marked as private (underscore prefix) and should
not be part of the public API.
"""

from typing import Optional, Any
import logging

# Constants used across parsing modules
SUPPORTED_FILE_EXTENSIONS = [
    ".txt",
    ".log",
    ".dat",
    ".bin",
    ".jpg",
    ".jpeg",
    ".png",
    ".csv",
    ".xml",
    ".json",
    ".gz",
]


def _normalize_numeric_value(raw_value: str, expected_type: type) -> str:
    """
    Normalize numeric values by replacing 'p' with '.' for float/int types.

    This handles the common pattern in filenames where decimal points are
    represented as 'p' (e.g., "2p5" instead of "2.5").

    Args:
        - raw_value: The raw string value extracted from filename or file
          contents
        - expected_type: The expected Python type for this value

    Returns:
        Normalized string value with 'p' replaced by '.' if applicable

    Example:
        >>> _normalize_numeric_value("2p5", float)
        "2.5"
        >>> _normalize_numeric_value("hello", str)
        "hello"
    """
    if "p" in raw_value and expected_type in {float, int}:
        return raw_value.replace("p", ".")
    return raw_value


def _convert_to_type(
    raw_value: str,
    expected_type: type,
    param_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[Any]:
    """
    Convert raw string value to expected type with error handling.

    This function handles the common pattern of type conversion with numeric
    normalization and proper error logging.

    Args:
        - raw_value: The raw string value to convert
        - expected_type: The target Python type (int, float, str, etc.)
        - param_name: Parameter name for error reporting
        - logger: Optional logger for warnings

    Returns:
        Converted value of expected_type, or None if conversion fails

    Example:
        >>> _convert_to_type("2p5", float, "beta_value")
        2.5
        >>> _convert_to_type("invalid", int, "count")  # logs warning
        None
    """
    # Normalize numeric values first
    normalized_value = _normalize_numeric_value(raw_value, expected_type)

    try:
        return expected_type(normalized_value)
    except ValueError as e:
        if logger:
            logger.warning(
                f"Could not convert '{raw_value}' to {expected_type.__name__} "
                f"for parameter '{param_name}': {e}"
            )
        return None
    except TypeError as e:
        if logger:
            logger.warning(
                f"Type error converting '{raw_value}' to {expected_type.__name__} "
                f"for parameter '{param_name}': {e}"
            )
        return None
