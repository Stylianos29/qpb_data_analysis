"""
Filename parameter extraction utilities.

This module provides functionality for extracting parameters from filenames
using regex patterns. It handles file extension removal, pattern matching, type
conversion, and special parameter transformations.
"""

import os
import re
from typing import Dict, List, Optional, Any, Tuple
import logging

import library.constants as constants
from ._shared_parsing_utils import SUPPORTED_FILE_EXTENSIONS, _convert_to_type


def _remove_file_extension(filename: str) -> str:
    """
    Remove known file extension from filename.

    Args:
        filename: Original filename with extension

    Returns:
        Filename without extension if extension is recognized, otherwise
        original filename

    Example:
        >>> _remove_file_extension("data.txt")
        "data"
        >>> _remove_file_extension("data.unknown")
        "data.unknown"
    """
    base_filename, ext = os.path.splitext(filename)
    if ext in SUPPORTED_FILE_EXTENSIONS:
        return base_filename
    return filename


def _extract_parameters_with_regex(
    filename: str, logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Extract parameters from filename using regex patterns from constants.

    Args:
        - filename: The filename to process (without extension)
        - logger: Optional logger for warnings

    Returns:
        Tuple of (extracted_parameters_dict, matched_segments_list)

    Example:
        >>> _extract_parameters_with_regex("test_beta2p5_config001")
        ({'QCD_beta_value': 2.5, 'Configuration_label': '001'}, ['beta2p5', 'config001'])
    """
    extracted_values = {}
    matched_segments = []

    # Apply each regex pattern to the filename
    for (
        param_name,
        param_info,
    ) in constants.FILENAME_SCALAR_PATTERNS_DICTIONARY.items():
        regex_pattern = param_info["pattern"]
        expected_type = param_info["type"]

        match = re.search(regex_pattern, filename)
        if match:
            raw_value = match.group(param_name)
            matched_segments.append(match.group(0))

            # Convert to expected type
            converted_value = _convert_to_type(
                raw_value, expected_type, param_name, logger
            )
            if converted_value is not None:
                extracted_values[param_name] = converted_value

    return extracted_values, matched_segments


def _extract_unmatched_segments(
    filename: str, matched_segments: List[str]
) -> List[str]:
    """
    Extract segments of filename that weren't matched by any regex pattern.

    Args:
        - filename: The original filename (without extension)
        - matched_segments: List of segments that were matched by regex patterns

    Returns:
        List of unmatched segments, cleaned of underscores

    Example:
        >>> _extract_unmatched_segments(
        >>>     "test_beta2p5_extra_config001", ["beta2p5", "config001"]
        >>>     )
        ["test", "extra"]
    """
    if not matched_segments:
        return []

    # Create combined pattern to match all matched segments
    matched_segments_pattern = "|".join(map(re.escape, matched_segments))

    # Split filename by matched patterns and filter out empty strings
    unmatched_segments = [
        segment.strip("_")
        for segment in re.split(matched_segments_pattern, filename)
        if segment.strip("_")
    ]

    return unmatched_segments


def _apply_special_transformations(extracted_values: Dict[str, Any]) -> None:
    """
    Apply special transformations to specific parameters.

    Note: This function modifies the dictionary in-place.
    Currently handles MPI_geometry formatting.

    Args:
        extracted_values: Dictionary of extracted parameters to transform

    Example:
        >>> params = {"MPI_geometry": "123"}
        >>> _apply_special_transformations(params)
        >>> params["MPI_geometry"]
        "(1, 1, 2, 3)"
    """
    # Special treatment for "MPI_geometry"
    if "MPI_geometry" in extracted_values:
        raw_cores = extracted_values["MPI_geometry"]
        # Format as string representation of tuple
        extracted_values["MPI_geometry"] = f"(1, {', '.join(raw_cores)})"


def extract_scalar_parameters_from_filename(
    filename: str, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Extract scalar parameters from a filename based on predefined regex
    patterns.

    This function parses filenames to extract parameter values using regex
    patterns defined in constants.FILENAME_SCALAR_PATTERNS_DICTIONARY. It
    handles type conversion, special formatting, and captures any unmatched text
    segments.

    The function processes filenames in the following steps: 1. Remove
    recognized file extensions 2. Apply regex patterns to extract known
    parameters 3. Identify unmatched segments for additional context 4. Apply
    special transformations (e.g., MPI_geometry formatting)

    Args:
        - filename: The filename to process
        - logger: Optional logger instance for warnings

    Returns:
        - Dictionary containing extracted parameter names as keys and their
          values.
        - May include an "Additional_text" key with unmatched filename segments.

    Raises:
        - ValueError: If filename is empty or None
        - TypeError: If filename is not a string

    Example:
        >>> extract_scalar_parameters_from_filename("test_beta2p3_config001.dat")
        {'QCD_beta_value': 2.3, 'Configuration_label': '001', 'Additional_text': ['test']}

        >>> extract_scalar_parameters_from_filename("Chebyshev_cores123.log")
        {'Overlap_operator_method': 'Chebyshev', 'MPI_geometry': '(1, 1, 2, 3)'}
    """
    # Input validation
    if not filename:
        raise ValueError("Filename cannot be empty or None")
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string")

    # Remove file extension
    clean_filename = _remove_file_extension(filename)

    # Extract parameters using regex patterns
    extracted_values, matched_segments = _extract_parameters_with_regex(
        clean_filename, logger
    )

    # Extract unmatched segments
    unmatched_segments = _extract_unmatched_segments(clean_filename, matched_segments)
    if unmatched_segments:
        extracted_values["Additional_text"] = unmatched_segments

    # Apply special transformations
    _apply_special_transformations(extracted_values)

    return extracted_values
