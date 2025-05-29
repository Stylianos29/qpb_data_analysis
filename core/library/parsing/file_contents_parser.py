"""
File contents parameter extraction utilities.

This module provides functionality for extracting both scalar and array
parameters from file contents using regex patterns. It handles line-by-line
processing, pattern matching, type conversion, and NumPy array creation.
"""

import re
from typing import Dict, List, Optional, Any
import logging

import numpy as np
import library.constants as constants
from ._shared_parsing_utils import _convert_to_type


def _extract_main_program_type(file_contents_list: List[str]) -> Optional[str]:
    """
    Extract the main program type based on specific key phrases in file
    contents.

    Args:
        file_contents_list: List of lines read from the file

    Returns:
        Main program type string if found, None otherwise

    Example:
        >>> lines = ["Some text", "|| Sign^2(X) - 1 ||^2 = 0.001", "More text"]
        >>> _extract_main_program_type(lines)
        "sign_squared_violation"
    """
    for line in file_contents_list:
        for key_string, program_type in constants.MAIN_PROGRAM_TYPE_MAPPING.items():
            if key_string in line:
                return program_type
    return None


def _find_line_with_identifier(
    file_contents_list: List[str], line_identifier: str
) -> Optional[str]:
    """
    Find the first line containing the specified identifier.

    Args:
        - file_contents_list: List of lines to search through
        - line_identifier: String identifier to search for

    Returns:
        First matching line or None if not found

    Example:
        >>> lines = ["data = 123", "mass = 0.01", "other = abc"]
        >>> _find_line_with_identifier(lines, "mass =")
        "mass = 0.01"
    """
    return next((line for line in file_contents_list if line_identifier in line), None)


def _extract_parameter_from_line(
    matched_line: str,
    regex_pattern: str,
    param_name: str,
    expected_type: type,
    logger: Optional[logging.Logger] = None,
) -> Optional[Any]:
    """
    Extract a single parameter value from a line using regex.

    Args:
        - matched_line: The line containing the parameter
        - regex_pattern: Regex pattern to extract the value
        - param_name: Parameter name for error reporting
        - expected_type: Expected Python type for the value
        - logger: Optional logger for warnings

    Returns:
        Extracted and converted parameter value, or None if extraction fails

    Example:
        >>> _extract_parameter_from_line("mass = 0.01", r"(\d+\.\d+)", "mass", float)
        0.01
    """
    match = re.search(regex_pattern, matched_line)
    if not match:
        if logger:
            logger.warning(
                f"Regex pattern '{regex_pattern}' did not match for parameter '{param_name}'."
            )
        return None

    # Extract the first capture group
    raw_value = match.group(1)

    # Convert to expected type
    return _convert_to_type(raw_value, expected_type, param_name, logger)


def _initialize_array_parameters_dict() -> Dict[str, List]:
    """
    Initialize dictionary with empty lists for each array parameter.

    Returns:
        Dictionary with parameter names as keys and empty lists as values

    Example:
        >>> result = _initialize_array_parameters_dict()
        >>> "Calculation_result_per_vector" in result
        True
        >>> result["Calculation_result_per_vector"]
        []
    """
    return {
        param_name: []
        for param_name in constants.FILE_CONTENTS_ARRAY_PATTERNS_DICTIONARY.keys()
    }


def _extract_matches_from_line(line: str, pattern_details: Dict[str, Any]) -> List[Any]:
    """
    Extract all matches from a line using the provided pattern details.

    Args:
        - line: Line to search for matches
        - pattern_details: Dictionary containing regex pattern and type
          information

    Returns:
        List of converted matches (empty if no matches found)

    Example:
        >>> pattern = {"regex_pattern": r"(\d+\.\d+)", "type": float}
        >>> _extract_matches_from_line("values: 1.23 and 4.56", pattern)
        [1.23, 4.56]
    """
    matches = re.findall(pattern_details["regex_pattern"], line)
    if not matches:
        return []

    # Convert matches to correct type
    converted_matches = []
    for match in matches:
        try:
            converted_value = pattern_details["type"](match)
            converted_matches.append(converted_value)
        except (ValueError, TypeError):
            # Skip invalid matches rather than stopping execution
            continue

    return converted_matches


def _convert_lists_to_arrays(
    array_parameters: Dict[str, List]
) -> Dict[str, np.ndarray]:
    """
    Convert all list values in dictionary to NumPy arrays.

    Args:
        array_parameters: Dictionary with parameter names as keys and lists as
        values

    Returns:
        Dictionary with same keys but NumPy arrays as values

    Example:
        >>> input_dict = {"param1": [1, 2, 3], "param2": [4.5, 6.7]}
        >>> result = _convert_lists_to_arrays(input_dict)
        >>> isinstance(result["param1"], np.ndarray)
        True
        >>> result["param1"].tolist()
        [1, 2, 3]
    """
    return {
        param_name: np.array(param_values)
        for param_name, param_values in array_parameters.items()
    }


def extract_scalar_parameters_from_file_contents(
    file_contents_list: List[str], logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Extract scalar parameter values from file contents based on predefined
    patterns.

    This function searches through file contents to extract parameter values
    using patterns defined in constants.FILE_CONTENTS_SCALAR_PATTERNS_DICTIONARY
    and determines the main program type based on key phrases.

    The function processes file contents in the following steps: 1. Extract main
    program type using special key phrase matching 2. For each defined parameter
    pattern: - Find lines containing the parameter
       identifier - Apply regex to extract the value - Convert to the expected
       type
    3. Return dictionary of all successfully extracted parameters

    Args:
        - file_contents_list: List of lines read from the file
        - logger: Optional logger for warnings

    Returns:
        Dictionary of extracted parameter values with parameter names as keys

    Raises:
        - ValueError: If file_contents_list is empty or None
        - TypeError: If file_contents_list is not a list

    Example:
        >>> lines = ["Partition: gpu", "mass = 0.01", "|| Sign^2(X) - 1 ||^2 = 0.001"]
        >>> result = extract_scalar_parameters_from_file_contents(lines)
        >>> result['Cluster_partition']
        'gpu'
        >>> result['Bare_mass']
        0.01
        >>> result['Main_program_type']
        'sign_squared_violation'
    """
    # Input validation
    if not file_contents_list:
        raise ValueError("File contents list cannot be empty or None")
    if not isinstance(file_contents_list, list):
        raise TypeError("File contents must be provided as a list of strings")

    extracted_values = {}

    # Extract main program type (special case)
    main_program_type = _extract_main_program_type(file_contents_list)
    if main_program_type:
        extracted_values["Main_program_type"] = main_program_type

    # Extract regular parameters
    for (
        param_name,
        param_info,
    ) in constants.FILE_CONTENTS_SCALAR_PATTERNS_DICTIONARY.items():
        line_identifier = param_info["line_identifier"]
        regex_pattern = param_info["regex_pattern"]
        expected_type = param_info["type"]

        # Find line containing the parameter
        matched_line = _find_line_with_identifier(file_contents_list, line_identifier)

        if matched_line:
            # Extract parameter value from the line
            extracted_value = _extract_parameter_from_line(
                matched_line, regex_pattern, param_name, expected_type, logger
            )
            if extracted_value is not None:
                extracted_values[param_name] = extracted_value

    return extracted_values


def extract_array_parameters_from_file_contents(
    file_contents_list: List[str], logger: Optional[logging.Logger] = None
) -> Dict[str, np.ndarray]:
    """
    Extract array parameters from file contents based on predefined patterns.

    This function searches through file contents to extract arrays of values
    using patterns defined in constants.FILE_CONTENTS_ARRAY_PATTERNS_DICTIONARY.
    All extracted arrays are returned as NumPy arrays.

    The function processes file contents in the following steps: 1. Initialize
    empty lists for each defined array parameter 2. Process each line in the
    file contents: - Check if line contains
       identifiers for any array parameters - Extract all matches using regex
       patterns - Convert matches to appropriate types and append to lists
    3. Convert all collected lists to NumPy arrays

    Args:
        - file_contents_list: List of lines read from the file
        - logger: Optional logger for warnings (currently unused but kept for
          consistency)

    Returns:
        Dictionary mapping parameter names to NumPy arrays of extracted values

    Raises:
        - ValueError: If file_contents_list is empty or None
        - TypeError: If file_contents_list is not a list

    Example:
        >>> lines = [
        ...     "Done vector = 1.23e-05",
        ...     "Done vector = 2.45e-05",
        ...     "After 10 iters, CG converged, t = 1.5 sec",
        ...     "After 15 iters, CG converged, t = 2.1 sec"
        ... ]
        >>> result = extract_array_parameters_from_file_contents(lines)
        >>> result['Calculation_result_per_vector'].tolist()
        [1.23e-05, 2.45e-05]
        >>> result['Total_number_of_CG_iterations_per_spinor'].tolist()
        [10, 15]
    """
    # Input validation
    if not file_contents_list:
        raise ValueError("File contents list cannot be empty or None")
    if not isinstance(file_contents_list, list):
        raise TypeError("File contents must be provided as a list of strings")

    # Initialize dictionary with empty lists for each parameter
    array_parameters = _initialize_array_parameters_dict()

    # Process each line in the file contents
    for line in file_contents_list:
        # Check each array parameter pattern
        for (
            param_name,
            pattern_details,
        ) in constants.FILE_CONTENTS_ARRAY_PATTERNS_DICTIONARY.items():
            # Skip if line doesn't contain the identifier
            if pattern_details["line_identifier"] not in line:
                continue

            # Extract matches from this line
            matches = _extract_matches_from_line(line, pattern_details)

            # Add matches to the parameter's list
            array_parameters[param_name].extend(matches)

    # Convert all lists to NumPy arrays
    return _convert_lists_to_arrays(array_parameters)
