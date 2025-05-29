import os
import re

import numpy as np


import library.constants as constants


def extract_parameters_values_from_filename(filename, logger=None):
    """
    Extract parameters from a filename based on predefined regex patterns.

    Parameters:
        filename (str): The filename to process. logger (logging.Logger,
        optional): Logger instance for warnings. Defaults to None.

    Returns:
        dict: Extracted values from the filename.
    """

    # List of known extensions to trim
    known_extensions = [
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

    # Trim trailing known extension if it exists
    base_filename, ext = os.path.splitext(filename)
    if ext in known_extensions:
        filename = base_filename

    # Initialize a dictionary to store extracted values from filename
    extracted_values_dictionary = {}
    matched_segments = []  # List to store all matched parts of the filename

    # Apply each regex pattern to the filename
    for (
        parameter_name,
        parameter_info,
    ) in constants.FILENAME_SINGLE_VALUE_PATTERNS_DICTIONARY.items():
        regex_pattern = parameter_info["pattern"]
        expected_type = parameter_info["type"]

        match = re.search(regex_pattern, filename)
        if match:
            raw_value = match.group(parameter_name)
            # Store the full matched substring
            matched_segments.append(match.group(0))

            # Preprocess the value if it contains 'p' instead of '.'
            if "p" in raw_value and expected_type in {float, int}:
                raw_value = raw_value.replace("p", ".")

            try:
                # Convert to the expected type
                extracted_values_dictionary[parameter_name] = expected_type(raw_value)
            except ValueError:
                # Handle type conversion errors
                if logger:
                    logger.warning(
                        f"Could not convert {raw_value} to "
                        f"{expected_type.__name__} for {parameter_name}."
                    )

    # Create a regex pattern to match all matched segments
    matched_segments_combined_pattern = "|".join(map(re.escape, matched_segments))
    # Split the filename by the matched pattern and filter out empty strings
    non_matched_segments = [
        segment.strip("_")
        for segment in re.split(matched_segments_combined_pattern, filename)
        if segment.strip("_")
    ]

    # Specific treatment for "MPI_geometry"
    if "MPI_geometry" in extracted_values_dictionary:
        raw_cores = extracted_values_dictionary["MPI_geometry"]
        extracted_values_dictionary["MPI_geometry"] = f"(1, {', '.join(raw_cores)})"

    # Add unmatched text as "Additional_text"
    if non_matched_segments:
        extracted_values_dictionary["Additional_text"] = non_matched_segments

    return extracted_values_dictionary


def extract_single_valued_parameter_values_from_file_contents(
    file_contents_list, logger=None
):
    """
    Extract parameter values from file contents based on patterns defined in
    FILE_CONTENTS_SINGLE_VALUE_PATTERNS_DICTIONARY, along with determining
    the 'Main_program_type' based on specific key phrases.

    Parameters:
        - file_contents_list (list): List of lines read from the file.
        - logger (logging.Logger, optional): Logger for logging warnings.

    Returns:
        dict: Dictionary of extracted parameter values.
    """

    # Initialize output dictionary
    extracted_values = {}

    # SPECIAL EXTRACTION OF THE MAIN PROGRAM TYPE

    # Initialize "Main_program_type" as None (or a default value if needed)
    main_program_type = None
    for line in file_contents_list:
        # Check for the special string matches
        for key_string, program_type in constants.MAIN_PROGRAM_TYPE_MAPPING.items():
            if key_string in line:
                main_program_type = program_type
                break  # Stop checking further if a match is found
    # If a match was found, store it in the dictionary
    if main_program_type:
        extracted_values["Main_program_type"] = main_program_type

    # EXTRACT THE REST PIECES OF INFORMATION

    for (
        parameter_name,
        parameter_info,
    ) in constants.FILE_CONTENTS_SINGLE_VALUE_PATTERNS_DICTIONARY.items():
        line_identifier = parameter_info["line_identifier"]
        regex_pattern = parameter_info["regex_pattern"]
        expected_type = parameter_info["type"]

        # Search for the line containing the parameter
        matched_line = next(
            (line for line in file_contents_list if line_identifier in line), None
        )

        if matched_line:
            # Apply regex to extract the value
            match = re.search(regex_pattern, matched_line)
            if match:
                # Assume the first capture group contains the value
                raw_value = match.group(1)

                # Preprocess the value if it contains 'p' instead of '.'
                if "p" in raw_value and expected_type in {float, int}:
                    raw_value = raw_value.replace("p", ".")

                try:
                    # Convert to the expected type
                    extracted_values[parameter_name] = expected_type(raw_value)
                except ValueError:
                    if logger:
                        logger.warning(
                            f"Failed to convert '{raw_value}' "
                            f"to {expected_type} for {parameter_name}."
                        )
            else:
                if logger:
                    logger.warning(
                        f"Regex pattern '{regex_pattern}' did not "
                        f"match for {parameter_name}."
                    )

    return extracted_values


def extract_multivalued_parameters_from_file_contents(file_contents_list, logger=None):
    # TODO: Include logging

    # Initialize a dictionary to store multivalued parameters
    multivalued_parameters = {}

    # Loop through each line in the file contents
    for line in file_contents_list:
        # Check for each multivalued parameter in the dictionary
        for (
            parameter,
            pattern_details,
        ) in constants.FILE_CONTENTS_MULTIVALUED_PATTERNS_DICTIONARY.items():
            # If the line contains the line identifier for the current parameter
            if pattern_details["line_identifier"] in line:
                # Find all matches using the regex pattern
                matches = re.findall(pattern_details["regex_pattern"], line)
                if matches:
                    # Initialize a non-existing key with an empty list
                    if parameter not in multivalued_parameters:
                        multivalued_parameters[parameter] = []

                    # Convert extracted values to correct type and add to list
                    for match in matches:
                        multivalued_parameters[parameter].append(
                            pattern_details["type"](match)
                        )

    # Convert lists to numpy arrays
    for key, value in multivalued_parameters.items():
        multivalued_parameters[key] = np.array(value)

    # Return the dictionary with Numpy arrays
    return multivalued_parameters
