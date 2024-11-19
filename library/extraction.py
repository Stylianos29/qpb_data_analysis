from inspect import Parameter
import os
import re


import library.constants as constants


def extract_parameters_values_from_filename(filename, logger=None):
    """
    Extract parameters from a filename based on predefined regex patterns.

    Parameters:
        filename (str): The filename to process.
        logger (logging.Logger, optional): Logger instance for warnings. Defaults to None.

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
    for (parameter_name, parameter_info) in \
                            constants.FILENAME_REGEX_PATTERNS_DICTIONARY.items():
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
                extracted_values_dictionary[parameter_name] = expected_type(
                                                                    raw_value)
            except ValueError:
                # Handle type conversion errors
                if logger:
                    logger.warning(f"Could not convert {raw_value} to "\
                            f"{expected_type.__name__} for {parameter_name}.")
        else:
            # Log a warning if no match is found for the parameter
            if logger:
                logger.warning(f"No match found for parameter "\
                        f"'{parameter_name}' using pattern '{regex_pattern}'.")

    # Create a regex pattern to match all matched segments
    matched_segments_combined_pattern = "|".join(map(re.escape, \
                                                            matched_segments))
    # Split the filename by the matched pattern and filter out empty strings
    non_matched_segments = [
        segment.strip("_")
        for segment in re.split(matched_segments_combined_pattern, filename)
        if segment.strip("_")
    ]

    # Add unmatched text as "Additional_text"
    if non_matched_segments:
        extracted_values_dictionary["Additional_text"] = non_matched_segments

    return extracted_values_dictionary
