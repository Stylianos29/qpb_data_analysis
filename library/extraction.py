import re


import library.constants as constants


def filename_extraction(filename):
    # Initialize a dictionary to store extracted values from filename
    extracted_values_dictionary = {}

    # Apply each regex pattern to the filename
    for parameter_name, parameter_info in constants.FILENAME_REGEX_PATTERNS_DICTIONARY.items():
        regex_pattern = parameter_info["pattern"]
        expected_type = parameter_info["type"]

        match = re.search(regex_pattern, filename)
        if match:
            raw_value = match.group(parameter_name)
            
            # Preprocess the value if it contains 'p' instead of '.'
            if "p" in raw_value and expected_type in {float, int}:
                raw_value = raw_value.replace("p", ".")
            
            try:
                # Convert to the expected type
                extracted_values_dictionary[parameter_name] = expected_type(raw_value)
            except ValueError:
                # Handle type conversion errors
                print(f"Warning: Could not convert {raw_value} to {expected_type.__name__} for {parameter_name}")
                extracted_values_dictionary[parameter_name] = raw_value  # Store raw value as fallback

    return extracted_values_dictionary
