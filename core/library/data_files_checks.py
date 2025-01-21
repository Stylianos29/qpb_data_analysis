import os
from statistics import correlation
import sys
import re


CORRELATOR_IDENTIFIERS_LIST = [
    "1-1",
    "g1-g1",
    "g2-g2",
    "g3-g3",
    "g5-g5",
    "g5-g4g5",
    "g4g5-g5",
    "g4g5-g4g5",
]
OPERATOR_METHODS_LIST = ["Operator", "Chebyshev", "KL"]
OPERATOR_TYPES_LIST = ["Standard", "Brillouin"]


def data_file_format_validation(data_file_full_path):
    """Checks if the data file contains data organized in 7 columns. Additionally, it verifies that every line in the file contains one of the specified correlator identifiers substrings.."""

    with open(data_file_full_path, "r") as data_file:
        for line in data_file:

            # Check for 7 columns in total
            parts = line.strip().split()
            if len(parts) != 7:
                return False

            # Look for any of the identifiers at each line
            if not any(substring in line for substring in CORRELATOR_IDENTIFIERS_LIST):
                return False

    return True


def data_file_non_empty_lines_count_validation(data_file_full_path):
    """Counts the number of non-empty lines in a file. If this number is divisible by the total number of correlator identifier substrings, then it returns the result of the integer division. Otherwise, it signals that the test has failed."""

    non_empty_lines_count = 0
    with open(data_file_full_path, "r") as file:
        for line in file:
            # Check if the line is not empty after stripping it (of empty spaces)
            if line.strip():
                non_empty_lines_count += 1

    number_of_correlator_identifiers = len(CORRELATOR_IDENTIFIERS_LIST)
    # If the number of non-empty lines is divisibleby the number of correlator identifiers, then return the quotient
    if not non_empty_lines_count % number_of_correlator_identifiers:
        return non_empty_lines_count // number_of_correlator_identifiers
    else:
        # otherwise it expects handling by the main program
        return False


def request_permission(request_message: str):
    """Request permission from the user to continue with the suggest tasks."""

    while True:
        request_response = input(request_message + " (y[es]/n[o]): ")

        if request_response.lower() in ["yes", "y"]:
            return True
        elif request_response.lower() in ["no", "n"]:
            return False
        else:
            # The loop will continue until a valid response is given
            print("Invalid input. Please enter 'yes' or 'no'.")


def extract_operator_classification(filename):
    """
    This function extracts information about the operator from the given filename and returns it as a tuple.
    It first identifies the operator type from the filename. If it cannot be determined, the function exits and expects handling by the main program.
    Once the operator type is identified, it searches for the operator method in the filename. If no method is mentioned, it assumes the method is "Operator".
    If the operator method is not "Operator", the function extracts the operator enumeration as a string of the form [nN][d].
    Otherwise, it sets the enumeration value to 'Not Applicable'.
    """

    # Operator type
    if OPERATOR_TYPES_LIST[0] in filename:
        operator_type = OPERATOR_TYPES_LIST[0]
    elif OPERATOR_TYPES_LIST[1] in filename:
        operator_type = OPERATOR_TYPES_LIST[1]
    else:
        # Exit the function, it needs to be handled from outside
        return None, None, None

    # Operator method
    if OPERATOR_METHODS_LIST[1] in filename:
        operator_method = OPERATOR_METHODS_LIST[1]
    elif OPERATOR_METHODS_LIST[2] in filename:
        operator_method = OPERATOR_METHODS_LIST[2]
    else:
        operator_method = OPERATOR_METHODS_LIST[0]

    # Operator enumeration
    if operator_method != OPERATOR_METHODS_LIST[0]:
        match = re.search(r"([nN])[=_]?(\d+)", filename, flags=re.IGNORECASE)
        if match:
            operator_enumeration = match.group(1) + match.group(2)
        else:
            operator_enumeration = None
    else:
        operator_enumeration = "Not Applicable"

    return operator_method, operator_type, operator_enumeration


def extract_bare_mass_value(filename):
    """Extract the bare mass value as float number."""

    match = re.search(r"mb[=_]?(-?\d*\.?\d+)", filename)
    if match:
        return float(match.group(1))
    else:
        return None


def extract_configuration_label(filename):
    """Extract the configuration label as a 3-digit string."""

    match = re.search(r"config[=_]?(\w+)\.dat", filename)
    if match:
        return match.group(1)
    else:
        return None


def extract_optimization_factor(filename):
    """Extract an optional optimization factor as a float number."""
    # TODO: Anticipate rho value?

    match = re.search(r"mu[=_]?(\d*\.?\d+)", filename)
    if match:
        return float(match.group(1))
    else:
        return None
