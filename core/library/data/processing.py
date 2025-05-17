import pandas as pd

from ..constants import DTYPE_MAPPING, CONVERTERS_MAPPING


def load_csv(input_csv_file_path, dtype_mapping=None, converters_mapping=None):
    """
    Loads a CSV file into a Pandas DataFrame with optional filtering of dtypes
    and converters based on the CSV file's header.

    Parameters:
    -----------
    input_csv_file_path : str
        Path to the input CSV file.
    dtype_mapping : dict, optional
        A dictionary mapping column names to data types.
    converters_mapping : dict, optional
        A dictionary mapping column names to converter functions.

    Returns:
    --------
    pd.DataFrame
        The loaded DataFrame with the specified dtypes and converters applied.
    """
    # Define default mappings if not provided
    if dtype_mapping is None:
        dtype_mapping = DTYPE_MAPPING

    if converters_mapping is None:
        converters_mapping = CONVERTERS_MAPPING

    # Read the header of the .csv file to determine available fields
    with open(input_csv_file_path, "r") as f:
        # Convert to a set for fast lookup
        csv_header = set(f.readline().strip().split(","))

    # Filter mappings based on the header
    filtered_dtype_mapping = {
        field: dtype for field, dtype in dtype_mapping.items() if field in csv_header
    }
    filtered_converters_mapping = {
        field: converter
        for field, converter in converters_mapping.items()
        if field in csv_header
    }

    # Load the CSV file with the filtered mappings
    dataframe = pd.read_csv(
        input_csv_file_path,
        dtype=filtered_dtype_mapping,
        converters=filtered_converters_mapping,
    )

    # Define the expected unique values
    expected_values = {"Wilson", "Brillouin"}
    # Check unique values in the column
    actual_values = set(dataframe["Kernel_operator_type"].unique())
    # Verify the conditions
    if actual_values == expected_values:
        # Set a categorical data type with a custom order
        dataframe["Kernel_operator_type"] = pd.Categorical(
            dataframe["Kernel_operator_type"],
            categories=["Wilson", "Brillouin"],  # Custom order
            ordered=True,
        )

    return dataframe
