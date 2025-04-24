import os

import numpy as np
import pandas as pd
import itertools
import h5py
from typing import List, Set, Dict

from library import constants


def print_dictionaries_side_by_side(
    left_dictionary,
    right_dictionary,
    line_width=80,
    left_column_title=None,
    right_column_title=None,
):
    """
    Print two dictionaries side by side, with the second dictionary starting at
    the middle of the line width.

    Parameters: - left_dictionary (dict): The first dictionary to print. -
    right_dictionary (dict): The second dictionary to print. - line_width (int,
    optional): The total width of the line. Default is 80.
    """
    # Calculate the middle position of the line
    middle_position = line_width // 2

    # Prepare keys and values as formatted strings
    left_dictionary_items = [f"{k}: {v}" for k, v in left_dictionary.items()]
    right_dictionary_items = [f"{k}: {v}" for k, v in right_dictionary.items()]

    # Determine the maximum number of lines to print
    max_lines = max(len(left_dictionary_items), len(right_dictionary_items))

    # Print titles if provided, aligned with the key-value pairs
    if left_column_title and right_column_title:
        # Format and align the two column titles Format the first title and add
        # the separator
        title_output = (
            f"{left_column_title:<{middle_position-3}} | {right_column_title}"
        )
        print(title_output)
        # print(f"{left_column_title:<{middle_position}}{right_column_title}")
        print("-" * (line_width))

    # Print dictionaries side by side
    for i in range(max_lines):
        # Get the current item from each dictionary, if it exists
        left = left_dictionary_items[i] if i < len(left_dictionary_items) else ""
        right = right_dictionary_items[i] if i < len(right_dictionary_items) else ""

        # Format and align the two outputs
        output = f"{left:<{middle_position}}{right}"
        print(output)


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
        dtype_mapping = constants.DTYPE_MAPPING

    if converters_mapping is None:
        converters_mapping = constants.CONVERTERS_MAPPING

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


def get_fields_with_multiple_values(df, excluded_fields=None) -> list:
    """
    Retrieves a list of column names from the given Pandas DataFrame that
    contain more than one unique value, excluding any columns specified in the
    'excluded_fields' set.

    Parameters:

    - df (pd.DataFrame): The Pandas DataFrame to analyze.
    - excluded_fields (set, optional): A set of column names to exclude from the
    results. If not provided or is empty, no columns will be excluded. Default
    is None.

    Returns:

    - List of column names that have more than one unique value,
    excluding any specified in 'excluded_fields'.

    Usage:

    - get_fields_with_multiple_values(df, {"Filename", "Plaquette"}) -
    get_fields_with_multiple_values(df)
    """

    if excluded_fields is None:
        excluded_fields = set()  # Default to an empty set if not provided

    # Filter columns with more than one unique value and exclude specified
    # fields
    varying_fields = [
        col
        for col in df.columns
        if df[col].nunique() > 1 and col not in excluded_fields
    ]

    return varying_fields


class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataAnalyzer instance.

        Parameters:
        -----------
        df (pd.DataFrame): The Pandas DataFrame to analyze.
        """
        self.df = df

        # Extract fields with unique values (constant for this DataFrame)
        self.fields_with_unique_values_dictionary = self.get_fields_with_unique_values()

        # Extract fields with multiple values (to be dynamically adjusted later)
        self.fields_with_multiple_values = []

        # Get combinations of unique field values for analysis (to be updated
        # later)
        self.unique_combinations = []

    def get_column_titles(self):
        """
        Retrieve a list of all column titles from a Pandas DataFrame.

        Parameters:
            dataframe (pd.DataFrame): The input DataFrame.

        Returns:
            list: A list of column titles as strings.

        Usage:
            >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            >>> get_column_titles(df)
            ['A', 'B']
        """
        return self.df.columns.tolist()

    def get_fields_with_unique_values(self) -> dict:
        """
        This function returns a dictionary where the keys are the column names
        from the DataFrame, and the values are the single unique value in the
        column for those columns that contain only a single unique value.

        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe whose columns are to be analyzed.

        Returns:
        --------
        dict
            A dictionary with column names as keys and the single unique value
            as values for those columns that have only one unique value.

        Example:
        --------
        df = pd.DataFrame({
            'col1': [1, 1, 1, 1], 'col2': [1, 1, 1, 1], 'col3': [5, 6, 7, 8]
        }) result = get_fields_with_unique_values(df) print(result) # Output:
        {'col1': 1, 'col2': 1}
        """
        single_unique_value_columns = {}

        # Iterate through each column
        for column in self.df.columns:
            unique_values = self.df[column].unique()

            # Only add columns where there is a single unique value
            if len(unique_values) == 1:
                # Store the unique value directly, not as a list
                single_unique_value_columns[column] = unique_values[0]

        return single_unique_value_columns

    def get_multivalued_fields(self) -> list:
        multivalued_columns_list = []

        # Iterate through each column
        for column in self.df.columns:
            unique_values = self.df[column].unique()

            # Only add columns where there is more than a single unique value
            if len(unique_values) > 1:
                multivalued_columns_list.append(column)

        return multivalued_columns_list

    def set_excluded_fields(self, excluded_fields: Set[str]):
        """
        Set the excluded fields and update the combinations and fields with
        multiple values.

        Parameters:
        -----------
        excluded_fields (set): A set of column names to exclude from the
        analysis.
        """
        # Update the excluded fields
        self.excluded_fields = excluded_fields

        # Recalculate fields with multiple values excluding the specified ones
        self.fields_with_multiple_values = get_fields_with_multiple_values(
            self.df, self.excluded_fields
        )

        # Update the unique combinations based on the new excluded fields
        self.unique_combinations = [
            self.df[field].unique() for field in self.fields_with_multiple_values
        ]

    def generate_combinations(self) -> List[Dict[str, any]]:
        """
        Generate all possible combinations of unique values across multiple
        fields.

        Returns:
        --------
        List of dictionaries, where each dictionary represents a combination of
        values for different fields.
        """
        return [
            dict(zip(self.fields_with_multiple_values, combination))
            for combination in itertools.product(*self.unique_combinations)
        ]

    def get_dataframe_group(self, filters: Dict[str, any]) -> pd.DataFrame:
        """
        Filters the dataframe based on the provided filters (a combination of
        field values).

        Parameters:
        -----------
        filters (dict): A dictionary with field names as keys and the
        corresponding values to filter on.

        Returns:
        --------
        pd.DataFrame: A filtered dataframe based on the provided filters.
        """
        dataframe_group = self.df
        for field, value in filters.items():
            dataframe_group = dataframe_group[dataframe_group[field] == value]

        return dataframe_group

    def get_valid_dataframe_groups_with_metadata(self) -> List[Dict[str, any]]:
        """
        Returns a list of dictionaries, where each dictionary contains: - The
        valid (non-empty) dataframe group. - The combined metadata for that
        group.

        Returns:
        --------
        List of dictionaries with keys 'dataframe_group' and 'metadata'.
        """
        valid_groups_with_metadata = []

        for combination in self.generate_combinations():
            # Filter the dataframe based on the current combination
            dataframe_group = self.get_dataframe_group(combination)

            # Skip empty dataframe_groups (no data for this combination)
            if not dataframe_group.empty:
                metadata = combination
                valid_groups_with_metadata.append(
                    {"dataframe_group": dataframe_group, "metadata": metadata}
                )

        return valid_groups_with_metadata


def extract_datasets_to_dict(hdf5_file_path, dataset_name, group_level=3):
    """
    Extracts datasets from a specified group level in an HDF5 file and stores
    them in a dictionary.

    Parameters: - hdf5_file_path (str): Path to the HDF5 file. - dataset_name
    (str): Name of the dataset to extract within each group. - group_level (int,
    optional): The level of groups containing the datasets (default: 3).

    Returns: - dict: A dictionary where keys are group names at the specified
    level, and values are the extracted datasets.

    Raises: - ValueError: If the group level is invalid or the dataset is not
    found at the specified level.
    """
    data_dict = {}

    # Open the HDF5 file
    with h5py.File(hdf5_file_path, "r") as hdf_file:
        # Recursively find all groups at the specified level
        def collect_groups(name, obj):
            if isinstance(obj, h5py.Group) and name.count("/") == group_level - 1:
                groups_at_level.append((name, obj))

        groups_at_level = []
        hdf_file.visititems(collect_groups)

        if not groups_at_level:
            raise ValueError(
                f"No groups found at level {group_level} in the HDF5 file."
            )

        # Extract datasets from the identified groups
        for group_name, group in groups_at_level:
            if dataset_name in group:
                data_dict[group_name.split("/")[-1]] = group[dataset_name][()]
            else:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found in group '{group_name}'."
                )

    return data_dict


class DataHandler:
    def __init__(
        self,
        dataframe,
        hdf5_path,
        base_hdf5_path="sign_squared_values/Chebyshev_several_configs_varying_Ν",
    ):
        """
        Initialize the DataHandler.

        Parameters: - ? - hdf5_path (str): Path to the HDF5 file containing
        datasets. - base_hdf5_path (str): Base path in the HDF5 file where
        level-3 groups are located. Default is
        "sign_squared_values/Chebyshev_several_configs_varying_Ν".
        """
        # Load the DataFrame
        self.df = dataframe

        # Open the HDF5 file
        self.hdf5 = h5py.File(hdf5_path, "r")

        # Store the base HDF5 path
        self.base_hdf5_path = base_hdf5_path

    def get_value(self, filename, field_or_dataset):
        """
        Retrieve a value associated with a specific filename.

        Parameters: - filename (str): The linking value (Filename field in the
        DataFrame). - field_or_dataset (str): The name of the DataFrame field or
        HDF5 dataset.

        Returns: - Value from the DataFrame or HDF5 file.
        """
        # Check if the field_or_dataset exists in the DataFrame
        if field_or_dataset in self.df.columns:
            # Retrieve value from the DataFrame
            row = self.df[self.df["Filename"] == filename]
            if row.empty:
                raise KeyError(f"Filename '{filename}' not found in DataFrame.")
            return row[field_or_dataset].iloc[0]

        # If not in DataFrame, check in HDF5 file
        hdf5_group_path = f"{self.base_hdf5_path}/{filename}"
        if hdf5_group_path in self.hdf5:
            if field_or_dataset in self.hdf5[hdf5_group_path]:
                return self.hdf5[hdf5_group_path][field_or_dataset][:]
            else:
                raise KeyError(
                    f"Dataset '{field_or_dataset}' not found in HDF5 group '{filename}'."
                )
        else:
            raise KeyError(f"Filename '{filename}' not found in HDF5 file.")

    def close(self):
        """Close the HDF5 file."""
        self.hdf5.close()


def extract_HDF5_datasets_to_dictionary(file_path, dataset_name):
    """
    Open an HDF5 file and extract datasets named 'Calculation_result_per_vector'
    stored in level-3 groups.

    Parameters:
        file_path (str): The physical path of the HDF5 file provided by the
        user.

    Returns:
        dict: A dictionary with group basenames as keys and the corresponding
        datasets (NumPy arrays) as values.
    """
    datasets_dictionary = {}

    # Open the HDF5 file in read mode
    with h5py.File(file_path, "r") as hdf_file:

        def traverse_groups(name, obj):
            # Only process if the object is a dataset with the target name
            if isinstance(obj, h5py.Dataset) and name.endswith(dataset_name):
                # Get the group name (parent group name) and extract its
                # basename
                group_name = name.rsplit("/", maxsplit=1)[0]
                basename = os.path.basename(group_name)
                # Store the dataset as a NumPy array
                datasets_dictionary[basename] = obj[:]

        # Recursively visit all objects in the file
        hdf_file.visititems(traverse_groups)

    return datasets_dictionary


def find_single_value_columns(df):
    """
    Finds columns in the DataFrame with a single unique value and returns a
    dictionary with column names as keys and their unique values as dictionary
    values.

    Parameters: df (pd.DataFrame): The input Pandas DataFrame.

    Returns: dict: A dictionary with column names as keys and their unique
    values as dictionary values.
    """
    single_value_columns = {}

    for col in df.columns:
        unique_values = df[
            col
        ].nunique()  # Get the number of unique values in the column
        if unique_values == 1:
            # If there's only one unique value, add it to the dictionary
            single_value_columns[col] = df[col].iloc[0]

    return single_value_columns


def find_multiple_value_columns(df):
    """
    Finds columns in the DataFrame with multiple unique values and returns a
    dictionary with column names as keys and the count of unique values as
    dictionary values.

    Parameters: df (pd.DataFrame): The input Pandas DataFrame.

    Returns: dict: A dictionary with column names as keys and the count of
    unique values as dictionary values.
    """
    multiple_value_columns = {}

    for col in df.columns:
        unique_values_count = df[
            col
        ].nunique()  # Get the number of unique values in the column
        if unique_values_count > 1:
            # If there's more than one unique value, add it to the dictionary
            multiple_value_columns[col] = unique_values_count

    return multiple_value_columns


class DataFrameAnalyzer:
    """
    A class for analyzing and manipulating Pandas DataFrames with a focus on
    distinguishing between tunable parameters and output quantities.

    This class provides functionality to categorize DataFrame columns, identify
    columns with single vs. multiple unique values, group data by parameters,
    and filter the DataFrame based on various conditions.

    Attributes:
        - original_dataframe (pd.DataFrame): A copy of the original input
          DataFrame.
        - dataframe (pd.DataFrame): The working DataFrame that can be filtered
          and manipulated.
        - list_of_dataframe_column_names (list): List of all column names in the
          DataFrame.
        - list_of_tunable_parameter_names_from_dataframe (list): Columns
          identified as tunable parameters.
        - list_of_output_quantity_names_from_dataframe (list): Columns
          identified as output quantities.
        - unique_value_columns_dictionary (dict): Columns with single unique
          values and their values.
        - multivalued_columns_count_dictionary (dict): Columns with multiple
          unique values and counts.
        - list_of_single_valued_column_names (list): Column names with only one
          unique value.
        - list_of_multivalued_column_names (list): Column names with multiple
          unique values.
        - list_of_single_valued_tunable_parameter_names (list): Tunable
          parameters with one unique value.
        - list_of_multivalued_tunable_parameter_names (list): Tunable parameters
          with multiple values.
        - list_of_single_valued_output_quantity_names (list): Output quantities
          with one unique value.
        - list_of_multivalued_output_quantity_names (list): Output quantities
          with multiple values.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the DataFrameAnalyzer with a DataFrame.

        We divide parameters into two categories:
        1. Tunable parameters: These are parameters that can be adjusted by the
           user.
        2. Output quantities: These are the results of calculations.

        The primary purpose of this class is to distinguish between tunable
        parameters and output quantities, and to construct DataFrame groups
        based on tunable parameters or subsets of them, according to user needs.

        Args:
            dataframe (pd.DataFrame): The input DataFrame to analyze.

        Raises:
            TypeError: If the input is not a Pandas DataFrame.
        """
        # Verify that the input is a Pandas DataFrame
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame")

        # Store a copy of the original DataFrame
        self.original_dataframe = dataframe.copy()

        # Initialize the dataframe attribute
        self.dataframe = self.original_dataframe.copy()

        # Initialize all lists of dataframe column name categories
        self._update_column_categories()

    def _update_column_categories(self):
        """
        Private method: Update and categorize column names from the dataframe
        into various lists and dictionaries.

        This internal method processes dataframe columns to identify:
        - Tunable parameters vs output quantities
        - Columns with single vs multiple unique values
        - Intersections of these categories

        The method updates multiple class attributes that categorize the columns
        in different ways for later use in analysis and filtering operations.

        Note: This method is called automatically during initialization and
        after operations that modify the dataframe structure. It should not be
        called directly by users.
        """
        self.list_of_dataframe_column_names = self.dataframe.columns.tolist()

        # Extract list of tunable parameter names
        self.list_of_tunable_parameter_names_from_dataframe = [
            column_name
            for column_name in self.list_of_dataframe_column_names
            if column_name in constants.TUNABLE_PARAMETER_NAMES_LIST
        ]

        # Extract list of output quantity names
        self.list_of_output_quantity_names_from_dataframe = [
            column_name
            for column_name in self.list_of_dataframe_column_names
            if column_name in constants.OUTPUT_QUANTITY_NAMES_LIST
        ]

        # Get dictionary of columns with single unique values and their values
        self.unique_value_columns_dictionary = self._get_unique_value_columns()

        # Get dictionary of multivalued columns with number of unique values
        self.multivalued_columns_count_dictionary = (
            self._get_multivalued_columns_count()
        )

        # Extract list of column names with single unique values
        self.list_of_single_valued_column_names = list(
            self.unique_value_columns_dictionary.keys()
        )

        # Extract list of column names with multiple unique values
        self.list_of_multivalued_column_names = list(
            self.multivalued_columns_count_dictionary.keys()
        )

        # Create list of tunable parameters with single unique values
        self.list_of_single_valued_tunable_parameter_names = list(
            set(self.list_of_tunable_parameter_names_from_dataframe)
            & set(self.list_of_single_valued_column_names)
        )

        # Create list of tunable parameters with multiple unique values
        self.list_of_multivalued_tunable_parameter_names = list(
            set(self.list_of_tunable_parameter_names_from_dataframe)
            & set(self.list_of_multivalued_column_names)
        )

        # Create list of output quantities with single unique values
        self.list_of_single_valued_output_quantity_names = list(
            set(self.list_of_output_quantity_names_from_dataframe)
            & set(self.list_of_single_valued_column_names)
        )

        # Create list of output quantities with multiple unique values
        self.list_of_multivalued_output_quantity_names = list(
            set(self.list_of_output_quantity_names_from_dataframe)
            & set(self.list_of_multivalued_column_names)
        )

    def _get_unique_value_columns(self) -> dict:
        """
        Private helper method: Identify columns that have only a single unique
        value.

        Used internally to populate class attributes related to single-valued
        columns.

        Returns:
            dict: A dictionary where keys are column names with only one unique
            value, and values are the actual unique values for those columns.
        """
        unique_value_columns_dictionary = {}
        for column in self.dataframe.columns:
            unique_values = self.dataframe[column].unique()

            # Only add columns where there is a single unique value
            if len(unique_values) == 1:
                # Store the unique value directly, not as a list
                unique_value_columns_dictionary[column] = unique_values[0]

        return unique_value_columns_dictionary

    def _get_multivalued_columns_count(self) -> dict:
        """
        Private helper method: Identify columns that have multiple unique values
        and count them.

        Used internally to populate class attributes related to multi-valued
        columns.

        Returns:
            dict: A dictionary where keys are column names with multiple unique
            values, and values are the counts of unique values in those columns.
        """
        multivalued_columns_count_dictionary = {}
        for column in self.dataframe.columns:
            # Get the number of unique values in the column
            unique_values_count = self.dataframe[column].nunique()
            if unique_values_count > 1:
                # If there's more than one unique value, add it to dictionary
                multivalued_columns_count_dictionary[column] = unique_values_count

        return multivalued_columns_count_dictionary

    def group_by_multivalued_tunable_parameters(
        self, filter_out_parameters_list: list = None
    ):
        """
        Group the DataFrame by multivalued tunable parameters, optionally
        excluding some parameters.

        This method allows for flexible grouping of data based on tunable
        parameters that have multiple unique values. Parameters can be excluded
        from the grouping if they are not relevant to the current analysis.

        Args:
            filter_out_parameters_list (list, optional): List of parameter names
            to exclude from the grouping operation. Default is None (include all
            multivalued parameters).

        Returns:
            pandas.core.groupby.DataFrameGroupBy: A GroupBy object created by
            grouping on the specified multivalued tunable parameters.

        Raises:
            TypeError: If filter_out_parameters_list is not a list. ValueError:
            If filter_out_parameters_list contains invalid parameter names.

        Example:
            # Group by all multivalued tunable parameters groups =
            analyzer.group_by_multivalued_tunable_parameters()

            # Group by all except 'temperature' and 'pressure'

            groups = analyzer.group_by_multivalued_tunable_parameters(
                filter_out_parameters_list=['temperature', 'pressure']
            )
        """
        if filter_out_parameters_list is not None:
            # Check if filter_out_parameters_list is a list
            if not isinstance(filter_out_parameters_list, list):
                raise TypeError("'filter_out_parameters_list' must be a list!")

            # Check if all elements in filter_out_parameters_list are valid
            # tunable parameters
            invalid_parameters = [
                param
                for param in filter_out_parameters_list
                if param not in self.list_of_multivalued_tunable_parameter_names
            ]
            if invalid_parameters:
                raise ValueError(
                    "Invalid parameters in filter_out_parameters_list: "
                    f"{invalid_parameters}. Must be from: "
                    f"{self.list_of_multivalued_tunable_parameter_names}."
                )

            # Create a reduced list of multivalued tunable parameters by
            # excluding the specified ones from the filter_out_parameters_list
            self.reduced_multivalued_tunable_parameter_names_list = [
                item
                for item in self.list_of_multivalued_tunable_parameter_names
                if item not in filter_out_parameters_list
            ]

        else:
            self.reduced_multivalued_tunable_parameter_names_list = (
                self.list_of_multivalued_tunable_parameter_names.copy()
            )

        # Create a groupby object based on the reduced list of multivalued
        # tunable parameters. If reduced list is empty, return entire dataframe
        if self.reduced_multivalued_tunable_parameter_names_list:
            return self.dataframe.groupby(
                self.reduced_multivalued_tunable_parameter_names_list, observed=True
            )
        else:
            # Group by a dummy value to return the entire dataframe
            return self.dataframe.groupby(lambda _: "Dummy", observed=True)

    def restrict_dataframe(self, condition=None, filter_func=None):
        """
        Restricts the DataFrame to rows that satisfy given conditions.

        This method provides two ways to filter the DataFrame: 1. Using a string
        condition with Pandas query syntax 2. Using a custom filter function for
        more complex filtering logic

        Args:
            - condition (str, optional): A string condition to filter rows using
              Pandas query syntax e.g., "MSCG_epsilon >= 1e-06 and
              Kernel_operator_type == 'Wilson'"
            - filter_func (callable, optional): A function that takes a
              DataFrame and returns a boolean Series for more complex filtering
              needs

        Returns:
            None: Modifies the DataFrameAnalyzer's DataFrame in-place.

        Raises:
            ValueError: If neither condition nor filter_func is provided, or if
            filtering fails.

        Examples:
            # Example 1: Using a query string for simpler conditions
            analyzer.restrict_dataframe(
                "(MSCG_epsilon >= 1e-06 and Kernel_operator_type == 'Wilson') "
                "or (MSCG_epsilon >= 1e-05 and Kernel_operator_type ==
                'Brillouin')"
            )

            # Example 2: Using a filter function for more complex conditions def
            complex_filter(df):
                return (
                    ((df["MSCG_epsilon"] >= 1e-06) & (df["Kernel_operator_type"]
                    == "Wilson")) | ((df["MSCG_epsilon"] >= 1e-05) &
                    (df["Kernel_operator_type"] == "Brillouin"))
                )
            analyzer.restrict_dataframe(filter_func=complex_filter)
        """
        try:
            if condition is not None:
                # Use `query` to filter the DataFrame based on the condition
                # string
                self.dataframe = self.dataframe.query(condition)
            elif filter_func is not None:
                # Apply the custom filter function for more complex filtering
                mask = filter_func(self.dataframe)
                self.dataframe = self.dataframe[mask]
            else:
                raise ValueError("Either condition or filter_func must be provided")

            # Recompute the column categories after filtering
            self._update_column_categories()

        except Exception as e:
            raise ValueError(f"Failed to apply filter: {e}")

    def restore_original_dataframe(self):
        """
        Reset the working DataFrame to the original, unfiltered state.

        This method replaces the current working DataFrame with a fresh copy of
        the original DataFrame that was provided when the object was created. It
        also updates all the column categories to reflect the original data.

        This is useful when you want to start a new analysis from scratch after
        applying various filters or transformations.

        Returns:
            None: Modifies the DataFrameAnalyzer's DataFrame in-place.
        """
        self.dataframe = self.original_dataframe.copy()
        self._update_column_categories()


class TableGenerator(DataFrameAnalyzer):
    """
    A class for generating formatted tables from DataFrame analysis results.

    This class extends DataFrameAnalyzer to provide methods for creating
    well-formatted tables in different output formats, with a focus on Markdown
    readability. TODO: Say say something about grouping in multivalued tunable
    parameters and pivoting.
    """

    def __init__(self, dataframe: pd.DataFrame, output_directory: str = "."):
        """
        Initialize the TableGenerator with a DataFrame and optional output
        directory.

        Args:
            - dataframe (pd.DataFrame): The input DataFrame to analyze and
              generate tables from.
            - output_directory (str, optional): Default directory to save
              generated table files. Defaults to current directory (".").

        Raises:
            TypeError: If the input is not a Pandas DataFrame.
        """
        # Initialize the parent DataFrameAnalyzer class
        super().__init__(dataframe)

        # Store the default output directory
        self.output_directory = output_directory

    def _save_table_to_file(
        self,
        table_string: str,
        output_directory: str = None,
        filename: str = "table",
        file_format: str = "md",
    ) -> None:
        """
        Save a table string to a file in the specified directory and format.

        Args:
            - table_string (str): The string representation of the table to
              save.
            - output_directory (str, optional): Directory path where the table
              will be saved. If None, uses the class's default output directory.
            - filename (str, optional): Base name of the file (without
              extension). Default is 'table'.
            - file_format (str, optional): Format of the file: 'md', 'txt', or
              'tex'. Default is 'md'.

        Raises:
            ValueError: If an unsupported file format is provided.
        """
        # Use class's default output directory if none is provided
        if output_directory is None:
            output_directory = self.output_directory

        # Normalize and create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Validate file format
        valid_formats = {"md", "txt", "tex"}
        if file_format not in valid_formats:
            raise ValueError(
                f"Unsupported format '{file_format}'. "
                f"Must be one of {valid_formats}."
            )

        # Compose full path
        file_path = os.path.join(output_directory, f"{filename}.{file_format}")

        # Write the table string to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(table_string)

    def generate_column_uniqueness_report(
        self,
        max_width=80,
        separate_by_type=True,
        export_to_file=False,
        output_directory=None,
        filename="column_uniqueness_report",
        file_format="md",
    ) -> str:
        """
        Generate a table reporting uniqueness characteristics of DataFrame
        columns, showing single-valued and multi-valued fields, optionally
        organized by parameter type (tunable parameters vs output quantities).

        This method creates a formatted table with two columns:
        1. Single-valued fields with their unique values
        2. Multi-valued fields with the count of their unique values

        When separate_by_type is True, each column is further divided into
        tunable parameters and output quantities, with headers shown only in the
        left column.

        Args:
            - max_width (int, optional): Maximum width of the table in
              characters. Default is 80.
            - separate_by_type (bool, optional): Whether to separate fields by
              their type (tunable parameters vs output quantities). Default is
              True.
            - export_to_file (bool, optional): Whether to export the table to a
              file. Default is False.
            - output_directory (str, optional): Directory to save the output
              file. If None, uses the class's default output directory.
            - filename (str, optional): Base name for the output file (without
              extension). Default is 'column_uniqueness_report'.
            - file_format (str, optional): Format of the file: 'md', 'txt', or
              'tex'. Default is 'md'.

        Returns:
            str: A formatted string containing the table, optimized for Markdown
            display.
        """
        # Calculate how much space to allocate for each column
        half_width = (max_width - 3) // 2  # -3 for the separator and spacing

        # Create the header
        header_left = "Single-valued fields: unique value"
        header_right = "Multivalued fields: No of unique values"
        header = f"{header_left:<{half_width}} | {header_right}"

        # Create the separator line
        separator = "-" * max_width

        # Start building the table with header and separator
        table_lines = [header, separator]

        if separate_by_type:
            # GROUP ITEMS BY TYPE AND VALUE COUNT
            single_valued_tunable = [
                (col, self.unique_value_columns_dictionary[col])
                for col in self.list_of_single_valued_tunable_parameter_names
            ]
            single_valued_output = [
                (col, self.unique_value_columns_dictionary[col])
                for col in self.list_of_single_valued_output_quantity_names
            ]

            multi_valued_tunable = [
                (col, self.multivalued_columns_count_dictionary[col])
                for col in self.list_of_multivalued_tunable_parameter_names
            ]
            multi_valued_output = [
                (col, self.multivalued_columns_count_dictionary[col])
                for col in self.list_of_multivalued_output_quantity_names
            ]

            # Sort items alphabetically by column name within each group
            single_valued_tunable.sort(key=lambda x: x[0])
            single_valued_output.sort(key=lambda x: x[0])
            multi_valued_tunable.sort(key=lambda x: x[0])
            multi_valued_output.sort(key=lambda x: x[0])

            # Index counters for left and right columns
            left_index = 0
            right_index = 0

            # TODO: No need to repeat "TUNABLE PARAMETERS:" and "OUTPUT
            # QUANTITIES:" twice, once for every column Add tunable parameters
            # header only on the left side. Simply add the substring in the
            # middle of the line once.
            if single_valued_tunable or multi_valued_tunable:
                row = f"TUNABLE PARAMETERS:{' ' * (half_width - 19)} | "
                table_lines.append(row)

                # Determine maximum rows needed for this section
                max_rows = max(
                    len(single_valued_tunable) if single_valued_tunable else 0,
                    len(multi_valued_tunable) if multi_valued_tunable else 0,
                )

                # Output rows in parallel
                for _ in range(max_rows):
                    # Left column (single-valued tunable parameters)
                    left_col = ""
                    if left_index < len(single_valued_tunable):
                        col_name, value = single_valued_tunable[left_index]
                        left_index += 1

                        # Format the value appropriately
                        if isinstance(value, float):
                            if value == int(value):
                                formatted_value = str(int(value))
                            else:
                                formatted_value = f"{value:.8g}"
                        else:
                            formatted_value = str(value)

                        left_col = f"{col_name}: {formatted_value}"

                    # Right column (multi-valued tunable parameters)
                    right_col = ""
                    if right_index < len(multi_valued_tunable):
                        col_name, count = multi_valued_tunable[right_index]
                        right_index += 1
                        right_col = f"{col_name}: {count}"

                    # Create the row
                    row = f"{left_col:<{half_width}} | {right_col}"
                    table_lines.append(row)

            # Add a blank row between sections if both sections have content
            if (single_valued_tunable or multi_valued_output) and (
                single_valued_output or multi_valued_output
            ):
                table_lines.append(f"{'':<{half_width}} | ")

            # Reset index counters for output quantities
            left_index = 0
            right_index = 0

            # Add output quantities header only on the left side
            if single_valued_output or multi_valued_output:
                row = f"OUTPUT QUANTITIES:{' ' * (half_width - 18)} | "
                table_lines.append(row)

                # Determine maximum rows needed for this section
                max_rows = max(
                    len(single_valued_output) if single_valued_output else 0,
                    len(multi_valued_output) if multi_valued_output else 0,
                )

                # Output rows in parallel
                for _ in range(max_rows):
                    # Left column (single-valued output quantities)
                    left_col = ""
                    if left_index < len(single_valued_output):
                        col_name, value = single_valued_output[left_index]
                        left_index += 1

                        # Format the value appropriately
                        if isinstance(value, float):
                            if value == int(value):
                                formatted_value = str(int(value))
                            else:
                                formatted_value = f"{value:.8g}"
                        else:
                            formatted_value = str(value)

                        left_col = f"{col_name}: {formatted_value}"

                    # Right column (multi-valued output quantities)
                    right_col = ""
                    if right_index < len(multi_valued_output):
                        col_name, count = multi_valued_output[right_index]
                        right_index += 1
                        right_col = f"{col_name}: {count}"

                    # Create the row
                    row = f"{left_col:<{half_width}} | {right_col}"
                    table_lines.append(row)
        else:
            # SIMPLER VERSION - NO SEPARATION BY TYPE
            # Get all single-valued and multi-valued fields
            single_valued_fields = [
                (col, self.unique_value_columns_dictionary[col])
                for col in self.list_of_single_valued_column_names
            ]

            multi_valued_fields = [
                (col, self.multivalued_columns_count_dictionary[col])
                for col in self.list_of_multivalued_column_names
            ]

            # Sort alphabetically by column name
            single_valued_fields.sort(key=lambda x: x[0])
            multi_valued_fields.sort(key=lambda x: x[0])

            # Determine the maximum number of rows needed
            max_rows = max(len(single_valued_fields), len(multi_valued_fields))

            # Build each row of the table
            for i in range(max_rows):
                # Prepare left column content (single-valued fields)
                left_col = ""
                if i < len(single_valued_fields):
                    col_name, value = single_valued_fields[i]

                    # Format the value appropriately
                    if isinstance(value, float):
                        if value == int(value):
                            formatted_value = str(int(value))
                        else:
                            formatted_value = (
                                f"{value:.8g}"  # Use general format with precision
                            )
                    else:
                        formatted_value = str(value)

                    left_col = f"{col_name}: {formatted_value}"

                # Prepare right column content (multi-valued fields)
                right_col = ""
                if i < len(multi_valued_fields):
                    col_name, count = multi_valued_fields[i]
                    right_col = f"{col_name}: {count}"

                # Create the row with perfectly aligned separator
                row = left_col.ljust(half_width)
                row += " | "
                row += right_col

                table_lines.append(row)

        # Join all lines into a single string
        table_string = "\n".join(table_lines)

        # Export to file if requested
        if export_to_file:
            self._save_table_to_file(
                table_string=table_string,
                output_directory=output_directory,
                filename=filename,
                file_format=file_format,
            )

        return table_string

    def generate_grouped_summary_tables(
        self,
        value_variable: str,
        row_variable: str = None,
        column_variable: str = None,
        aggregation: str = "count",
        exclude_from_grouping: list = None,
        export_to_file: bool = False,
        output_directory: str = None,
        filename: str = "summary_tables",
        file_format: str = "md",
    ) -> str:
        """
        Generate grouped summary tables with optional row/column pivots and
        aggregations.

        Args:
            - value_variable (str): The variable to summarize.
            - row_variable (str, optional): Row variable in table.
            - column_variable (str, optional): Column variable in table.
            - aggregation (str, optional): One of 'count', 'list', 'len', 'min',
              'max', 'mean'.
            - exclude_from_grouping (list, optional): Additional tunable
              parameters to exclude from grouping.
            - export_to_file (bool, optional): Whether to export the table to a
              file. Default is False.
            - output_directory (str, optional): Directory to save the output
              file. If None, uses the class's default output directory.
            - filename (str, optional): Filename (without extension). Default:
              "summary_tables".
            - file_format (str, optional): Format to save in ('md', 'txt',
              etc.). Default: 'md'.

        Returns:
            str: A single formatted string containing all tables.
        """
        supported_aggregations = {"count", "list", "len", "min", "max", "mean"}
        if aggregation not in supported_aggregations:
            raise ValueError(
                f"Unsupported aggregation: '{aggregation}'. "
                f"Must be one of {supported_aggregations}."
            )

        for var in [value_variable, row_variable, column_variable]:
            if var is not None and var not in self.dataframe.columns:
                raise ValueError(f"'{var}' is not a column in the DataFrame.")

        # Determine which tunable parameters to exclude from grouping
        filter_out_vars = set()
        for var in [value_variable, row_variable, column_variable]:
            if (
                var is not None
                and var in self.list_of_multivalued_tunable_parameter_names
            ):
                filter_out_vars.add(var)

        if exclude_from_grouping:
            for var in exclude_from_grouping:
                if var not in self.dataframe.columns:
                    raise ValueError(
                        f"'{var}' in exclude_from_grouping is not a valid column."
                    )
                filter_out_vars.add(var)

        groupby_obj = self.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=list(filter_out_vars)
        )

        all_tables = []

        for idx, (group_keys, group_df) in enumerate(groupby_obj):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            group_dict = dict(
                zip(self.reduced_multivalued_tunable_parameter_names_list, group_keys)
            )

            table_lines = [f"#### Group {idx+1}:\n{group_dict}", ""]

            # 0D case
            if row_variable is None and column_variable is None:
                values = group_df[value_variable]
                if aggregation == "count":
                    result = values.nunique()
                elif aggregation == "list":
                    result = ", ".join(str(v) for v in sorted(values.unique()))
                elif aggregation == "len":
                    result = len(values)
                elif aggregation == "min":
                    result = values.min()
                elif aggregation == "max":
                    result = values.max()
                elif aggregation == "mean":
                    result = values.mean()
                table_lines.append(f"{value_variable} ({aggregation}): {result}")

            # 1D case
            elif row_variable is not None and column_variable is None:
                group = group_df.groupby(row_variable)[value_variable]
            elif column_variable is not None and row_variable is None:
                group = group_df.groupby(column_variable)[value_variable]

            if (row_variable is not None) ^ (column_variable is not None):
                result_dict = {}
                for key, subset in group:
                    if aggregation == "count":
                        result = subset.nunique()
                    elif aggregation == "list":
                        result = ", ".join(str(v) for v in sorted(subset.unique()))
                    elif aggregation == "len":
                        result = len(subset)
                    elif aggregation == "min":
                        result = subset.min()
                    elif aggregation == "max":
                        result = subset.max()
                    elif aggregation == "mean":
                        result = subset.mean()
                    result_dict[key] = result

                label = row_variable or column_variable
                table_lines.append(f"{label} | {aggregation} of {value_variable}")
                table_lines.append("-- | --")
                for k in sorted(result_dict):
                    table_lines.append(f"{k} | {result_dict[k]}")

            # 2D case
            elif row_variable is not None and column_variable is not None:
                pivot_df = group_df.groupby([row_variable, column_variable])[
                    value_variable
                ]

                if aggregation == "count":
                    table_df = pivot_df.nunique().unstack(fill_value=0)
                elif aggregation == "list":
                    table_df = pivot_df.apply(
                        lambda x: ", ".join(str(v) for v in sorted(x.unique()))
                    ).unstack(fill_value="")
                elif aggregation == "len":
                    table_df = pivot_df.size().unstack(fill_value=0)
                elif aggregation == "min":
                    table_df = pivot_df.min().unstack()
                elif aggregation == "max":
                    table_df = pivot_df.max().unstack()
                elif aggregation == "mean":
                    table_df = pivot_df.mean().unstack()

                # Format
                headers = [f"{row_variable} \\ {column_variable}"] + [
                    str(col) for col in table_df.columns
                ]
                table_lines.append(" | ".join(headers))
                table_lines.append(
                    " | ".join([":" + "-" * (len(h) - 1) for h in headers])
                )

                for row_index, row in table_df.iterrows():
                    row_str = f"{row_index} | " + " | ".join(
                        str(row[col]) for col in table_df.columns
                    )
                    table_lines.append(row_str)

            # Append final table
            all_tables.append("\n".join(table_lines))

        # Join all tables into one string with separators
        full_output = "\n\n---\n\n".join(all_tables)

        # Save if requested
        if export_to_file:
            self._save_table_to_file(
                table_string=full_output,
                output_directory=output_directory,
                filename=filename,
                file_format=file_format,
            )

        return full_output

    def generate_comparison_table_by_pivot(
        self,
        value_variable: str,
        pivot_variable: str,
        id_variable: str,
        comparison: str = "ratio",
        exclude_from_grouping: list = None,
        export_to_file: bool = False,
        output_directory: str = None,
        filename: str = "comparison_table",
        file_format: str = "md",
    ) -> str:
        """
        Generate comparison tables (e.g., ratio or difference) of a value
        variable across two categories of a pivot variable, grouped by the rest
        of the multivalued tunable parameters.

        Args:
            - value_variable (str): The numeric variable to compare (e.g.,
              'Condition_number').
            - pivot_variable (str): The categorical variable to compare across
              (e.g., 'Kernel_operator_type'). Must have exactly two unique
              values per group.
            - id_variable (str): The ID variable to match entries between pivot
              values (e.g., 'Configuration_label').
            - comparison (str): 'ratio' or 'difference'. Default is 'ratio'.
            - exclude_from_grouping (list, optional): Extra parameters to
              exclude from grouping.
            - export_to_file (bool, optional): Whether to export the table to a
              file. Default is False.
            - output_directory (str, optional): Directory to save output file.
              If None, uses the class's default output directory.
            - filename (str, optional): Name of the file to save (no extension).
              Default is 'comparison_table'.
            - file_format (str, optional): Output format: 'md', 'txt', etc.
              Default is 'md'.

        Returns:
            str: A Markdown-formatted string containing all group tables.
        """
        if comparison not in {"ratio", "difference"}:
            raise ValueError("comparison must be either 'ratio' or 'difference'")

        for var in [value_variable, pivot_variable, id_variable]:
            if var not in self.dataframe.columns:
                raise ValueError(f"'{var}' is not a valid column in the DataFrame.")

        # Determine which parameters to exclude from grouping
        filter_out_vars = {pivot_variable, id_variable}
        if exclude_from_grouping:
            filter_out_vars.update(exclude_from_grouping)

        filter_out_vars = list(filter_out_vars)

        groupby_obj = self.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=filter_out_vars
        )

        all_tables = []

        for idx, (group_keys, group_df) in enumerate(groupby_obj):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            group_dict = dict(
                zip(self.reduced_multivalued_tunable_parameter_names_list, group_keys)
            )

            # Pivot to wide format: one row per id_variable, columns are pivot values
            pivoted = group_df.pivot_table(
                index=id_variable,
                columns=pivot_variable,
                values=value_variable,
                aggfunc="mean",  # Can be changed to other strategies if needed
            )

            # Ensure exactly two pivot values exist per group
            pivot_columns = pivoted.columns.tolist()
            if len(pivot_columns) != 2:
                continue  # skip incomplete comparisons

            a, b = pivot_columns
            if comparison == "ratio":
                comparison_series = pivoted[a] / pivoted[b]
                comparison_label = f"{a} / {b}"
            elif comparison == "difference":
                comparison_series = pivoted[a] - pivoted[b]
                comparison_label = f"{a} - {b}"

            # Combine all into final DataFrame
            result_df = pivoted.copy()
            result_df[comparison_label] = comparison_series

            # Format table header
            table_lines = [f"#### Group {idx+1}:\n{group_dict}", ""]
            headers = (
                [id_variable] + [str(col) for col in pivot_columns] + [comparison_label]
            )
            table_lines.append(" | ".join(headers))
            table_lines.append(" | ".join([":" + "-" * (len(h) - 1) for h in headers]))

            for row_idx, row in result_df.iterrows():
                formatted_row = [str(row_idx)] + [
                    f"{row[col]:.6g}" if pd.notnull(row[col]) else "NaN"
                    for col in pivot_columns + [comparison_label]
                ]
                table_lines.append(" | ".join(formatted_row))

            all_tables.append("\n".join(table_lines))

        # Join all tables
        full_output = "\n\n---\n\n".join(all_tables)

        # Save if requested
        if export_to_file:
            self._save_table_to_file(
                table_string=full_output,
                output_directory=output_directory,
                filename=filename,
                file_format=file_format,
            )

        return full_output


class HDF5Analyzer:
    def __init__(self, hdf5_file_path: str):
        """
        Initialize the HDF5Analyzer.

        Parameters:
        -----------
        hdf5_file_path : str
            Path to the HDF5 file to analyze.
        """
        self.hdf5_file_path = hdf5_file_path
        self.hdf5_file = h5py.File(hdf5_file_path, "r")

    def get_group(self, group_path: str) -> h5py.Group:
        """
        Retrieve a group from the HDF5 file.

        Parameters:
        -----------
        group_path : str
            Path to the group to retrieve.

        Returns:
        --------
        h5py.Group
            The group object.
        """
        return self.hdf5_file[group_path]

    def list_datasets(self, group_path: str = "/") -> List[str]:
        """
        List all datasets in the specified group.

        Parameters:
        -----------
        group_path : str, optional
            Path to the group to list datasets from. Default is the root group.

        Returns:
        --------
        List[str]
            A list of dataset names in the specified group.
        """
        datasets = []

        def visitor_func(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append(name)

        self.hdf5_file[group_path].visititems(visitor_func)
        return datasets

    def get_dataset(self, dataset_path: str):
        """
        Retrieve a dataset from the HDF5 file.

        Parameters:
        -----------
        dataset_path : str
            Path to the dataset to retrieve.

        Returns:
        --------
        np.ndarray
            The dataset as a NumPy array.
        """
        return self.hdf5_file[dataset_path][()]

    def get_attributes(self, object_path: str) -> Dict[str, any]:
        """
        Retrieve attributes of a specified object (group or dataset).

        Parameters:
        -----------
        object_path : str
            Path to the object to retrieve attributes from.

        Returns:
        --------
        Dict[str, any]
            A dictionary of attributes.
        """
        obj = self.hdf5_file[object_path]
        return dict(obj.attrs)

    def list_common_datasets(self, group_path: str = "/") -> List[str]:
        """
        List common dataset names for all subgroups of a specific group.

        Parameters:
        -----------
        group_path : str
            Path to the group to analyze.

        Returns:
        --------
        List[str]
            A list of common dataset names.
        """
        subgroups = [
            obj
            for obj in self.hdf5_file[group_path].values()
            if isinstance(obj, h5py.Group)
        ]
        if not subgroups:
            return []

        common_datasets = set(subgroups[0].keys())
        for subgroup in subgroups[1:]:
            common_datasets.intersection_update(subgroup.keys())

        return list(common_datasets)

    def get_common_attributes(self, group_path: str) -> Dict[str, any]:
        """
        Get common attributes from all subgroups of a specific group.

        Parameters:
        -----------
        group_path : str
            Path to the group to analyze.

        Returns:
        --------
        Dict[str, any]
            A dictionary of common attribute names and their values.
        """
        subgroups = [
            obj
            for obj in self.hdf5_file[group_path].values()
            if isinstance(obj, h5py.Group)
        ]
        if not subgroups:
            return {}

        # Initialize with attributes of the first subgroup
        common_attributes = set(dict(subgroups[0].attrs).keys())

        for subgroup in subgroups[1:]:
            subgroup_attribute_names = set(dict(subgroups[0].attrs).keys())
            common_attributes.intersection_update(subgroup_attribute_names)

        return list(common_attributes)

    def get_common_dataset_values(
        self, group_path: str, dataset_name: str
    ) -> Dict[str, np.ndarray]:
        """
        Get values of a common dataset from all subgroups of a specific group.

        Parameters:
        -----------
        group_path : str
            Path to the group to analyze.
        dataset_name : str
            Name of the common dataset to retrieve values for.

        Returns:
        --------
        Dict[str, np.ndarray]
            A dictionary with subgroup names as keys and dataset values as NumPy
            arrays.
        """
        dataset_values = {}
        subgroups = [
            obj
            for obj in self.hdf5_file[group_path].values()
            if isinstance(obj, h5py.Group)
        ]

        for subgroup in subgroups:
            if dataset_name in subgroup:
                dataset_values[subgroup.name] = subgroup[dataset_name][()]

        return dataset_values

    def close(self):
        """Close the HDF5 file."""
        self.hdf5_file.close()
