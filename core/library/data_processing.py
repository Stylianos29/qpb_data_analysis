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
    def __init__(self, dataframe: pd.DataFrame):

        self.dataframe = dataframe
        self.original_dataframe = self.dataframe.copy()

        self.list_of_dataframe_fields = self.dataframe.columns.tolist()

        # Extract list of tunable parameter names for the provided dataframe
        self.list_of_tunable_parameter_names_from_dataframe = [
            column_name
            for column_name in self.list_of_dataframe_fields
            if column_name in constants.TUNABLE_PARAMETER_NAMES_LIST
        ]

        # Extract list of output quantity names for the provided dataframe
        self.list_of_output_quantity_names_from_dataframe = [
            column_name
            for column_name in self.list_of_dataframe_fields
            if column_name in constants.OUTPUT_QUANTITY_NAMES_LIST
        ]

        self.set_data_based_attributes()

    # TODO: This must be renamed as the update method
    def set_data_based_attributes(self):

        self.list_of_dataframe_fields = self.dataframe.columns.tolist()

        # Extract dictionary with single-valued fields and their values
        self.single_valued_fields_dictionary = self.get_single_valued_fields()

        # Extract dictionary of multivalued fields with number of unique values
        self.multivalued_fields_dictionary = self.get_multivalued_fields()

        self.list_of_tunable_multivalued_parameter_names = list(
            set(self.list_of_tunable_parameter_names_from_dataframe)
            & set(self.multivalued_fields_dictionary.keys())
        )

        self.tunable_single_valued_parameters_dictionary = {
            parameter: value
            for parameter, value in self.single_valued_fields_dictionary.items()
            if parameter in self.list_of_tunable_parameter_names_from_dataframe
        }

    def get_single_valued_fields(self) -> dict:

        single_valued_fields_dictionary = {}

        for column in self.dataframe.columns:
            unique_values = self.dataframe[column].unique()

            # Only add columns where there is a single unique value
            if len(unique_values) == 1:
                # Store the unique value directly, not as a list
                single_valued_fields_dictionary[column] = unique_values[0]

        return single_valued_fields_dictionary

    def get_multivalued_fields(self) -> dict:

        multivalued_fields_dictionary = {}

        for column in self.dataframe.columns:
            # Get the number of unique values in the column
            unique_values_count = self.dataframe[column].nunique()
            if unique_values_count > 1:
                # If there's more than one unique value, add it to dictionary
                multivalued_fields_dictionary[column] = unique_values_count

        return multivalued_fields_dictionary

    def group_by_reduced_tunable_parameters_list(self, filter_out_parameters: list):

        # self.reduced_tunable_parameters_list = list(
        #     set(self.list_of_tunable_parameter_names_from_dataframe)
        #     - set(self.single_valued_fields_dictionary)
        #     - set(filter_out_parameters)
        # )

        self.reduced_tunable_parameters_list = [
            item
            for item in self.list_of_tunable_multivalued_parameter_names
            if item not in filter_out_parameters
        ]

        self.list_of_tunable_multivalued_parameter_names

        print(self.reduced_tunable_parameters_list)

        if self.reduced_tunable_parameters_list:
            return self.dataframe.groupby(
                self.reduced_tunable_parameters_list, observed=True
            )
        else:
            return [(None, self.dataframe)]

    def restrict_data(self, condition: str):
        """
        Restricts the DataFrame to rows that satisfy the given condition.

        Args:
            condition (str): A condition to filter rows, e.g., "parameter >= 1".

        Returns:
            None: Modifies the DataFrameAnalyzer's DataFrame in-place.
        """
        try:
            # Use `query` to filter the DataFrame based on the condition
            self.dataframe = self.dataframe.query(condition)
            # Recompute data-based dictionaries as the DataFrame has changed
            self.set_data_based_attributes()

        except Exception as e:
            raise ValueError(f"Failed to apply condition '{condition}': {e}")

    def restore_entire_dataframe(self):

        self.dataframe = self.original_dataframe
        self.set_data_based_attributes()


class TableGenerator(DataFrameAnalyzer):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def generate_table(self, group_by_columns: list, aggregation_functions: dict):
        """
        Generate a table by grouping the DataFrame based on the specified columns
        and applying the specified aggregation functions.

        Parameters:
        -----------
        group_by_columns : list
            A list of column names to group the DataFrame by.
        aggregation_functions : dict
            A dictionary mapping column names to aggregation functions.

        Returns:
        --------
        pd.DataFrame
            The generated table.
        """
        return self.dataframe.groupby(group_by_columns).agg(aggregation_functions)


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

    def list_common_datasets(self, group_path: str) -> List[str]:
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
        subgroups = [obj for obj in self.hdf5_file[group_path].values() if isinstance(obj, h5py.Group)]
        if not subgroups:
            return []

        common_datasets = set(subgroups[0].keys())
        for subgroup in subgroups[1:]:
            common_datasets.intersection_update(subgroup.keys())

        return list(common_datasets)

    def get_common_dataset_values(self, group_path: str, dataset_name: str) -> Dict[str, np.ndarray]:
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
            A dictionary with subgroup names as keys and dataset values as NumPy arrays.
        """
        dataset_values = {}
        subgroups = [obj for obj in self.hdf5_file[group_path].values() if isinstance(obj, h5py.Group)]
        
        for subgroup in subgroups:
            if dataset_name in subgroup:
                dataset_values[subgroup.name] = subgroup[dataset_name][()]
        
        return dataset_values

    def close(self):
        """Close the HDF5 file."""
        self.hdf5_file.close()
