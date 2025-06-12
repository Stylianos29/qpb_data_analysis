import os

import numpy as np
import pandas as pd

from ..data.hdf5_analyzer import HDF5Analyzer
from .data_plotter import DataPlotter


class HDF5Plotter(HDF5Analyzer):
    """
    A class for plotting datasets from HDF5 files using DataPlotter.

    This class extends HDF5Analyzer to provide methods for converting
    HDF5 datasets into DataFrames compatible with the DataPlotter class.
    It uses composition to work with DataPlotter instances for visualizing
    HDF5 data without needing to modify the DataPlotter class.

    Attributes:
        default_output_directory (str): Default directory to save plots
    """

    def __init__(self, hdf5_file_path, default_output_directory=None):
        """
        Initialize the HDF5Plotter with a path to an HDF5 file.

        Args:
            hdf5_file_path (str): Path to the HDF5 file to analyze and plot
            default_output_directory (str, optional): Default directory where plots will be saved
                If None, a temporary directory will be used.
        """
        # Initialize the parent HDF5Analyzer
        super().__init__(hdf5_file_path)

        # Set up default output directory
        if default_output_directory is None:
            import tempfile

            self.default_output_directory = tempfile.mkdtemp()
            print(
                f"No output directory specified. Using temporary directory: {self.default_output_directory}"
            )
        else:
            self.default_output_directory = default_output_directory
            os.makedirs(self.default_output_directory, exist_ok=True)

        # Cache for merged datasets
        self._merged_datasets_cache = {}

    def create_dataset_dataframe(
        self,
        dataset_name,
        add_time_column=True,
        time_offset=0,
        filter_func=None,
        include_group_path=False,
        flatten_arrays=True,
    ):
        """
        Create a pandas DataFrame from a specific dataset across all compatible groups.

        This method generates a DataFrame optimized for use with DataPlotter, with:
        - One row per dataset value (when flatten_arrays=True)
        - Columns for all relevant tunable parameters
        - Optional time/index column

        Args:
            dataset_name (str): Name of the dataset to extract
            add_time_column (bool, optional): Whether to add a time/index column. Defaults to True.
            time_offset (int, optional): Offset to apply to the time/index values. Defaults to 0.
            filter_func (callable, optional): Function to filter groups, takes group path and returns bool
            include_group_path (bool, optional): Include the full HDF5 group path as a column. Defaults to False.
            flatten_arrays (bool, optional): Whether to convert each array element to a separate row.
                                            If False, stores whole arrays as DataFrame values. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame containing the dataset values and associated parameters

        Raises:
            ValueError: If the dataset doesn't exist in the HDF5 file
        """
        if dataset_name not in self.output_quantities["paths_by_name"]:
            raise ValueError(f"Dataset '{dataset_name}' not found in HDF5 file.")

        # Collect all paths to the specified dataset
        dataset_paths = self.output_quantities["paths_by_name"][dataset_name]

        # Initialize lists to hold data for DataFrame construction
        data_rows = []

        # Process each dataset instance
        for dataset_path in dataset_paths:
            # Get the parent group of this dataset
            group_path = os.path.dirname(dataset_path)

            # Apply filter if provided
            if filter_func is not None and not filter_func(group_path):
                continue

            # Get dataset values
            dataset_values = self.file[dataset_path][()]

            # Get all parameter values for this group by traversing up the hierarchy
            group_parameters = {}

            # First, add all single-valued tunable parameters (global across file)
            group_parameters.update(self.tunable_parameters["single_valued"])

            # Then, add multi-valued tunable parameters by traversing up the hierarchy
            current_path = group_path
            while current_path:
                if current_path in self.parameters_by_group:
                    # Add parameters from the current level
                    for param_name, param_value in self.parameters_by_group[
                        current_path
                    ].items():
                        # Don't overwrite parameters found at lower levels
                        if param_name not in group_parameters:
                            group_parameters[param_name] = param_value

                # Move up to parent group
                current_path = os.path.dirname(current_path)

            # Handle the dataset values based on the flatten_arrays flag
            if (
                flatten_arrays
                and hasattr(dataset_values, "shape")
                and dataset_values.size > 1
            ):
                # Create a row for each element in the dataset array
                for idx, value in enumerate(dataset_values):
                    row = group_parameters.copy()
                    row[dataset_name] = value

                    if add_time_column:
                        row["time_index"] = idx + time_offset

                    if include_group_path:
                        row["group_path"] = group_path

                    data_rows.append(row)
            else:
                # Create a single row for the whole dataset
                row = group_parameters.copy()
                row[dataset_name] = dataset_values

                if include_group_path:
                    row["group_path"] = group_path

                data_rows.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(data_rows)

        # Ensure consistent column ordering
        column_order = []

        # Start with time_index if present
        if add_time_column and "time_index" in df.columns:
            column_order.append("time_index")

        # Add tunable parameters in a consistent order
        for param_name in sorted(self.tunable_parameters["single_valued"].keys()):
            if param_name in df.columns:
                column_order.append(param_name)

        for param_name in sorted(self.tunable_parameters["multi_valued"].keys()):
            if param_name in df.columns:
                column_order.append(param_name)

        # Add the dataset column
        column_order.append(dataset_name)

        # Add group_path at the end if present
        if include_group_path and "group_path" in df.columns:
            column_order.append("group_path")

        # Reorder columns (only include columns that exist in the DataFrame)
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]

        return df

    def create_multi_dataset_dataframe(
        self,
        dataset_names,
        time_aligned=True,
        add_time_column=True,
        time_offset=0,
        filter_func=None,
        include_group_path=False,
    ):
        """
        Create a DataFrame containing multiple datasets, with values aligned by time index.

        This method is useful when you want to plot multiple related datasets together,
        such as means and errors, or different correlation functions.

        Args:
            dataset_names (list): List of dataset names to include
            time_aligned (bool, optional): Whether datasets should align by time index.
                                        If True, assumes all datasets have same length. Defaults to True.
            add_time_column (bool, optional): Whether to add a time/index column. Defaults to True.
            time_offset (int, optional): Offset to apply to the time/index values. Defaults to 0.
            filter_func (callable, optional): Function to filter groups, takes group path and returns bool
            include_group_path (bool, optional): Include the full HDF5 group path as a column. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing all specified datasets and parameters

        Raises:
            ValueError: If any dataset doesn't exist or datasets have incompatible shapes
        """
        # Validate all dataset names
        for dataset_name in dataset_names:
            if dataset_name not in self.output_quantities["paths_by_name"]:
                raise ValueError(f"Dataset '{dataset_name}' not found in HDF5 file.")

        # Maps group path to parameter values
        group_params_cache = {}

        # Function to get parameters for a group path with caching
        def get_group_parameters(group_path):
            if group_path in group_params_cache:
                return group_params_cache[group_path]

            # Start with single-valued parameters
            params = dict(self.tunable_parameters["single_valued"])

            # Add multi-valued parameters by traversing hierarchy
            current_path = group_path
            while current_path:
                if current_path in self.parameters_by_group:
                    for param_name, param_value in self.parameters_by_group[
                        current_path
                    ].items():
                        if param_name not in params:
                            params[param_name] = param_value
                current_path = os.path.dirname(current_path)

            group_params_cache[group_path] = params
            return params

        # Initialize data collection for DataFrame
        data_rows = []

        if time_aligned:
            # Get all unique group paths containing all requested datasets
            group_paths = set()
            for dataset_name in dataset_names:
                dataset_paths = self.output_quantities["paths_by_name"][dataset_name]
                dataset_groups = {os.path.dirname(path) for path in dataset_paths}

                if not group_paths:
                    group_paths = dataset_groups
                else:
                    group_paths = group_paths.intersection(dataset_groups)

            # Process each group that contains all datasets
            for group_path in group_paths:
                # Apply filter if provided
                if filter_func is not None and not filter_func(group_path):
                    continue

                # Get the first dataset to determine length
                first_dataset = self.file[os.path.join(group_path, dataset_names[0])][
                    ()
                ]
                if not hasattr(first_dataset, "shape") or first_dataset.size < 2:
                    # Skip scalar or single-value datasets if we need time alignment
                    continue

                n_time_points = len(first_dataset)

                # Get parameter values for this group
                group_parameters = get_group_parameters(group_path)

                # Create a row for each time point
                for idx in range(n_time_points):
                    row = group_parameters.copy()

                    # Add each dataset value at this time point
                    for dataset_name in dataset_names:
                        dataset_path = os.path.join(group_path, dataset_name)
                        dataset_values = self.file[dataset_path][()]

                        # Ensure all datasets have the same length
                        if len(dataset_values) != n_time_points:
                            # Skip this group if datasets don't align
                            break

                        row[dataset_name] = dataset_values[idx]

                    # If we added all datasets successfully, add the time index and include the row
                    if len(row) == len(group_parameters) + len(dataset_names):
                        if add_time_column:
                            row["time_index"] = idx + time_offset

                        if include_group_path:
                            row["group_path"] = group_path

                        data_rows.append(row)

        else:
            # Non-time-aligned mode: process each dataset separately
            for dataset_name in dataset_names:
                dataset_paths = self.output_quantities["paths_by_name"][dataset_name]

                for dataset_path in dataset_paths:
                    group_path = os.path.dirname(dataset_path)

                    # Apply filter if provided
                    if filter_func is not None and not filter_func(group_path):
                        continue

                    # Get dataset values
                    dataset_values = self.file[dataset_path][()]

                    # Get parameter values for this group
                    group_parameters = get_group_parameters(group_path)

                    # Handle scalar datasets
                    if not hasattr(dataset_values, "shape") or dataset_values.size == 1:
                        row = group_parameters.copy()
                        row[dataset_name] = dataset_values

                        if include_group_path:
                            row["group_path"] = group_path

                        data_rows.append(row)
                    else:
                        # Create a row for each element in the dataset array
                        for idx, value in enumerate(dataset_values):
                            row = group_parameters.copy()
                            row[dataset_name] = value

                            if add_time_column:
                                row["time_index"] = idx + time_offset

                            if include_group_path:
                                row["group_path"] = group_path

                            data_rows.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(data_rows)

        # Ensure consistent column ordering as in create_dataset_dataframe
        column_order = []

        if add_time_column and "time_index" in df.columns:
            column_order.append("time_index")

        # Add tunable parameters
        for param_name in sorted(self.tunable_parameters["single_valued"].keys()):
            if param_name in df.columns:
                column_order.append(param_name)

        for param_name in sorted(self.tunable_parameters["multi_valued"].keys()):
            if param_name in df.columns:
                column_order.append(param_name)

        # Add dataset columns
        for dataset_name in dataset_names:
            if dataset_name in df.columns:
                column_order.append(dataset_name)

        if include_group_path and "group_path" in df.columns:
            column_order.append("group_path")

        # Reorder columns (only include columns that exist in the DataFrame)
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]

        return df

    def _merge_value_error_datasets(self, base_name, group_path=None):
        """
        Private method to merge corresponding mean and error datasets into value-error tuples.

        This method identifies and combines dataset pairs following the pattern:
        - "[base_name]_mean_values" - Contains the mean/central values
        - "[base_name]_error_values" - Contains the corresponding error values

        Args:
            base_name (str): Base name of the dataset without the "_mean_values" or "_error_values" suffix
            group_path (str, optional): If provided, only look for datasets in this specific group path

        Returns:
            dict: A dictionary mapping group paths to numpy arrays of tuples (value, error)

        Raises:
            ValueError: If corresponding mean/error datasets can't be found or have mismatched shapes
        """
        # Generate the full dataset names for mean and error values
        if "_mean_values" in base_name or "_error_values" in base_name:
            # Strip existing suffixes if present
            base_name = base_name.replace("_mean_values", "").replace(
                "_error_values", ""
            )

        mean_dataset_name = f"{base_name}_mean_values"
        error_dataset_name = f"{base_name}_error_values"

        # Check if both datasets exist
        if mean_dataset_name not in self.output_quantities["paths_by_name"]:
            raise ValueError(
                f"Mean dataset '{mean_dataset_name}' not found in HDF5 file."
            )
        if error_dataset_name not in self.output_quantities["paths_by_name"]:
            raise ValueError(
                f"Error dataset '{error_dataset_name}' not found in HDF5 file."
            )

        # Get all paths for both datasets
        mean_paths = self.output_quantities["paths_by_name"][mean_dataset_name]
        error_paths = self.output_quantities["paths_by_name"][error_dataset_name]

        # Create lookup dictionaries for faster access
        mean_by_group = {os.path.dirname(path): path for path in mean_paths}
        error_by_group = {os.path.dirname(path): path for path in error_paths}

        # Find common groups
        if group_path:
            common_groups = (
                [group_path]
                if group_path in mean_by_group and group_path in error_by_group
                else []
            )
        else:
            common_groups = [
                group for group in mean_by_group if group in error_by_group
            ]

        if not common_groups:
            raise ValueError(
                f"No matching groups found for both '{mean_dataset_name}' and '{error_dataset_name}'"
            )

        # Create result dictionary
        merged_data = {}

        # Process each common group
        for group in common_groups:
            mean_values = self.file[mean_by_group[group]][()]
            error_values = self.file[error_by_group[group]][()]

            # Ensure shapes match
            if mean_values.shape != error_values.shape:
                raise ValueError(
                    f"Shape mismatch in group '{group}': "
                    f"mean shape {mean_values.shape}, error shape {error_values.shape}"
                )

            # Create array of tuples
            merged_array = np.array(list(zip(mean_values, error_values)))
            merged_data[group] = merged_array

        return merged_data, base_name

    def create_merged_value_error_dataframe(
        self,
        base_name,
        add_time_column=True,
        time_offset=0,
        filter_func=None,
        include_group_path=False,
    ):
        """
        Create a DataFrame with value-error tuples for a given base dataset name.

        This method automatically finds and merges corresponding mean and error datasets,
        creating a DataFrame where each value is a tuple of (mean, error), which is directly
        compatible with error bar plotting in DataPlotter.

        Args:
            base_name (str): Base name of the dataset (without "_mean_values" or "_error_values" suffix)
            add_time_column (bool, optional): Whether to add a time/index column. Defaults to True.
            time_offset (int, optional): Offset to apply to the time/index values. Defaults to 0.
            filter_func (callable, optional): Function to filter groups
            include_group_path (bool, optional): Include the full HDF5 group path as a column

        Returns:
            pd.DataFrame: DataFrame with merged value-error tuples and associated parameters

        Raises:
            ValueError: If corresponding datasets can't be found or have incompatible shapes
        """
        # Get merged data for all matching groups
        merged_data, clean_base_name = self._merge_value_error_datasets(base_name)

        # Output column name (clean base name without any suffixes)
        output_column_name = clean_base_name

        # Initialize data rows
        data_rows = []

        # Process each group
        for group_path, merged_array in merged_data.items():
            # Apply filter if provided
            if filter_func is not None and not filter_func(group_path):
                continue

            # Get parameters for this group
            group_parameters = {}

            # First, add all single-valued tunable parameters
            group_parameters.update(self.tunable_parameters["single_valued"])

            # Then, add multi-valued tunable parameters by traversing up the hierarchy
            current_path = group_path
            while current_path:
                if current_path in self.parameters_by_group:
                    for param_name, param_value in self.parameters_by_group[
                        current_path
                    ].items():
                        if param_name not in group_parameters:
                            group_parameters[param_name] = param_value
                current_path = os.path.dirname(current_path)

            # Create a row for each value-error pair
            for idx, value_error_pair in enumerate(merged_array):
                row = group_parameters.copy()
                row[output_column_name] = tuple(value_error_pair)  # Ensure it's a tuple

                if add_time_column:
                    row["time_index"] = idx + time_offset

                if include_group_path:
                    row["group_path"] = group_path

                data_rows.append(row)

        # Convert to DataFrame with consistent column ordering
        df = pd.DataFrame(data_rows)

        # Ensure consistent column ordering
        column_order = []

        if add_time_column and "time_index" in df.columns:
            column_order.append("time_index")

        for param_name in sorted(self.tunable_parameters["single_valued"].keys()):
            if param_name in df.columns:
                column_order.append(param_name)

        for param_name in sorted(self.tunable_parameters["multi_valued"].keys()):
            if param_name in df.columns:
                column_order.append(param_name)

        column_order.append(output_column_name)

        if include_group_path and "group_path" in df.columns:
            column_order.append("group_path")

        # Apply column ordering
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]

        return df

    def plot_datasets(
        self,
        dataset_names,
        output_directory=None,
        x_axis="time_index",
        time_offset=0,
        filter_func=None,
        plot_kwargs=None,
        group_by=None,
        exclude_from_grouping=None,
        merge_value_error=False,
    ):
        """
        Plot datasets directly from the HDF5 file using DataPlotter.

        This is a convenience method that combines dataframe creation and
        plotting.

        Args:
            - dataset_names (str or list): Name(s) of dataset(s) to plot
            - output_directory (str, optional): Directory where plots will be
              saved
            - x_axis (str, optional): Column to use as x-axis. Defaults to
              'time_index'.
            - filter_func (callable, optional): Function to filter groups
            - plot_kwargs (dict, optional): Additional keyword arguments for
              DataPlotter.plot()
            - group_by (str, optional): Column to use for grouping in combined
              plots
            - exclude_from_grouping (list, optional): Parameters to exclude from
              grouping
            - merge_value_error (bool, optional): If True, try to merge
              value/error datasets. Default is False.

        Returns:
            DataPlotter: The configured DataPlotter instance

        Raises:
            ValueError: If datasets don't exist or are incompatible
        """
        # Use default output directory if not provided
        if output_directory is None:
            output_directory = self.default_output_directory

        # Handle value-error merging option
        if merge_value_error:
            if isinstance(dataset_names, str):
                # Single base dataset name
                df = self.create_merged_value_error_dataframe(
                    dataset_names,
                    add_time_column=(x_axis == "time_index"),
                    time_offset=time_offset,
                    filter_func=filter_func,
                )
                # Update dataset_names to be the cleaned base name
                dataset_names = [
                    dataset_names.replace("_mean_values", "").replace(
                        "_error_values", ""
                    )
                ]
            else:
                # Can't merge multiple datasets at once
                raise ValueError(
                    "merge_value_error=True can only be used with a single dataset name. "
                    "For multiple datasets, create merged dataframes individually."
                )
        else:
            # Standard dataset handling (no merging)
            if isinstance(dataset_names, str):
                dataset_names = [dataset_names]
                df = self.create_dataset_dataframe(
                    dataset_names[0],
                    add_time_column=(x_axis == "time_index"),
                    filter_func=filter_func,
                )
            else:
                # Multiple datasets - use time-aligned method
                df = self.create_multi_dataset_dataframe(
                    dataset_names,
                    add_time_column=(x_axis == "time_index"),
                    filter_func=filter_func,
                )

        # Initialize DataPlotter with the created DataFrame
        plotter = DataPlotter(df, output_directory)

        # Plot each dataset against the x-axis
        for dataset_name in dataset_names:
            # Clean the dataset name if it contains suffixes
            clean_name = dataset_name.replace("_mean_values", "").replace(
                "_error_values", ""
            )

            # Find the matching column name in the DataFrame
            plot_column = None
            for col in df.columns:
                if col == dataset_name or col == clean_name:
                    plot_column = col
                    break

            if plot_column:
                # Set the x and y variables for plotting
                plotter.set_plot_variables(x_axis, plot_column)

                # Setup plot kwargs
                if plot_kwargs is None:
                    plot_kwargs = {}

                # Plot the dataset
                plot_args = {
                    "grouping_variable": group_by,
                    "excluded_from_grouping_list": exclude_from_grouping,
                }
                plot_args.update(plot_kwargs)

                plotter.plot(**plot_args)
