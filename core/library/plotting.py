import os, shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import h5py
from collections import defaultdict

from library import DataFrameAnalyzer, TableGenerator, constants


class EnhancedHDF5Analyzer:
    """
    An enhanced analyzer for HDF5 files, with a focus on parameter categorization and dataset analysis aligned with the
    project's conceptual framework.

    This class analyzes HDF5 files with the following structure and assumptions:
    - Second level groups contain single-valued tunable parameters as attributes
    - Subsequent subgroups contain multi-valued tunable parameters as attributes
    - Datasets are output quantities, stored in the subgroups

    Attributes:
        file_path (str): Path to the HDF5 file file (h5py.File): The opened HDF5 file object

        tunable_parameters (dict): Dictionary categorizing tunable parameters:
            - single_valued: Dict mapping parameter names to their single values
            - multi_valued: Dict mapping parameter names to sets of unique values

        output_quantities (dict): Dictionary of all datasets (output quantities):
            - names: List of all dataset names found in the file
            - paths_by_name: Dict mapping dataset names to lists of full paths

        groups_by_level (dict): Dict mapping hierarchy levels to lists of group paths parameters_by_group (dict): Dict
        mapping group paths to their parameter dicts
    """

    def __init__(self, hdf5_file_path: str):
        """
        Initialize the analyzer with the path to an HDF5 file.

        Args:
            hdf5_file_path (str): Path to the HDF5 file to analyze
        """
        if not os.path.exists(hdf5_file_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_file_path}")

        self.file_path = hdf5_file_path
        self.file = h5py.File(hdf5_file_path, "r")

        # Storage for parameter categorization
        self.tunable_parameters = {
            "single_valued": {},  # Maps parameter names to their single values
            "multi_valued": defaultdict(set),  # Maps parameter names to sets of values
        }

        # Storage for output quantities (datasets)
        self.output_quantities = {
            "names": [],  # List of all dataset names
            "paths_by_name": defaultdict(list),  # Maps dataset names to their paths
            "single_valued": {},  # Will store dataset names with only one unique value
            "multi_valued": defaultdict(
                int
            ),  # Will store dataset names with count of unique values
        }

        # Path mappings
        self.groups_by_level = defaultdict(list)  # Maps level to list of group paths
        self.parameters_by_group = {}  # Maps group paths to their parameters

        # Perform initial analysis
        self._analyze_structure()

    def _analyze_structure(self):
        """
        Analyze the HDF5 file structure, categorizing parameters and datasets.

        This method:
        1. Identifies all groups and their hierarchy levels
        2. Extracts parameters (attributes) from each group
        3. Categorizes parameters as single-valued or multi-valued tunable parameters
        4. Identifies all datasets (output quantities)
        """
        # Clear previous analysis results
        self.tunable_parameters = {
            "single_valued": {},
            "multi_valued": defaultdict(set),
        }
        self.output_quantities = {
            "names": [],
            "paths_by_name": defaultdict(list),
            "single_valued": {},
            "multi_valued": defaultdict(int),
        }
        self.groups_by_level = defaultdict(list)
        self.parameters_by_group = {}

        # Track all values for each parameter across groups
        all_param_values = defaultdict(set)

        # First pass: Collect groups, parameters, and datasets
        def visit_node(name, obj):
            # Track group levels (depth in the hierarchy)
            if isinstance(obj, h5py.Group):
                level = name.count("/") + 1  # Root is level 0
                self.groups_by_level[level].append(name)

                # Extract and store parameters (attributes)
                params = dict(obj.attrs)
                self.parameters_by_group[name] = params

                # Track parameter values for categorization
                for param_name, param_value in params.items():
                    all_param_values[param_name].add(param_value)

            # Track datasets (output quantities)
            elif isinstance(obj, h5py.Dataset):
                dataset_name = os.path.basename(name)

                # Add to list of dataset names if not already there
                if dataset_name not in self.output_quantities["names"]:
                    self.output_quantities["names"].append(dataset_name)

                # Map dataset name to its full path(s)
                self.output_quantities["paths_by_name"][dataset_name].append(name)

        # Visit all nodes in the file
        self.file.visititems(visit_node)

        # Second pass: Categorize parameters by value count
        for param_name, values in all_param_values.items():
            if len(values) == 1:
                # Single-valued tunable parameter
                self.tunable_parameters["single_valued"][param_name] = next(
                    iter(values)
                )
            else:
                # Multi-valued tunable parameter
                self.tunable_parameters["multi_valued"][param_name] = values

        # Third pass: Analyze datasets for uniqueness
        for dataset_name, paths in self.output_quantities["paths_by_name"].items():
            # Collect all unique values of this dataset across the file
            all_values = set()
            try:
                for path in paths:
                    dataset_value = self.file[path][()]

                    # Handle different dataset shapes
                    if dataset_value.shape == ():  # Scalar
                        all_values.add(float(dataset_value))
                    else:
                        # For array datasets, we consider each one unique unless
                        # exactly equal Adding a tuple representation of the
                        # array
                        all_values.add(tuple(dataset_value.flatten()))

                # Categorize the dataset
                if len(all_values) == 1:
                    # Single-valued output quantity
                    value = next(iter(all_values))
                    if isinstance(value, tuple):
                        # If it's a flattened array, get the original array back
                        for path in paths:
                            self.output_quantities["single_valued"][dataset_name] = (
                                self.file[path][()]
                            )
                            break
                    else:
                        # It's a scalar
                        self.output_quantities["single_valued"][dataset_name] = value
                else:
                    # Multi-valued output quantity
                    self.output_quantities["multi_valued"][dataset_name] = len(
                        all_values
                    )

            except (TypeError, ValueError):
                # Some datasets might not be easily comparable or have complex structures
                # Consider them as multi-valued in this case
                self.output_quantities["multi_valued"][dataset_name] = len(paths)

    def print_unique_values(self, parameter_name):
        """
        Print the count and list of unique values for a specified parameter.

        Args:
            parameter_name (str): The name of the parameter to analyze.

        Raises:
            ValueError: If the parameter doesn't exist or isn't multi-valued.
        """
        if parameter_name not in self.tunable_parameters["multi_valued"]:
            if parameter_name in self.tunable_parameters["single_valued"]:
                value = self.tunable_parameters["single_valued"][parameter_name]
                print(f"Parameter '{parameter_name}' has only one value: {value}")
                return
            else:
                raise ValueError(f"Parameter '{parameter_name}' not found.")

        values = sorted(self.tunable_parameters["multi_valued"][parameter_name])
        print(f"Parameter '{parameter_name}' has {len(values)} unique values:")
        print(values)

    def get_dataset_values_by_parameters(self, dataset_name, parameters_dict):
        """
        Get values of a specific dataset that match the given parameter values.

        Args:
            dataset_name (str): Name of the dataset to retrieve parameters_dict (dict): Dictionary of parameter
            name/value pairs to filter by.
                Example: {'resolution': 32, 'kernel_type': 'Wilson'}

        Returns:
            list: Values of the specified dataset from groups matching the parameter criteria

        Raises:
            ValueError: If the dataset doesn't exist
        """
        if dataset_name not in self.output_quantities["paths_by_name"]:
            raise ValueError(f"Dataset '{dataset_name}' not found in HDF5 file.")

        # Find all groups that contain this dataset
        dataset_paths = self.output_quantities["paths_by_name"][dataset_name]
        matching_values = []

        for path in dataset_paths:
            # Get the parent group of this dataset
            group_path = os.path.dirname(path)

            # Check if group matches all specified parameters
            match = True
            for param_name, param_value in parameters_dict.items():
                # We need to look up the hierarchy to find parameter values
                current_path = group_path
                param_found = False

                while current_path:
                    if (
                        current_path in self.parameters_by_group
                        and param_name in self.parameters_by_group[current_path]
                    ):
                        # Parameter found in this group or parent group
                        group_value = self.parameters_by_group[current_path][param_name]
                        if group_value != param_value:
                            match = False
                        param_found = True
                        break
                    # Move up to parent group
                    current_path = os.path.dirname(current_path)

                # If parameter not found anywhere in hierarchy, it's not a match
                if not param_found:
                    match = False
                    break

            # If all parameters matched, add the dataset value
            if match:
                matching_values.append(self.file[path][()])

        return matching_values

    def get_dataset_values(self, dataset_name):
        """
        Get all values for a specific dataset across the file.

        Args:
            dataset_name (str): Name of the dataset

        Returns:
            list: List of all values for the dataset from across the file
        """
        values = []
        for path in self.output_quantities["paths_by_name"].get(dataset_name, []):
            values.append(self.file[path][()])
        return values

    def generate_uniqueness_report(self, max_width=80, separate_by_type=True):
        """
        Generate a report on the uniqueness of parameters and datasets.

        This method creates a formatted table showing:
        1. Single-valued parameters/datasets with their values
        2. Multi-valued parameters/datasets with counts of unique values

        Args:
            max_width (int): Maximum width of the report in characters
            separate_by_type (bool): Whether to separate by tunable parameters
            vs output quantities

        Returns:
            str: A formatted string containing the report
        """
        # Calculate how much space to allocate for each column
        half_width = (max_width - 3) // 2  # -3 for separator and spacing

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

            # Get single-valued tunable parameters
            single_valued_tunable = [
                (param, value)
                for param, value in self.tunable_parameters["single_valued"].items()
            ]
            single_valued_tunable.sort(key=lambda x: x[0])

            # Get multi-valued tunable parameters
            multi_valued_tunable = [
                (param, len(values))
                for param, values in self.tunable_parameters["multi_valued"].items()
            ]
            multi_valued_tunable.sort(key=lambda x: x[0])

            # Get single-valued output quantities (datasets)
            single_valued_output = [
                (name, value)
                for name, value in self.output_quantities["single_valued"].items()
            ]
            single_valued_output.sort(key=lambda x: x[0])

            # Get multi-valued output quantities (datasets)
            multi_valued_output = [
                (name, count)
                for name, count in self.output_quantities["multi_valued"].items()
            ]
            multi_valued_output.sort(key=lambda x: x[0])

            # Index counters for left and right columns
            left_index = 0
            right_index = 0

            # Add tunable parameters header centered across the entire line width
            if single_valued_tunable or multi_valued_tunable:
                header_text = "TUNABLE PARAMETERS"
                padding = (max_width - len(header_text)) // 2
                row = " " * padding + header_text
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
            if (single_valued_tunable or multi_valued_tunable) and (
                single_valued_output or multi_valued_output
            ):
                table_lines.append("")

            # Reset index counters for output quantities
            left_index = 0
            right_index = 0

            # Add output quantities header centered across the entire line width
            if single_valued_output or multi_valued_output:
                header_text = "OUTPUT QUANTITIES"
                padding = (max_width - len(header_text)) // 2
                row = " " * padding + header_text
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

                        # Format the value appropriately - handle array values
                        if isinstance(value, np.ndarray):
                            if value.size == 1:  # Single element array
                                val = value.item()
                                if isinstance(val, float) and val == int(val):
                                    formatted_value = str(int(val))
                                else:
                                    formatted_value = (
                                        f"{val:.8g}"
                                        if isinstance(val, float)
                                        else str(val)
                                    )
                            else:
                                # For arrays, show shape
                                formatted_value = f"array{value.shape}"
                        elif isinstance(value, float):
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
            single_valued_fields = list(single_valued_tunable) + list(
                single_valued_output
            )
            multi_valued_fields = list(multi_valued_tunable) + list(multi_valued_output)

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

                    # Format the value appropriately (similar to above)
                    if isinstance(value, np.ndarray):
                        if value.size == 1:
                            val = value.item()
                            if isinstance(val, float) and val == int(val):
                                formatted_value = str(int(val))
                            else:
                                formatted_value = (
                                    f"{val:.8g}" if isinstance(val, float) else str(val)
                                )
                        else:
                            formatted_value = f"array{value.shape}"
                    elif isinstance(value, float):
                        if value == int(value):
                            formatted_value = str(int(value))
                        else:
                            formatted_value = f"{value:.8g}"
                    else:
                        formatted_value = str(value)

                    left_col = f"{col_name}: {formatted_value}"

                # Prepare right column content (multi-valued fields)
                right_col = ""
                if i < len(multi_valued_fields):
                    col_name, count = multi_valued_fields[i]
                    right_col = f"{col_name}: {count}"

                # Create the row with aligned separator
                row = f"{left_col:<{half_width}} | {right_col}"
                table_lines.append(row)

        # Join all lines into a single string
        return "\n".join(table_lines)

    def group_by_multivalued_tunable_parameters(self, filter_out_parameters_list=None):
        """
        Group the HDF5 data by multivalued tunable parameters, with options to
        exclude parameters.

        This method creates groups based on unique combinations of the
        multi-valued tunable parameters, similar to the DataFrameAnalyzer
        method.

        Args:
            filter_out_parameters_list (list, optional): Parameters to exclude
            from grouping

        Returns:
            dict: A mapping of parameter combinations (as tuples) to lists of
            dataset paths
        """
        # Start with all multi-valued tunable parameters
        grouping_params = list(self.tunable_parameters["multi_valued"].keys())

        # Filter out specified parameters
        if filter_out_parameters_list:
            grouping_params = [
                p for p in grouping_params if p not in filter_out_parameters_list
            ]

        # Store this for use in other methods
        self.reduced_multivalued_tunable_parameter_names_list = grouping_params

        # If no grouping parameters remain, return a single group with all paths
        if not grouping_params:
            return {"all_data": self.groups_by_level[max(self.groups_by_level.keys())]}

        # Group paths by parameter value combinations
        grouped_paths = defaultdict(list)

        # Get leaf groups (those at the deepest level)
        max_level = max(self.groups_by_level.keys())
        leaf_groups = self.groups_by_level[max_level]

        # For each leaf group, extract the values of grouping parameters
        for group_path in leaf_groups:
            # Find parameter values for this group
            param_values = []

            # We need to look up the hierarchy to find all parameter values
            current_path = group_path
            while current_path:
                # Check if this path has parameters
                if current_path in self.parameters_by_group:
                    group_params = self.parameters_by_group[current_path]
                    for param in grouping_params:
                        if param in group_params:
                            param_values.append((param, group_params[param]))

                # Move up one level
                current_path = os.path.dirname(current_path)

            # Create a dict of parameter values
            param_dict = dict(param_values)

            # Only include groups that have all the grouping parameters
            if len(param_dict) == len(grouping_params):
                # Create a tuple of parameter values (in the same order as grouping_params)
                group_key = tuple(param_dict[param] for param in grouping_params)
                grouped_paths[group_key].append(group_path)

        return grouped_paths

    def _get_dataset_values_by_group(self, dataset_name, group_paths):
        """
        Get values of a dataset from specified group paths.

        Args:
            - dataset_name (str): Name of the dataset to retrieve
            - group_paths (list): List of group paths to look for the dataset

        Returns:
            list: Dataset values from the specified groups
        """
        values = []
        for group_path in group_paths:
            dataset_path = os.path.join(group_path, dataset_name)
            if dataset_path in self.file:
                values.append(self.file[dataset_path][()])
        return values

    def close(self):
        """Close the HDF5 file."""
        self.file.close()

    def __del__(self):
        """Ensure file is closed when object is garbage collected."""
        try:
            self.file.close()
        except:
            pass


class DataPlotter(DataFrameAnalyzer):
    """
    A plotting interface for visualizing grouped data from a DataFrame using
    matplotlib.

    This class extends `DataFrameAnalyzer` to provide plotting functionality
    tailored to data that varies across subsets of tunable parameters. It
    supports generating visual summaries of results for each combination of
    multivalued tunable parameters, and is designed to produce
    publication-quality figures in a structured directory hierarchy.

    Attributes:
    -----------
    dataframe : pd.DataFrame
        The main DataFrame containing raw or processed data to be visualized.
    plots_directory : str
        The base output directory where all plots will be saved.
    individual_plots_subdirectory : str
        Subdirectory within `plots_directory` used for individual plots.
    combined_plots_subdirectory : str
        Subdirectory within `plots_directory` used for combined plots.
    xaxis_variable_name : Optional[str]
        Name of the variable to be used as x-axis in plots.
    yaxis_variable_name : Optional[str]
        Name of the variable to be used as y-axis in plots.
    plots_base_name : Optional[str]
        Base string used in naming plot files and directories.
    """

    def __init__(self, dataframe: pd.DataFrame, plots_directory: str):
        """
        Initialize the DataPlotter with a DataFrame and an output directory for
        plots.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input data for plotting. This is expected to include tunable
            parameters and output quantities as defined in the analysis logic.
        plots_directory : str
            The base path where plots will be saved. Subdirectories will be
            created inside this path for individual and grouped plots.

        Raises:
        -------
        TypeError:
            If the input is not a Pandas DataFrame.
        ValueError:
            If the provided `plots_directory` does not point to a valid
            directory.
        """
        super().__init__(dataframe)

        if not os.path.isdir(plots_directory):
            raise ValueError(f"Invalid plots directory: '{plots_directory}'")

        self.plots_directory = plots_directory
        self.individual_plots_subdirectory = plots_directory
        self.combined_plots_subdirectory = plots_directory

        self.xaxis_variable_name = None
        self.yaxis_variable_name = None
        self.plots_base_name = None

    def generate_column_uniqueness_report(
        self, max_width=80, separate_by_type=True
    ) -> str:
        table_generator = TableGenerator(self.dataframe)
        return table_generator.generate_column_uniqueness_report(
            max_width=max_width,
            separate_by_type=separate_by_type,
            export_to_file=False,
        )

    def _prepare_plot_subdirectory(
        self, subdir_name: str, clear_existing: bool = False
    ) -> str:
        """
        Create or clean a subdirectory for storing plots.

        Parameters:
        -----------
        subdir_name : str
            The name of the subdirectory to create inside the main plots
            directory.
        clear_existing : bool, optional
            If True, delete all contents of the subdirectory if it already
            exists.

        Returns:
        --------
        str:
            The full path to the prepared subdirectory.
        """
        full_path = os.path.join(self.plots_directory, subdir_name)
        os.makedirs(full_path, exist_ok=True)

        if clear_existing:
            for item in os.listdir(full_path):
                item_path = os.path.join(full_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

        return full_path

    def _plot_group(
        self,
        ax,
        group_df: pd.DataFrame,
        label: str = None,
        color: str = "blue",
        marker: str = "o",
        marker_size: int = 6,
        capsize: float = 5,
        empty_markers: bool = False,
    ):
        """
        Plot a single data group on the provided Matplotlib Axes object.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes on which to plot.
        group_df : pd.DataFrame
            The DataFrame containing the data for one group.
        label : str, optional
            Label to use in the legend.
        color : str, optional
            Color of the plotted line or points.
        marker : str, optional
            Marker style for scatter/errorbar.
        """
        x_raw = group_df[self.xaxis_variable_name].to_numpy()
        y_raw = group_df[self.yaxis_variable_name].to_numpy()

        # print(x_raw)
        # print(y_raw)

        # is_tuple_array = lambda arr: (isinstance(arr[0], tuple) and len(arr[0]) == 2)

        def is_tuple_array(arr):
            # Filter out any NaNs or Nones
            for element in arr:
                if isinstance(element, tuple) and len(element) == 2:
                    return True
            return False

        x_is_tuple = is_tuple_array(x_raw)
        y_is_tuple = is_tuple_array(y_raw)

        # if not (np.issubdtype(x_raw.dtype, np.number) or x_is_tuple):
        #     raise TypeError(f"x-axis data has unsupported type: {x_raw.dtype}")
        # if not (np.issubdtype(y_raw.dtype, np.number) or y_is_tuple):
        #     raise TypeError(f"y-axis data has unsupported type: {y_raw.dtype}")

        # if not (np.issubdtype(x_raw.dtype, np.number) or x_is_tuple):
        #     example = x_raw[0]
        #     raise TypeError(
        #         f"x-axis data has unsupported type: {x_raw.dtype}. "
        #         f"Example value: {example} (type: {type(example).__name__})"
        #     )

        def is_numeric_array(arr):
            return all(
                isinstance(x, (int, float, np.integer, np.floating))
                or (np.isscalar(x) and np.isnan(x))
                for x in arr
            )

        # # Then use:
        # if not (is_numeric_array(y_raw) or y_is_tuple):
        #     example = y_raw[0]
        #     raise TypeError(
        #         f"y-axis data has unsupported type: {y_raw.dtype}. "
        #         f"Example value: {example} (type: {type(example).__name__})"
        #     )

        if not (is_numeric_array(x_raw) or x_is_tuple):
            example = x_raw[0]
            raise TypeError(
                f"x-axis data has unsupported type: {x_raw.dtype}. "
                f"Example value: {example} (type: {type(example).__name__})"
            )

        if not (is_numeric_array(y_raw) or y_is_tuple):
            example = y_raw[0]
            raise TypeError(
                f"y-axis data has unsupported type: {y_raw.dtype}. "
                f"Example value: {example} (type: {type(example).__name__})"
            )

        # if not (np.issubdtype(y_raw.dtype, np.number) or y_is_tuple):
        #     example = y_raw[0]
        #     raise TypeError(
        #         f"y-axis data has unsupported type: {y_raw.dtype}. "
        #         f"Example value: {example} (type: {type(example).__name__})"
        #     )

        # Case 1: both scalar
        if not x_is_tuple and not y_is_tuple:
            # ax.scatter(
            #     x_raw, y_raw, color=color, marker=marker, label=label, s=marker_size**2
            # )
            if empty_markers:
                # ax.scatter(
                #     x_raw,
                #     y_raw,
                #     marker=marker,
                #     s=marker_size**2,
                #     markerfacecolor="none",
                #     markeredgecolor=color,
                #     label=label,
                # )
                ax.scatter(
                    x_raw,
                    y_raw,
                    marker=marker,
                    s=marker_size**2,
                    facecolors="none",  # hollow marker
                    edgecolors=color,  # border color
                    label=label,
                )
            else:
                ax.scatter(
                    x_raw,
                    y_raw,
                    marker=marker,
                    s=marker_size**2,
                    color=color,
                    label=label,
                )

        # Case 2: scalar x, tuple y
        elif not x_is_tuple and y_is_tuple:
            y_val = np.array([val for val, _ in y_raw])
            y_err = np.array([err for _, err in y_raw])
            # ax.errorbar(
            #     x_raw,
            #     y_val,
            #     yerr=y_err,
            #     fmt=marker,
            #     capsize=capsize,
            #     color=color,
            #     label=label,
            #     markersize=marker_size,
            # )
            if empty_markers:
                ax.errorbar(
                    x_raw,
                    y_val,
                    yerr=y_err,
                    fmt=marker,
                    markersize=marker_size,
                    capsize=capsize,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    color=color,
                    label=label,
                )
            else:
                ax.errorbar(
                    x_raw,
                    y_val,
                    yerr=y_err,
                    fmt=marker,
                    markersize=marker_size,
                    capsize=capsize,
                    color=color,
                    label=label,
                )

        # Case 3: tuple x, scalar y
        elif x_is_tuple and not y_is_tuple:
            x_val = np.array([val for val, _ in x_raw])
            # ax.scatter(x_val, y_raw, color=color, marker=marker, label=label)
            if empty_markers:
                # ax.scatter(
                #     x_raw,
                #     y_raw,
                #     marker=marker,
                #     s=marker_size**2,
                #     markerfacecolor="none",
                #     markeredgecolor=color,
                #     label=label,
                # )
                ax.scatter(
                    x_raw,
                    y_raw,
                    marker=marker,
                    s=marker_size**2,
                    facecolors="none",  # hollow marker
                    edgecolors=color,  # border color
                    label=label,
                )
            else:
                ax.scatter(
                    x_raw,
                    y_raw,
                    marker=marker,
                    s=marker_size**2,
                    color=color,
                    label=label,
                )

        # Case 4: tuple x and tuple y
        elif x_is_tuple and y_is_tuple:
            x_val = np.array([val for val, _ in x_raw])
            y_val = np.array([val for val, _ in y_raw])
            y_err = np.array([err for _, err in y_raw])
            # ax.errorbar(
            #     x_val,
            #     y_val,
            #     yerr=y_err,
            #     fmt=marker,
            #     capsize=capsize,
            #     color=color,
            #     label=label,
            #     markersize=marker_size,
            # )
            if empty_markers:
                ax.errorbar(
                    x_val,
                    y_val,
                    yerr=y_err,
                    fmt=marker,
                    markersize=marker_size,
                    capsize=capsize,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    color=color,
                    label=label,
                )
            else:
                ax.errorbar(
                    x_val,
                    y_val,
                    yerr=y_err,
                    fmt=marker,
                    markersize=marker_size,
                    capsize=capsize,
                    color=color,
                    label=label,
                )

    def _generate_marker_color_map(
        self, grouping_values: list, custom_map: dict = None
    ) -> dict:
        """
        Generate a stable mapping from grouping values to (marker, color) pairs.

        Parameters:
        -----------
        grouping_values : list
            List of unique values of the grouping variable.
        custom_map : dict, optional
            Custom mapping from value to (marker, color). Values not included will be auto-assigned.

        Returns:
        --------
        dict:
            Complete mapping from value → (marker, color)
        """
        sorted_values = sorted(grouping_values, key=lambda x: str(x))
        num_markers = len(constants.MARKER_STYLES)
        num_colors = len(constants.DEFAULT_COLORS)

        style_map = {}

        for idx, value in enumerate(sorted_values):
            if custom_map and value in custom_map:
                style_map[value] = custom_map[value]
            else:
                marker = constants.MARKER_STYLES[idx % num_markers]
                color = constants.DEFAULT_COLORS[idx % num_colors]
                style_map[value] = (marker, color)

        return style_map

    def _construct_plot_title(
        self,
        metadata_dict: dict,
        grouping_variable: str = None,
        labeling_variable: str = None,
        leading_plot_substring: str = None,
        excluded_from_title_list: list = None,
        title_number_format: str = ".2f",  # ".4g" for scientific/float hybrid format
        title_wrapping_length: int = 90,
    ) -> str:
        """
        Construct an informative plot title based on metadata and user
        preferences.

        Parameters:
        -----------
        metadata_dict : dict
            Dictionary containing key-value pairs for this plot group.
        grouping_variable : str, optional
            If provided, exclude this variable from appearing in the title.
        labeling_variable : str, optional
            If provided, exclude this variable from appearing in the title.
        leading_plot_substring : str, optional
            Optional leading string to prepend to the title.
        excluded_from_title_list : list, optional
            List of additional variable names to exclude from the title.

        Returns:
        --------
        str
            The constructed plot title.
        """
        title_parts = []

        # 1. Include leading substring if given
        if leading_plot_substring:
            title_parts.append(leading_plot_substring)

        # 2. Handle Overlap/Kernel/KL_or_Chebyshev_Order special logic
        excluded = set(excluded_from_title_list or [])
        if grouping_variable:
            excluded.add(grouping_variable)
        if labeling_variable:
            excluded.add(labeling_variable)

        overlap_method = metadata_dict.get("Overlap_operator_method")
        kernel_type = metadata_dict.get("Kernel_operator_type")
        chebyshev_terms = metadata_dict.get("Number_of_Chebyshev_terms")
        kl_order = metadata_dict.get("KL_diagonal_order")

        if overlap_method and "Overlap_operator_method" not in excluded:
            if overlap_method == "Bare":
                # Only Overlap_operator_method + Kernel_operator_type
                temp = []
                temp.append(str(overlap_method))
                if kernel_type and "Kernel_operator_type" not in excluded:
                    temp.append(str(kernel_type))
                title_parts.append(" ".join(temp) + ",")
            else:
                # Chebyshev or KL
                temp = []
                temp.append(str(overlap_method))
                if kernel_type and "Kernel_operator_type" not in excluded:
                    temp.append(str(kernel_type))
                if (
                    overlap_method == "Chebyshev"
                    and "Number_of_Chebyshev_terms" not in excluded
                    and chebyshev_terms is not None
                ):
                    temp.append(str(chebyshev_terms))
                if (
                    overlap_method == "KL"
                    and "KL_diagonal_order" not in excluded
                    and kl_order is not None
                ):
                    temp.append(str(kl_order))
                title_parts.append(" ".join(temp) + ",")

        # 3. Handle all remaining tunable parameters
        for param_name in self.list_of_tunable_parameter_names_from_dataframe:
            if param_name in excluded:
                continue
            if param_name not in metadata_dict:
                continue

            value = metadata_dict[param_name]
            label = constants.TITLE_LABELS_BY_COLUMN_NAME.get(param_name, param_name)

            # Format number values cleanly
            if isinstance(value, (int, float)):
                formatted_value = format(value, title_number_format)
            else:
                formatted_value = str(value)

            title_parts.append(f"{label} {formatted_value},")

        # 4. Merge and clean up final title
        full_title = " ".join(title_parts).strip()

        # Remove trailing comma if any
        if full_title.endswith(","):
            full_title = full_title[:-1]

        if title_wrapping_length and len(full_title) > title_wrapping_length:
            # Find a good place (comma) to insert newline
            comma_positions = [
                pos for pos, char in enumerate(full_title) if char == ","
            ]

            if comma_positions:
                # Find best comma to split around the middle
                split_pos = min(
                    comma_positions, key=lambda x: abs(x - len(full_title) // 2)
                )
                full_title = (
                    full_title[: split_pos + 1] + "\n" + full_title[split_pos + 1 :]
                )

        return full_title

    def _construct_plot_filename(
        self,
        metadata_dict: dict,
        include_combined_prefix: bool = False,
        custom_leading_substring: str = None,
        grouping_variable: str = None,
    ) -> str:
        """
        Construct a plot filename based on metadata and class configuration.

        Parameters:
        -----------
        metadata_dict : dict
            Dictionary containing values of tunable parameters for this plot
            group.
        include_combined_prefix : bool, optional
            Whether to prepend "Combined_" to the filename (used when
            grouping_variable is defined).
        custom_leading_substring : str, optional
            An optional custom prefix that overrides "Combined_".
        grouping_variable : str, optional
            If provided, appends "_grouped_by_{grouping_variable}" to the
            filename.

        Returns:
        --------
        str:
            A string to use as the plot filename (without extension).
        """

        def sanitize(value):
            return (
                str(value)
                .replace(".", "p")
                .replace(",", "")
                .replace("(", "")
                .replace(")", "")
            )

        # -- Build filename in parts
        filename_parts = []

        # 1. Overlap_operator_method
        overlap_method = metadata_dict.get("Overlap_operator_method")
        if overlap_method in {"KL", "Chebyshev", "Bare"}:
            filename_parts.append(overlap_method)
            metadata_dict.pop("Overlap_operator_method", None)

        # 2. plots_base_name (y_Vs_x)
        filename_parts.append(self.plots_base_name)

        # 3. Kernel_operator_type
        kernel_type = metadata_dict.get("Kernel_operator_type")
        if kernel_type in {"Brillouin", "Wilson"}:
            filename_parts.append(kernel_type)
            metadata_dict.pop("Kernel_operator_type", None)

        # 4. Parameters from reduced tunable parameter list
        for param in self.reduced_multivalued_tunable_parameter_names_list:
            if param in metadata_dict:
                label = constants.FILENAME_LABELS_BY_COLUMN_NAME.get(param, param)
                value = sanitize(metadata_dict[param])
                filename_parts.append(f"{label}{value}")

        # 5. Optional prefix override
        if custom_leading_substring is not None:
            prefix = custom_leading_substring
        elif include_combined_prefix:
            prefix = "Combined_"
        else:
            prefix = ""

        # 6. Optional grouping variable suffix
        if grouping_variable:
            if isinstance(grouping_variable, str):
                suffix = f"_grouped_by_{grouping_variable}"
            else:
                suffix = "_grouped_by_" + "_and_".join(grouping_variable)
            # suffix = f"_grouped_by_{grouping_variable}"
        else:
            suffix = ""

        return prefix + "_".join(filename_parts) + suffix

    def set_plot_variables(
        self, x_variable: str, y_variable: str, clear_existing: bool = False
    ) -> None:
        """
        Set the x- and y-axis variables for plotting and prepare the
        corresponding subdirectory.

        Parameters:
        -----------
        x_variable : str
            The name of the DataFrame column to use as the x-axis variable.
        y_variable : str
            The name of the DataFrame column to use as the y-axis variable.
        clear_existing : bool, optional
            If True and the plot subdirectory already exists, clear its
            contents. Default is False.

        Raises:
        -------
        ValueError:
            If either `x_variable` or `y_variable` is not a column in the
            DataFrame.
        """
        if x_variable not in self.dataframe.columns:
            raise ValueError(f"'{x_variable}' is not a column in the DataFrame.")
        if y_variable not in self.dataframe.columns:
            raise ValueError(f"'{y_variable}' is not a column in the DataFrame.")

        self.xaxis_variable_name = x_variable
        self.yaxis_variable_name = y_variable
        self.plots_base_name = f"{y_variable}_Vs_{x_variable}"

        self.individual_plots_subdirectory = self._prepare_plot_subdirectory(
            self.plots_base_name, clear_existing=clear_existing
        )

    def plot(
        self,
        grouping_variable: str = None,
        excluded_from_grouping_list: list = None,
        labeling_variable: str = None,
        legend_number_format: str = ".2f",
        include_legend_title: bool = True,
        sorting_variable: str = None,
        sort_ascending: bool = None,
        figure_size=(7, 5),
        xaxis_label: str = None,
        yaxis_label: str = None,
        xaxis_log_scale: bool = False,
        yaxis_log_scale: bool = False,
        invert_xaxis: bool = False,
        invert_yaxis: bool = False,
        left_margin_adjustment: float = 0.14,
        right_margin_adjustment: float = 0.9,
        bottom_margin_adjustment: float = 0.12,
        top_margin_adjustment: float = 0.88,
        legend_location: str = "upper left",
        styling_variable: str = None,
        marker_color_map: dict = None,
        marker_size: int = 6,
        empty_markers: bool = False,
        alternate_filled_markers: bool = False,
        capsize: float = 5,
        include_plot_title: bool = False,
        custom_plot_title: str = None,
        leading_plot_substring: str = None,
        excluded_from_title_list: list = None,
        title_number_format: str = ".2f",
        title_wrapping_length: int = 90,
        customization_function: callable = None,
        verbose: bool = True,
    ):
        """
        Plot data from the DataFrame, optionally grouped by a specific
        multivalued parameter.

        Parameters:
        -----------
        grouping_variable : str, optional
            If provided, combine plots grouped by this variable into single
            plots.
        excluded_from_grouping_list : list, optional
            Additional multivalued parameters to exclude from grouping.
        figure_size : tuple, optional
            Size of each plot figure.
        xaxis_log_scale : bool, optional
            Use log scale for x-axis. Defaults to True if parameter is in
            EXPONENTIAL_FORMAT.
        yaxis_log_scale : bool, optional
            Use log scale for y-axis. Defaults to True if parameter is in
            EXPONENTIAL_FORMAT.
        """
        if self.xaxis_variable_name is None or self.yaxis_variable_name is None:
            raise ValueError("Call 'set_plot_variables()' before plotting.")

        if alternate_filled_markers:
            empty_markers = (
                False  # ignore user's empty_markers setting when alternating
            )

        # Determine which tunable parameters to exclude from grouping
        excluded = set(excluded_from_grouping_list or [])
        for axis_variable in [self.xaxis_variable_name, self.yaxis_variable_name]:
            if axis_variable in self.list_of_multivalued_tunable_parameter_names:
                excluded.add(axis_variable)

        if grouping_variable:
            grouping_columns = (
                [grouping_variable]
                if isinstance(grouping_variable, str)
                else grouping_variable
            )
            for grouping_column in grouping_columns:
                if (
                    grouping_column
                    not in self.list_of_multivalued_tunable_parameter_names
                ):
                    raise ValueError(
                        f"'{grouping_column}' is not a multivalued tunable parameter."
                    )
                excluded.add(grouping_column)

        # Get the grouped DataFrame
        grouped = self.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=list(excluded),
            verbose=verbose,
        )

        # styling_unique_group_values = grouped[styling_variable].unique().tolist()
        # print(styling_unique_group_values)

        for group_keys, group_df in grouped:
            fig, ax = plt.subplots(figsize=figure_size)
            ax.grid(True, linestyle="--", alpha=0.5)

            # Reconstruct metadata dict
            if not isinstance(group_keys, tuple):
                # group_keys = [group_keys]
                group_keys = (group_keys,)
            metadata = dict(
                zip(self.reduced_multivalued_tunable_parameter_names_list, group_keys)
            )

            # Add method and kernel type if constant
            for special in ["Overlap_operator_method", "Kernel_operator_type"]:
                if special in group_df.columns:
                    unique_vals = group_df[special].unique()
                    if len(unique_vals) == 1:
                        metadata[special] = unique_vals[0]

            # Determine axes labels
            if xaxis_label is not None:
                ax.set_xlabel(xaxis_label)
            else:
                ax.set_xlabel(
                    constants.AXES_LABELS_BY_COLUMN_NAME.get(
                        self.xaxis_variable_name, ""
                    )
                )

            if yaxis_label is not None:
                ax.set_ylabel(yaxis_label)
            else:
                ax.set_ylabel(
                    constants.AXES_LABELS_BY_COLUMN_NAME.get(
                        self.yaxis_variable_name, ""
                    )
                )

            # Axes scaling
            if (
                self.xaxis_variable_name in constants.PARAMETERS_WITH_EXPONENTIAL_FORMAT
                or xaxis_log_scale
            ):
                ax.set_xscale("log")
            if (
                self.yaxis_variable_name in constants.PARAMETERS_WITH_EXPONENTIAL_FORMAT
                or yaxis_log_scale
            ):
                ax.set_yscale("log")

            # Integer ticks
            if self.xaxis_variable_name in constants.PARAMETERS_OF_INTEGER_VALUE:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if self.yaxis_variable_name in constants.PARAMETERS_OF_INTEGER_VALUE:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            if invert_xaxis:
                ax.invert_xaxis()
            if invert_yaxis:
                ax.invert_yaxis()

            # Apply any custom user modifications
            if customization_function is not None:
                customization_function(ax)

            if grouping_variable:
                # # Determine order of unique grouping values
                # if sorting_variable:
                #     unique_group_values = (
                #         group_df.sort_values(
                #             by=sorting_variable, ascending=(sort_ascending is not False)
                #         )[grouping_columns]
                #         .unique()
                #         .tolist()
                #     )
                # else:
                #     unique_group_values = (
                #         group_df[grouping_columns[0]].unique().tolist()
                #     )

                if sorting_variable:
                    unique_group_values = (
                        group_df.sort_values(
                            by=sorting_variable, ascending=(sort_ascending is not False)
                        )[grouping_columns]
                        .drop_duplicates()
                        .apply(tuple, axis=1)
                        .tolist()
                    )
                else:
                    unique_group_values = (
                        group_df[grouping_columns]
                        .drop_duplicates()
                        .apply(tuple, axis=1)
                        .tolist()
                    )

                if sort_ascending is True and not sorting_variable:
                    unique_group_values = sorted(unique_group_values)
                elif sort_ascending is False and not sorting_variable:
                    unique_group_values = sorted(unique_group_values, reverse=True)
                # else sort_ascending=None => preserve order

                # Build color/marker map once per plot
                style_map = self._generate_marker_color_map(
                    unique_group_values, custom_map=marker_color_map
                )

                # print(style_map)

                # Combine subgroups in one plot
                # for value, subgroup in group_df.groupby(
                #     grouping_variable, observed=True, sort=False
                # ):
                for curve_index, value in enumerate(unique_group_values):
                    if labeling_variable:
                        # Fetch unique value of labeling_variable for the current group
                        # label_value = group_df.loc[
                        #     group_df[grouping_variable] == value, labeling_variable
                        # ].unique()

                        # if isinstance(grouping_variable, str):
                        #     label_rows = group_df[group_df[grouping_variable] == value]
                        if isinstance(grouping_variable, str):
                            actual_value = (
                                value[0]
                                if isinstance(value, tuple) and len(value) == 1
                                else value
                            )
                            label_rows = group_df[
                                group_df[grouping_variable] == actual_value
                            ]
                        else:
                            mask = (
                                group_df[grouping_variable].apply(tuple, axis=1)
                                == value
                            )
                            label_rows = group_df[mask]

                        label_value = label_rows[labeling_variable].unique()


                        if len(label_value) == 1:
                            label_value = label_value[0]
                        # else:
                        #     raise ValueError(
                        #         f"Multiple values found for '{labeling_variable}' within group '{value}'."
                        #     )

                            # Format numerical labels nicely
                            if isinstance(label_value, (int, float)):
                                label_value = format(label_value, legend_number_format)
                        label = str(label_value)
                    else:
                        # label = str(value)
                        if isinstance(value, tuple):
                            label = " ".join(str(v) for v in value)
                        else:
                            label = str(value)

                    # subgroup = group_df[group_df[grouping_variable] == value]

                    # print(grouping_variable, value)

                    # if isinstance(grouping_variable, str):
                    #     subgroup = group_df[group_df[grouping_variable] == value]
                    if isinstance(grouping_variable, str):
                        # Value is a 1-tuple like ('0004200',), extract the scalar
                        actual_value = (
                            value[0]
                            if isinstance(value, tuple) and len(value) == 1
                            else value
                        )
                        subgroup = group_df[group_df[grouping_variable] == actual_value]
                    else:
                        # grouping_variable is a list of column names
                        mask = group_df[grouping_variable].apply(tuple, axis=1) == value
                        subgroup = group_df[mask]

                    marker, color = style_map[value]
                    # Alternate filling
                    if alternate_filled_markers:
                        empty_marker = curve_index % 2 == 1  # odd indices → empty
                    else:
                        empty_marker = (
                            empty_markers  # regular user setting (could be False)
                        )
                    self._plot_group(
                        ax,
                        subgroup,
                        label=label,
                        color=color,
                        marker=marker,
                        marker_size=marker_size,
                        capsize=capsize,
                        empty_markers=empty_marker,
                    )
                legend = ax.legend(loc=legend_location)
                if include_legend_title:
                    legend_title = constants.LEGEND_LABELS_BY_COLUMN_NAME.get(
                        labeling_variable if labeling_variable else grouping_variable,
                        labeling_variable if labeling_variable else grouping_variable,
                    )
                    # If the title is not a LaTeX string (no $ symbols), replace
                    # underscores with spaces
                    if "$" not in legend_title:
                        legend_title = legend_title.replace("_", " ")
                    legend.set_title(legend_title)
                # ax.legend(title=legend_title, loc=legend_location)

            else:
                # Individual plot
                self._plot_group(
                    ax,
                    group_df,
                    marker_size=marker_size,
                    capsize=capsize,
                    empty_markers=empty_markers,
                )

            fig.subplots_adjust(
                left=left_margin_adjustment,
                right=right_margin_adjustment,
                bottom=bottom_margin_adjustment,
                top=top_margin_adjustment,
            )

            if include_plot_title:
                if custom_plot_title:
                    ax.set_title(custom_plot_title)
                else:
                    title = self._construct_plot_title(
                        metadata_dict=metadata,
                        grouping_variable=grouping_variable,
                        labeling_variable=labeling_variable,
                        leading_plot_substring=leading_plot_substring,
                        excluded_from_title_list=excluded_from_title_list,
                        title_number_format=title_number_format,
                        title_wrapping_length=title_wrapping_length,
                    )
                    ax.set_title(title)

            # Construct filename and save
            filename = self._construct_plot_filename(
                metadata_dict=metadata,
                include_combined_prefix=(grouping_variable is not None),
                grouping_variable=grouping_variable,
            )
            if grouping_variable:
                # nested_dirname = f"Grouped_by_{grouping_variable}"
                if isinstance(grouping_variable, str):
                    nested_dirname = f"Grouped_by_{grouping_variable}"
                else:
                    nested_dirname = "Grouped_by_" + "_and_".join(grouping_variable)
                self.combined_plots_subdirectory = self._prepare_plot_subdirectory(
                    os.path.join(self.plots_base_name, nested_dirname)
                )
                full_path = os.path.join(
                    self.combined_plots_subdirectory, f"{filename}.png"
                )
            else:
                full_path = os.path.join(
                    self.individual_plots_subdirectory, f"{filename}.png"
                )
            fig.savefig(full_path)
            plt.close(fig)


class HDF5Plotter(EnhancedHDF5Analyzer):
    """
    A class for plotting datasets from HDF5 files using DataPlotter.

    This class extends EnhancedHDF5Analyzer to provide methods for converting
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
        # Initialize the parent EnhancedHDF5Analyzer
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

        return plotter
