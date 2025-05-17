import os

import numpy as np
import h5py
from collections import defaultdict


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

    def get_unique_values(self, parameter_name, print_output: bool = False):
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

        if print_output:
            print(f"Parameter '{parameter_name}' has {len(values)} unique values:")
            print(values)

        return values

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
