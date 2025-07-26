"""
Private base class for read-only HDF5 file inspection.

This module provides the _HDF5Inspector class, which handles the
analysis of HDF5 file structure, parameter categorization, and dataset
discovery. It serves as the foundation for the HDF5Analyzer public API.
"""

import os
from collections import defaultdict
from typing import List, Set, Any

import h5py
import numpy as np


class _HDF5Inspector:
    """
    Private base class for read-only inspection of HDF5 files.

    This class analyzes HDF5 file structure and categorizes parameters
    and datasets according to the project's conventions:
    - Directory hierarchy mirroring in top-level groups
    - Single-valued parameters as attributes at second-to-last level
    - Multi-valued parameters as attributes at deepest level
    - Datasets stored in deepest level groups

    Attributes follow the same naming convention as DataFrameAnalyzer
    for consistency.
    """

    def __init__(self, hdf5_file_path: str):
        """
        Initialize the inspector with an HDF5 file path.

        Args:
            hdf5_file_path: Path to the HDF5 file to inspect

        Raises:
            FileNotFoundError: If the HDF5 file doesn't exist
            ValueError: If the file cannot be opened as HDF5
        """
        if not os.path.exists(hdf5_file_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_file_path}")

        self.file_path = hdf5_file_path

        # Try to open the file to validate it's HDF5
        try:
            self._file = h5py.File(hdf5_file_path, "r")
        except Exception as e:
            raise ValueError(f"Cannot open file as HDF5: {hdf5_file_path}") from e

        # Initialize storage structures matching DataFrameAnalyzer naming
        self._initialize_storage()

        # Perform initial structure analysis
        self._analyze_structure()

    def _initialize_storage(self):
        """Initialize internal storage structures with
        DataFrameAnalyzer-compatible names."""
        # Column (parameter/dataset) lists
        self.list_of_dataframe_column_names = []  # All parameters and datasets
        self.list_of_tunable_parameter_names_from_dataframe = []
        self.list_of_output_quantity_names_from_dataframe = []

        # Value analysis dictionaries
        self.unique_value_columns_dictionary = {}  # Single-valued params -> value
        self.multivalued_columns_count_dictionary = defaultdict(
            int
        )  # Multi-valued -> count

        # Categorized lists
        self.list_of_single_valued_column_names = []
        self.list_of_multivalued_column_names = []
        self.list_of_single_valued_tunable_parameter_names = []
        self.list_of_multivalued_tunable_parameter_names = []
        self.list_of_single_valued_output_quantity_names = []
        self.list_of_multivalued_output_quantity_names = []

        # HDF5-specific storage
        self._groups_by_level = defaultdict(list)  # level -> [group_paths]
        self._parameters_by_group = {}  # group_path -> {param: value}
        self._datasets_by_group = defaultdict(list)  # group_path -> [dataset_names]
        self._dataset_paths = defaultdict(list)  # dataset_name -> [full_paths]

        # NEW: Single-valued parameters from second-to-deepest level
        # groups
        self._single_valued_parameters_from_parent = {}  # param_name -> value

        # Cache for gvar dataset pairs
        self._gvar_dataset_pairs = {}  # base_name -> (mean_dataset, error_dataset)

    def _analyze_structure(self):
        """
        Analyze the HDF5 file structure and categorize parameters and
        datasets.

        This method identifies:
        - Group hierarchy levels
        - Single-valued parameters (second-to-last level attributes)
        - Multi-valued parameters (deepest level attributes)
        - Output datasets and their locations
        - Gvar dataset pairs (mean/error)
        """
        # First pass: collect all groups and their levels
        self._collect_groups()

        # Second pass: extract parameters from appropriate levels
        self._extract_parameters()

        # Third pass: collect datasets and identify gvar pairs
        self._collect_datasets()

        # Fourth pass: categorize parameters and datasets
        self._categorize_columns()

    def _collect_groups(self):
        """Collect all groups and organize by hierarchy level."""

        def visit_group(name, obj):
            if isinstance(obj, h5py.Group):
                level = name.count("/")
                self._groups_by_level[level].append(name)

        self._file.visititems(visit_group)

    def _extract_parameters(self):
        """Extract and categorize parameters from group attributes."""
        all_param_values = defaultdict(set)
        max_level = max(self._groups_by_level.keys()) if self._groups_by_level else -1

        # Extract single-valued parameters from second-to-deepest level
        self._single_valued_parameters_from_parent = {}
        if max_level > 0:  # Ensure there's a second-to-deepest level
            second_deepest_level = max_level - 1
            second_deepest_groups = self._groups_by_level.get(second_deepest_level, [])

            for group_path in second_deepest_groups:
                if group_path in self._file:
                    group = self._file[group_path]
                    attrs = dict(group.attrs)
                    if attrs:
                        # Store parameters from second-to-deepest level
                        self._parameters_by_group[group_path] = attrs
                        # These are single-valued by definition
                        # (constant across all deepest groups)
                        for param_name, value in attrs.items():
                            # Convert numpy types to native Python types
                            if isinstance(value, (np.integer, np.floating)):
                                value = value.item()
                            self._single_valued_parameters_from_parent[param_name] = (
                                value
                            )

        # Extract multi-valued parameters from deepest level
        deepest_groups = self._groups_by_level.get(max_level, [])

        for group_path in deepest_groups:
            if group_path in self._file:
                group = self._file[group_path]
                attrs = dict(group.attrs)
                if attrs:
                    self._parameters_by_group[group_path] = attrs
                    # Track all values for each parameter
                    for param_name, value in attrs.items():
                        # Convert numpy arrays to tuples for hashability
                        if isinstance(value, np.ndarray):
                            value = tuple(value.flatten())
                        elif isinstance(value, (np.integer, np.floating)):
                            value = value.item()  # Convert to native Python type
                        all_param_values[param_name].add(value)

        # Now categorize ALL parameters based on their value counts
        # First add the single-valued parameters from parent groups
        for param_name, value in self._single_valued_parameters_from_parent.items():
            self.unique_value_columns_dictionary[param_name] = value

        # Then categorize parameters from deepest level groups
        for param_name, values in all_param_values.items():
            if len(values) == 1:
                # Single-valued parameter
                value = next(iter(values))
                # Convert back from tuple if it was an array
                if isinstance(value, tuple):
                    value = np.array(value)
                self.unique_value_columns_dictionary[param_name] = value
            else:
                # Multi-valued parameter
                self.multivalued_columns_count_dictionary[param_name] = len(values)

    def _collect_datasets(self):
        """Collect all datasets and identify gvar pairs."""
        max_level = max(self._groups_by_level.keys()) if self._groups_by_level else -1
        deepest_groups = self._groups_by_level[max_level] if max_level >= 0 else []

        all_dataset_names = set()
        dataset_value_tracker = defaultdict(list)

        for group_path in deepest_groups:
            group = self._file[group_path]

            # Ensure we're working with a Group, not a Dataset
            if not isinstance(group, h5py.Group):
                continue  # Skip if it's not a group

            # Collect datasets in this group
            for dataset_name in group.keys():
                # if isinstance(group[dataset_name], h5py.Dataset):
                dataset_obj = group[dataset_name]
                if isinstance(dataset_obj, h5py.Dataset):
                    full_path = f"{group_path}/{dataset_name}"
                    self._dataset_paths[dataset_name].append(full_path)
                    self._datasets_by_group[group_path].append(dataset_name)
                    all_dataset_names.add(dataset_name)

                    # Track entire arrays for single/multi classification
                    # Use dataset_obj which is confirmed to be h5py.Dataset
                    dataset_value_tracker[dataset_name].append(dataset_obj[()])

        # Identify gvar pairs
        self._identify_gvar_pairs(all_dataset_names)

        # Classify datasets as single/multi valued
        for dataset_name, arrays_list in dataset_value_tracker.items():
            if self._is_single_valued_dataset(arrays_list):
                # Store the common array value
                self.unique_value_columns_dictionary[dataset_name] = arrays_list[0]
            else:
                # Count unique arrays (different array values across groups)
                unique_arrays = []
                for arr in arrays_list:
                    # Check if this array is already in our unique list
                    is_unique = True
                    for unique_arr in unique_arrays:
                        if np.array_equal(arr, unique_arr):
                            is_unique = False
                            break
                    if is_unique:
                        unique_arrays.append(arr)

                self.multivalued_columns_count_dictionary[dataset_name] = len(
                    unique_arrays
                )

    def _identify_gvar_pairs(self, dataset_names: Set[str]):
        """Identify dataset pairs following the _mean_values/_error_values
        convention."""
        # Look for matching pairs
        for name in dataset_names:
            if name.endswith("_mean_values"):
                base_name = name[:-12]  # Remove '_mean_values'
                error_name = f"{base_name}_error_values"
                if error_name in dataset_names:
                    self._gvar_dataset_pairs[base_name] = (name, error_name)

    def _is_single_valued_dataset(self, values_list: List[np.ndarray]) -> bool:
        """
        Check if all dataset values are identical across all groups.

        A dataset is single-valued if the entire array is the same
        across all subgroups at the same level (analogous to a column
        having the same value across all DataFrame rows).

        Args:
            values_list: List of numpy arrays from different groups

        Returns:
            True if all arrays are identical, False otherwise
        """
        if not values_list:
            return True

        if len(values_list) == 1:
            return True  # Only one instance, so it's "single-valued"

        first_array = values_list[0]
        for array in values_list[1:]:
            # Check if the entire array is identical to the first one
            if not np.array_equal(array, first_array):
                return False
        return True

    def _categorize_columns(self):
        """Categorize all parameters and datasets into appropriate
        lists."""
        # Combine all column names
        all_params = set(self.unique_value_columns_dictionary.keys()) | set(
            self.multivalued_columns_count_dictionary.keys()
        )
        self.list_of_dataframe_column_names = sorted(list(all_params))

        # Separate tunable parameters from output quantities
        # For now, all attributes are considered tunable parameters
        # and all datasets are output quantities
        param_names = set()
        for params in self._parameters_by_group.values():
            param_names.update(params.keys())

        self.list_of_tunable_parameter_names_from_dataframe = sorted(list(param_names))

        dataset_names = set(self._dataset_paths.keys())
        self.list_of_output_quantity_names_from_dataframe = sorted(list(dataset_names))

        # Single vs multi-valued lists
        self.list_of_single_valued_column_names = sorted(
            list(self.unique_value_columns_dictionary.keys())
        )
        self.list_of_multivalued_column_names = sorted(
            list(self.multivalued_columns_count_dictionary.keys())
        )

        # Categorized parameter lists
        param_set = set(self.list_of_tunable_parameter_names_from_dataframe)
        self.list_of_single_valued_tunable_parameter_names = [
            name
            for name in self.list_of_single_valued_column_names
            if name in param_set
        ]
        self.list_of_multivalued_tunable_parameter_names = [
            name for name in self.list_of_multivalued_column_names if name in param_set
        ]

        # Categorized output quantity lists
        output_set = set(self.list_of_output_quantity_names_from_dataframe)
        self.list_of_single_valued_output_quantity_names = [
            name
            for name in self.list_of_single_valued_column_names
            if name in output_set
        ]
        self.list_of_multivalued_output_quantity_names = [
            name for name in self.list_of_multivalued_column_names if name in output_set
        ]

    def column_unique_values(self, column_name: str) -> List[Any]:
        """
        Return sorted list of unique values for the specified column.

        This method matches DataFrameAnalyzer's API for consistency.

        Args:
            column_name: Name of the parameter to analyze

        Returns:
            Sorted list of unique values

        Raises:
            ValueError: If the column doesn't exist
        """
        # Check single-valued columns first
        if column_name in self.unique_value_columns_dictionary:
            return [self.unique_value_columns_dictionary[column_name]]

        # Check multi-valued parameters
        if column_name in self.list_of_multivalued_tunable_parameter_names:
            values = set()
            for _, params in self._parameters_by_group.items():
                if column_name in params:
                    value = params[column_name]
                    # Convert numpy arrays to tuples for hashability
                    if isinstance(value, np.ndarray):
                        value = tuple(value.flatten())
                    elif isinstance(value, (np.integer, np.floating)):
                        value = value.item()  # Convert to native Python type
                    values.add(value)

            # Convert back to numpy arrays if needed
            result = []
            for value in values:
                if isinstance(value, tuple):
                    value = np.array(value)
                result.append(value)
            return sorted(result)

        raise ValueError(f"Column '{column_name}' not found in HDF5 file")

    def __del__(self):
        """Ensure HDF5 file is properly closed."""
        if hasattr(self, "_file") and self._file:
            try:
                self._file.close()
            except:
                pass
