"""
Private class for HDF5 data management and manipulation.

This module provides the _HDF5DataManager class, which extends _HDF5Inspector to
add data filtering, transformation, and virtual dataset capabilities while
maintaining the read-only nature of the underlying HDF5 file.
"""

from typing import Dict, List, Set, Any, Optional, Callable, Union, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict
import gvar

from .inspector import _HDF5Inspector


class _HDF5DataManager(_HDF5Inspector):
    """
    Private class that adds data manipulation capabilities to HDF5 inspection.

    This class extends _HDF5Inspector to provide:
    - Data filtering/restriction based on parameter values
    - Virtual dataset transformations
    - Gvar array handling (automatic merging of mean/error pairs)
    - DataFrame generation from filtered data
    - Context manager support for temporary restrictions

    The HDF5 file remains read-only; all operations work on virtual views.
    """

    def __init__(self, hdf5_file_path: str):
        """
        Initialize the data manager.

        Args:
            hdf5_file_path: Path to the HDF5 file
        """
        super().__init__(hdf5_file_path)

        # State management for restrictions
        self._active_groups = None  # None means all groups are active
        self._restriction_stack = []  # For context manager support

        # Virtual datasets (transformations)
        self._virtual_datasets = {}  # name -> (transform_func, source_datasets)

        # Cache for loaded data
        self._data_cache = {}  # (group_path, dataset_name) -> array

        # Initialize active groups to all deepest groups
        max_level = max(self._groups_by_level.keys()) if self._groups_by_level else -1
        self._all_deepest_groups = set(self._groups_by_level.get(max_level, []))

    @property
    def active_groups(self) -> Set[str]:
        """Get currently active groups (after restrictions)."""
        if self._active_groups is None:
            return self._all_deepest_groups.copy()
        return self._active_groups.copy()

    @property
    def reduced_multivalued_tunable_parameter_names_list(self) -> List[str]:
        """
        Get list of multivalued parameters after considering active groups.

        This property matches DataFrameAnalyzer's API.
        """
        if not self._active_groups:
            return self.list_of_multivalued_tunable_parameter_names.copy()

        # Find parameters that still have multiple values in active groups
        param_values = defaultdict(list)  # Use list instead of set
        for group_path in self.active_groups:
            params = self._get_all_parameters_for_group(group_path)
            for param_name, value in params.items():
                # Convert to hashable type if needed
                if isinstance(value, np.ndarray):
                    # Convert array to tuple for comparison
                    hashable_value = tuple(value.flatten())
                else:
                    hashable_value = value

                # Check if this value is already in the list
                already_exists = False
                for existing in param_values[param_name]:
                    if isinstance(existing, tuple) and isinstance(
                        hashable_value, tuple
                    ):
                        if existing == hashable_value:
                            already_exists = True
                            break
                    elif existing == hashable_value:
                        already_exists = True
                        break

                if not already_exists:
                    param_values[param_name].append(hashable_value)

        # Return parameters that still have multiple values
        return sorted(
            [
                param
                for param, values in param_values.items()
                if len(values) > 1
                and param in self.list_of_multivalued_tunable_parameter_names
            ]
        )

    def _get_all_parameters_for_group(self, group_path: str) -> Dict[str, Any]:
        """
        Get all parameters (single and multi-valued) for a specific group.

        Args:
            group_path: Path to the deepest level group

        Returns:
            Dictionary of all parameters affecting this group
        """
        all_params = {}

        # Start with single-valued parameters
        all_params.update(self.unique_value_columns_dictionary)

        # Add parameters from all levels up to this group
        current_path = group_path
        while current_path:
            if current_path in self._parameters_by_group:
                # Add parameters from this level (don't overwrite existing)
                for param, value in self._parameters_by_group[current_path].items():
                    if param not in all_params:
                        all_params[param] = value
            current_path = current_path.rsplit("/", 1)[0] if "/" in current_path else ""

        return all_params

    def restrict_data(self, condition: str = None, filter_func: Callable = None):
        """
        Restrict data to groups matching the given condition.

        This method mirrors DataFrameAnalyzer's restrict_dataframe API.

        Args:
            condition: Pandas-style query string (e.g., "param > 5 and param2 ==
            'value'")
            filter_func: Custom function that takes parameters dict and returns
            bool

        Returns:
            self for method chaining

        Raises:
            ValueError: If neither condition nor filter_func provided
        """
        if condition is None and filter_func is None:
            raise ValueError("Either condition or filter_func must be provided")

        # Start with all deepest groups or current active groups
        candidate_groups = self.active_groups
        filtered_groups = set()

        for group_path in candidate_groups:
            params = self._get_all_parameters_for_group(group_path)

            # Apply filtering
            include_group = False

            if filter_func is not None:
                include_group = filter_func(params)
            elif condition is not None:
                # Create a temporary DataFrame for query evaluation
                temp_df = pd.DataFrame([params])
                try:
                    mask = temp_df.eval(condition)
                    include_group = bool(mask.iloc[0])
                except Exception as e:
                    raise ValueError(f"Failed to evaluate condition: {e}")

            if include_group:
                filtered_groups.add(group_path)

        self._active_groups = filtered_groups
        return self

    def restore_all_groups(self):
        """
        Restore all groups (remove all restrictions).

        This mirrors DataFrameAnalyzer's restore_original_dataframe.

        Returns:
            self for method chaining
        """
        self._active_groups = None
        self._virtual_datasets.clear()
        self._data_cache.clear()
        return self

    def __enter__(self):
        """Enter context manager - save current state."""
        # Save current active groups
        current_state = (
            self._active_groups.copy() if self._active_groups is not None else None
        )
        self._restriction_stack.append(current_state)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - restore previous state."""
        if self._restriction_stack:
            self._active_groups = self._restriction_stack.pop()
        # Clear any cached data from the context
        self._data_cache.clear()
        return False

    def get_dataset_values(
        self,
        dataset_name: str,
        return_gvar: bool = True,
        group_path: Optional[str] = None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get dataset values, with automatic gvar merging if applicable.

        Args:
            dataset_name: Name of the dataset (or base name for gvar pairs)
            return_gvar: If True, automatically merge mean/error pairs into gvar
            arrays
            group_path: Specific group to get data from (None = all active
            groups)

        Returns:
            Array of values or list of arrays (one per group)
        """
        # Check if this is a virtual dataset
        if dataset_name in self._virtual_datasets:
            return self._get_virtual_dataset_values(dataset_name, group_path)

        # Check if we should look for gvar pairs
        if return_gvar and dataset_name in self._gvar_dataset_pairs:
            mean_name, error_name = self._gvar_dataset_pairs[dataset_name]
            return self._get_gvar_dataset_values(mean_name, error_name, group_path)

        # Standard dataset retrieval
        return self._get_standard_dataset_values(dataset_name, group_path)

    def _get_standard_dataset_values(
        self, dataset_name: str, group_path: Optional[str] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get values for a standard dataset."""
        if dataset_name not in self._dataset_paths:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        if group_path:
            # Single group request
            cache_key = (group_path, dataset_name)
            if cache_key not in self._data_cache:
                full_path = f"{group_path}/{dataset_name}"
                if full_path in self._file:
                    self._data_cache[cache_key] = self._file[full_path][()]
                else:
                    raise ValueError(f"Dataset not found in group: {group_path}")
            return self._data_cache[cache_key]
        else:
            # All active groups
            values = []
            for group in sorted(self.active_groups):
                cache_key = (group, dataset_name)
                if cache_key not in self._data_cache:
                    full_path = f"{group}/{dataset_name}"
                    if full_path in self._file:
                        self._data_cache[cache_key] = self._file[full_path][()]
                if cache_key in self._data_cache:
                    values.append(self._data_cache[cache_key])
            return values

    def _get_gvar_dataset_values(
        self, mean_name: str, error_name: str, group_path: Optional[str] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get merged gvar values from mean/error dataset pairs."""
        mean_values = self._get_standard_dataset_values(mean_name, group_path)
        error_values = self._get_standard_dataset_values(error_name, group_path)

        if isinstance(mean_values, list):
            # Multiple groups
            gvar_arrays = []
            for mean_arr, error_arr in zip(mean_values, error_values):
                gvar_arrays.append(gvar.gvar(mean_arr, error_arr))
            return gvar_arrays
        else:
            # Single group/array
            return gvar.gvar(mean_values, error_values)

    def transform_dataset(
        self,
        source_dataset: str,
        transform_func: Callable[[np.ndarray], np.ndarray],
        new_name: str,
    ) -> "_HDF5DataManager":
        """
        Create a virtual dataset by transforming an existing one.

        Args:
            source_dataset: Name of the source dataset
            transform_func: Function to apply to the dataset values
            new_name: Name for the virtual dataset

        Returns:
            self for method chaining
        """
        self._virtual_datasets[new_name] = (transform_func, [source_dataset])

        # Add to output quantities list
        if new_name not in self.list_of_output_quantity_names_from_dataframe:
            self.list_of_output_quantity_names_from_dataframe.append(new_name)
            self.list_of_dataframe_column_names.append(new_name)
            # Assume transformed datasets are multi-valued
            self.list_of_multivalued_column_names.append(new_name)
            self.list_of_multivalued_output_quantity_names.append(new_name)

        return self

    def _get_virtual_dataset_values(
        self, dataset_name: str, group_path: Optional[str] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get values for a virtual (transformed) dataset."""
        transform_func, source_datasets = self._virtual_datasets[dataset_name]

        # Get source values
        source_values = self.get_dataset_values(
            source_datasets[0], return_gvar=True, group_path=group_path
        )

        # Apply transformation
        if isinstance(source_values, list):
            return [transform_func(val) for val in source_values]
        else:
            return transform_func(source_values)

    def to_dataframe(
        self,
        datasets: Optional[List[str]] = None,
        include_parameters: bool = True,
        flatten_arrays: bool = True,
    ) -> pd.DataFrame:
        """
        Convert active groups' data to a pandas DataFrame.

        Args:
            datasets: List of datasets to include (None = all)
            include_parameters: Whether to include parameters as columns
            flatten_arrays: Whether to create one row per array element

        Returns:
            DataFrame with requested data
        """
        if datasets is None:
            datasets = self.list_of_output_quantity_names_from_dataframe.copy()
            # Include virtual datasets
            datasets.extend(self._virtual_datasets.keys())

        rows = []

        for group_path in sorted(self.active_groups):
            # Get parameters for this group
            if include_parameters:
                group_params = self._get_all_parameters_for_group(group_path)
            else:
                group_params = {}

            # Get dataset values
            group_data = {}
            for dataset_name in datasets:
                try:
                    values = self.get_dataset_values(
                        dataset_name, return_gvar=True, group_path=group_path
                    )
                    group_data[dataset_name] = values
                except ValueError:
                    continue  # Dataset not in this group

            # Create rows
            if flatten_arrays and any(
                isinstance(v, np.ndarray) and v.size > 1 for v in group_data.values()
            ):
                # Find the length of arrays (assuming all are same length)
                array_length = None
                for val in group_data.values():
                    if isinstance(val, np.ndarray) and val.size > 1:
                        array_length = len(val)
                        break

                if array_length:
                    for idx in range(array_length):
                        row = group_params.copy()
                        row["time_index"] = idx
                        for ds_name, ds_values in group_data.items():
                            if isinstance(ds_values, np.ndarray) and len(ds_values) > 1:
                                row[ds_name] = ds_values[idx]
                            else:
                                row[ds_name] = ds_values
                        rows.append(row)
            else:
                # Single row per group
                row = group_params.copy()
                row.update(group_data)
                rows.append(row)

        return pd.DataFrame(rows)

    def group_by_multivalued_tunable_parameters(
        self,
        filter_out_parameters_list: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> Dict[Tuple, List[str]]:
        """
        Group active data by multivalued tunable parameters.

        This mirrors DataFrameAnalyzer's grouping functionality.

        Args:
            filter_out_parameters_list: Parameters to exclude from grouping
            verbose: Whether to print grouping information

        Returns:
            Dictionary mapping parameter value tuples to lists of group paths
        """
        # Get parameters that are still multivalued in active groups
        grouping_params = self.reduced_multivalued_tunable_parameter_names_list

        if filter_out_parameters_list:
            grouping_params = [
                p for p in grouping_params if p not in filter_out_parameters_list
            ]

        if verbose:
            print(f"Grouping by parameters: {grouping_params}")

        # Group active groups by parameter values
        grouped = defaultdict(list)

        for group_path in self.active_groups:
            params = self._get_all_parameters_for_group(group_path)

            # Extract grouping parameter values
            group_key = tuple(params.get(p) for p in grouping_params)
            grouped[group_key].append(group_path)

        return dict(grouped)
