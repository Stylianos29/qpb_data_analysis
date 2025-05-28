"""
Public HDF5Analyzer class for analyzing HDF5 files with structured data.

This module provides the main public interface for HDF5 file analysis, combining
inspection and data management capabilities while maintaining a familiar API
similar to DataFrameAnalyzer.
"""

from typing import List, Any, Optional, Callable, Union
import numpy as np
import pandas as pd
import h5py
import gvar
from pathlib import Path

from .data_manager import _HDF5DataManager


class HDF5Analyzer(_HDF5DataManager):
    """
    A comprehensive analyzer for HDF5 files following the project's data
    structure.

    This class provides a familiar API similar to DataFrameAnalyzer while
    working directly with HDF5 files. It supports:
    - Parameter categorization (single/multi-valued, tunable/output)
    - Data filtering and restrictions
    - Automatic gvar array handling
    - Virtual dataset transformations
    - DataFrame export for compatibility with other tools

    The HDF5 file remains read-only; all operations work on virtual views.

    Examples:
        >>> analyzer = HDF5Analyzer('data.h5')
        >>> print(analyzer.list_of_multivalued_tunable_parameter_names)
        ['configuration_id', 'measurement_type']

        >>> # Filter data
        >>> analyzer.restrict_data("temperature > 300")
        >>>
        >>> # Work with gvar arrays automatically
        >>> pcac_mass = analyzer.dataset_values('PCAC_mass')  # Auto-merges mean/error
        >>>
        >>> # Transform datasets
        >>> analyzer.transform_dataset('energy', lambda x: x**2, 'energy_squared')
        >>>
        >>> # Export to DataFrame
        >>> df = analyzer.to_dataframe(datasets=['energy', 'energy_squared'])
    """

    def __init__(self, hdf5_file_path: Union[str, Path]):
        """
        Initialize the HDF5Analyzer with a file path.

        Args:
            hdf5_file_path: Path to the HDF5 file to analyze

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid HDF5 file
        """
        # Convert Path to string if needed
        if isinstance(hdf5_file_path, Path):
            hdf5_file_path = str(hdf5_file_path)

        super().__init__(hdf5_file_path)

        # Additional initialization for public API if needed
        self._original_file_path = hdf5_file_path

    def generate_uniqueness_report(
        self, max_width: int = 80, separate_by_type: bool = True
    ) -> str:
        """
        Generate a formatted report on parameter and dataset uniqueness.

        This method creates a table showing:

        - Single-valued parameters/datasets with their values
        - Multi-valued parameters/datasets with counts of unique values

        Args:
            max_width: Maximum width of the report in characters
            separate_by_type: Whether to separate tunable parameters from output
            quantities

        Returns:
            Formatted string containing the uniqueness report
        """
        # Calculate column widths
        half_width = (max_width - 3) // 2

        # Create header
        header_left = "Single-valued fields: unique value"
        header_right = "Multivalued fields: No of unique values"
        header = f"{header_left:<{half_width}} | {header_right}"
        separator = "-" * max_width

        lines = [header, separator]

        if separate_by_type:
            # Separate by type
            sections = [
                (
                    "TUNABLE PARAMETERS",
                    self.list_of_single_valued_tunable_parameter_names,
                    self.list_of_multivalued_tunable_parameter_names,
                ),
                (
                    "OUTPUT QUANTITIES",
                    self.list_of_single_valued_output_quantity_names,
                    self.list_of_multivalued_output_quantity_names,
                ),
            ]

            for section_name, single_list, multi_list in sections:
                if single_list or multi_list:
                    # Add section header
                    if lines[-1] != separator:
                        lines.append("")
                    padding = (max_width - len(section_name)) // 2
                    lines.append(" " * padding + section_name)

                    # Create rows
                    max_rows = max(len(single_list), len(multi_list))
                    for i in range(max_rows):
                        left_col = ""
                        right_col = ""

                        # Single-valued column
                        if i < len(single_list):
                            name = single_list[i]
                            value = self.unique_value_columns_dictionary[name]
                            left_col = self._format_uniqueness_entry(name, value)

                        # Multi-valued column
                        if i < len(multi_list):
                            name = multi_list[i]
                            count = self.multivalued_columns_count_dictionary[name]
                            right_col = f"{name}: {count}"

                        lines.append(f"{left_col:<{half_width}} | {right_col}")
        else:
            # Combined view
            single_items = [
                (name, self.unique_value_columns_dictionary[name])
                for name in self.list_of_single_valued_column_names
            ]
            multi_items = [
                (name, self.multivalued_columns_count_dictionary[name])
                for name in self.list_of_multivalued_column_names
            ]

            max_rows = max(len(single_items), len(multi_items))
            for i in range(max_rows):
                left_col = ""
                right_col = ""

                if i < len(single_items):
                    name, value = single_items[i]
                    left_col = self._format_uniqueness_entry(name, value)

                if i < len(multi_items):
                    name, count = multi_items[i]
                    right_col = f"{name}: {count}"

                lines.append(f"{left_col:<{half_width}} | {right_col}")

        return "\n".join(lines)

    def _format_uniqueness_entry(self, name: str, value: Any) -> str:
        """Format a single uniqueness report entry."""
        if isinstance(value, np.ndarray):
            if value.size == 1:
                formatted_value = self._format_single_value(value.item())
            else:
                formatted_value = f"array{value.shape}"
        else:
            formatted_value = self._format_single_value(value)

        return f"{name}: {formatted_value}"

    def _format_single_value(self, value: Any) -> str:
        """Format a single value for display."""
        if isinstance(value, float):
            if value == int(value):
                return str(int(value))
            else:
                return f"{value:.8g}"
        else:
            return str(value)

    def unique_values(
        self, parameter_name: str, print_output: bool = False
    ) -> List[Any]:
        """
        Get sorted list of unique values for a parameter.

        This method provides backward compatibility with the original
        HDF5Analyzer.

        Args:
            parameter_name: Name of the parameter to analyze print_output:
            Whether to print the values (default: False)

        Returns:
            Sorted list of unique values

        Raises:
            ValueError: If the parameter doesn't exist
        """
        try:
            values = self.column_unique_values(parameter_name)

            if print_output:
                if len(values) == 1:
                    print(
                        f"Parameter '{parameter_name}' has only one "
                        f"value: {values[0]}"
                    )
                else:
                    print(
                        f"Parameter '{parameter_name}' has {len(values)} "
                        f"unique values:"
                    )
                    print(values)

            return values

        except ValueError:
            # Provide more helpful error message
            if parameter_name in self.list_of_output_quantity_names_from_dataframe:
                raise ValueError(
                    f"'{parameter_name}' is a dataset (output quantity), "
                    "not a parameter. Use dataset_values() instead."
                )
            else:
                raise ValueError(
                    f"Parameter '{parameter_name}' not found in HDF5 file."
                )

    def create_dataset_dataframe(
        self,
        dataset_name: str,
        add_time_column: bool = True,
        time_offset: int = 0,
        filter_func: Optional[Callable] = None,
        include_group_path: bool = False,
        flatten_arrays: bool = True,
    ) -> pd.DataFrame:
        """
        Create a DataFrame from a specific dataset across all active groups.

        This method provides compatibility with the original HDF5Analyzer API.

        Args:
            dataset_name: Name of the dataset (or base name for gvar pairs)
            add_time_column: Whether to add time_index column
            time_offset: Offset for time indices
            filter_func: Optional function to filter groups
            include_group_path: Whether to include group paths in DataFrame
            flatten_arrays: Whether to create one row per array element

        Returns:
            DataFrame containing the dataset and associated parameters
        """
        # Apply filter if provided
        if filter_func:
            with self:  # Use context manager to restore state after
                self.restrict_data(filter_func=filter_func)
                df = self.to_dataframe(
                    datasets=[dataset_name], flatten_arrays=flatten_arrays
                )
        else:
            df = self.to_dataframe(
                datasets=[dataset_name], flatten_arrays=flatten_arrays
            )

        # Adjust time indices if needed
        if add_time_column and "time_index" in df.columns and time_offset != 0:
            df["time_index"] = df["time_index"] + time_offset

        # Add group paths if requested
        if include_group_path:
            # This requires tracking which row came from which group
            # For now, we'll skip this feature in the simplified version
            pass

        return df

    def create_merged_value_error_dataframe(
        self,
        base_name: str,
        add_time_column: bool = True,
        time_offset: int = 0,
        filter_func: Optional[Callable] = None,
        include_group_path: bool = False,
    ) -> pd.DataFrame:
        """
        Create a DataFrame with automatic gvar merging for mean/error pairs.

        Args:
            base_name: Base dataset name (without _mean_values/_error_values)
            add_time_column: Whether to add time_index column
            time_offset: Offset for time indices
            filter_func: Optional function to filter groups
            include_group_path: Whether to include group paths

        Returns:
            DataFrame with gvar values
        """
        # Check if this is a valid gvar pair
        if base_name not in self._gvar_dataset_pairs:
            # Try adding suffixes to see if they exist
            if f"{base_name}_mean_values" in self._dataset_paths:
                # The user provided the base name correctly
                pass
            else:
                raise ValueError(
                    f"No gvar dataset pair found for '{base_name}'. "
                    f"Expected to find '{base_name}_mean_values' "
                    f"and '{base_name}_error_values'."
                )

        # Use standard DataFrame creation with gvar support
        return self.create_dataset_dataframe(
            base_name,
            add_time_column=add_time_column,
            time_offset=time_offset,
            filter_func=filter_func,
            include_group_path=include_group_path,
        )

    def save_transformed_data(
        self,
        output_path: Union[str, Path],
        include_virtual: bool = True,
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> None:
        """
        Save current data view (with restrictions and transformations) to a new
        HDF5 file.

        The output file maintains the same hierarchical structure as the input.

        Args:
            output_path: Path for the output HDF5 file include_virtual: Whether
            to include virtual (transformed) datasets compression: HDF5
            compression filter ('gzip', 'lzf', or None) compression_opts:
            Compression level (1-9 for gzip)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(output_path), "w") as out_file:
            # Recreate the group hierarchy for active groups
            created_parents = set()

            for group_path in sorted(self.active_groups):
                # Create all parent groups
                parts = group_path.split("/")
                for i in range(1, len(parts) + 1):
                    parent_path = "/".join(parts[:i])
                    if parent_path not in created_parents:
                        if parent_path not in out_file:
                            out_file.create_group(parent_path)
                        created_parents.add(parent_path)

                        # Copy attributes for this group
                        if parent_path in self._parameters_by_group:
                            out_group = out_file[parent_path]
                            for attr_name, attr_value in self._parameters_by_group[
                                parent_path
                            ].items():
                                out_group.attrs[attr_name] = attr_value

            # Copy datasets from active groups
            for group_path in self.active_groups:
                out_group = out_file[group_path]

                # Copy original datasets
                if group_path in self._datasets_by_group:
                    for dataset_name in self._datasets_by_group[group_path]:
                        # Handle gvar splitting
                        if dataset_name in self._gvar_dataset_pairs.values():
                            # This is part of a gvar pair, copy as-is
                            data = self.dataset_values(
                                dataset_name, return_gvar=False, group_path=group_path
                            )
                        else:
                            # Check if this should be split into mean/error
                            base_name = dataset_name
                            if base_name in self._gvar_dataset_pairs:
                                # Skip - will be handled by mean/error separately
                                continue
                            else:
                                data = self.dataset_values(
                                    dataset_name,
                                    return_gvar=False,
                                    group_path=group_path,
                                )

                        # Create dataset with compression
                        if compression:
                            out_group.create_dataset(
                                dataset_name,
                                data=data,
                                compression=compression,
                                compression_opts=compression_opts,
                            )
                        else:
                            out_group.create_dataset(dataset_name, data=data)

                # Add virtual datasets if requested
                if include_virtual:
                    for virtual_name in self._virtual_datasets:
                        data = self.dataset_values(
                            virtual_name, group_path=group_path
                        )

                        # Handle gvar arrays
                        if (
                            isinstance(data, np.ndarray)
                            and len(data) > 0
                            and isinstance(data[0], gvar.GVar)
                        ):
                            # Split into mean and error
                            mean_data = gvar.mean(data)
                            error_data = gvar.sdev(data)

                            if compression:
                                out_group.create_dataset(
                                    f"{virtual_name}_mean_values",
                                    data=mean_data,
                                    compression=compression,
                                    compression_opts=compression_opts,
                                )
                                out_group.create_dataset(
                                    f"{virtual_name}_error_values",
                                    data=error_data,
                                    compression=compression,
                                    compression_opts=compression_opts,
                                )
                            else:
                                out_group.create_dataset(
                                    f"{virtual_name}_mean_values", data=mean_data
                                )
                                out_group.create_dataset(
                                    f"{virtual_name}_error_values", data=error_data
                                )
                        else:
                            # Regular dataset
                            if compression:
                                out_group.create_dataset(
                                    virtual_name,
                                    data=data,
                                    compression=compression,
                                    compression_opts=compression_opts,
                                )
                            else:
                                out_group.create_dataset(virtual_name, data=data)

    def close(self):
        """Close the HDF5 file."""
        if hasattr(self, "_file") and self._file:
            self._file.close()

    def __repr__(self) -> str:
        """String representation of the analyzer."""
        return (
            f"HDF5Analyzer('{self._original_file_path}')\n"
            f"  Total groups: {len(self._all_deepest_groups)}\n"
            f"  Active groups: {len(self.active_groups)}\n"
            f"  Tunable parameters: {len(self.list_of_tunable_parameter_names_from_dataframe)}\n"
            f"  Output quantities: {len(self.list_of_output_quantity_names_from_dataframe)}\n"
            f"  Virtual datasets: {len(self._virtual_datasets)}"
        )
