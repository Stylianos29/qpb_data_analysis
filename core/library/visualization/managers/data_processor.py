"""
Plot data processing and validation for visualization components.

This module provides the PlotDataProcessor class for handling data
validation, filtering, transformation, and preparation for plotting
operations.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
import warnings


class PlotDataProcessor:
    """
    Processes and validates data for plotting operations.

    This class handles all data-related operations needed before
    plotting, including:
    - Data validation and type checking
    - Filtering invalid values (NaN, Inf)
    - Extracting values and errors from tuple data
    - Sorting and grouping data
    - Interpolation for smooth curves
    - Data transformation for different plot types

    Attributes:
        _validation_cache: Cache for validation results to improve
        performance
    """

    def __init__(self):
        """Initialize the data processor."""
        self._validation_cache = {}

    def validate_plot_data(
        self, x_data: np.ndarray, y_data: np.ndarray, label: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that data is suitable for plotting.

        Args:
            - x_data: X-axis data array
            - y_data: Y-axis data array
            - label: Optional label for error messages

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for empty data
        if len(x_data) == 0 or len(y_data) == 0:
            return False, f"Empty data arrays for {label or 'plot'}"

        # Check for length mismatch
        if len(x_data) != len(y_data):
            return False, f"Data length mismatch: x={len(x_data)}, y={len(y_data)}"

        # Check data types
        x_is_tuple = self.is_tuple_array(x_data)
        y_is_tuple = self.is_tuple_array(y_data)

        if not (self.is_numeric_array(x_data) or x_is_tuple):
            example = x_data[0] if len(x_data) > 0 else None
            return False, (
                f"X-axis data has unsupported type. "
                f"Example value: {example} (type: {type(example).__name__})"
            )

        if not (self.is_numeric_array(y_data) or y_is_tuple):
            example = y_data[0] if len(y_data) > 0 else None
            return False, (
                f"Y-axis data has unsupported type. "
                f"Example value: {example} (type: {type(example).__name__})"
            )

        return True, None

    def is_tuple_array(self, arr: np.ndarray) -> bool:
        """
        Check if array contains (value, error) tuples.

        Args:
            arr: Array to check

        Returns:
            True if array contains 2-element tuples or is a 2D array
            with shape (n, 2)
        """
        if len(arr) == 0:
            return False

        # Check if it's a 2D array with 2 columns (common when tuples
        # are passed to np.array)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return True

        # Check if it's an object array with actual tuples
        if arr.dtype == object:
            # Check first few non-None elements
            for element in arr[: min(5, len(arr))]:
                if (
                    element is not None
                    and isinstance(element, tuple)
                    and len(element) == 2
                ):
                    return True

        return False

    def is_numeric_array(self, arr: np.ndarray) -> bool:
        """
        Check if array contains numeric data that can be plotted.

        Args:
            arr: Array to check

        Returns:
            True if array can be converted to numeric values
        """
        try:
            # Filter out None values and try to convert to float
            clean_arr = np.array([x for x in arr if x is not None], dtype=float)
            return len(clean_arr) > 0
        except (ValueError, TypeError):
            return False

    def filter_valid_data(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        additional_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Filter out invalid data points (NaN, Inf) from arrays.

        Args:
            - x_data: X-axis data
            - y_data: Y-axis data
            - additional_data: Optional dict of additional arrays to
              filter

        Returns:
            Tuple of (filtered_x, filtered_y, filtered_additional_data)
        """
        x_is_tuple = self.is_tuple_array(x_data)
        y_is_tuple = self.is_tuple_array(y_data)

        valid_indices = []

        if x_is_tuple and y_is_tuple:
            # Both are tuple arrays
            for i, ((x_val, x_err), (y_val, y_err)) in enumerate(zip(x_data, y_data)):
                if (
                    np.isfinite(x_val)
                    and np.isfinite(x_err)
                    and np.isfinite(y_val)
                    and np.isfinite(y_err)
                ):
                    valid_indices.append(i)

        elif x_is_tuple and not y_is_tuple:
            # X is tuple, Y is scalar
            for i, ((x_val, x_err), y_val) in enumerate(zip(x_data, y_data)):
                if np.isfinite(x_val) and np.isfinite(x_err) and np.isfinite(y_val):
                    valid_indices.append(i)

        elif not x_is_tuple and y_is_tuple:
            # X is scalar, Y is tuple
            for i, (x_val, (y_val, y_err)) in enumerate(zip(x_data, y_data)):
                if np.isfinite(x_val) and np.isfinite(y_val) and np.isfinite(y_err):
                    valid_indices.append(i)

        else:
            # Both are scalar arrays
            valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
            valid_indices = np.where(valid_mask)[0].tolist()

        if not valid_indices:
            warnings.warn("No valid data points after filtering")
            return np.array([]), np.array([]), {}

        # Filter arrays
        x_filtered = np.array([x_data[i] for i in valid_indices])
        y_filtered = np.array([y_data[i] for i in valid_indices])

        # Filter additional data if provided
        filtered_additional = {}
        if additional_data:
            for key, arr in additional_data.items():
                filtered_additional[key] = np.array([arr[i] for i in valid_indices])

        return x_filtered, y_filtered, filtered_additional

    def extract_values_and_errors(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract values and errors from data that may contain tuples.

        Args:
            data: Array that may contain scalar values or (value, error)
            tuples

        Returns:
            Tuple of (values_array, errors_array_or_none)
        """
        if not self.is_tuple_array(data):
            return data, None

        # Handle 2D arrays (common when tuples are passed to np.array)
        if data.ndim == 2 and data.shape[1] == 2:
            values = data[:, 0]
            errors = data[:, 1]
            return values, errors

        # Handle actual tuple arrays
        values = np.array([val for val, _ in data])
        errors = np.array([err for _, err in data])

        return values, errors

    def prepare_data_for_interpolation(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        min_points: int = 2,  # Changed from 4 to 2 to allow linear interpolation
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare data for interpolation by ensuring monotonic x values.

        Args:
            - x_data: X-axis data (may contain tuples)
            - y_data: Y-axis data (may contain tuples)
            - min_points: Minimum points required for interpolation

        Returns:
            Tuple of (x_unique, y_unique) or (None, None) if
            interpolation not possible
        """
        # Extract values if data contains tuples
        x_values, _ = self.extract_values_and_errors(x_data)
        y_values, _ = self.extract_values_and_errors(y_data)

        if len(x_values) < min_points:
            return None, None

        # Sort by x values
        sort_indices = np.argsort(x_values)
        x_sorted = x_values[sort_indices]
        y_sorted = y_values[sort_indices]

        # Remove duplicates
        unique_indices = np.unique(x_sorted, return_index=True)[1]
        x_unique = x_sorted[unique_indices]
        y_unique = y_sorted[unique_indices]

        if len(x_unique) < min_points:
            return None, None

        return x_unique, y_unique

    def create_interpolation(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        num_points: int = 100,
        kind: str = "cubic",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Create interpolated smooth curve from data points.

        Args:
            - x_data: X-axis data
            - y_data: Y-axis data
            - num_points: Number of points in interpolated curve
            - kind: Interpolation type ('cubic', 'linear')

        Returns:
            Tuple of (x_smooth, y_smooth) or (None, None) if
            interpolation fails
        """
        # Prepare data
        x_unique, y_unique = self.prepare_data_for_interpolation(x_data, y_data)

        if x_unique is None:
            return None, None

        try:
            x_smooth = np.linspace(x_unique.min(), x_unique.max(), num_points)

            if kind == "cubic" and len(x_unique) >= 4:
                # Use cubic spline
                spl = make_interp_spline(x_unique, y_unique, k=3)
                y_smooth = spl(x_smooth)
            else:
                # Fall back to linear interpolation (for any kind if not enough
                # points)

                # Ensure all arrays are 1D float arrays for np.interp
                # compatibility and to avoid type errors
                x_unique = np.asarray(x_unique, dtype=float).flatten()
                y_unique = np.asarray(y_unique, dtype=float).flatten()
                x_smooth = np.asarray(x_smooth, dtype=float).flatten()

                y_smooth = np.interp(x_smooth, x_unique, y_unique)

            return x_smooth, y_smooth

        except Exception as e:
            warnings.warn(f"Interpolation failed: {e}")
            return None, None

    def group_data_by_variable(
        self,
        dataframe: pd.DataFrame,
        grouping_columns: Union[str, List[str]],
        sorting_variable: Optional[str] = None,
        sort_ascending: Optional[bool] = None,
    ) -> List[Tuple[Any, pd.DataFrame]]:
        """
        Group dataframe by specified columns and return sorted groups.

        Args:
            - dataframe: Input dataframe
            - grouping_columns: Column(s) to group by
            - sorting_variable: Optional column to sort groups by
            - sort_ascending: Sort order (None preserves original order)

        Returns:
            List of (group_value, group_dataframe) tuples
        """
        if isinstance(grouping_columns, str):
            grouping_columns = [grouping_columns]

        # Check if dataframe is empty or columns don't exist
        if len(dataframe) == 0:
            return []

        # Check if all grouping columns exist
        missing_cols = [col for col in grouping_columns if col not in dataframe.columns]
        if missing_cols:
            return []

        # Get unique group values
        if sorting_variable:
            # Sort by specified variable
            sorted_df = dataframe.sort_values(
                by=sorting_variable, ascending=(sort_ascending is not False)
            )
            unique_groups = (
                sorted_df[grouping_columns]
                .drop_duplicates()
                .apply(tuple, axis=1)
                .tolist()
            )
        else:
            # Get unique values without sorting or with simple sorting
            unique_groups = (
                dataframe[grouping_columns]
                .drop_duplicates()
                .apply(tuple, axis=1)
                .tolist()
            )

            if sort_ascending is True:
                unique_groups = sorted(unique_groups)
            elif sort_ascending is False:
                unique_groups = sorted(unique_groups, reverse=True)
            # If sort_ascending is None, preserve original order

        # Create group list
        groups = []
        for group_value in unique_groups:
            if len(grouping_columns) == 1:
                # Single column grouping
                mask = dataframe[grouping_columns[0]] == (
                    group_value[0] if isinstance(group_value, tuple) else group_value
                )
            else:
                # Multiple column grouping
                mask = dataframe[grouping_columns].apply(tuple, axis=1) == group_value

            group_df = dataframe[mask]
            groups.append((group_value, group_df))

        return groups

    def prepare_annotation_data(
        self,
        dataframe: pd.DataFrame,
        x_variable: str,
        y_variable: str,
        annotation_variable: str,
        annotation_range: Optional[Tuple[int, Optional[int], int]] = None,
    ) -> List[Tuple[float, float, Any]]:
        """
        Prepare data for annotations on plot.

        Args:
            - dataframe: Input dataframe
            - x_variable: X-axis column name
            - y_variable: Y-axis column name
            - annotation_variable: Column containing annotation values
            - annotation_range: Optional (start, end, step) for
              annotation indices

        Returns:
            List of (x, y, annotation_value) tuples
        """
        if annotation_variable not in dataframe.columns:
            return []

        # Get data and sort by x values
        x_data = dataframe[x_variable].to_numpy()
        y_data = dataframe[y_variable].to_numpy()
        ann_data = dataframe[annotation_variable].to_numpy()

        # Extract values if tuples
        x_values, _ = self.extract_values_and_errors(x_data)
        y_values, _ = self.extract_values_and_errors(y_data)

        # Sort by x values
        sort_indices = np.argsort(x_values)
        x_sorted = x_values[sort_indices]
        y_sorted = y_values[sort_indices]
        ann_sorted = ann_data[sort_indices]

        # Determine annotation indices
        if annotation_range is None:
            indices = range(len(x_sorted))
        else:
            start = annotation_range[0] if len(annotation_range) > 0 else 0
            end = annotation_range[1] if len(annotation_range) > 1 else None
            step = annotation_range[2] if len(annotation_range) > 2 else 1

            if end is None:
                indices = range(start, len(x_sorted), step)
            else:
                indices = range(start, min(end, len(x_sorted)), step)

        # Create annotation list
        annotations = []
        for idx in indices:
            if idx < len(x_sorted):
                # Skip NaN/Inf annotation values
                if isinstance(ann_sorted[idx], (int, float)) and not np.isfinite(
                    ann_sorted[idx]
                ):
                    continue

                annotations.append((x_sorted[idx], y_sorted[idx], ann_sorted[idx]))

        return annotations

    def format_annotation_value(self, value: Any) -> str:
        """
        Format annotation value for display.

        Args:
            value: Value to format

        Returns:
            Formatted string
        """
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                return ""
            # Check if it's effectively an integer
            if float(value).is_integer():
                return str(int(value))
            return str(value)
        return str(value)

    def clear_cache(self):
        """Clear internal caches."""
        self._validation_cache.clear()
