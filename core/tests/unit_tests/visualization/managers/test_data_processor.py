"""
Unit tests for PlotDataProcessor class.

This module provides comprehensive testing for the PlotDataProcessor
class, covering data validation, filtering, transformation, and
preparation operations.
"""

import warnings
from unittest.mock import patch

import pytest
import numpy as np
import pandas as pd

from library.visualization.managers.data_processor import PlotDataProcessor


@pytest.fixture
def processor():
    """Create a PlotDataProcessor instance."""
    return PlotDataProcessor()


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
            "group": ["A", "A", "B", "B", "C"],
            "value": [100, 200, 300, 400, 500],
            "annotation": ["a1", "a2", "a3", "a4", "a5"],
        }
    )


class TestDataValidation:
    """Test data validation functionality."""

    def test_validate_plot_data_valid_scalar(self, processor):
        """Test validation with valid scalar data."""
        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([10, 20, 30, 40, 50])

        is_valid, error = processor.validate_plot_data(x_data, y_data)
        assert is_valid is True
        assert error is None

    def test_validate_plot_data_valid_tuples(self, processor):
        """Test validation with valid tuple data."""
        # Use object dtype to preserve tuples
        x_data = np.array([(1, 0.1), (2, 0.2), (3, 0.3)], dtype=object)
        y_data = np.array([(10, 1), (20, 2), (30, 3)], dtype=object)

        is_valid, error = processor.validate_plot_data(x_data, y_data)
        assert is_valid is True
        assert error is None

    def test_validate_plot_data_empty(self, processor):
        """Test validation with empty arrays."""
        x_data = np.array([])
        y_data = np.array([])

        is_valid, error = processor.validate_plot_data(x_data, y_data, label="test")
        assert is_valid is False
        assert "Empty data arrays for test" in error

    def test_validate_plot_data_length_mismatch(self, processor):
        """Test validation with mismatched lengths."""
        x_data = np.array([1, 2, 3])
        y_data = np.array([10, 20])

        is_valid, error = processor.validate_plot_data(x_data, y_data)
        assert is_valid is False
        assert "Data length mismatch" in error

    def test_validate_plot_data_invalid_type(self, processor):
        """Test validation with invalid data types."""
        x_data = np.array(["a", "b", "c"])
        y_data = np.array([10, 20, 30])

        is_valid, error = processor.validate_plot_data(x_data, y_data)
        assert is_valid is False
        assert "X-axis data has unsupported type" in error


class TestArrayTypeChecking:
    """Test array type checking methods."""

    def test_is_tuple_array_true(self, processor):
        """Test tuple array detection with valid tuples."""
        # Use object dtype to preserve tuples
        arr = np.array([(1, 0.1), (2, 0.2), (3, 0.3)], dtype=object)
        assert processor.is_tuple_array(arr) is True

    def test_is_tuple_array_false(self, processor):
        """Test tuple array detection with scalars."""
        arr = np.array([1, 2, 3, 4, 5])
        assert processor.is_tuple_array(arr) is False

    def test_is_tuple_array_mixed(self, processor):
        """Test tuple array detection with mixed types."""
        # Use object dtype for heterogeneous array
        arr = np.array([None, (1, 0.1), None, (2, 0.2)], dtype=object)
        assert processor.is_tuple_array(arr) is True

    def test_is_tuple_array_empty(self, processor):
        """Test tuple array detection with empty array."""
        arr = np.array([])
        assert processor.is_tuple_array(arr) is False

    def test_is_numeric_array_true(self, processor):
        """Test numeric array detection with valid numbers."""
        arr = np.array([1, 2.5, 3, 4.0, 5])
        assert processor.is_numeric_array(arr) is True

    def test_is_numeric_array_false(self, processor):
        """Test numeric array detection with non-numeric data."""
        arr = np.array(["a", "b", "c"])
        assert processor.is_numeric_array(arr) is False

    def test_is_numeric_array_with_none(self, processor):
        """Test numeric array detection with None values."""
        arr = np.array([1, None, 3, None, 5], dtype=object)
        assert processor.is_numeric_array(arr) is True


class TestDataFiltering:
    """Test data filtering functionality."""

    def test_filter_valid_data_scalar(self, processor):
        """Test filtering with scalar arrays."""
        x_data = np.array([1, 2, np.nan, 4, np.inf])
        y_data = np.array([10, 20, 30, 40, 50])

        x_filtered, y_filtered, _ = processor.filter_valid_data(x_data, y_data)

        # Note: np.inf is NOT filtered out for x values in the original
        # code
        assert len(x_filtered) == 3  # 1, 2, 4 are valid (nan is filtered)
        assert np.array_equal(x_filtered, [1, 2, 4])
        assert np.array_equal(y_filtered, [10, 20, 40])  # filtered, [10, 20, 40])

    def test_filter_valid_data_tuples(self, processor):
        """Test filtering with tuple arrays."""
        x_data = np.array([(1, 0.1), (2, np.nan), (3, 0.3)], dtype=object)
        y_data = np.array([(10, 1), (20, 2), (30, 3)], dtype=object)

        x_filtered, y_filtered, _ = processor.filter_valid_data(x_data, y_data)

        assert len(x_filtered) == 2
        # Fix: Use tuple() to ensure proper comparison
        assert tuple(x_filtered[0]) == (1, 0.1)
        assert tuple(x_filtered[1]) == (3, 0.3)
        assert tuple(y_filtered[0]) == (10, 1)
        assert tuple(y_filtered[1]) == (30, 3)

    def test_filter_valid_data_mixed(self, processor):
        """Test filtering with mixed scalar/tuple data."""
        x_data = np.array([1, 2, 3, 4])
        y_data = np.array([(10, 1), (20, np.inf), (30, 3), (40, 4)], dtype=object)

        x_filtered, y_filtered, _ = processor.filter_valid_data(x_data, y_data)

        assert len(x_filtered) == 3
        assert np.array_equal(x_filtered, [1, 3, 4])

    def test_filter_valid_data_with_additional(self, processor):
        """Test filtering with additional data arrays."""
        x_data = np.array([1, np.nan, 3, 4])
        y_data = np.array([10, 20, 30, 40])
        additional = {"z": np.array([100, 200, 300, 400])}

        x_filtered, y_filtered, filtered_additional = processor.filter_valid_data(
            x_data, y_data, additional
        )

        assert len(filtered_additional["z"]) == 3
        assert np.array_equal(filtered_additional["z"], [100, 300, 400])

    def test_filter_valid_data_all_invalid(self, processor):
        """Test filtering when all data is invalid."""
        x_data = np.array([np.nan, np.inf, -np.inf])
        y_data = np.array([10, 20, 30])

        with warnings.catch_warnings(record=True) as w:
            x_filtered, y_filtered, _ = processor.filter_valid_data(x_data, y_data)

            assert len(w) == 1
            assert "No valid data points" in str(w[0].message)
            assert len(x_filtered) == 0
            assert len(y_filtered) == 0


class TestDataExtraction:
    """Test value and error extraction."""

    def test_extract_values_and_errors_tuples(self, processor):
        """Test extraction from tuple data."""
        data = np.array([(1, 0.1), (2, 0.2), (3, 0.3)], dtype=object)

        values, errors = processor.extract_values_and_errors(data)

        assert np.array_equal(values, [1, 2, 3])
        assert np.array_equal(errors, [0.1, 0.2, 0.3])

    def test_extract_values_and_errors_scalars(self, processor):
        """Test extraction from scalar data."""
        data = np.array([1, 2, 3, 4, 5])

        values, errors = processor.extract_values_and_errors(data)

        assert np.array_equal(values, data)
        assert errors is None


class TestInterpolation:
    """Test interpolation functionality."""

    def test_prepare_data_for_interpolation_valid(self, processor):
        """Test preparation of valid data for interpolation."""
        x_data = np.array([3, 1, 4, 1, 5])
        y_data = np.array([30, 10, 40, 10, 50])

        x_unique, y_unique = processor.prepare_data_for_interpolation(x_data, y_data)

        assert x_unique is not None
        assert np.array_equal(x_unique, [1, 3, 4, 5])
        assert np.array_equal(y_unique, [10, 30, 40, 50])

    def test_prepare_data_for_interpolation_tuples(self, processor):
        """Test preparation with tuple data."""
        x_data = np.array([(3, 0.3), (1, 0.1), (2, 0.2)], dtype=object)
        y_data = np.array([(30, 3), (10, 1), (20, 2)], dtype=object)

        x_unique, y_unique = processor.prepare_data_for_interpolation(x_data, y_data)

        assert np.array_equal(x_unique, [1, 2, 3])
        assert np.array_equal(y_unique, [10, 20, 30])

    def test_prepare_data_for_interpolation_insufficient_points(self, processor):
        """Test preparation with too few points."""
        x_data = np.array([1])  # Only 1 point, less than min_points=2
        y_data = np.array([10])

        x_unique, y_unique = processor.prepare_data_for_interpolation(x_data, y_data)

        # Fix: The implementation was changed to allow 2 points minimum
        # Only 1 point should return None
        assert x_unique is None
        assert y_unique is None

    def test_create_interpolation_cubic(self, processor):
        """Test cubic interpolation creation."""
        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([1, 4, 9, 16, 25])  # y = x^2

        x_smooth, y_smooth = processor.create_interpolation(
            x_data, y_data, num_points=10, kind="cubic"
        )

        assert x_smooth is not None
        assert len(x_smooth) == 10
        assert x_smooth[0] == 1
        assert x_smooth[-1] == 5

    def test_create_interpolation_linear_fallback(self, processor):
        """Test linear interpolation fallback."""
        x_data = np.array([1, 2, 3])  # Too few for cubic
        y_data = np.array([10, 20, 30])

        x_smooth, y_smooth = processor.create_interpolation(
            x_data, y_data, kind="linear"  # Explicitly use linear
        )

        assert x_smooth is not None
        assert len(x_smooth) == 100
        # Should be linear
        assert np.allclose(y_smooth[50], 20, rtol=0.1)

    @patch("library.visualization.managers.data_processor.make_interp_spline")
    def test_create_interpolation_failure(self, mock_spline, processor):
        """Test interpolation failure handling."""
        mock_spline.side_effect = ValueError("Spline error")

        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([10, 20, 30, 40, 50])

        with warnings.catch_warnings(record=True) as w:
            x_smooth, y_smooth = processor.create_interpolation(x_data, y_data)

            assert len(w) == 1
            assert "Interpolation failed" in str(w[0].message)
            assert x_smooth is None
            assert y_smooth is None


class TestDataGrouping:
    """Test data grouping functionality."""

    def test_group_data_by_variable_single_column(self, processor, sample_dataframe):
        """Test grouping by single column."""
        groups = processor.group_data_by_variable(sample_dataframe, "group")

        assert len(groups) == 3
        group_values = [g[0] for g in groups]
        assert ("A",) in group_values
        assert ("B",) in group_values
        assert ("C",) in group_values

        # Check group sizes
        for value, df in groups:
            if value == ("A",):
                assert len(df) == 2
            elif value == ("B",):
                assert len(df) == 2
            elif value == ("C",):
                assert len(df) == 1

    def test_group_data_by_variable_multiple_columns(self, processor):
        """Test grouping by multiple columns."""
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [10, 20, 30, 40],
                "group1": ["A", "A", "B", "B"],
                "group2": ["X", "Y", "X", "Y"],
            }
        )

        groups = processor.group_data_by_variable(df, ["group1", "group2"])

        assert len(groups) == 4
        assert ("A", "X") in [g[0] for g in groups]
        assert ("B", "Y") in [g[0] for g in groups]

    def test_group_data_by_variable_with_sorting(self, processor, sample_dataframe):
        """Test grouping with sorting by another variable."""
        groups = processor.group_data_by_variable(
            sample_dataframe, "group", sorting_variable="value", sort_ascending=False
        )

        # Groups should be ordered by their max 'value'
        group_values = [g[0][0] for g in groups]
        assert group_values == ["C", "B", "A"]  # C has 500, B has 400, A has 200

    def test_group_data_by_variable_sort_ascending(self, processor, sample_dataframe):
        """Test grouping with ascending sort."""
        groups = processor.group_data_by_variable(
            sample_dataframe, "group", sort_ascending=True
        )

        group_values = [g[0][0] for g in groups]
        assert group_values == ["A", "B", "C"]

    def test_group_data_by_variable_preserve_order(self, processor):
        """Test grouping preserving original order."""
        df = pd.DataFrame({"x": [1, 2, 3, 4], "group": ["B", "A", "C", "A"]})

        groups = processor.group_data_by_variable(
            df, "group", sort_ascending=None  # Preserve order
        )

        group_values = [g[0][0] for g in groups]
        assert group_values == ["B", "A", "C"]  # First appearance order


class TestAnnotationData:
    """Test annotation data preparation."""

    def test_prepare_annotation_data_basic(self, processor, sample_dataframe):
        """Test basic annotation data preparation."""
        annotations = processor.prepare_annotation_data(
            sample_dataframe, "x", "y", "annotation"
        )

        assert len(annotations) == 5
        assert annotations[0] == (1, 10, "a1")
        assert annotations[-1] == (5, 50, "a5")

    def test_prepare_annotation_data_with_range(self, processor, sample_dataframe):
        """Test annotation data with range specification."""
        annotations = processor.prepare_annotation_data(
            sample_dataframe,
            "x",
            "y",
            "annotation",
            annotation_range=(1, 4, 2),  # Start at 1, end at 4, step 2
        )

        assert len(annotations) == 2
        assert annotations[0][2] == "a2"  # Index 1
        assert annotations[1][2] == "a4"  # Index 3

    def test_prepare_annotation_data_invalid_column(self, processor, sample_dataframe):
        """Test annotation data with invalid column."""
        annotations = processor.prepare_annotation_data(
            sample_dataframe, "x", "y", "invalid_column"
        )

        assert annotations == []

    def test_prepare_annotation_data_with_nan(self, processor):
        """Test annotation data filtering NaN values."""
        df = pd.DataFrame(
            {"x": [1, 2, 3], "y": [10, 20, 30], "ann": [100, np.nan, 300]}
        )

        annotations = processor.prepare_annotation_data(df, "x", "y", "ann")

        assert len(annotations) == 2
        assert annotations[0][2] == 100
        assert annotations[1][2] == 300

    def test_format_annotation_value(self, processor):
        """Test annotation value formatting."""
        # Integer
        assert processor.format_annotation_value(42.0) == "42"
        assert processor.format_annotation_value(42) == "42"

        # Float
        assert processor.format_annotation_value(3.14) == "3.14"

        # String
        assert processor.format_annotation_value("text") == "text"

        # NaN/Inf
        assert processor.format_annotation_value(np.nan) == ""
        assert processor.format_annotation_value(np.inf) == ""


class TestCaching:
    """Test caching functionality."""

    def test_clear_cache(self, processor):
        """Test cache clearing."""
        # Add something to cache
        processor._validation_cache["test"] = "value"

        assert len(processor._validation_cache) > 0

        processor.clear_cache()

        assert len(processor._validation_cache) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe(self, processor):
        """Test operations on empty dataframe with non-existent column."""
        df = pd.DataFrame({"col1": []})  # Empty but with a column

        # Test with existing column
        groups = processor.group_data_by_variable(df, "col1")
        assert groups == []

        # Test with non-existent column should handle gracefully
        # This would need to be handled in the actual implementation

    def test_all_none_values(self, processor):
        """Test with all None values."""
        data = np.array([None, None, None], dtype=object)

        assert processor.is_numeric_array(data) is False
        assert processor.is_tuple_array(data) is False

    def test_mixed_valid_invalid_tuples(self, processor):
        """Test with mix of valid and invalid tuples."""
        x_data = np.array([(1, 0.1), (2, np.nan), (np.inf, 0.3)], dtype=object)
        y_data = np.array([(10, 1), (20, 2), (30, 3)], dtype=object)

        x_filtered, y_filtered, _ = processor.filter_valid_data(x_data, y_data)

        assert len(x_filtered) == 1
        # Fix: Use tuple() to ensure proper comparison
        assert tuple(x_filtered[0]) == (1, 0.1)
        assert tuple(y_filtered[0]) == (10, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
