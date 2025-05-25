import pytest

import pandas as pd

from library.data.analyzer import _DataFrameInspector, DataFrameAnalyzer


# MODULE-LEVEL FIXTURE
@pytest.fixture
def simple_dataframe():
    """Create an extended DataFrame for analyzer testing."""
    return pd.DataFrame(
        {
            "Kernel_operator_type": ["Wilson", "Wilson", "Brillouin", "Brillouin"],
            "MSCG_epsilon": [1e-6, 1e-5, 1e-6, 1e-5],
            "Configuration_label": ["0000100", "0000100", "0000200", "0000200"],
            "APE_alpha": [0.72, 0.72, 0.72, 0.72],  # Single unique value
            "Total_calculation_time": [10.5, 12.3, 11.8, 13.2],
            "Spatial_lattice_size": [24, 24, 24, 24],  # Single unique value
        }
    )


class TestDataFrameInspector:
    """Test suite for the _DataFrameInspector class using synthetic data."""

    @pytest.fixture
    def empty_dataframe(self):
        """Create an empty DataFrame for edge case testing."""
        return pd.DataFrame()

    @pytest.fixture
    def single_row_dataframe(self):
        """Create a DataFrame with only one row."""
        return pd.DataFrame(
            {
                "Kernel_operator_type": ["Wilson"],
                "MSCG_epsilon": [1e-6],
                "Total_calculation_time": [10.5],
            }
        )

    @pytest.fixture
    def all_single_valued_dataframe(self):
        """Create a DataFrame where all columns have single unique values."""
        return pd.DataFrame(
            {
                "Kernel_operator_type": ["Wilson", "Wilson", "Wilson"],
                "APE_alpha": [0.72, 0.72, 0.72],
                "Spatial_lattice_size": [24, 24, 24],
            }
        )

    def test_initialization_with_valid_dataframe(self, simple_dataframe):
        """Test that inspector initializes correctly with valid DataFrame."""
        inspector = _DataFrameInspector(simple_dataframe)
        assert inspector.dataframe is simple_dataframe  # Should be same reference
        assert hasattr(inspector, "list_of_dataframe_column_names")

    def test_initialization_with_invalid_input(self):
        """Test that inspector raises TypeError for non-DataFrame input."""
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            _DataFrameInspector([1, 2, 3])

        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            _DataFrameInspector("not a dataframe")

        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            _DataFrameInspector(None)

    def test_column_categorization(self, simple_dataframe):
        """Test that columns are correctly categorized."""
        inspector = _DataFrameInspector(simple_dataframe)

        # Check column names list
        expected_columns = [
            "Kernel_operator_type",
            "MSCG_epsilon",
            "Configuration_label",
            "APE_alpha",
            "Total_calculation_time",
            "Spatial_lattice_size",
        ]
        assert inspector.list_of_dataframe_column_names == expected_columns

        # Check tunable parameters
        tunable_params = inspector.list_of_tunable_parameter_names_from_dataframe
        assert set(tunable_params) == {
            "Kernel_operator_type",
            "MSCG_epsilon",
            "Configuration_label",
            "APE_alpha",
        }

        # Check output quantities
        output_quantities = inspector.list_of_output_quantity_names_from_dataframe
        assert set(output_quantities) == {
            "Total_calculation_time",
            "Spatial_lattice_size",
        }

    def test_single_valued_columns_identification(self, simple_dataframe):
        """Test identification of columns with single unique values."""
        inspector = _DataFrameInspector(simple_dataframe)

        # Check single-valued columns dictionary
        assert inspector.unique_value_columns_dictionary == {
            "APE_alpha": 0.72,
            "Spatial_lattice_size": 24,
        }

        # Check single-valued column names list
        assert set(inspector.list_of_single_valued_column_names) == {
            "APE_alpha",
            "Spatial_lattice_size",
        }

    def test_multivalued_columns_identification(self, simple_dataframe):
        """Test identification of columns with multiple unique values."""
        inspector = _DataFrameInspector(simple_dataframe)

        # Check multi-valued columns dictionary
        assert inspector.multivalued_columns_count_dictionary == {
            "Kernel_operator_type": 2,  # Wilson, Brillouin
            "MSCG_epsilon": 2,  # 1e-6, 1e-5
            "Configuration_label": 2,  # "0000100", "0000200"
            "Total_calculation_time": 4,  # 10.5, 12.3, 11.8, 13.2
        }

        # Check multi-valued column names list
        assert set(inspector.list_of_multivalued_column_names) == {
            "Kernel_operator_type",
            "MSCG_epsilon",
            "Configuration_label",
            "Total_calculation_time",
        }

    def test_parameter_value_categorization(self, simple_dataframe):
        """Test categorization of parameters by single/multi values."""
        inspector = _DataFrameInspector(simple_dataframe)

        # Single-valued tunable parameters
        assert inspector.list_of_single_valued_tunable_parameter_names == ["APE_alpha"]

        # Multi-valued tunable parameters
        assert set(inspector.list_of_multivalued_tunable_parameter_names) == {
            "Kernel_operator_type",
            "MSCG_epsilon",
            "Configuration_label",
        }

        # Single-valued output quantities
        assert inspector.list_of_single_valued_output_quantity_names == [
            "Spatial_lattice_size"
        ]

        # Multi-valued output quantities
        assert inspector.list_of_multivalued_output_quantity_names == [
            "Total_calculation_time"
        ]

    def test_column_unique_values_method(self, simple_dataframe):
        """Test the column_unique_values method."""
        inspector = _DataFrameInspector(simple_dataframe)

        # Test with multi-valued column
        kernel_values = inspector.column_unique_values("Kernel_operator_type")
        assert kernel_values == ["Brillouin", "Wilson"]  # Sorted

        # Test with single-valued column
        ape_values = inspector.column_unique_values("APE_alpha")
        assert ape_values == [0.72]

        # Test with column containing numpy types
        epsilon_values = inspector.column_unique_values("MSCG_epsilon")
        assert epsilon_values == [1e-6, 1e-5]  # Should be Python floats, not numpy
        assert all(isinstance(v, (int, float, str)) for v in epsilon_values)

    def test_column_unique_values_invalid_column(self, simple_dataframe):
        """Test column_unique_values raises error for invalid column."""
        inspector = _DataFrameInspector(simple_dataframe)

        with pytest.raises(ValueError, match="Column 'nonexistent' does not exist"):
            inspector.column_unique_values("nonexistent")

    def test_empty_dataframe_handling(self, empty_dataframe):
        """Test inspector handles empty DataFrame correctly."""
        inspector = _DataFrameInspector(empty_dataframe)

        assert inspector.list_of_dataframe_column_names == []
        assert inspector.list_of_tunable_parameter_names_from_dataframe == []
        assert inspector.list_of_output_quantity_names_from_dataframe == []
        assert inspector.unique_value_columns_dictionary == {}
        assert inspector.multivalued_columns_count_dictionary == {}

    def test_single_row_dataframe_handling(self, single_row_dataframe):
        """Test inspector handles single-row DataFrame correctly."""
        inspector = _DataFrameInspector(single_row_dataframe)

        # All columns should be single-valued
        assert len(inspector.list_of_single_valued_column_names) == 3
        assert len(inspector.list_of_multivalued_column_names) == 0

        # Check values
        assert inspector.unique_value_columns_dictionary == {
            "Kernel_operator_type": "Wilson",
            "MSCG_epsilon": 1e-6,
            "Total_calculation_time": 10.5,
        }

    def test_all_single_valued_dataframe(self, all_single_valued_dataframe):
        """Test inspector with DataFrame where all columns have single
        values."""
        inspector = _DataFrameInspector(all_single_valued_dataframe)

        # All columns should be single-valued
        assert len(inspector.list_of_multivalued_column_names) == 0
        assert len(inspector.list_of_single_valued_column_names) == 3

        # No multi-valued parameters
        assert inspector.list_of_multivalued_tunable_parameter_names == []
        assert inspector.list_of_multivalued_output_quantity_names == []

    def test_dataframe_not_copied(self, simple_dataframe):
        """Test that inspector doesn't create a copy of the DataFrame."""
        inspector = _DataFrameInspector(simple_dataframe)

        # Modify the original DataFrame
        simple_dataframe.loc[0, "MSCG_epsilon"] = 1e-4

        # Change should be reflected in inspector's dataframe
        assert inspector.dataframe.loc[0, "MSCG_epsilon"] == 1e-4

    def test_update_column_categories_called_on_init(self, simple_dataframe):
        """Test that _update_column_categories is called during
        initialization."""
        inspector = _DataFrameInspector(simple_dataframe)

        # All category attributes should be populated
        assert hasattr(inspector, "list_of_dataframe_column_names")
        assert hasattr(inspector, "list_of_tunable_parameter_names_from_dataframe")
        assert hasattr(inspector, "unique_value_columns_dictionary")
        assert hasattr(inspector, "multivalued_columns_count_dictionary")
        # ... etc


class TestDataFrameAnalyzer:
    """Test suite for the DataFrameAnalyzer class using synthetic data."""

    @pytest.fixture
    def analyzer(self, simple_dataframe):
        """Create a DataFrameAnalyzer instance."""
        return DataFrameAnalyzer(simple_dataframe)

    def test_initialization_creates_copies(self, simple_dataframe):
        """Test that analyzer creates copies of the input DataFrame."""
        analyzer = DataFrameAnalyzer(simple_dataframe)

        # Check that both original and working copies exist
        assert hasattr(analyzer, "original_dataframe")
        assert hasattr(analyzer, "dataframe")

        # Both should be copies, not references
        assert analyzer.original_dataframe is not simple_dataframe
        assert analyzer.dataframe is not simple_dataframe
        assert analyzer.dataframe is not analyzer.original_dataframe

        # But should have same content
        pd.testing.assert_frame_equal(analyzer.original_dataframe, simple_dataframe)
        pd.testing.assert_frame_equal(analyzer.dataframe, simple_dataframe)

    def test_context_manager_basic(self, analyzer):
        """Test basic context manager functionality."""
        original_len = len(analyzer.dataframe)

        with analyzer:
            # Make changes inside context
            analyzer.restrict_dataframe("MSCG_epsilon == 1e-6")
            assert len(analyzer.dataframe) == 2  # Should have filtered rows

        # Should be restored after context
        assert len(analyzer.dataframe) == original_len

    def test_context_manager_with_exception(self, analyzer):
        """Test context manager restores state even with exceptions."""
        original_len = len(analyzer.dataframe)

        with pytest.raises(ValueError):
            with analyzer:
                analyzer.restrict_dataframe("MSCG_epsilon == 1e-6")
                assert len(analyzer.dataframe) == 2
                raise ValueError("Test exception")

        # Should still be restored after exception
        assert len(analyzer.dataframe) == original_len

    def test_context_manager_nested(self, analyzer):
        """Test nested context managers."""
        original_len = len(analyzer.dataframe)

        with analyzer:
            analyzer.restrict_dataframe("Kernel_operator_type == 'Wilson'")
            wilson_len = len(analyzer.dataframe)

            with analyzer:
                analyzer.restrict_dataframe("MSCG_epsilon == 1e-6")
                assert len(analyzer.dataframe) == 1

            # Should restore to outer context state
            assert len(analyzer.dataframe) == wilson_len

        # Should restore to original state
        assert len(analyzer.dataframe) == original_len

    def test_restrict_dataframe_with_condition(self, analyzer):
        """Test restrict_dataframe with string condition."""
        # Single condition
        analyzer.restrict_dataframe("Kernel_operator_type == 'Wilson'")
        assert len(analyzer.dataframe) == 2
        assert all(analyzer.dataframe["Kernel_operator_type"] == "Wilson")

        # Multiple conditions
        analyzer.restore_original_dataframe()
        analyzer.restrict_dataframe(
            "Kernel_operator_type == 'Wilson' and MSCG_epsilon == 1e-6"
        )
        assert len(analyzer.dataframe) == 1

    def test_restrict_dataframe_with_filter_func(self, analyzer):
        """Test restrict_dataframe with filter function."""

        def custom_filter(df):
            return df["Total_calculation_time"] > 12.0

        analyzer.restrict_dataframe(filter_func=custom_filter)
        assert len(analyzer.dataframe) == 2
        assert all(analyzer.dataframe["Total_calculation_time"] > 12.0)

    def test_restrict_dataframe_method_chaining(self, analyzer):
        """Test that restrict_dataframe supports method chaining."""
        result = analyzer.restrict_dataframe("Kernel_operator_type == 'Wilson'")
        assert result is analyzer  # Should return self

        # Chain multiple operations
        analyzer.restore_original_dataframe()
        analyzer.restrict_dataframe(
            "Kernel_operator_type == 'Wilson'"
        ).restrict_dataframe("MSCG_epsilon == 1e-6")
        assert len(analyzer.dataframe) == 1

    def test_restrict_dataframe_errors(self, analyzer):
        """Test restrict_dataframe error handling."""
        # No arguments
        with pytest.raises(ValueError, match="Either condition or filter_func"):
            analyzer.restrict_dataframe()

        # Invalid condition type
        with pytest.raises(TypeError, match="condition must be a string"):
            analyzer.restrict_dataframe(condition=123)

        # Invalid filter_func type
        with pytest.raises(TypeError, match="filter_func must be callable"):
            analyzer.restrict_dataframe(filter_func="not a function")

        # Invalid query syntax
        with pytest.raises(ValueError, match="Failed to apply filter"):
            analyzer.restrict_dataframe("invalid syntax ===")

    def test_add_derived_column_with_function(self, analyzer):
        """Test add_derived_column with derivation function."""
        analyzer.add_derived_column(
            "time_per_epsilon",
            derivation_function=lambda df: df["Total_calculation_time"]
            / df["MSCG_epsilon"],
        )

        assert "time_per_epsilon" in analyzer.dataframe.columns
        assert (
            "time_per_epsilon" in analyzer.list_of_output_quantity_names_from_dataframe
        )

        # Check calculation
        expected = (
            analyzer.dataframe["Total_calculation_time"]
            / analyzer.dataframe["MSCG_epsilon"]
        )
        pd.testing.assert_series_equal(analyzer.dataframe["time_per_epsilon"], expected)

    def test_add_derived_column_with_expression(self, analyzer):
        """Test add_derived_column with string expression."""
        analyzer.add_derived_column(
            "double_time", expression="Total_calculation_time * 2"
        )

        assert "double_time" in analyzer.dataframe.columns
        expected = analyzer.dataframe["Total_calculation_time"] * 2
        pd.testing.assert_series_equal(analyzer.dataframe["double_time"], expected)

    def test_add_derived_column_method_chaining(self, analyzer):
        """Test that add_derived_column supports method chaining."""
        result = analyzer.add_derived_column(
            "new_col", expression="Total_calculation_time + 1"
        )
        assert result is analyzer

    def test_add_derived_column_errors(self, analyzer):
        """Test add_derived_column error handling."""
        # Column already exists
        with pytest.raises(ValueError, match="already exists"):
            analyzer.add_derived_column("Kernel_operator_type", expression="1")

        # No derivation method provided
        with pytest.raises(
            ValueError, match="Either derivation_function or expression"
        ):
            analyzer.add_derived_column("new_col")

        # Invalid expression
        with pytest.raises(ValueError, match="Failed to create derived column"):
            analyzer.add_derived_column("bad_col", expression="nonexistent_column * 2")

    def test_add_derived_column_rollback_on_error(self, analyzer):
        """Test that failed column addition doesn't leave partial changes."""
        original_columns = analyzer.dataframe.columns.tolist()

        # Try to add a column with bad expression
        with pytest.raises(ValueError):
            analyzer.add_derived_column(
                "bad_col", expression="1 / 0"
            )  # Division by zero

        # Columns should be unchanged
        assert analyzer.dataframe.columns.tolist() == original_columns
        assert "bad_col" not in analyzer.dataframe.columns

    def test_restore_original_dataframe(self, analyzer):
        """Test restore_original_dataframe functionality."""
        # Make multiple changes
        analyzer.restrict_dataframe("Kernel_operator_type == 'Wilson'")
        analyzer.add_derived_column("new_col", expression="1")

        # Verify changes
        assert len(analyzer.dataframe) == 2
        assert "new_col" in analyzer.dataframe.columns

        # Restore
        result = analyzer.restore_original_dataframe()
        assert result is analyzer  # Should return self for chaining

        # Should be back to original
        assert len(analyzer.dataframe) == 4
        assert "new_col" not in analyzer.dataframe.columns
        pd.testing.assert_frame_equal(analyzer.dataframe, analyzer.original_dataframe)

    def test_group_by_multivalued_tunable_parameters_basic(self, analyzer):
        """Test basic groupby functionality."""
        # Group by all multivalued tunable parameters
        grouped = analyzer.group_by_multivalued_tunable_parameters()
        assert isinstance(grouped, pd.core.groupby.generic.DataFrameGroupBy)

        # Should group by all multivalued tunable parameters
        expected_grouping_cols = [
            "Configuration_label",
            "Kernel_operator_type",
            "MSCG_epsilon",
        ]
        assert grouped.keys.tolist() == expected_grouping_cols

    def test_group_by_with_filter_out_parameters(self, analyzer):
        """Test groupby with filtered parameters."""
        grouped = analyzer.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=["MSCG_epsilon"]
        )

        # Should exclude MSCG_epsilon from grouping
        expected_cols = ["Configuration_label", "Kernel_operator_type"]
        assert grouped.keys.tolist() == expected_cols

    def test_group_by_filter_all_parameters(self, analyzer):
        """Test groupby when all parameters are filtered out."""
        all_multivalued = analyzer.list_of_multivalued_tunable_parameter_names
        grouped = analyzer.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=all_multivalued
        )

        # Should return a dummy group containing all data
        assert len(list(grouped)) == 1
        group_name, group_df = list(grouped)[0]
        assert group_name == "Dummy"
        assert len(group_df) == len(analyzer.dataframe)

    def test_group_by_errors(self, analyzer):
        """Test groupby error handling."""
        # filter_out_parameters_list not a list
        with pytest.raises(TypeError, match="must be a list"):
            analyzer.group_by_multivalued_tunable_parameters(
                filter_out_parameters_list="not a list"
            )

        # Invalid parameter in filter list
        with pytest.raises(ValueError, match="Invalid parameters"):
            analyzer.group_by_multivalued_tunable_parameters(
                filter_out_parameters_list=["nonexistent_param"]
            )

    def test_reduced_multivalued_tunable_parameter_names_property(self, analyzer):
        """Test the computed property for reduced parameter names."""
        # Before calling group_by, should return all multivalued parameters
        initial = analyzer.reduced_multivalued_tunable_parameter_names_list
        assert set(initial) == {
            "Configuration_label",
            "Kernel_operator_type",
            "MSCG_epsilon",
        }

        # After calling group_by with filter
        analyzer.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=["MSCG_epsilon"]
        )
        reduced = analyzer.reduced_multivalued_tunable_parameter_names_list
        assert set(reduced) == {"Configuration_label", "Kernel_operator_type"}

        # Property should always return sorted list
        assert reduced == sorted(reduced)

    def test_state_consistency_after_operations(self, analyzer):
        """Test that state remains consistent after various operations."""
        # Perform multiple operations
        analyzer.restrict_dataframe("Kernel_operator_type == 'Wilson'")
        analyzer.add_derived_column("test_col", expression="1")
        analyzer.group_by_multivalued_tunable_parameters(["Configuration_label"])

        # Check that categorizations are updated correctly
        assert "test_col" in analyzer.list_of_output_quantity_names_from_dataframe
        assert (
            len(analyzer.list_of_multivalued_tunable_parameter_names) == 2
        )  # One filtered

        # Restore and check
        analyzer.restore_original_dataframe()
        assert "test_col" not in analyzer.dataframe.columns
        assert analyzer._filter_out_parameters_list is None  # Reset

    def test_complex_workflow(self, analyzer):
        """Test a complex workflow combining multiple operations."""
        # Filter data
        analyzer.restrict_dataframe("Total_calculation_time > 11")

        # Add derived column
        analyzer.add_derived_column(
            "efficiency",
            derivation_function=lambda df: 100 / df["Total_calculation_time"],
        )

        # Group by parameters
        grouped = analyzer.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=["Configuration_label"]
        )

        # Verify groups
        for name, group in grouped:
            assert "efficiency" in group.columns
            assert all(group["Total_calculation_time"] > 11)

        # Use context manager for temporary change
        with analyzer:
            analyzer.restrict_dataframe("Kernel_operator_type == 'Brillouin'")
            assert all(analyzer.dataframe["Kernel_operator_type"] == "Brillouin")

        # Should be back to previous state
        assert len(analyzer.dataframe) > len(
            analyzer.dataframe[
                analyzer.dataframe["Kernel_operator_type"] == "Brillouin"
            ]
        )


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
