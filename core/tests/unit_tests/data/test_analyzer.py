import pytest
import pandas as pd
import numpy as np

from library.data.analyzer import _DataFrameInspector
from library.constants import TUNABLE_PARAMETER_NAMES_LIST


class TestDataFrameInspector:
    """Test suite for the _DataFrameInspector class using synthetic data."""

    @pytest.fixture
    def simple_dataframe(self):
        """Create a simple DataFrame with mixed parameter types."""
        return pd.DataFrame(
            {
                "Kernel_operator_type": ["Wilson", "Wilson", "Brillouin"],
                "MSCG_epsilon": [1e-6, 1e-5, 1e-6],
                "APE_alpha": [0.72, 0.72, 0.72],  # Single value
                "Total_calculation_time": [10.5, 12.3, 11.8],
                "Spatial_lattice_size": [24, 24, 24],  # Single value
            }
        )

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
            "Total_calculation_time": 3,  # 10.5, 12.3, 11.8
        }

        # Check multi-valued column names list
        assert set(inspector.list_of_multivalued_column_names) == {
            "Kernel_operator_type",
            "MSCG_epsilon",
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


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
