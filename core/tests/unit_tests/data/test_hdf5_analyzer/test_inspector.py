"""
Unit tests for the _HDF5Inspector private base class using pytest.

Tests the read-only inspection capabilities of HDF5 files following the
project's specific structure conventions.
"""

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from library.data.hdf5_analyzer.inspector import _HDF5Inspector


@pytest.fixture(scope="module")
def test_hdf5_path():
    """Provide path to the test HDF5 file."""
    test_dir = Path(__file__).parent.parent.parent.parent  # Navigate to core/tests/
    test_file = (
        test_dir
        / "mock_data"
        / "valid"
        / "KL_several_n_and_m_varying_cores_correlators_jackknife_analysis.h5"
    )

    if not test_file.exists():
        pytest.fail(f"Test HDF5 file not found at: {test_file}")

    return str(test_file)


@pytest.fixture(scope="module")
def inspector(test_hdf5_path):
    """Create an inspector instance for testing."""
    return _HDF5Inspector(test_hdf5_path)


class TestHDF5InspectorInitialization:
    """Test initialization and file handling."""

    def test_valid_file_initialization(self, test_hdf5_path):
        """Test that inspector initializes correctly with a valid HDF5 file."""
        inspector = _HDF5Inspector(test_hdf5_path)
        assert inspector.file_path == test_hdf5_path
        assert isinstance(inspector._file, h5py.File)

    def test_nonexistent_file_raises_error(self):
        """Test that inspector raises FileNotFoundError for non-existent
        files."""
        with pytest.raises(FileNotFoundError, match="HDF5 file not found"):
            _HDF5Inspector("non_existent_file.h5")

    def test_invalid_hdf5_file_raises_error(self):
        """Test that inspector raises ValueError for invalid HDF5 files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".h5", delete=False) as f:
            f.write("This is not an HDF5 file")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Cannot open file as HDF5"):
                _HDF5Inspector(temp_path)
        finally:
            os.unlink(temp_path)

    def test_file_cleanup(self, test_hdf5_path):
        """Test that HDF5 file is properly closed on deletion."""
        inspector = _HDF5Inspector(test_hdf5_path)
        file_id = id(inspector._file)

        # Delete the inspector
        del inspector

        # We can open the file again (would fail if still locked)
        with h5py.File(test_hdf5_path, "r") as f:
            assert f is not None


class TestHDF5InspectorStructure:
    """Test structure analysis capabilities."""

    def test_group_hierarchy_detection(self, inspector):
        """Test that group hierarchy levels are correctly identified."""
        assert isinstance(inspector._groups_by_level, dict)
        assert len(inspector._groups_by_level) > 0

        # Verify levels are consecutive integers
        levels = sorted(inspector._groups_by_level.keys())
        for i in range(1, len(levels)):
            assert levels[i] == levels[i - 1] + 1

    def test_parameter_hierarchy_levels(self, inspector):
        """Test that parameters are extracted from correct hierarchy levels."""
        max_level = max(inspector._groups_by_level.keys())

        # Check second-to-last level for single-valued parameters
        if max_level >= 1:
            second_to_last_groups = inspector._groups_by_level[max_level - 1]

            # Find groups with attributes at this level
            groups_with_attrs = [
                g for g in second_to_last_groups if g in inspector._parameters_by_group
            ]

            if groups_with_attrs:
                # Parameters at this level should be single-valued
                for group_path in groups_with_attrs:
                    params = inspector._parameters_by_group[group_path]
                    for param_name in params:
                        if (
                            param_name
                            in inspector.list_of_tunable_parameter_names_from_dataframe
                        ):
                            assert (
                                param_name
                                in inspector.list_of_single_valued_tunable_parameter_names
                            )


class TestHDF5InspectorParameters:
    """Test parameter extraction and categorization."""

    def test_parameter_categorization(self, inspector):
        """Test that parameters are correctly categorized as
        single/multi-valued."""
        # Should have both types of parameters
        assert len(inspector.list_of_single_valued_tunable_parameter_names) > 0
        assert len(inspector.list_of_multivalued_tunable_parameter_names) > 0

        # No overlap between categories
        single_set = set(inspector.list_of_single_valued_tunable_parameter_names)
        multi_set = set(inspector.list_of_multivalued_tunable_parameter_names)
        assert len(single_set & multi_set) == 0

        # All tunable parameters should be categorized
        all_tunable = set(inspector.list_of_tunable_parameter_names_from_dataframe)
        categorized = single_set | multi_set
        assert all_tunable == categorized

    def test_column_unique_values_single_valued(self, inspector):
        """Test column_unique_values with single-valued parameters."""
        if not inspector.list_of_single_valued_tunable_parameter_names:
            pytest.skip("No single-valued parameters in test file")

        param_name = inspector.list_of_single_valued_tunable_parameter_names[0]
        values = inspector.column_unique_values(param_name)
        assert len(values) == 1

    def test_column_unique_values_multi_valued(self, inspector):
        """Test column_unique_values with multi-valued parameters."""
        if not inspector.list_of_multivalued_tunable_parameter_names:
            pytest.skip("No multi-valued parameters in test file")

        param_name = inspector.list_of_multivalued_tunable_parameter_names[0]
        values = inspector.column_unique_values(param_name)
        assert len(values) > 1
        assert values == sorted(values)  # Should be sorted

    def test_column_unique_values_nonexistent(self, inspector):
        """Test column_unique_values with non-existent column."""
        with pytest.raises(ValueError, match="Column 'non_existent_column' not found"):
            inspector.column_unique_values("non_existent_column")


class TestHDF5InspectorDatasets:
    """Test dataset collection and analysis."""

    def test_dataset_collection(self, inspector):
        """Test that datasets are correctly identified."""
        assert len(inspector.list_of_output_quantity_names_from_dataframe) > 0
        assert len(inspector._dataset_paths) > 0

        # Each dataset should have at least one path
        for dataset_name, paths in inspector._dataset_paths.items():
            assert len(paths) > 0
            # Paths should be valid HDF5 paths
            for path in paths:
                assert "/" in path

    def test_gvar_pair_detection(self, inspector):
        """Test detection of gvar dataset pairs."""
        assert len(inspector._gvar_dataset_pairs) > 0

        for base_name, (mean_name, error_name) in inspector._gvar_dataset_pairs.items():
            # Check naming convention
            assert mean_name.endswith("_mean_values")
            assert error_name.endswith("_error_values")
            assert mean_name == f"{base_name}_mean_values"
            assert error_name == f"{base_name}_error_values"

            # Both datasets should exist
            assert mean_name in inspector._dataset_paths
            assert error_name in inspector._dataset_paths

    def test_single_valued_dataset_detection(self, inspector):
        """Test detection of single-valued datasets (rare but possible)."""
        single_valued_datasets = inspector.list_of_single_valued_output_quantity_names

        for dataset_name in single_valued_datasets:
            assert dataset_name in inspector.unique_value_columns_dictionary
            value = inspector.unique_value_columns_dictionary[dataset_name]
            assert isinstance(value, np.ndarray)


class TestHDF5InspectorConsistency:
    """Test internal consistency of the inspector."""

    def test_list_completeness(self, inspector):
        """Test that all columns are properly categorized."""
        all_columns = set(inspector.list_of_dataframe_column_names)
        single_columns = set(inspector.list_of_single_valued_column_names)
        multi_columns = set(inspector.list_of_multivalued_column_names)

        # Every column is either single or multi-valued
        assert all_columns == single_columns | multi_columns
        assert len(single_columns & multi_columns) == 0

    def test_parameter_output_separation(self, inspector):
        """Test that parameters and outputs are properly separated."""
        tunable = set(inspector.list_of_tunable_parameter_names_from_dataframe)
        outputs = set(inspector.list_of_output_quantity_names_from_dataframe)

        # No overlap between parameters and outputs
        assert len(tunable & outputs) == 0

        # Together they should cover all columns
        all_columns = set(inspector.list_of_dataframe_column_names)
        assert all_columns == tunable | outputs

    def test_storage_dictionaries_consistency(self, inspector):
        """Test consistency between storage dictionaries and lists."""
        # Single-valued columns
        single_valued_keys = set(inspector.unique_value_columns_dictionary.keys())
        single_valued_list = set(inspector.list_of_single_valued_column_names)
        assert single_valued_keys == single_valued_list

        # Multi-valued columns
        multi_valued_keys = set(inspector.multivalued_columns_count_dictionary.keys())
        multi_valued_list = set(inspector.list_of_multivalued_column_names)
        assert multi_valued_keys == multi_valued_list


@pytest.mark.parametrize(
    "property_name",
    [
        "list_of_dataframe_column_names",
        "list_of_tunable_parameter_names_from_dataframe",
        "list_of_output_quantity_names_from_dataframe",
        "list_of_single_valued_column_names",
        "list_of_multivalued_column_names",
        "list_of_single_valued_tunable_parameter_names",
        "list_of_multivalued_tunable_parameter_names",
        "list_of_single_valued_output_quantity_names",
        "list_of_multivalued_output_quantity_names",
    ],
)
def test_dataframe_analyzer_compatible_properties(inspector, property_name):
    """Test that all DataFrameAnalyzer-compatible properties exist and are lists."""
    assert hasattr(inspector, property_name)
    value = getattr(inspector, property_name)
    assert isinstance(value, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
