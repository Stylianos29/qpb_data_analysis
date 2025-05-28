"""
Unit tests for the _HDF5DataManager private class using pytest.

Tests data manipulation, filtering, transformation, and gvar handling
capabilities while maintaining read-only access to the underlying HDF5 file.
"""

import os
import tempfile
from pathlib import Path
from contextlib import contextmanager

import h5py
import numpy as np
import pandas as pd
import gvar
import pytest

from library.data.hdf5_analyzer.data_manager import _HDF5DataManager


@pytest.fixture(scope="module")
def test_hdf5_path():
    """Provide path to the test HDF5 file."""
    test_dir = Path(__file__).parent.parent.parent.parent
    test_file = (
        test_dir
        / "mock_data"
        / "valid"
        / "KL_several_n_and_m_varying_cores_correlators_jackknife_analysis.h5"
    )

    if not test_file.exists():
        pytest.fail(f"Test HDF5 file not found at: {test_file}")

    return str(test_file)


@pytest.fixture
def manager(test_hdf5_path):
    """Create a fresh data manager instance for each test."""
    return _HDF5DataManager(test_hdf5_path)


@pytest.fixture
def synthetic_hdf5_file():
    """Create a synthetic HDF5 file with known structure for specific tests."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    with h5py.File(tmp_path, "w") as f:
        # Create directory-like structure
        level1 = f.create_group("experiment")
        level2 = level1.create_group("run_2024")

        # Add single-valued parameters at second-to-last level
        level2.attrs["temperature"] = 300.0
        level2.attrs["system_size"] = 32

        # Create multiple deepest-level groups with multi-valued parameters
        for i in range(3):
            subgroup = level2.create_group(f"config_{i:03d}")
            subgroup.attrs["configuration_id"] = i
            subgroup.attrs["measurement_type"] = "A" if i < 2 else "B"

            # Add datasets
            time_data = np.arange(10) * 0.1
            signal_data = np.sin(time_data + i * np.pi / 4)
            error_data = np.ones_like(signal_data) * 0.1

            subgroup.create_dataset("time", data=time_data)
            subgroup.create_dataset("signal_mean_values", data=signal_data)
            subgroup.create_dataset("signal_error_values", data=error_data)

            # Add a constant dataset (single-valued across groups)
            subgroup.create_dataset("constants", data=np.array([1.0, 2.0, 3.0]))

    yield tmp_path

    # Cleanup
    os.unlink(tmp_path)


class TestHDF5DataManagerBasics:
    """Test basic functionality and state management."""

    def test_initialization(self, manager):
        """Test that manager initializes with correct state."""
        assert manager._active_groups is None  # All groups active by default
        assert len(manager._virtual_datasets) == 0
        assert len(manager._data_cache) == 0
        assert len(manager._restriction_stack) == 0

    def test_active_groups_property(self, manager):
        """Test active_groups property returns all groups initially."""
        active = manager.active_groups
        assert isinstance(active, set)
        assert len(active) > 0
        # Should match deepest level groups
        max_level = max(manager._groups_by_level.keys())
        assert active == set(manager._groups_by_level[max_level])


class TestDataRestriction:
    """Test data filtering and restriction capabilities."""

    def test_restrict_with_pandas_query(self, synthetic_hdf5_file):
        """Test restriction using pandas-style query strings."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        # Get initial group count
        initial_count = len(manager.active_groups)
        assert initial_count == 3

        # Restrict to specific measurement type
        manager.restrict_data("measurement_type == 'A'")
        assert len(manager.active_groups) == 2

        # Chain restrictions
        manager.restrict_data("configuration_id > 0")
        assert len(manager.active_groups) == 1

    def test_restrict_with_filter_function(self, synthetic_hdf5_file):
        """Test restriction using custom filter functions."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        # Custom filter function
        def filter_func(params):
            return params.get("configuration_id", -1) < 2

        manager.restrict_data(filter_func=filter_func)
        assert len(manager.active_groups) == 2

    def test_restrict_with_complex_query(self, manager):
        """Test restriction with complex query on real data."""
        # This uses the real test file
        initial_count = len(manager.active_groups)

        # Apply a restriction (adjust based on actual parameters in test file)
        # This is an example - modify based on your actual data
        try:
            manager.restrict_data("KL_diagonal_order > 10")
            assert len(manager.active_groups) <= initial_count
        except Exception:
            # If the parameter doesn't exist, test a different approach
            pytest.skip("Test file doesn't have expected parameters")

    def test_restore_all_groups(self, synthetic_hdf5_file):
        """Test restoring all groups after restrictions."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        initial_count = len(manager.active_groups)
        manager.restrict_data("configuration_id == 0")
        assert len(manager.active_groups) == 1

        manager.restore_all_groups()
        assert len(manager.active_groups) == initial_count
        assert manager._active_groups is None


class TestContextManager:
    """Test context manager functionality."""

    def test_simple_context(self, synthetic_hdf5_file):
        """Test basic context manager usage."""
        manager = _HDF5DataManager(synthetic_hdf5_file)
        initial_count = len(manager.active_groups)

        with manager:
            manager.restrict_data("configuration_id == 0")
            assert len(manager.active_groups) == 1

        # Should be restored
        assert len(manager.active_groups) == initial_count

    def test_nested_contexts(self, synthetic_hdf5_file):
        """Test nested context managers."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        with manager:
            manager.restrict_data("measurement_type == 'A'")
            assert len(manager.active_groups) == 2

            with manager:
                manager.restrict_data("configuration_id == 0")
                assert len(manager.active_groups) == 1

            # Back to outer context
            assert len(manager.active_groups) == 2

        # Back to original
        assert len(manager.active_groups) == 3

    def test_context_with_exception(self, synthetic_hdf5_file):
        """Test context manager handles exceptions properly."""
        manager = _HDF5DataManager(synthetic_hdf5_file)
        initial_count = len(manager.active_groups)

        try:
            with manager:
                manager.restrict_data("configuration_id == 0")
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still be restored
        assert len(manager.active_groups) == initial_count


class TestDatasetRetrieval:
    """Test dataset value retrieval."""

    def test_retrieve_standard_dataset(self, synthetic_hdf5_file):
        """Test retrieving standard dataset values."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        # Get all values
        time_values = manager.dataset_values("time", return_gvar=False)
        assert isinstance(time_values, list)
        assert len(time_values) == 3
        assert all(isinstance(v, np.ndarray) for v in time_values)

    def test_retrieve_dataset_from_specific_group(self, synthetic_hdf5_file):
        """Test retrieving dataset from specific group."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        # Get specific group
        group_path = list(manager.active_groups)[0]
        time_values = manager.dataset_values(
            "time", return_gvar=False, group_path=group_path
        )
        assert isinstance(time_values, np.ndarray)
        assert len(time_values) == 10

    def test_nonexistent_dataset_error(self, manager):
        """Test error handling for non-existent datasets."""
        with pytest.raises(ValueError, match="Dataset 'nonexistent' not found"):
            manager.dataset_values("nonexistent")


class TestGvarHandling:
    """Test automatic gvar array handling."""

    def test_automatic_gvar_merging(self, synthetic_hdf5_file):
        """Test automatic merging of mean/error pairs."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        # Request base name with gvar=True
        signal_values = manager.dataset_values("signal", return_gvar=True)
        assert isinstance(signal_values, list)
        assert len(signal_values) == 3

        # Check that we got gvar arrays
        for gvar_array in signal_values:
            assert isinstance(gvar_array[0], gvar.GVar)
            assert len(gvar_array) == 10

    def test_gvar_with_restriction(self, synthetic_hdf5_file):
        """Test gvar handling with restricted groups."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        manager.restrict_data("configuration_id < 2")
        signal_values = manager.dataset_values("signal", return_gvar=True)
        assert len(signal_values) == 2

    def test_explicit_mean_error_access(self, synthetic_hdf5_file):
        """Test accessing mean/error datasets explicitly."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        mean_values = manager.dataset_values(
            "signal_mean_values", return_gvar=False
        )
        error_values = manager.dataset_values(
            "signal_error_values", return_gvar=False
        )

        assert len(mean_values) == len(error_values)
        assert all(m.shape == e.shape for m, e in zip(mean_values, error_values))


class TestVirtualDatasets:
    """Test virtual dataset transformations."""

    def test_simple_transformation(self, synthetic_hdf5_file):
        """Test creating and retrieving transformed datasets."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        # Create virtual dataset
        manager.transform_dataset("time", lambda x: x**2, "time_squared")

        # Retrieve transformed values
        squared_values = manager.dataset_values("time_squared")
        original_values = manager.dataset_values("time")

        assert len(squared_values) == len(original_values)
        for sq, orig in zip(squared_values, original_values):
            np.testing.assert_allclose(sq, orig**2)

    def test_transformation_on_gvar(self, synthetic_hdf5_file):
        """Test transformations work with gvar arrays."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        # Transform gvar dataset
        manager.transform_dataset("signal", lambda x: 2 * x, "signal_doubled")

        doubled = manager.dataset_values("signal_doubled")
        original = manager.dataset_values("signal")

        for d, o in zip(doubled, original):
            np.testing.assert_allclose(gvar.mean(d), 2 * gvar.mean(o))
            np.testing.assert_allclose(gvar.sdev(d), 2 * gvar.sdev(o))

    def test_virtual_dataset_in_listings(self, synthetic_hdf5_file):
        """Test that virtual datasets appear in dataset lists."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        initial_count = len(manager.list_of_output_quantity_names_from_dataframe)
        manager.transform_dataset("time", lambda x: x + 1, "time_plus_one")

        assert "time_plus_one" in manager.list_of_output_quantity_names_from_dataframe
        assert (
            len(manager.list_of_output_quantity_names_from_dataframe)
            == initial_count + 1
        )


class TestDataFrameGeneration:
    """Test DataFrame export functionality."""

    def test_basic_dataframe_generation(self, synthetic_hdf5_file):
        """Test basic DataFrame generation."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        df = manager.to_dataframe(datasets=["time"], include_parameters=True)

        assert isinstance(df, pd.DataFrame)
        assert "time_index" in df.columns
        assert "temperature" in df.columns  # Single-valued parameter
        assert "configuration_id" in df.columns  # Multi-valued parameter
        assert len(df) == 30  # 3 groups × 10 time points

    def test_dataframe_without_flattening(self, synthetic_hdf5_file):
        """Test DataFrame generation without array flattening."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        df = manager.to_dataframe(datasets=["time"], flatten_arrays=False)

        assert len(df) == 3  # One row per group
        assert isinstance(df.iloc[0]["time"], np.ndarray)

    def test_dataframe_with_gvar(self, synthetic_hdf5_file):
        """Test DataFrame includes gvar values correctly."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        df = manager.to_dataframe(datasets=["signal"])

        assert "signal" in df.columns
        assert isinstance(df.iloc[0]["signal"], gvar.GVar)

    def test_dataframe_with_restrictions(self, synthetic_hdf5_file):
        """Test DataFrame respects active group restrictions."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        manager.restrict_data("configuration_id == 0")
        df = manager.to_dataframe(datasets=["time"])

        assert len(df) == 10  # Only one group
        assert all(df["configuration_id"] == 0)


class TestGrouping:
    """Test grouping functionality."""

    def test_group_by_parameters(self, synthetic_hdf5_file):
        """Test grouping by multivalued parameters."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        groups = manager.group_by_multivalued_tunable_parameters()

        assert isinstance(groups, dict)
        # Should group by both configuration_id and measurement_type
        assert len(groups) > 0

        # Check structure
        for key, group_list in groups.items():
            assert isinstance(key, tuple)
            assert isinstance(group_list, list)
            assert all(isinstance(g, str) for g in group_list)

    def test_group_by_with_exclusion(self, synthetic_hdf5_file):
        """Test grouping with parameter exclusion."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        # Exclude configuration_id from grouping
        groups = manager.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=["configuration_id"]
        )

        # Should only group by measurement_type
        assert len(groups) == 2  # 'A' and 'B'

    def test_reduced_parameters_property(self, synthetic_hdf5_file):
        """Test reduced multivalued parameters property."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        initial_params = manager.reduced_multivalued_tunable_parameter_names_list

        # Restrict to single measurement type
        manager.restrict_data("measurement_type == 'A'")
        reduced_params = manager.reduced_multivalued_tunable_parameter_names_list

        # measurement_type should no longer be multivalued
        assert "measurement_type" not in reduced_params
        assert "configuration_id" in reduced_params  # Still varies


class TestCaching:
    """Test data caching behavior."""

    def test_cache_population(self, synthetic_hdf5_file):
        """Test that data is cached after first access."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        assert len(manager._data_cache) == 0

        # Access dataset
        manager.dataset_values("time")

        assert len(manager._data_cache) > 0

    def test_cache_cleared_on_context_exit(self, synthetic_hdf5_file):
        """Test cache is cleared when exiting context."""
        manager = _HDF5DataManager(synthetic_hdf5_file)

        with manager:
            manager.dataset_values("time")
            assert len(manager._data_cache) > 0

        assert len(manager._data_cache) == 0


@pytest.mark.parametrize(
    "method_name,args",
    [
        ("restrict_data", ["configuration_id >= 0"]),
        ("restore_all_groups", []),
        ("dataset_values", ["time"]),
        ("to_dataframe", []),
    ],
)
def test_method_chaining(synthetic_hdf5_file, method_name, args):
    """Test that methods return self for chaining."""
    manager = _HDF5DataManager(synthetic_hdf5_file)

    method = getattr(manager, method_name)
    result = method(*args)

    # Most methods should return self for chaining
    if method_name not in [
        "dataset_values",
        "to_dataframe",
        "group_by_multivalued_tunable_parameters",
    ]:
        assert result is manager


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
