"""
Unit tests for the HDF5Analyzer public class using pytest.

Tests the complete public API including backward compatibility methods, report
generation, DataFrame creation, and data export functionality.
"""

import os
import tempfile
from pathlib import Path
import shutil

import h5py
import numpy as np
import pandas as pd
import gvar
import pytest

from library.data.hdf5_analyzer.analyzer import HDF5Analyzer


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

    return test_file


@pytest.fixture
def analyzer(test_hdf5_path):
    """Create a fresh analyzer instance for each test."""
    return HDF5Analyzer(test_hdf5_path)


@pytest.fixture
def synthetic_hdf5_with_gvar():
    """Create a synthetic HDF5 file with gvar dataset pairs."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    with h5py.File(tmp_path, "w") as f:
        # Create structure
        level1 = f.create_group("analysis")
        level2 = level1.create_group("run_001")

        # Single-valued parameters
        level2.attrs["temperature"] = 298.15
        level2.attrs["lattice_size"] = 32

        # Create groups with multi-valued parameters
        for i in range(4):
            config = level2.create_group(f"config_{i:04d}")
            config.attrs["configuration_number"] = i
            config.attrs["algorithm"] = "HMC" if i < 2 else "RHMC"

            # Time series data
            time_points = np.arange(20)

            # Regular dataset
            config.create_dataset("evolution_time", data=time_points * 0.5)

            # Gvar pair datasets
            signal = np.sin(time_points * 0.1 + i * np.pi / 6) + np.random.normal(
                0, 0.05, 20
            )
            errors = np.full_like(signal, 0.1) + np.random.normal(0, 0.01, 20)

            config.create_dataset("correlator_mean_values", data=signal)
            config.create_dataset("correlator_error_values", data=np.abs(errors))

            # Another gvar pair
            mass = (
                0.5 + 0.1 * np.cos(time_points * 0.05) + np.random.normal(0, 0.02, 20)
            )
            mass_err = np.full_like(mass, 0.05)

            config.create_dataset("effective_mass_mean_values", data=mass)
            config.create_dataset("effective_mass_error_values", data=mass_err)

            # Constant dataset across all configs
            config.create_dataset("lattice_spacing", data=np.array(0.09))

    yield Path(tmp_path)
    os.unlink(tmp_path)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestHDF5AnalyzerInitialization:
    """Test initialization and basic properties."""

    def test_init_with_string_path(self, test_hdf5_path):
        """Test initialization with string path."""
        analyzer = HDF5Analyzer(str(test_hdf5_path))
        assert analyzer._original_file_path == str(test_hdf5_path)

    def test_init_with_path_object(self, test_hdf5_path):
        """Test initialization with Path object."""
        analyzer = HDF5Analyzer(test_hdf5_path)
        assert analyzer._original_file_path == str(test_hdf5_path)

    def test_repr_output(self, analyzer):
        """Test string representation."""
        repr_str = repr(analyzer)
        assert "HDF5Analyzer" in repr_str
        assert "Total groups:" in repr_str
        assert "Active groups:" in repr_str
        assert "Tunable parameters:" in repr_str
        assert "Output quantities:" in repr_str
        assert "Virtual datasets:" in repr_str

    def test_inherited_properties(self, analyzer):
        """Test that all DataFrameAnalyzer-like properties are accessible."""
        # These should all work from inheritance
        assert hasattr(analyzer, "list_of_dataframe_column_names")
        assert hasattr(analyzer, "list_of_tunable_parameter_names_from_dataframe")
        assert hasattr(analyzer, "list_of_output_quantity_names_from_dataframe")
        assert hasattr(analyzer, "list_of_multivalued_tunable_parameter_names")
        assert hasattr(analyzer, "list_of_single_valued_tunable_parameter_names")


class TestUniquenessReport:
    """Test uniqueness report generation."""

    def test_basic_report_generation(self, analyzer):
        """Test basic report generation."""
        report = analyzer.generate_uniqueness_report()
        assert isinstance(report, str)
        assert "Single-valued fields" in report
        assert "Multivalued fields" in report
        assert "TUNABLE PARAMETERS" in report
        assert "OUTPUT QUANTITIES" in report

    def test_report_without_type_separation(self, analyzer):
        """Test report without separating by type."""
        report = analyzer.generate_uniqueness_report(separate_by_type=False)
        assert "TUNABLE PARAMETERS" not in report
        assert "OUTPUT QUANTITIES" not in report
        # But should still have the data
        assert "Single-valued fields" in report

    def test_report_value_formatting(self, synthetic_hdf5_with_gvar):
        """Test value formatting in report."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)
        report = analyzer.generate_uniqueness_report()

        # Check float formatting
        assert "298.15" in report  # temperature
        assert "32" in report  # integer value

        # Check array formatting
        if "lattice_spacing" in report:
            # Single-valued dataset should show as single value
            assert "0.09" in report or "array()" in report


class TestUniqueValuesMethod:
    """Test the unique_values method."""

    def test_unique_values_single_valued(self, synthetic_hdf5_with_gvar):
        """Test unique_values with single-valued parameter."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)
        values = analyzer.unique_values("temperature")
        assert len(values) == 1
        assert values[0] == 298.15

    def test_unique_values_multi_valued(self, synthetic_hdf5_with_gvar):
        """Test unique_values with multi-valued parameter."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)
        values = analyzer.unique_values("configuration_number")
        assert len(values) == 4
        assert values == [0, 1, 2, 3]  # Should be sorted

    def test_unique_values_with_print(self, synthetic_hdf5_with_gvar, capsys):
        """Test unique_values with print_output=True."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)
        values = analyzer.unique_values("algorithm", print_output=True)

        captured = capsys.readouterr()
        assert "2 unique values" in captured.out
        assert str(values) in captured.out

    def test_unique_values_dataset_error(self, synthetic_hdf5_with_gvar):
        """Test error when trying to get unique values for a dataset."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)
        with pytest.raises(
            ValueError, match="is a dataset.*output quantity.*not a parameter"
        ):
            analyzer.unique_values("evolution_time")

    def test_unique_values_nonexistent_error(self, analyzer):
        """Test error for non-existent parameter."""
        with pytest.raises(ValueError, match="not found in HDF5 file"):
            analyzer.unique_values("nonexistent_parameter")


class TestDataFrameCreation:
    """Test DataFrame creation methods."""

    def test_create_dataset_dataframe_basic(self, synthetic_hdf5_with_gvar):
        """Test basic dataset DataFrame creation."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)
        df = analyzer.create_dataset_dataframe("evolution_time")

        assert isinstance(df, pd.DataFrame)
        assert "evolution_time" in df.columns
        assert "time_index" in df.columns
        assert "temperature" in df.columns  # Single-valued parameter
        assert "configuration_number" in df.columns  # Multi-valued parameter
        assert len(df) == 80  # 4 configs × 20 time points

    def test_create_dataset_dataframe_without_time(self, synthetic_hdf5_with_gvar):
        """Test DataFrame creation without time column."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)
        df = analyzer.create_dataset_dataframe("evolution_time", add_time_column=False)

        assert "time_index" not in df.columns
        assert "evolution_time" in df.columns

    def test_create_dataset_dataframe_with_time_offset(self, synthetic_hdf5_with_gvar):
        """Test DataFrame creation with time offset."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)
        df = analyzer.create_dataset_dataframe("evolution_time", time_offset=10)

        assert df["time_index"].min() == 10
        assert df["time_index"].max() == 29  # 19 + 10

    def test_create_dataset_dataframe_with_filter(self, synthetic_hdf5_with_gvar):
        """Test DataFrame creation with filter function."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)

        def filter_hmc(params):
            return params.get("algorithm") == "HMC"

        df = analyzer.create_dataset_dataframe("evolution_time", filter_func=filter_hmc)

        assert len(df) == 40  # 2 configs × 20 time points
        assert all(df["algorithm"] == "HMC")

    def test_create_dataset_dataframe_without_flattening(
        self, synthetic_hdf5_with_gvar
    ):
        """Test DataFrame creation without array flattening."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)
        df = analyzer.create_dataset_dataframe("evolution_time", flatten_arrays=False)

        assert len(df) == 4  # One row per config
        assert isinstance(df.iloc[0]["evolution_time"], np.ndarray)
        assert len(df.iloc[0]["evolution_time"]) == 20


class TestGvarDataFrameCreation:
    """Test gvar-specific DataFrame creation."""

    def test_create_merged_value_error_dataframe(self, synthetic_hdf5_with_gvar):
        """Test creating DataFrame with automatic gvar merging."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)
        df = analyzer.create_merged_value_error_dataframe("correlator")

        assert "correlator" in df.columns
        assert isinstance(df.iloc[0]["correlator"], gvar.GVar)
        assert len(df) == 80  # 4 configs × 20 points

    def test_merged_dataframe_with_filter(self, synthetic_hdf5_with_gvar):
        """Test merged DataFrame with filtering."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)

        def filter_first_config(params):
            return params.get("configuration_number") == 0

        df = analyzer.create_merged_value_error_dataframe(
            "effective_mass", filter_func=filter_first_config
        )

        assert len(df) == 20  # 1 config × 20 points
        assert all(isinstance(val, gvar.GVar) for val in df["effective_mass"])

    def test_merged_dataframe_invalid_base_name(self, synthetic_hdf5_with_gvar):
        """Test error handling for invalid gvar base name."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)

        with pytest.raises(ValueError, match="No gvar dataset pair found"):
            analyzer.create_merged_value_error_dataframe("nonexistent_dataset")


class TestDataTransformation:
    """Test virtual dataset transformations."""

    def test_transform_and_save(self, synthetic_hdf5_with_gvar, temp_output_dir):
        """Test transforming data and saving to new file."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)

        # Create transformation
        analyzer.transform_dataset(
            "evolution_time", lambda x: x**2, "evolution_time_squared"
        )

        # Save to new file
        output_path = temp_output_dir / "transformed.h5"
        analyzer.save_transformed_data(output_path)

        # Verify output file
        assert output_path.exists()

        # Check contents
        with h5py.File(output_path, "r") as f:
            # Should have same structure
            assert "analysis/run_001/config_0000" in f
            group = f["analysis/run_001/config_0000"]
            assert isinstance(group, h5py.Group), "Expected a Group at this path"

            # Check original dataset
            assert "evolution_time" in group
            assert isinstance(group["evolution_time"], h5py.Dataset)
            # Check transformed dataset
            assert "evolution_time_squared" in group
            assert isinstance(group["evolution_time_squared"], h5py.Dataset)

            # Verify transformation
            evolution_time_dataset = group["evolution_time"]
            assert isinstance(evolution_time_dataset, h5py.Dataset)
            original = evolution_time_dataset[()]
            evolution_time_squared_dataset = group["evolution_time"]
            assert isinstance(evolution_time_squared_dataset, h5py.Dataset)
            transformed = evolution_time_squared_dataset[()]
            np.testing.assert_allclose(transformed, original**2)

    def test_save_with_restrictions(self, synthetic_hdf5_with_gvar, temp_output_dir):
        """Test saving with active restrictions."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)

        # Apply restriction
        analyzer.restrict_data("algorithm == 'HMC'")

        # Save filtered data
        output_path = temp_output_dir / "filtered.h5"
        analyzer.save_transformed_data(output_path)

        # Verify only HMC configs are saved
        with h5py.File(output_path, "r") as f:
            analysis_group = f["analysis/run_001"]
            assert isinstance(
                analysis_group, h5py.Group
            ), "Expected a Group at this path"
            # Count config groups
            config_count = 0
            for key in analysis_group.keys():
                if key.startswith("config_"):
                    config_count += 1
                    # Check algorithm attribute
                    assert f[f"analysis/run_001/{key}"].attrs["algorithm"] == "HMC"

            assert config_count == 2  # Only 2 HMC configs

    def test_save_gvar_transformation(self, synthetic_hdf5_with_gvar, temp_output_dir):
        """Test saving transformed gvar datasets."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)

        # Transform a gvar dataset
        analyzer.transform_dataset(
            "correlator",  # This will use gvar values
            lambda x: 2 * x,
            "correlator_doubled",
        )

        output_path = temp_output_dir / "gvar_transformed.h5"
        analyzer.save_transformed_data(output_path)

        # Check that gvar is properly split
        with h5py.File(output_path, "r") as f:
            group = f["analysis/run_001/config_0000"]
            assert isinstance(group, h5py.Group), "Expected a Group at this path"

            assert "correlator_doubled_mean_values" in group
            dataset = group["correlator_doubled_mean_values"]
            assert isinstance(dataset, h5py.Dataset)
            mean = dataset[()]

            assert "correlator_doubled_error_values" in group
            error_dataset = group["correlator_doubled_error_values"]
            assert isinstance(error_dataset, h5py.Dataset)
            error = error_dataset[()]

            assert "correlator_mean_values" in group
            orig_mean_dataset = group["correlator_mean_values"]
            assert isinstance(orig_mean_dataset, h5py.Dataset)
            orig_mean = orig_mean_dataset[()]

            assert "correlator_error_values" in group
            orig_error_dataset = group["correlator_error_values"]
            assert isinstance(orig_error_dataset, h5py.Dataset)
            orig_error = orig_error_dataset[()]

            np.testing.assert_allclose(mean, 2 * orig_mean, rtol=1e-10)
            np.testing.assert_allclose(error, 2 * orig_error, rtol=1e-10)

    def test_save_without_virtual_datasets(
        self, synthetic_hdf5_with_gvar, temp_output_dir
    ):
        """Test saving without including virtual datasets."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)

        # Create transformation
        analyzer.transform_dataset("evolution_time", lambda x: x + 100, "time_plus_100")

        # Save without virtual datasets
        output_path = temp_output_dir / "no_virtual.h5"
        analyzer.save_transformed_data(output_path, include_virtual=False)

        # Verify virtual dataset is not included
        with h5py.File(output_path, "r") as f:
            group = f["analysis/run_001/config_0000"]
            assert isinstance(group, h5py.Group), "Expected a Group at this path"
            assert "evolution_time" in group  # Original dataset
            assert "time_plus_100" not in group  # Virtual dataset excluded

    def test_save_with_compression_options(
        self, synthetic_hdf5_with_gvar, temp_output_dir
    ):
        """Test different compression options."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)

        # Test with different compressions
        test_cases = [
            ("gzip", 9, "gzip_max.h5"),
            ("gzip", 1, "gzip_min.h5"),
            ("lzf", None, "lzf.h5"),
            (None, None, "no_compression.h5"),
        ]

        for compression, opts, filename in test_cases:
            output_path = temp_output_dir / filename
            if compression and opts is not None:
                analyzer.save_transformed_data(
                    output_path, compression=compression, compression_opts=opts
                )
            else:
                analyzer.save_transformed_data(output_path, compression=compression)

            assert output_path.exists()

            # Verify data integrity
            with h5py.File(output_path, "r") as f:
                dataset = f["analysis/run_001/config_0000/evolution_time"]
                assert isinstance(
                    dataset, h5py.Dataset
                ), "Expected a Dataset at this path"
                data = dataset[()]
                assert len(data) == 20


class TestContextManagerIntegration:
    """Test context manager functionality at the public API level."""

    def test_context_preserves_state(self, synthetic_hdf5_with_gvar):
        """Test that context manager preserves analyzer state."""
        analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)

        initial_groups = len(analyzer.active_groups)

        with analyzer:
            analyzer.restrict_data("configuration_number < 2")
            assert len(analyzer.active_groups) == 2

            # Create and check DataFrame inside context
            df = analyzer.create_dataset_dataframe("evolution_time")
            assert len(df) == 40  # 2 configs × 20 points

        # After context, should be restored
        assert len(analyzer.active_groups) == initial_groups
        df_after = analyzer.create_dataset_dataframe("evolution_time")
        assert len(df_after) == 80  # All configs


class TestCloseMethod:
    """Test file closing."""

    def test_close_method(self, test_hdf5_path):
        """Test that close method properly closes the file."""
        analyzer = HDF5Analyzer(test_hdf5_path)

        # File should be open
        assert analyzer._file is not None

        # Close it
        analyzer.close()

        # Should handle multiple closes gracefully
        analyzer.close()  # Should not raise


class TestRealFileIntegration:
    """Integration tests with the real test file."""

    def test_real_file_report(self, analyzer):
        """Test report generation with real file."""
        report = analyzer.generate_uniqueness_report()

        # Should contain actual parameters from the file
        assert len(report) > 0
        lines = report.split("\n")
        assert len(lines) > 5  # Should have multiple lines

    def test_real_file_gvar_handling(self, analyzer):
        """Test gvar handling with real file data."""
        # Find a gvar dataset pair from the real file
        gvar_pairs = list(analyzer._gvar_dataset_pairs.keys())

        if gvar_pairs:
            base_name = gvar_pairs[0]

            # Test DataFrame creation
            df = analyzer.create_merged_value_error_dataframe(base_name)
            assert isinstance(df, pd.DataFrame)
            assert base_name in df.columns

            # Check that values are gvar
            if len(df) > 0:
                assert isinstance(df.iloc[0][base_name], gvar.GVar)


@pytest.mark.parametrize(
    "method,args",
    [
        ("restrict_data", ["configuration_number >= 0"]),
        ("restore_all_groups", []),
        ("transform_dataset", ["evolution_time", lambda x: x, "test"]),
    ],
)
def test_method_chaining(synthetic_hdf5_with_gvar, method, args):
    """Test that methods support chaining where appropriate."""
    analyzer = HDF5Analyzer(synthetic_hdf5_with_gvar)

    # These methods should return self for chaining
    result = getattr(analyzer, method)(*args)

    if method != "dataset_values":  # This returns data, not self
        assert result is analyzer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
