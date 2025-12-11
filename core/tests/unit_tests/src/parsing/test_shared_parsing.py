"""
Unit tests for shared parsing functions in _shared_parsing.py module.

Tests cover parameter classification, HDF5 structure creation, data
export, and parameter validation functionality used by both log and
correlator parsing scripts.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
import h5py
from unittest.mock import Mock, patch, ANY

from src.parsing._shared_parsing import (
    _classify_parameters_by_uniqueness,
    _create_hdf5_group_structure,
    _create_hdf5_structure_with_constant_params,
    _export_dataframe_to_csv,
    _export_arrays_to_hdf5_with_proper_structure,
    _check_parameter_mismatches,
)


class TestClassifyParametersByUniqueness:
    """Test parameter classification into constant and multivalued
    categories."""

    def test_all_constant_parameters(self):
        """Test case where all parameters have constant values across
        files."""
        scalar_params_list = [
            {"Filename": "file1.txt", "beta": 6.0, "mass": 0.1, "method": "KL"},
            {"Filename": "file2.txt", "beta": 6.0, "mass": 0.1, "method": "KL"},
            {"Filename": "file3.txt", "beta": 6.0, "mass": 0.1, "method": "KL"},
        ]

        dataframe, constant_params_dict, multivalued_params_list = (
            _classify_parameters_by_uniqueness(scalar_params_list)
        )

        # Check DataFrame
        assert len(dataframe) == 3
        assert list(dataframe.columns) == ["Filename", "beta", "mass", "method"]

        # Check constant parameters (excluding Filename)
        expected_constant = {"beta": 6.0, "mass": 0.1, "method": "KL"}
        assert constant_params_dict == expected_constant

        # Check multivalued parameters (only Filename should vary)
        assert multivalued_params_list == ["Filename"]

    def test_all_multivalued_parameters(self):
        """Test case where all parameters vary across files."""
        scalar_params_list = [
            {"Filename": "file1.txt", "beta": 6.0, "mass": 0.1},
            {"Filename": "file2.txt", "beta": 6.2, "mass": 0.2},
            {"Filename": "file3.txt", "beta": 6.4, "mass": 0.3},
        ]

        dataframe, constant_params_dict, multivalued_params_list = (
            _classify_parameters_by_uniqueness(scalar_params_list)
        )

        # Check that no constant parameters exist
        assert constant_params_dict == {}

        # Check that all parameters are multivalued
        assert set(multivalued_params_list) == {"Filename", "beta", "mass"}

    def test_mixed_constant_and_multivalued(self):
        """Test typical case with both constant and multivalued
        parameters."""
        scalar_params_list = [
            {"Filename": "file1.txt", "beta": 6.0, "mass": 0.1, "method": "KL"},
            {"Filename": "file2.txt", "beta": 6.0, "mass": 0.2, "method": "KL"},
            {"Filename": "file3.txt", "beta": 6.0, "mass": 0.3, "method": "KL"},
        ]

        dataframe, constant_params_dict, multivalued_params_list = (
            _classify_parameters_by_uniqueness(scalar_params_list)
        )

        # Check constant parameters
        expected_constant = {"beta": 6.0, "method": "KL"}
        assert constant_params_dict == expected_constant

        # Check multivalued parameters
        assert set(multivalued_params_list) == {"Filename", "mass"}

    def test_empty_input(self):
        """Test behavior with empty input list."""
        scalar_params_list = []

        dataframe, constant_params_dict, multivalued_params_list = (
            _classify_parameters_by_uniqueness(scalar_params_list)
        )

        assert len(dataframe) == 0
        assert constant_params_dict == {}
        assert multivalued_params_list == []

    def test_single_file(self):
        """Test behavior with only one file."""
        scalar_params_list = [{"Filename": "file1.txt", "beta": 6.0, "mass": 0.1}]

        dataframe, constant_params_dict, multivalued_params_list = (
            _classify_parameters_by_uniqueness(scalar_params_list)
        )

        # With only one file, all parameters should be constant
        expected_constant = {"Filename": "file1.txt", "beta": 6.0, "mass": 0.1}
        assert constant_params_dict == expected_constant
        assert multivalued_params_list == []

    def test_tuple_values_consistency(self):
        """Test that tuple values are handled correctly in
        classification."""
        scalar_params_list = [
            {"Filename": "file1.txt", "geometry": (4, 4, 4, 8), "beta": 6.0},
            {"Filename": "file2.txt", "geometry": (4, 4, 4, 8), "beta": 6.2},
        ]

        dataframe, constant_params_dict, multivalued_params_list = (
            _classify_parameters_by_uniqueness(scalar_params_list)
        )

        # Tuple parameter should be constant
        assert constant_params_dict["geometry"] == (4, 4, 4, 8)
        assert set(multivalued_params_list) == {"Filename", "beta"}


# =============================================================================
# NEW: Tests for _create_hdf5_group_structure (direct testing)
# =============================================================================


class TestCreateHdf5GroupStructure:
    """Test HDF5 group structure creation that mirrors directory hierarchy."""

    def test_single_level_directory_structure(self):
        """Test creating structure for single-level subdirectory."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            hdf5_path = tmp_file.name

        try:
            base_directory = "/data/raw"
            target_directory = "/data/raw/study1"

            with h5py.File(hdf5_path, "w") as hdf5_file:
                result_group = _create_hdf5_group_structure(
                    hdf5_file, base_directory, target_directory
                )

                # Check that the group was created
                assert "study1" in hdf5_file
                assert result_group.name == "/study1"

        finally:
            os.unlink(hdf5_path)

    def test_multi_level_directory_structure(self):
        """Test creating structure for multi-level subdirectory."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            hdf5_path = tmp_file.name

        try:
            base_directory = "/data/raw"
            target_directory = "/data/raw/project/experiment/run1"

            with h5py.File(hdf5_path, "w") as hdf5_file:
                result_group = _create_hdf5_group_structure(
                    hdf5_file, base_directory, target_directory
                )

                # Check hierarchical structure
                assert "project" in hdf5_file
                assert "experiment" in hdf5_file["project"]
                assert "run1" in hdf5_file["project/experiment"]
                assert result_group.name == "/project/experiment/run1"

        finally:
            os.unlink(hdf5_path)

    def test_same_base_and_target_directory(self):
        """Test when base and target directories are the same."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            hdf5_path = tmp_file.name

        try:
            base_directory = "/data/raw"
            target_directory = "/data/raw"
            logger = Mock()

            with h5py.File(hdf5_path, "w") as hdf5_file:
                result_group = _create_hdf5_group_structure(
                    hdf5_file, base_directory, target_directory, logger
                )

                # Should return root group when directories are identical
                assert result_group.name == "/"

        finally:
            os.unlink(hdf5_path)

    def test_existing_groups_are_reused(self):
        """Test that existing groups are reused (require_group behavior)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            hdf5_path = tmp_file.name

        try:
            base_directory = "/data"
            target_directory = "/data/level1/level2"

            with h5py.File(hdf5_path, "w") as hdf5_file:
                # Pre-create a group with an attribute
                existing_group = hdf5_file.create_group("level1")
                existing_group.attrs["marker"] = "pre-existing"

                # Now call the function
                result_group = _create_hdf5_group_structure(
                    hdf5_file, base_directory, target_directory
                )

                # The existing group should be reused (attribute preserved)
                assert hdf5_file["level1"].attrs["marker"] == "pre-existing"
                assert result_group.name == "/level1/level2"

        finally:
            os.unlink(hdf5_path)

    def test_with_real_paths(self, tmp_path):
        """Test with actual filesystem paths."""
        # Create actual directory structure
        base_dir = tmp_path / "raw_data"
        target_dir = base_dir / "qcd_beta_6.0" / "mass_0.1"
        target_dir.mkdir(parents=True)

        hdf5_path = tmp_path / "test.h5"

        with h5py.File(hdf5_path, "w") as hdf5_file:
            result_group = _create_hdf5_group_structure(
                hdf5_file, str(base_dir), str(target_dir)
            )

            # Check structure matches directory hierarchy
            assert "qcd_beta_6.0" in hdf5_file
            assert "mass_0.1" in hdf5_file["qcd_beta_6.0"]
            assert result_group.name == "/qcd_beta_6.0/mass_0.1"

    def test_returns_deepest_group(self):
        """Test that the function returns the deepest group created."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            hdf5_path = tmp_file.name

        try:
            with h5py.File(hdf5_path, "w") as hdf5_file:
                result = _create_hdf5_group_structure(
                    hdf5_file, "/base", "/base/a/b/c/d"
                )

                # Should return the deepest group
                assert result.name == "/a/b/c/d"
                # Verify it's an actual group object
                assert isinstance(result, h5py.Group)

        finally:
            os.unlink(hdf5_path)


# =============================================================================
# UPDATED: Tests for _create_hdf5_structure_with_constant_params
# (now testing directly without mocking)
# =============================================================================


class TestCreateHdf5StructureWithConstantParams:
    """Test HDF5 group structure creation with constant parameters."""

    def test_basic_structure_creation(self):
        """Test basic HDF5 structure creation with constant parameters."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            hdf5_path = tmp_file.name

        try:
            constant_params_dict = {"beta": 6.0, "method": "KL", "mass": 0.1}
            base_directory = "/data/raw"
            target_directory = "/data/raw/study1"
            logger = Mock()

            with h5py.File(hdf5_path, "w") as hdf5_file:
                result_group = _create_hdf5_structure_with_constant_params(
                    hdf5_file,
                    constant_params_dict,
                    base_directory,
                    target_directory,
                    logger,
                )

                # Check that the group was created
                assert result_group.name == "/study1"

                # Check that attributes were set
                assert result_group.attrs["beta"] == 6.0
                assert result_group.attrs["method"] == "KL"
                assert result_group.attrs["mass"] == 0.1

            # Verify persistence after closing
            with h5py.File(hdf5_path, "r") as hdf5_file:
                assert hdf5_file["study1"].attrs["beta"] == 6.0
                assert hdf5_file["study1"].attrs["method"] == "KL"
                assert hdf5_file["study1"].attrs["mass"] == 0.1

        finally:
            os.unlink(hdf5_path)

    def test_empty_constant_params(self):
        """Test structure creation with no constant parameters."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            hdf5_path = tmp_file.name

        try:
            constant_params_dict = {}
            logger = Mock()

            with h5py.File(hdf5_path, "w") as hdf5_file:
                result_group = _create_hdf5_structure_with_constant_params(
                    hdf5_file, constant_params_dict, "/base", "/base/target", logger
                )

                # Group should be created but with no attributes
                assert result_group.name == "/target"
                assert len(result_group.attrs) == 0

        finally:
            os.unlink(hdf5_path)

    def test_nested_directory_structure(self):
        """Test structure creation with deeply nested directories."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            hdf5_path = tmp_file.name

        try:
            constant_params_dict = {"beta": 6.0}
            base_directory = "/data"
            target_directory = "/data/project/experiment/run1"
            logger = Mock()

            with h5py.File(hdf5_path, "w") as hdf5_file:
                result_group = _create_hdf5_structure_with_constant_params(
                    hdf5_file,
                    constant_params_dict,
                    base_directory,
                    target_directory,
                    logger,
                )

                # Check nested structure
                assert result_group.name == "/project/experiment/run1"
                assert result_group.attrs["beta"] == 6.0

                # Verify intermediate groups exist
                assert "project" in hdf5_file
                assert "experiment" in hdf5_file["project"]

        finally:
            os.unlink(hdf5_path)

    def test_various_attribute_types(self):
        """Test that various Python types are stored correctly as
        attributes."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            hdf5_path = tmp_file.name

        try:
            constant_params_dict = {
                "int_param": 42,
                "float_param": 3.14159,
                "string_param": "Wilson",
                "tuple_param": (4, 4, 4, 8),
            }
            logger = Mock()

            with h5py.File(hdf5_path, "w") as hdf5_file:
                result_group = _create_hdf5_structure_with_constant_params(
                    hdf5_file,
                    constant_params_dict,
                    "/base",
                    "/base/study",
                    logger,
                )

                assert result_group.attrs["int_param"] == 42
                assert result_group.attrs["float_param"] == pytest.approx(3.14159)
                assert result_group.attrs["string_param"] == "Wilson"
                # Tuples are stored as arrays in HDF5
                np.testing.assert_array_equal(
                    result_group.attrs["tuple_param"], [4, 4, 4, 8]
                )

        finally:
            os.unlink(hdf5_path)


class TestExportDataframeToCsv:
    """Test CSV export functionality with logging."""

    def test_successful_export(self):
        """Test successful DataFrame export to CSV."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                "Filename": ["file1.txt", "file2.txt"],
                "beta": [6.0, 6.2],
                "mass": [0.1, 0.2],
            }
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            csv_path = tmp_file.name

        try:
            logger = Mock()

            _export_dataframe_to_csv(df, csv_path, logger, "test parameters")

            # Check that file was created and contains correct data
            assert os.path.exists(csv_path)
            df_read = pd.read_csv(csv_path)
            pd.testing.assert_frame_equal(df, df_read)

            # Check that logging occurred
            logger.info.assert_called_once()
            log_call = logger.info.call_args[0][0]
            assert "test parameters" in log_call
            assert os.path.basename(csv_path) in log_call

        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_custom_description(self):
        """Test export with custom description."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        logger = Mock()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            csv_path = tmp_file.name

        try:
            _export_dataframe_to_csv(df, csv_path, logger, "custom description")

            log_call = logger.info.call_args[0][0]
            assert "custom description" in log_call

        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)


# =============================================================================
# UPDATED: Tests for _export_arrays_to_hdf5_with_proper_structure
# (now testing directly without mocking)
# =============================================================================


class TestExportArraysToHdf5WithProperStructure:
    """Test HDF5 array export with proper hierarchical structure."""

    def test_complete_export_structure(self):
        """Test complete HDF5 export with proper structure."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            hdf5_path = tmp_file.name

        try:
            # Test data
            constant_params_dict = {"beta": 6.0, "method": "KL"}
            multivalued_params_list = ["mass", "geometry"]
            arrays_dict = {
                "file1.dat": {
                    "correlator_1-1": np.array([1.0, 0.8, 0.6]),
                    "correlator_g5-g5": np.array([0.9, 0.7, 0.5]),
                },
                "file2.dat": {
                    "correlator_1-1": np.array([1.1, 0.9, 0.7]),
                    "correlator_g5-g5": np.array([1.0, 0.8, 0.6]),
                },
            }
            scalar_params_list = [
                {
                    "Filename": "file1.dat",
                    "beta": 6.0,
                    "method": "KL",
                    "mass": 0.1,
                    "geometry": (4, 4, 4, 8),
                },
                {
                    "Filename": "file2.dat",
                    "beta": 6.0,
                    "method": "KL",
                    "mass": 0.2,
                    "geometry": (8, 8, 8, 16),
                },
            ]
            logger = Mock()

            _export_arrays_to_hdf5_with_proper_structure(
                constant_params_dict,
                multivalued_params_list,
                arrays_dict,
                scalar_params_list,
                hdf5_path,
                "/base",
                "/base/target",
                logger,
                "test arrays",
            )

            # Verify the structure
            with h5py.File(hdf5_path, "r") as hdf5_file:
                # Check top-level group with constant params
                target_group = hdf5_file["target"]
                assert target_group.attrs["beta"] == 6.0
                assert target_group.attrs["method"] == "KL"

                # Check file-level groups
                assert "file1.dat" in target_group
                assert "file2.dat" in target_group

                # Check multivalued params as attributes on file groups
                file1_group = target_group["file1.dat"]
                assert file1_group.attrs["mass"] == 0.1
                np.testing.assert_array_equal(
                    file1_group.attrs["geometry"], [4, 4, 4, 8]
                )

                file2_group = target_group["file2.dat"]
                assert file2_group.attrs["mass"] == 0.2
                np.testing.assert_array_equal(
                    file2_group.attrs["geometry"], [8, 8, 8, 16]
                )

                # Check datasets
                np.testing.assert_array_almost_equal(
                    file1_group["correlator_1-1"][:], [1.0, 0.8, 0.6]
                )
                np.testing.assert_array_almost_equal(
                    file1_group["correlator_g5-g5"][:], [0.9, 0.7, 0.5]
                )

            # Check logging
            logger.info.assert_called_once()
            log_call = logger.info.call_args[0][0]
            assert "test arrays" in log_call

        finally:
            if os.path.exists(hdf5_path):
                os.unlink(hdf5_path)

    def test_missing_filename_in_scalar_params(self):
        """Test handling when filename is missing from scalar params lookup."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            hdf5_path = tmp_file.name

        try:
            arrays_dict = {"missing_file.dat": {"array1": np.array([1, 2, 3])}}
            scalar_params_list = [{"Filename": "different_file.dat", "param": "value"}]
            logger = Mock()

            # Should not raise error, just handle gracefully
            _export_arrays_to_hdf5_with_proper_structure(
                {},  # constant_params_dict
                [],  # multivalued_params_list
                arrays_dict,
                scalar_params_list,
                hdf5_path,
                "/base",
                "/base/target",
                logger,
            )

            # Verify file was created and dataset exists
            with h5py.File(hdf5_path, "r") as hdf5_file:
                assert "target" in hdf5_file
                assert "missing_file.dat" in hdf5_file["target"]
                np.testing.assert_array_equal(
                    hdf5_file["target/missing_file.dat/array1"][:], [1, 2, 3]
                )

        finally:
            if os.path.exists(hdf5_path):
                os.unlink(hdf5_path)

    def test_empty_arrays_dict(self):
        """Test export with no arrays (edge case)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            hdf5_path = tmp_file.name

        try:
            logger = Mock()

            _export_arrays_to_hdf5_with_proper_structure(
                {"beta": 6.0},  # constant_params_dict
                [],  # multivalued_params_list
                {},  # empty arrays_dict
                [],  # empty scalar_params_list
                hdf5_path,
                "/base",
                "/base/target",
                logger,
            )

            # Verify structure was created even with no data
            with h5py.File(hdf5_path, "r") as hdf5_file:
                assert "target" in hdf5_file
                assert hdf5_file["target"].attrs["beta"] == 6.0
                # No file groups should exist
                assert len(list(hdf5_file["target"].keys())) == 0

        finally:
            if os.path.exists(hdf5_path):
                os.unlink(hdf5_path)


class TestCheckParameterMismatches:
    """Test parameter mismatch detection and logging."""

    def test_no_mismatches(self):
        """Test case with no parameter mismatches."""
        source1_params = {"beta": 6.0, "mass": 0.1, "method": "KL"}
        source2_params = {"beta": 6.0, "mass": 0.1, "method": "KL", "extra": "value"}
        logger = Mock()

        _check_parameter_mismatches(
            source1_params, source2_params, "test_file.txt", logger
        )

        # No warnings should be logged
        logger.warning.assert_not_called()

    def test_single_mismatch(self):
        """Test case with one parameter mismatch."""
        source1_params = {"beta": 6.0, "mass": 0.1}
        source2_params = {"beta": 6.2, "mass": 0.1}
        logger = Mock()

        _check_parameter_mismatches(
            source1_params, source2_params, "test_file.txt", logger
        )

        # Should log one warning
        logger.warning.assert_called_once()
        warning_msg = logger.warning.call_args[0][0]
        assert "beta" in warning_msg
        assert "test_file.txt" in warning_msg
        assert "6.0" in warning_msg
        assert "6.2" in warning_msg

    def test_multiple_mismatches(self):
        """Test case with multiple parameter mismatches."""
        source1_params = {"beta": 6.0, "mass": 0.1, "method": "KL"}
        source2_params = {"beta": 6.2, "mass": 0.2, "method": "KL"}
        logger = Mock()

        _check_parameter_mismatches(
            source1_params, source2_params, "test_file.txt", logger
        )

        # Should log two warnings
        assert logger.warning.call_count == 2

    def test_custom_source_names(self):
        """Test with custom source names in logging."""
        source1_params = {"beta": 6.0}
        source2_params = {"beta": 6.2}
        logger = Mock()

        _check_parameter_mismatches(
            source1_params,
            source2_params,
            "test_file.txt",
            logger,
            "custom_source1",
            "custom_source2",
        )

        warning_msg = logger.warning.call_args[0][0]
        assert "custom_source1" in warning_msg
        assert "custom_source2" in warning_msg

    def test_no_common_parameters(self):
        """Test case where sources have no parameters in common."""
        source1_params = {"param1": "value1", "param2": "value2"}
        source2_params = {"param3": "value3", "param4": "value4"}
        logger = Mock()

        _check_parameter_mismatches(
            source1_params, source2_params, "test_file.txt", logger
        )

        # No warnings should be logged
        logger.warning.assert_not_called()

    def test_tuple_parameter_mismatch(self):
        """Test mismatch detection with tuple parameters."""
        source1_params = {"geometry": (4, 4, 4, 8)}
        source2_params = {"geometry": (8, 8, 8, 16)}
        logger = Mock()

        _check_parameter_mismatches(
            source1_params, source2_params, "test_file.txt", logger
        )

        logger.warning.assert_called_once()
        warning_msg = logger.warning.call_args[0][0]
        assert "geometry" in warning_msg
        assert "(4, 4, 4, 8)" in warning_msg
        assert "(8, 8, 8, 16)" in warning_msg
