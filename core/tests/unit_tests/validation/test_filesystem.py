"""
Unit tests for filesystem validation utilities.

This module provides comprehensive tests for the filesystem validation
functions:
    - is_valid_directory
    - is_valid_file
    - validate_path (with optional type checking)

The tests use pytest fixtures to create temporary files and directories
for testing various validation scenarios.
"""

import os
import pytest

from library.validation.filesystem import (
    is_valid_directory,
    is_valid_file,
    validate_path,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def temp_test_structure(tmp_path):
    """
    Create a temporary directory structure for testing.

    Structure:
        tmp_path/ ├── test_file.txt ├── test_file.csv ├── empty_dir/ └──
        nested_dir/
            ├── nested_file.txt └── deeply_nested_dir/
                └── deep_file.txt
    """
    # Create files
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("test content")

    csv_file = tmp_path / "test_file.csv"
    csv_file.write_text("col1,col2\n1,2")

    # Create empty directory
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    # Create nested structure
    nested_dir = tmp_path / "nested_dir"
    nested_dir.mkdir()

    nested_file = nested_dir / "nested_file.txt"
    nested_file.write_text("nested content")

    deeply_nested_dir = nested_dir / "deeply_nested_dir"
    deeply_nested_dir.mkdir()

    deep_file = deeply_nested_dir / "deep_file.txt"
    deep_file.write_text("deep content")

    return {
        "root": tmp_path,
        "test_file": test_file,
        "csv_file": csv_file,
        "empty_dir": empty_dir,
        "nested_dir": nested_dir,
        "nested_file": nested_file,
        "deeply_nested_dir": deeply_nested_dir,
        "deep_file": deep_file,
    }


# ==============================================================================
# TESTS FOR is_valid_file()
# ==============================================================================


class TestIsValidFile:
    """Tests for the is_valid_file function."""

    def test_valid_file_returns_true(self, temp_test_structure):
        """Test that a valid file returns True."""
        test_file = temp_test_structure["test_file"]
        assert is_valid_file(str(test_file)) is True

    def test_valid_file_with_pathlib_path(self, temp_test_structure):
        """Test that function accepts pathlib.Path objects."""
        test_file = temp_test_structure["test_file"]
        assert is_valid_file(test_file) is True

    def test_nested_file_returns_true(self, temp_test_structure):
        """Test that a nested file is validated correctly."""
        nested_file = temp_test_structure["nested_file"]
        assert is_valid_file(str(nested_file)) is True

    def test_nonexistent_file_raises_error(self, temp_test_structure):
        """Test that a non-existent file raises ValueError."""
        nonexistent = temp_test_structure["root"] / "nonexistent.txt"
        with pytest.raises(ValueError, match="File does not exist"):
            is_valid_file(str(nonexistent))

    def test_directory_as_file_raises_error(self, temp_test_structure):
        """Test that passing a directory raises ValueError."""
        directory = temp_test_structure["empty_dir"]
        with pytest.raises(ValueError, match="Path exists but is not a file"):
            is_valid_file(str(directory))

    def test_different_file_extensions(self, temp_test_structure):
        """Test that files with different extensions are validated."""
        txt_file = temp_test_structure["test_file"]
        csv_file = temp_test_structure["csv_file"]

        assert is_valid_file(str(txt_file)) is True
        assert is_valid_file(str(csv_file)) is True


# ==============================================================================
# TESTS FOR is_valid_directory()
# ==============================================================================


class TestIsValidDirectory:
    """Tests for the is_valid_directory function."""

    def test_valid_directory_returns_true(self, temp_test_structure):
        """Test that a valid directory returns True."""
        directory = temp_test_structure["empty_dir"]
        assert is_valid_directory(str(directory)) is True

    def test_valid_directory_with_pathlib_path(self, temp_test_structure):
        """Test that function accepts pathlib.Path objects."""
        directory = temp_test_structure["nested_dir"]
        assert is_valid_directory(directory) is True

    def test_nested_directory_returns_true(self, temp_test_structure):
        """Test that a nested directory is validated correctly."""
        nested_dir = temp_test_structure["deeply_nested_dir"]
        assert is_valid_directory(str(nested_dir)) is True

    def test_empty_directory_returns_true(self, temp_test_structure):
        """Test that an empty directory is valid."""
        empty_dir = temp_test_structure["empty_dir"]
        assert is_valid_directory(str(empty_dir)) is True

    def test_nonexistent_directory_raises_error(self, temp_test_structure):
        """Test that a non-existent directory raises ValueError."""
        nonexistent = temp_test_structure["root"] / "nonexistent_dir"
        with pytest.raises(ValueError, match="Directory does not exist"):
            is_valid_directory(str(nonexistent))

    def test_file_as_directory_raises_error(self, temp_test_structure):
        """Test that passing a file raises ValueError."""
        file_path = temp_test_structure["test_file"]
        with pytest.raises(ValueError, match="Path exists but is not a directory"):
            is_valid_directory(str(file_path))


# ==============================================================================
# TESTS FOR validate_path() - Basic Functionality
# ==============================================================================


class TestValidatePathBasic:
    """Tests for basic validate_path functionality without type
    checking."""

    def test_validate_single_path_file(self, temp_test_structure):
        """Test validating a single file path."""
        test_file = temp_test_structure["test_file"]
        result = validate_path(str(test_file))

        assert result == str(test_file.absolute())

    def test_validate_single_path_directory(self, temp_test_structure):
        """Test validating a single directory path."""
        directory = temp_test_structure["empty_dir"]
        result = validate_path(str(directory))

        assert result == str(directory.absolute())

    def test_validate_multiple_path_components_file(self, temp_test_structure):
        """Test validating with multiple path components for a file."""
        root = temp_test_structure["root"]
        result = validate_path(str(root), "nested_dir", "nested_file.txt")

        expected = temp_test_structure["nested_file"]
        assert result == str(expected.absolute())

    def test_validate_multiple_path_components_directory(self, temp_test_structure):
        """Test validating with multiple path components for a
        directory."""
        root = temp_test_structure["root"]
        result = validate_path(str(root), "nested_dir", "deeply_nested_dir")

        expected = temp_test_structure["deeply_nested_dir"]
        assert result == str(expected.absolute())

    def test_validate_returns_absolute_path(self, temp_test_structure):
        """Test that validate_path returns an absolute path."""
        test_file = temp_test_structure["test_file"]
        result = validate_path(str(test_file))

        assert os.path.isabs(result)

    def test_validate_nonexistent_path_raises_error(self, temp_test_structure):
        """Test that a non-existent path raises ValueError."""
        root = temp_test_structure["root"]
        with pytest.raises(ValueError, match="Path does not exist"):
            validate_path(str(root), "nonexistent.txt")

    def test_validate_no_arguments_raises_error(self):
        """Test that calling validate_path with no arguments raises
        ValueError."""
        with pytest.raises(ValueError, match="At least one path component"):
            validate_path()

    def test_validate_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            validate_path("")


# ==============================================================================
# TESTS FOR validate_path() - Type Checking with must_be_file
# ==============================================================================


class TestValidatePathMustBeFile:
    """Tests for validate_path with must_be_file parameter."""

    def test_must_be_file_true_with_file(self, temp_test_structure):
        """Test must_be_file=True succeeds with an actual file."""
        test_file = temp_test_structure["test_file"]
        result = validate_path(str(test_file), must_be_file=True)

        assert result == str(test_file.absolute())

    def test_must_be_file_true_with_directory_raises_error(self, temp_test_structure):
        """Test must_be_file=True raises error with a directory."""
        directory = temp_test_structure["empty_dir"]
        with pytest.raises(ValueError, match="Path exists but is not a file"):
            validate_path(str(directory), must_be_file=True)

    def test_must_be_file_false_with_directory(self, temp_test_structure):
        """Test must_be_file=False succeeds with a directory."""
        directory = temp_test_structure["empty_dir"]
        result = validate_path(str(directory), must_be_file=False)

        assert result == str(directory.absolute())

    def test_must_be_file_false_with_file_raises_error(self, temp_test_structure):
        """Test must_be_file=False raises error with a file."""
        test_file = temp_test_structure["test_file"]
        with pytest.raises(ValueError, match="Path exists but is a file"):
            validate_path(str(test_file), must_be_file=False)

    def test_must_be_file_none_accepts_both(self, temp_test_structure):
        """Test must_be_file=None (default) accepts both files and
        directories."""
        test_file = temp_test_structure["test_file"]
        directory = temp_test_structure["empty_dir"]

        file_result = validate_path(str(test_file), must_be_file=None)
        dir_result = validate_path(str(directory), must_be_file=None)

        assert file_result == str(test_file.absolute())
        assert dir_result == str(directory.absolute())

    def test_must_be_file_with_multiple_path_components(self, temp_test_structure):
        """Test must_be_file works with multiple path components."""
        root = temp_test_structure["root"]
        result = validate_path(
            str(root), "nested_dir", "nested_file.txt", must_be_file=True
        )

        expected = temp_test_structure["nested_file"]
        assert result == str(expected.absolute())


# ==============================================================================
# TESTS FOR validate_path() - Type Checking with must_be_dir
# ==============================================================================


class TestValidatePathMustBeDir:
    """Tests for validate_path with must_be_dir parameter."""

    def test_must_be_dir_true_with_directory(self, temp_test_structure):
        """Test must_be_dir=True succeeds with an actual directory."""
        directory = temp_test_structure["empty_dir"]
        result = validate_path(str(directory), must_be_dir=True)

        assert result == str(directory.absolute())

    def test_must_be_dir_true_with_file_raises_error(self, temp_test_structure):
        """Test must_be_dir=True raises error with a file."""
        test_file = temp_test_structure["test_file"]
        with pytest.raises(ValueError, match="Path exists but is not a directory"):
            validate_path(str(test_file), must_be_dir=True)

    def test_must_be_dir_false_with_file(self, temp_test_structure):
        """Test must_be_dir=False succeeds with a file."""
        test_file = temp_test_structure["test_file"]
        result = validate_path(str(test_file), must_be_dir=False)

        assert result == str(test_file.absolute())

    def test_must_be_dir_false_with_directory_raises_error(self, temp_test_structure):
        """Test must_be_dir=False raises error with a directory."""
        directory = temp_test_structure["empty_dir"]
        with pytest.raises(ValueError, match="Path exists but is a directory"):
            validate_path(str(directory), must_be_dir=False)

    def test_must_be_dir_none_accepts_both(self, temp_test_structure):
        """Test must_be_dir=None (default) accepts both files and
        directories."""
        test_file = temp_test_structure["test_file"]
        directory = temp_test_structure["empty_dir"]

        file_result = validate_path(str(test_file), must_be_dir=None)
        dir_result = validate_path(str(directory), must_be_dir=None)

        assert file_result == str(test_file.absolute())
        assert dir_result == str(directory.absolute())

    def test_must_be_dir_with_multiple_path_components(self, temp_test_structure):
        """Test must_be_dir works with multiple path components."""
        root = temp_test_structure["root"]
        result = validate_path(
            str(root), "nested_dir", "deeply_nested_dir", must_be_dir=True
        )

        expected = temp_test_structure["deeply_nested_dir"]
        assert result == str(expected.absolute())


# ==============================================================================
# TESTS FOR validate_path() - Conflicting Parameters
# ==============================================================================


class TestValidatePathConflictingParameters:
    """Tests for validate_path with conflicting or invalid
    parameters."""

    def test_both_must_be_file_and_must_be_dir_true_raises_error(
        self, temp_test_structure
    ):
        """Test that setting both must_be_file=True and must_be_dir=True
        raises error."""
        test_file = temp_test_structure["test_file"]
        with pytest.raises(
            ValueError,
            match="Cannot specify both must_be_file=True and must_be_dir=True",
        ):
            validate_path(str(test_file), must_be_file=True, must_be_dir=True)

    def test_must_be_file_false_and_must_be_dir_false_with_file(
        self, temp_test_structure
    ):
        """Test conflicting False parameters with a file."""
        test_file = temp_test_structure["test_file"]
        # File fails must_be_file=False (expects not a file)
        with pytest.raises(ValueError, match="Path exists but is a file"):
            validate_path(str(test_file), must_be_file=False, must_be_dir=False)

    def test_must_be_file_false_and_must_be_dir_false_with_directory(
        self, temp_test_structure
    ):
        """Test conflicting False parameters with a directory."""
        directory = temp_test_structure["empty_dir"]
        # Directory fails must_be_dir=False (expects not a directory)
        with pytest.raises(ValueError, match="Path exists but is a directory"):
            validate_path(str(directory), must_be_file=False, must_be_dir=False)

    def test_must_be_file_true_and_must_be_dir_false_with_file(
        self, temp_test_structure
    ):
        """Test must_be_file=True and must_be_dir=False with a file."""
        test_file = temp_test_structure["test_file"]
        result = validate_path(str(test_file), must_be_file=True, must_be_dir=False)

        assert result == str(test_file.absolute())

    def test_must_be_file_false_and_must_be_dir_true_with_directory(
        self, temp_test_structure
    ):
        """Test must_be_file=False and must_be_dir=True with a
        directory."""
        directory = temp_test_structure["empty_dir"]
        result = validate_path(str(directory), must_be_file=False, must_be_dir=True)

        assert result == str(directory.absolute())


# ==============================================================================
# TESTS FOR validate_path() - Edge Cases
# ==============================================================================


class TestValidatePathEdgeCases:
    """Tests for edge cases in validate_path."""

    def test_deeply_nested_path(self, temp_test_structure):
        """Test validation of a deeply nested path."""
        deep_file = temp_test_structure["deep_file"]
        result = validate_path(str(deep_file))

        assert result == str(deep_file.absolute())

    def test_path_with_spaces(self, tmp_path):
        """Test validation of paths with spaces in names."""
        file_with_spaces = tmp_path / "file with spaces.txt"
        file_with_spaces.write_text("content")

        result = validate_path(str(file_with_spaces))
        assert result == str(file_with_spaces.absolute())

    def test_path_with_special_characters(self, tmp_path):
        """Test validation of paths with special characters."""
        special_file = tmp_path / "file-name_123.txt"
        special_file.write_text("content")

        result = validate_path(str(special_file))
        assert result == str(special_file.absolute())

    def test_validate_root_directory(self, temp_test_structure):
        """Test validation of the root test directory itself."""
        root = temp_test_structure["root"]
        result = validate_path(str(root))

        assert result == str(root.absolute())

    def test_validate_with_trailing_slash(self, temp_test_structure):
        """Test that trailing slashes don't affect validation."""
        directory = temp_test_structure["empty_dir"]
        path_with_slash = str(directory) + os.sep

        result = validate_path(path_with_slash)
        assert result == str(directory.absolute())


# ==============================================================================
# TESTS FOR validate_path() - Real-world Use Cases
# ==============================================================================


class TestValidatePathRealWorld:
    """Tests simulating real-world usage patterns."""

    def test_jupyter_notebook_csv_file_pattern(self, temp_test_structure):
        """Simulate typical Jupyter notebook usage for CSV file
        validation."""
        root = temp_test_structure["root"]
        csv_file = temp_test_structure["csv_file"]

        # This is how it would be used in a notebook
        PROCESSED_PARAMETER_VALUES_CSV_FILE_FULL_PATH = validate_path(
            str(root), "test_file.csv", must_be_file=True
        )

        assert PROCESSED_PARAMETER_VALUES_CSV_FILE_FULL_PATH == str(csv_file.absolute())

    def test_jupyter_notebook_directory_pattern(self, temp_test_structure):
        """Simulate typical Jupyter notebook usage for directory
        validation."""
        root = temp_test_structure["root"]
        nested_dir = temp_test_structure["nested_dir"]

        # This is how it would be used in a notebook
        DATA_FILES_SET_TABLES_DIRECTORY = validate_path(
            str(root), "nested_dir", must_be_dir=True
        )

        assert DATA_FILES_SET_TABLES_DIRECTORY == str(nested_dir.absolute())

    def test_full_path_validation_pattern(self, temp_test_structure):
        """Simulate validation of a complete absolute path."""
        test_file = temp_test_structure["test_file"]

        # User provides full path
        FULL_PATH = validate_path(str(test_file.absolute()))

        assert FULL_PATH == str(test_file.absolute())

    def test_multiple_validations_in_sequence(self, temp_test_structure):
        """Test multiple validations as would occur in a typical
        notebook."""
        root = temp_test_structure["root"]

        csv_file = validate_path(str(root), "test_file.csv", must_be_file=True)
        data_dir = validate_path(str(root), "nested_dir", must_be_dir=True)
        nested_file = validate_path(
            str(root), "nested_dir", "nested_file.txt", must_be_file=True
        )

        assert all(os.path.isabs(p) for p in [csv_file, data_dir, nested_file])


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestIntegration:
    """Integration tests comparing all three validation functions."""

    def test_validate_path_equivalent_to_is_valid_file(self, temp_test_structure):
        """Test that validate_path with must_be_file=True behaves like
        is_valid_file."""
        test_file = temp_test_structure["test_file"]

        # Both should succeed
        is_valid_file(str(test_file))
        result = validate_path(str(test_file), must_be_file=True)

        assert result == str(test_file.absolute())

    def test_validate_path_equivalent_to_is_valid_directory(self, temp_test_structure):
        """Test that validate_path with must_be_dir=True behaves like
        is_valid_directory."""
        directory = temp_test_structure["empty_dir"]

        # Both should succeed
        is_valid_directory(str(directory))
        result = validate_path(str(directory), must_be_dir=True)

        assert result == str(directory.absolute())

    def test_all_three_functions_reject_nonexistent_path(self, temp_test_structure):
        """Test that all functions properly reject non-existent
        paths."""
        root = temp_test_structure["root"]
        nonexistent = str(root / "does_not_exist.txt")

        with pytest.raises(ValueError):
            is_valid_file(nonexistent)

        with pytest.raises(ValueError):
            is_valid_directory(nonexistent)

        with pytest.raises(ValueError):
            validate_path(nonexistent)


# ==============================================================================
# PERFORMANCE TESTS (Optional)
# ==============================================================================


class TestPerformance:
    """Basic performance tests to ensure validation is efficient."""

    def test_validate_many_files_quickly(self, temp_test_structure):
        """Test that validation of many files completes in reasonable
        time."""
        import time

        root = temp_test_structure["root"]

        # Create 100 test files
        files = []
        for i in range(100):
            test_file = root / f"perf_test_{i}.txt"
            test_file.write_text("test")
            files.append(test_file)

        # Measure validation time
        start = time.time()
        for file_path in files:
            validate_path(str(file_path), must_be_file=True)
        elapsed = time.time() - start

        # Should complete in under 1 second
        assert elapsed < 1.0, f"Validation took {elapsed:.2f}s, expected < 1s"


if __name__ == "__main__":
    # Run with: pytest test_filesystem.py -v
    pytest.main([__file__, "-v", "--tb=short"])
