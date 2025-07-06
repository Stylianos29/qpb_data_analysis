import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open

from library.visualization.managers import PlotFileManager


# Module-level fixtures
@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing file operations."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def file_manager(temp_directory):
    """Create a file manager with temporary directory."""
    return PlotFileManager(temp_directory)


@pytest.fixture
def file_manager_pdf(temp_directory):
    """Create a file manager with PDF as default format."""
    return PlotFileManager(temp_directory, default_format="pdf")


@pytest.fixture
def existing_files_directory(temp_directory):
    """Create a directory with some existing files for testing."""
    # Create subdirectory with files
    subdir = Path(temp_directory) / "test_subdir"
    subdir.mkdir()

    # Create some test files
    (subdir / "file1.txt").write_text("content1")
    (subdir / "file2.png").write_text("content2")
    (subdir / "nested_dir").mkdir()
    (subdir / "nested_dir" / "file3.txt").write_text("content3")

    yield temp_directory
    # Cleanup handled by temp_directory fixture


class TestInitialization:
    """Test initialization and configuration."""

    def test_init_basic(self, temp_directory):
        """Test basic initialization."""
        manager = PlotFileManager(temp_directory)

        assert manager.base_directory == Path(temp_directory).resolve()
        assert manager.default_format == "png"
        assert manager.base_directory.exists()

    def test_init_custom_format(self, temp_directory):
        """Test initialization with custom default format."""
        manager = PlotFileManager(temp_directory, default_format="pdf")

        assert manager.default_format == "pdf"

    def test_init_nonexistent_directory(self):
        """Test initialization creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_path = Path(temp_dir) / "new_directory"

            manager = PlotFileManager(str(nonexistent_path))

            assert manager.base_directory.exists()
            assert manager.base_directory.is_dir()

    def test_init_invalid_format(self, temp_directory):
        """Test initialization with invalid format raises error."""
        with pytest.raises(ValueError, match="Unsupported format"):
            PlotFileManager(temp_directory, default_format="invalid")

    def test_init_empty_base_directory(self):
        """Test initialization with empty directory string resolves to current directory."""
        manager = PlotFileManager("")

        # Empty string should resolve to current working directory
        assert manager.base_directory == Path("").resolve()
        assert manager.base_directory.exists()

    @patch("os.access")
    def test_init_no_write_permission(self, mock_access, temp_directory):
        """Test initialization fails when directory is not writable."""
        # Mock read access as True, write access as False
        mock_access.side_effect = lambda path, mode: mode != os.W_OK

        with pytest.raises(OSError, match="Cannot write to directory"):
            PlotFileManager(temp_directory)

    @patch("os.access")
    def test_init_no_read_permission(self, mock_access, temp_directory):
        """Test initialization fails when directory is not readable."""
        # Mock write access as True, read access as False
        mock_access.side_effect = lambda path, mode: mode != os.R_OK

        with pytest.raises(OSError, match="Cannot read from directory"):
            PlotFileManager(temp_directory)


class TestSupportedFormats:
    """Test supported format functionality."""

    def test_get_supported_formats(self, file_manager):
        """Test retrieving list of supported formats."""
        formats = file_manager.get_supported_formats()

        expected_formats = ["png", "pdf", "svg", "eps", "jpg", "jpeg", "tiff", "ps"]
        assert all(fmt in formats for fmt in expected_formats)
        assert isinstance(formats, list)

    def test_get_format_info_valid(self, file_manager):
        """Test getting format information for valid format."""
        info = file_manager.get_format_info("png")

        assert info["extension"] == ".png"
        assert info["description"] == "Portable Network Graphics"
        assert isinstance(info, dict)

    def test_get_format_info_case_insensitive(self, file_manager):
        """Test format info is case insensitive."""
        info_lower = file_manager.get_format_info("png")
        info_upper = file_manager.get_format_info("PNG")

        assert info_lower == info_upper

    def test_get_format_info_invalid(self, file_manager):
        """Test getting format info for invalid format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            file_manager.get_format_info("invalid")

    def test_set_default_format_valid(self, file_manager):
        """Test setting valid default format."""
        file_manager.set_default_format("pdf")
        assert file_manager.default_format == "pdf"

    def test_set_default_format_case_insensitive(self, file_manager):
        """Test setting default format is case insensitive."""
        file_manager.set_default_format("PDF")
        assert file_manager.default_format == "pdf"

    def test_set_default_format_invalid(self, file_manager):
        """Test setting invalid default format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            file_manager.set_default_format("invalid")


class TestSubdirectoryOperations:
    """Test subdirectory creation and management."""

    def test_prepare_subdirectory_basic(self, file_manager):
        """Test basic subdirectory preparation."""
        subdir_path = file_manager.prepare_subdirectory("test_plots")

        assert Path(subdir_path).exists()
        assert Path(subdir_path).is_dir()
        assert Path(subdir_path).name == "test_plots"

    def test_prepare_subdirectory_existing(self, file_manager):
        """Test preparing subdirectory that already exists."""
        # Create subdirectory first
        first_path = file_manager.prepare_subdirectory("existing")

        # Prepare again
        second_path = file_manager.prepare_subdirectory("existing")

        assert first_path == second_path
        assert Path(second_path).exists()

    def test_prepare_subdirectory_sanitization(self, file_manager):
        """Test subdirectory name sanitization."""
        subdir_path = file_manager.prepare_subdirectory("test<>plots|invalid")

        # Should replace invalid characters with underscores
        assert "test_plots_invalid" in subdir_path
        assert Path(subdir_path).exists()

    def test_prepare_subdirectory_whitespace_handling(self, file_manager):
        """Test handling of whitespace in subdirectory names."""
        subdir_path = file_manager.prepare_subdirectory("  test   plots  ")

        # Should clean up whitespace
        assert "test_plots" in Path(subdir_path).name
        assert Path(subdir_path).exists()

    def test_prepare_subdirectory_empty_name(self, file_manager):
        """Test preparation with empty subdirectory name."""
        with pytest.raises(ValueError, match="Directory name cannot be empty"):
            file_manager.prepare_subdirectory("")

    def test_prepare_subdirectory_invalid_chars_only(self, file_manager):
        """Test preparation with only invalid characters."""
        with pytest.raises(ValueError, match="contains only invalid characters"):
            file_manager.prepare_subdirectory("<>|")

    def test_directory_exists_true(self, file_manager):
        """Test directory_exists returns True for existing directory."""
        file_manager.prepare_subdirectory("test_dir")

        assert file_manager.directory_exists("test_dir") is True

    def test_directory_exists_false(self, file_manager):
        """Test directory_exists returns False for non-existing directory."""
        assert file_manager.directory_exists("nonexistent") is False

    def test_directory_exists_sanitized_name(self, file_manager):
        """Test directory_exists with name that needs sanitization."""
        file_manager.prepare_subdirectory("test<>dir")

        # Should find directory even with unsanitized input
        assert file_manager.directory_exists("test<>dir") is True


class TestDirectoryClearing:
    """Test directory clearing functionality."""

    def test_clear_directory_no_contents(self, file_manager):
        """Test clearing empty directory."""
        subdir_path = file_manager.prepare_subdirectory("empty_dir")

        # Should not raise error
        cleared_path = file_manager.prepare_subdirectory(
            "empty_dir", clear_existing=True, confirm_clear=False
        )

        assert cleared_path == subdir_path
        assert Path(cleared_path).exists()

    def test_clear_directory_with_files(self, file_manager):
        """Test clearing directory with files."""
        subdir_path = file_manager.prepare_subdirectory(
            "files_dir", confirm_clear=False
        )

        # Create some files
        test_file1 = Path(subdir_path) / "test1.txt"
        test_file2 = Path(subdir_path) / "test2.png"
        test_file1.write_text("content1")
        test_file2.write_text("content2")

        # Clear directory
        file_manager.prepare_subdirectory(
            "files_dir", clear_existing=True, confirm_clear=False
        )

        assert not test_file1.exists()
        assert not test_file2.exists()
        assert Path(subdir_path).exists()  # Directory itself should remain

    def test_clear_directory_with_subdirs(self, file_manager):
        """Test clearing directory with subdirectories."""
        subdir_path = file_manager.prepare_subdirectory(
            "nested_dir", confirm_clear=False
        )

        # Create nested structure
        nested = Path(subdir_path) / "nested"
        nested.mkdir()
        (nested / "nested_file.txt").write_text("nested content")

        # Clear directory
        file_manager.prepare_subdirectory(
            "nested_dir", clear_existing=True, confirm_clear=False
        )

        assert not nested.exists()
        assert Path(subdir_path).exists()

    @patch("builtins.input", return_value="y")
    @patch.object(PlotFileManager, "_is_interactive", return_value=True)
    def test_clear_directory_user_confirms(
        self, mock_interactive, mock_input, file_manager
    ):
        """Test clearing directory when user confirms."""
        subdir_path = file_manager.prepare_subdirectory("confirm_dir")

        # Create a test file
        test_file = Path(subdir_path) / "test.txt"
        test_file.write_text("content")

        # Should clear when user confirms
        file_manager.prepare_subdirectory(
            "confirm_dir", clear_existing=True, confirm_clear=True
        )

        assert not test_file.exists()
        mock_input.assert_called_once()

    @patch("builtins.input", return_value="n")
    @patch.object(PlotFileManager, "_is_interactive", return_value=True)
    def test_clear_directory_user_cancels(
        self, mock_interactive, mock_input, file_manager
    ):
        """Test clearing directory when user cancels."""
        subdir_path = file_manager.prepare_subdirectory("cancel_dir")

        # Create a test file
        test_file = Path(subdir_path) / "test.txt"
        test_file.write_text("content")

        # Should raise error when user cancels
        with pytest.raises(RuntimeError, match="cancelled by user"):
            file_manager.prepare_subdirectory(
                "cancel_dir", clear_existing=True, confirm_clear=True
            )

        assert test_file.exists()  # File should still exist


class TestPlotPathConstruction:
    """Test plot file path construction."""

    def test_plot_path_basic(self, file_manager):
        """Test basic plot path construction."""
        directory = str(file_manager.base_directory)
        path = file_manager.plot_path(directory, "test_plot")

        expected = str(file_manager.base_directory / "test_plot.png")
        assert path == expected

    def test_plot_path_custom_format(self, file_manager):
        """Test plot path with custom format."""
        directory = str(file_manager.base_directory)
        path = file_manager.plot_path(directory, "test_plot", format="pdf")

        expected = str(file_manager.base_directory / "test_plot.pdf")
        assert path == expected

    def test_plot_path_case_insensitive_format(self, file_manager):
        """Test plot path with case-insensitive format."""
        directory = str(file_manager.base_directory)
        path = file_manager.plot_path(directory, "test_plot", format="PDF")

        expected = str(file_manager.base_directory / "test_plot.pdf")
        assert path == expected

    def test_plot_path_invalid_format(self, file_manager):
        """Test plot path with invalid format."""
        directory = str(file_manager.base_directory)

        with pytest.raises(ValueError, match="Unsupported format"):
            file_manager.plot_path(directory, "test_plot", format="invalid")

    def test_plot_path_filename_sanitization(self, file_manager):
        """Test filename sanitization in plot path."""
        directory = str(file_manager.base_directory)
        path = file_manager.plot_path(directory, "test<>plot|invalid")

        # Should sanitize filename
        assert "test_plot_invalid.png" in path

    def test_plot_path_empty_filename(self, file_manager):
        """Test plot path with empty filename."""
        directory = str(file_manager.base_directory)

        with pytest.raises(ValueError, match="Filename cannot be empty"):
            file_manager.plot_path(directory, "")

    def test_plot_path_ensure_unique_new_file(self, file_manager):
        """Test ensure_unique with non-existing file."""
        directory = str(file_manager.base_directory)
        path = file_manager.plot_path(directory, "unique_test", ensure_unique=True)

        expected = str(file_manager.base_directory / "unique_test.png")
        assert path == expected

    def test_plot_path_ensure_unique_existing_file(self, file_manager):
        """Test ensure_unique with existing file."""
        directory = str(file_manager.base_directory)

        # Create existing file
        existing_file = file_manager.base_directory / "existing.png"
        existing_file.write_text("content")

        path = file_manager.plot_path(directory, "existing", ensure_unique=True)

        expected = str(file_manager.base_directory / "existing_1.png")
        assert path == expected

    def test_plot_path_ensure_unique_multiple_existing(self, file_manager):
        """Test ensure_unique with multiple existing files."""
        directory = str(file_manager.base_directory)

        # Create multiple existing files
        base_file = file_manager.base_directory / "multiple.png"
        file1 = file_manager.base_directory / "multiple_1.png"
        file2 = file_manager.base_directory / "multiple_2.png"

        base_file.write_text("content")
        file1.write_text("content")
        file2.write_text("content")

        path = file_manager.plot_path(directory, "multiple", ensure_unique=True)

        expected = str(file_manager.base_directory / "multiple_3.png")
        assert path == expected


class TestPrivateMethodValidation:
    """Test private validation methods."""

    def test_validate_format_valid(self, file_manager):
        """Test _validate_format with valid formats."""
        # Should not raise any errors
        file_manager._validate_format("png")
        file_manager._validate_format("PDF")
        file_manager._validate_format("svg")

    def test_validate_format_invalid(self, file_manager):
        """Test _validate_format with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            file_manager._validate_format("invalid")

    def test_validate_format_non_string(self, file_manager):
        """Test _validate_format with non-string input."""
        with pytest.raises(ValueError, match="Format must be a string"):
            file_manager._validate_format(123)

    def test_sanitize_directory_name_valid(self, file_manager):
        """Test _sanitize_directory_name with valid names."""
        assert file_manager._sanitize_directory_name("valid_name") == "valid_name"
        assert file_manager._sanitize_directory_name("Valid Name") == "Valid_Name"

    def test_sanitize_directory_name_invalid_chars(self, file_manager):
        """Test _sanitize_directory_name with invalid characters."""
        result = file_manager._sanitize_directory_name("test<>name|invalid")
        assert result == "test_name_invalid"

    def test_sanitize_directory_name_whitespace(self, file_manager):
        """Test _sanitize_directory_name with whitespace."""
        result = file_manager._sanitize_directory_name("  test   name  ")
        assert result == "test_name"

    def test_sanitize_directory_name_empty(self, file_manager):
        """Test _sanitize_directory_name with empty string."""
        with pytest.raises(ValueError, match="Directory name cannot be empty"):
            file_manager._sanitize_directory_name("")

    def test_sanitize_filename_valid(self, file_manager):
        """Test _sanitize_filename with valid names."""
        assert file_manager._sanitize_filename("valid_file") == "valid_file"
        assert file_manager._sanitize_filename("Valid File") == "Valid_File"

    def test_sanitize_filename_invalid_chars(self, file_manager):
        """Test _sanitize_filename with invalid characters."""
        result = file_manager._sanitize_filename("test<>file/invalid")
        assert result == "test_file_invalid"

    def test_sanitize_filename_empty(self, file_manager):
        """Test _sanitize_filename with empty string."""
        with pytest.raises(ValueError, match="Filename cannot be empty"):
            file_manager._sanitize_filename("")

    @patch("sys.stdin.isatty", return_value=True)
    def test_is_interactive_true(self, mock_isatty, file_manager):
        """Test _is_interactive returns True when stdin is a terminal."""
        assert file_manager._is_interactive() is True

    @patch("sys.stdin.isatty", return_value=False)
    def test_is_interactive_false(self, mock_isatty, file_manager):
        """Test _is_interactive returns False when stdin is not a terminal."""
        assert file_manager._is_interactive() is False

    @patch("sys.stdin.isatty", side_effect=Exception("Error"))
    def test_is_interactive_exception(self, mock_isatty, file_manager):
        """Test _is_interactive returns False when exception occurs."""
        assert file_manager._is_interactive() is False


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_ensure_directory_exists_permission_error(self, file_manager):
        """Test directory creation with permission error."""
        # Mock Path.mkdir to raise OSError
        with patch.object(Path, "mkdir", side_effect=OSError("Permission denied")):
            with pytest.raises(OSError, match="Cannot create directory"):
                file_manager._ensure_directory_exists(Path("/invalid/path"))

    def test_clear_directory_access_error(self, file_manager):
        """Test directory clearing with access error."""
        subdir_path = file_manager.prepare_subdirectory("access_error")

        # Mock iterdir to raise OSError
        with patch.object(Path, "iterdir", side_effect=OSError("Access denied")):
            with pytest.raises(OSError, match="Cannot access directory"):
                file_manager._clear_directory_contents(Path(subdir_path), False)

    def test_clear_directory_removal_error(self, file_manager):
        """Test directory clearing with file removal error."""
        subdir_path = file_manager.prepare_subdirectory("removal_error")
        test_file = Path(subdir_path) / "test.txt"
        test_file.write_text("content")

        # Mock unlink to raise OSError
        with patch.object(Path, "unlink", side_effect=OSError("Cannot remove")):
            with pytest.raises(OSError, match="Cannot remove"):
                file_manager._clear_directory_contents(Path(subdir_path), False)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_make_unique_path_high_count(self, file_manager):
        """Test _make_unique_path with many existing files."""
        directory = file_manager.base_directory
        base_name = "busy_file"

        # Create many existing files
        for i in range(10):
            if i == 0:
                (directory / f"{base_name}.png").write_text("content")
            else:
                (directory / f"{base_name}_{i}.png").write_text("content")

        base_path = directory / f"{base_name}.png"
        unique_path = file_manager._make_unique_path(base_path)

        expected = directory / f"{base_name}_10.png"
        assert unique_path == expected

    def test_complex_subdirectory_structure(self, file_manager):
        """Test creating complex nested subdirectory structure."""
        # Create nested subdirectories
        level1 = file_manager.prepare_subdirectory("level1")
        level2_path = Path(level1) / "level2"
        level2_path.mkdir()

        # Prepare subdirectory within existing structure
        level2_prepared = file_manager.prepare_subdirectory("level1")

        assert level1 == level2_prepared
        assert Path(level1).exists()
        assert level2_path.exists()

    def test_unicode_filename_handling(self, file_manager):
        """Test handling of unicode characters in filenames."""
        directory = str(file_manager.base_directory)

        # Test with unicode characters
        path = file_manager.plot_path(directory, "test_ñäme_测试")

        # Should handle unicode gracefully
        assert ".png" in path
        assert Path(path).parent == file_manager.base_directory


# Parametrized tests for comprehensive coverage
class TestParametrized:
    """Parametrized tests for comprehensive scenarios."""

    @pytest.mark.parametrize(
        "format_name,expected_extension",
        [
            ("png", ".png"),
            ("PDF", ".pdf"),
            ("svg", ".svg"),
            ("EPS", ".eps"),
            ("jpg", ".jpg"),
            ("JPEG", ".jpeg"),
            ("tiff", ".tiff"),
            ("ps", ".ps"),
        ],
    )
    def test_all_supported_formats(self, file_manager, format_name, expected_extension):
        """Test all supported formats work correctly."""
        directory = str(file_manager.base_directory)
        path = file_manager.plot_path(directory, "test", format=format_name)

        assert path.endswith(expected_extension)

    @pytest.mark.parametrize(
        "filename,expected_sanitized",
        [
            ("normal_file", "normal_file"),
            ("file with spaces", "file_with_spaces"),
            ("file<>with|invalid", "file_with_invalid"),
            ('file:with"quotes', "file_with_quotes"),
            ("file?with*wildcards", "file_with_wildcards"),
            ("  spaced  file  ", "spaced_file"),
            ("multiple___underscores", "multiple_underscores"),
        ],
    )
    def test_filename_sanitization_cases(
        self, file_manager, filename, expected_sanitized
    ):
        """Test various filename sanitization scenarios."""
        result = file_manager._sanitize_filename(filename)
        assert result == expected_sanitized

    @pytest.mark.parametrize(
        "dirname,expected_sanitized",
        [
            ("normal_dir", "normal_dir"),
            ("dir with spaces", "dir_with_spaces"),
            ("dir<>with|invalid", "dir_with_invalid"),
            ("  spaced  dir  ", "spaced_dir"),
            ("multiple   spaces", "multiple_spaces"),
        ],
    )
    def test_directory_sanitization_cases(
        self, file_manager, dirname, expected_sanitized
    ):
        """Test various directory name sanitization scenarios."""
        result = file_manager._sanitize_directory_name(dirname)
        assert result == expected_sanitized

    @pytest.mark.parametrize("confirm_input", ["y", "yes", "Y", "YES"])
    def test_clear_confirmation_yes_variants(self, file_manager, confirm_input):
        """Test various ways to confirm directory clearing."""
        with patch("builtins.input", return_value=confirm_input), patch.object(
            PlotFileManager, "_is_interactive", return_value=True
        ):

            subdir_path = file_manager.prepare_subdirectory("confirm_test")
            test_file = Path(subdir_path) / "test.txt"
            test_file.write_text("content")

            # Should clear successfully
            file_manager.prepare_subdirectory(
                "confirm_test", clear_existing=True, confirm_clear=True
            )

            assert not test_file.exists()

    @pytest.mark.parametrize("reject_input", ["n", "no", "N", "NO", "cancel", ""])
    def test_clear_confirmation_no_variants(self, file_manager, reject_input):
        """Test various ways to reject directory clearing."""
        with patch("builtins.input", return_value=reject_input), patch.object(
            PlotFileManager, "_is_interactive", return_value=True
        ):

            subdir_path = file_manager.prepare_subdirectory("reject_test")
            test_file = Path(subdir_path) / "test.txt"
            test_file.write_text("content")

            # Should raise error
            with pytest.raises(RuntimeError, match="cancelled by user"):
                file_manager.prepare_subdirectory(
                    "reject_test", clear_existing=True, confirm_clear=True
                )

            assert test_file.exists()


class TestIntegrationScenarios:
    """Test complete workflows and integration scenarios."""

    def test_complete_workflow(self, file_manager):
        """Test a complete workflow from directory creation to file path generation."""
        # Step 1: Create subdirectory
        subdir = file_manager.prepare_subdirectory("energy_plots")

        # Step 2: Generate plot paths
        plot1_path = file_manager.plot_path(subdir, "energy_vs_time", format="png")
        plot2_path = file_manager.plot_path(subdir, "energy_distribution", format="pdf")

        # Step 3: Verify paths
        assert Path(plot1_path).parent.name == "energy_plots"
        assert plot1_path.endswith("energy_vs_time.png")
        assert plot2_path.endswith("energy_distribution.pdf")

        # Step 4: Check directory exists
        assert file_manager.directory_exists("energy_plots")

    def test_workflow_with_clearing(self, file_manager):
        """Test workflow involving directory clearing."""
        # Create directory with files
        subdir = file_manager.prepare_subdirectory("temp_plots")
        old_file = Path(subdir) / "old_plot.png"
        old_file.write_text("old content")

        # Clear and recreate
        cleaned_subdir = file_manager.prepare_subdirectory(
            "temp_plots", clear_existing=True, confirm_clear=False
        )

        # Generate new plot path
        new_plot_path = file_manager.plot_path(cleaned_subdir, "new_plot")

        # Verify
        assert not old_file.exists()
        assert cleaned_subdir == subdir
        assert new_plot_path.endswith("new_plot.png")

    def test_format_switching_workflow(self, file_manager):
        """Test workflow with format switching."""
        subdir = file_manager.prepare_subdirectory("format_test")

        # Start with default format
        path1 = file_manager.plot_path(subdir, "plot1")
        assert path1.endswith(".png")

        # Switch default format
        file_manager.set_default_format("pdf")
        path2 = file_manager.plot_path(subdir, "plot2")
        assert path2.endswith(".pdf")

        # Override with specific format
        path3 = file_manager.plot_path(subdir, "plot3", format="svg")
        assert path3.endswith(".svg")


# Running specific test groups
if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
