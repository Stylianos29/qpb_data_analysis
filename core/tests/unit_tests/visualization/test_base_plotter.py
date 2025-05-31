import pytest
import os
import tempfile
from library.visualization.base_plotter import (
    _PlotFileManager,
    _PlotTitleBuilder,
    _PlotFilenameBuilder,
)


class TestPlotFileManager:
    """Test the _PlotFileManager class."""

    def test_init_with_valid_directory(self):
        """Test initialization with a valid directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _PlotFileManager(tmpdir)
            assert manager.base_directory == tmpdir

    def test_init_with_invalid_directory_raises_error(self):
        """Test initialization with an invalid directory raises ValueError."""
        with pytest.raises(ValueError, match="Invalid plots directory"):
            _PlotFileManager("/this/does/not/exist")

    def test_prepare_subdirectory_creates_new_directory(self):
        """Test that prepare_subdirectory creates a new directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _PlotFileManager(tmpdir)
            subdir_path = manager.prepare_subdirectory("test_plots")

            assert os.path.exists(subdir_path)
            assert os.path.isdir(subdir_path)
            assert subdir_path == os.path.join(tmpdir, "test_plots")

    def test_prepare_subdirectory_with_existing_directory(self):
        """Test prepare_subdirectory with an existing directory (no clear)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _PlotFileManager(tmpdir)

            # Create directory with a file
            subdir_path = manager.prepare_subdirectory("test_plots")
            test_file = os.path.join(subdir_path, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            # Call again without clearing
            result_path = manager.prepare_subdirectory(
                "test_plots", clear_existing=False
            )

            assert result_path == subdir_path
            assert os.path.exists(test_file)  # File should still exist

    def test_prepare_subdirectory_with_clear_existing(self):
        """Test prepare_subdirectory clears existing content when requested."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _PlotFileManager(tmpdir)

            # Create directory with files and subdirectories
            subdir_path = manager.prepare_subdirectory("test_plots")
            test_file = os.path.join(subdir_path, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            nested_dir = os.path.join(subdir_path, "nested")
            os.makedirs(nested_dir)

            # Call with clear_existing=True
            result_path = manager.prepare_subdirectory(
                "test_plots", clear_existing=True
            )

            assert result_path == subdir_path
            assert not os.path.exists(test_file)  # File should be deleted
            assert not os.path.exists(nested_dir)  # Subdirectory should be deleted
            assert os.path.exists(subdir_path)  # Main directory should still exist

    def test_plot_path_construction(self):
        """Test plot_path constructs correct full path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _PlotFileManager(tmpdir)

            path = manager.plot_path("/some/directory", "my_plot")
            assert path == "/some/directory/my_plot.png"


class TestPlotTitleBuilder:
    """Test the _PlotTitleBuilder class."""

    @pytest.fixture
    def title_labels(self):
        """Sample title labels for testing."""
        return {
            "temperature": "Temperature",
            "pressure": "Pressure",
            "Overlap_operator_method": "Overlap",
            "Kernel_operator_type": "Kernel",
            "Number_of_Chebyshev_terms": "Chebyshev Terms",
            "KL_diagonal_order": "KL Order",
        }

    def test_init(self, title_labels):
        """Test initialization."""
        builder = _PlotTitleBuilder(title_labels)
        assert builder.title_labels == title_labels
        assert builder.title_number_format == ".2f"

    def test_init_with_custom_format(self, title_labels):
        """Test initialization with custom number format."""
        builder = _PlotTitleBuilder(title_labels, ".3e")
        assert builder.title_number_format == ".3e"

    def test_build_simple_title_from_columns(self, title_labels):
        """Test building a simple title from selected columns."""
        builder = _PlotTitleBuilder(title_labels)
        metadata = {"temperature": 300.5, "pressure": 1.234, "other": "ignored"}

        title = builder.build(
            metadata,
            ["temperature", "pressure"],
            title_from_columns=["temperature", "pressure"],
        )

        assert title == "Temperature=300.50, Pressure=1.23"

    def test_build_with_kernel_operator_type(self, title_labels):
        """Test special handling of Kernel_operator_type in simple titles."""
        builder = _PlotTitleBuilder(title_labels)
        metadata = {"Kernel_operator_type": "Wilson"}

        title = builder.build(metadata, [], title_from_columns=["Kernel_operator_type"])

        assert title == "Wilson Kernel"

    def test_build_with_leading_substring(self, title_labels):
        """Test title with leading substring."""
        builder = _PlotTitleBuilder(title_labels)
        metadata = {"temperature": 300}

        title = builder.build(metadata, ["temperature"], leading_substring="Fig 1:")

        assert title.startswith("Fig 1:")

    def test_build_with_bare_overlap_method(self, title_labels):
        """Test title building with Bare overlap method."""
        builder = _PlotTitleBuilder(title_labels)
        metadata = {"Overlap_operator_method": "Bare", "Kernel_operator_type": "Wilson"}

        title = builder.build(metadata, [])
        assert title == "Bare Wilson"

    def test_build_with_chebyshev_overlap_method(self, title_labels):
        """Test title building with Chebyshev overlap method."""
        builder = _PlotTitleBuilder(title_labels)
        metadata = {
            "Overlap_operator_method": "Chebyshev",
            "Kernel_operator_type": "Brillouin",
            "Number_of_Chebyshev_terms": 50,
        }

        title = builder.build(metadata, ["Number_of_Chebyshev_terms"])
        assert title == "Chebyshev Brillouin 50"

    def test_build_with_kl_overlap_method(self, title_labels):
        """Test title building with KL overlap method."""
        builder = _PlotTitleBuilder(title_labels)
        metadata = {
            "Overlap_operator_method": "KL",
            "Kernel_operator_type": "Wilson",
            "KL_diagonal_order": 10,
        }

        title = builder.build(metadata, ["KL_diagonal_order"])
        assert title == "KL Wilson 10"

    def test_build_with_excluded_parameters(self, title_labels):
        """Test that excluded parameters don't appear in title."""
        builder = _PlotTitleBuilder(title_labels)
        metadata = {
            "temperature": 300,
            "pressure": 1.5,
            "excluded_param": "should not appear",
        }

        title = builder.build(
            metadata,
            ["temperature", "pressure", "excluded_param"],
            excluded={"excluded_param", "pressure"},
        )

        assert "Temperature 300.00" in title
        assert "pressure" not in title.lower()
        assert "excluded_param" not in title

    def test_build_with_title_wrapping(self, title_labels):
        """Test title wrapping at appropriate length."""
        builder = _PlotTitleBuilder(title_labels)
        metadata = {
            "param1": "very_long_value",
            "param2": "another_long_value",
            "param3": "yet_another_value",
        }

        title = builder.build(
            metadata, ["param1", "param2", "param3"], wrapping_length=30
        )

        assert "\n" in title  # Should contain a newline

    def test_format_value_with_numbers(self, title_labels):
        """Test number formatting."""
        builder = _PlotTitleBuilder(title_labels, ".3f")

        assert builder._format_value(1.23456) == "1.235"
        assert builder._format_value(42) == "42.000"
        assert builder._format_value("text") == "text"


class TestPlotFilenameBuilder:
    """Test the _PlotFilenameBuilder class."""

    @pytest.fixture
    def filename_labels(self):
        """Sample filename labels for testing."""
        return {"temperature": "T", "pressure": "P", "lattice_size": "L"}

    def test_init(self, filename_labels):
        """Test initialization."""
        builder = _PlotFilenameBuilder(filename_labels)
        assert builder.filename_labels == filename_labels

    def test_build_basic_filename(self, filename_labels):
        """Test basic filename construction."""
        builder = _PlotFilenameBuilder(filename_labels)
        metadata = {"temperature": 300, "pressure": 1.5}

        filename = builder.build(
            metadata, "energy_vs_time", ["temperature", "pressure"]
        )

        assert filename == "energy_vs_time_T300_P1p5"

    def test_build_with_overlap_method(self, filename_labels):
        """Test filename with overlap method."""
        builder = _PlotFilenameBuilder(filename_labels)
        metadata = {"Overlap_operator_method": "Chebyshev", "temperature": 300}

        filename = builder.build(metadata, "plot_base", ["temperature"])

        assert filename == "Chebyshev_plot_base_T300"

    def test_build_with_kernel_type(self, filename_labels):
        """Test filename with kernel type."""
        builder = _PlotFilenameBuilder(filename_labels)
        metadata = {"Kernel_operator_type": "Wilson", "pressure": 2.0}

        filename = builder.build(metadata, "plot_base", ["pressure"])

        assert filename == "plot_base_Wilson_P2p0"

    def test_build_with_combined_prefix(self, filename_labels):
        """Test filename with combined prefix."""
        builder = _PlotFilenameBuilder(filename_labels)
        metadata = {"temperature": 300}

        filename = builder.build(
            metadata, "plot_base", ["temperature"], include_combined_prefix=True
        )

        assert filename == "Combined_plot_base_T300"

    def test_build_with_custom_prefix(self, filename_labels):
        """Test filename with custom prefix."""
        builder = _PlotFilenameBuilder(filename_labels)
        metadata = {"temperature": 300}

        filename = builder.build(
            metadata, "plot_base", ["temperature"], custom_prefix="Custom_"
        )

        assert filename == "Custom_plot_base_T300"

    def test_build_with_grouping_variable_string(self, filename_labels):
        """Test filename with single grouping variable."""
        builder = _PlotFilenameBuilder(filename_labels)
        metadata = {"temperature": 300}

        filename = builder.build(
            metadata, "plot_base", ["temperature"], grouping_variable="pressure"
        )

        assert filename == "plot_base_T300_grouped_by_pressure"

    def test_build_with_grouping_variable_list(self, filename_labels):
        """Test filename with multiple grouping variables."""
        builder = _PlotFilenameBuilder(filename_labels)
        metadata = {"temperature": 300}

        filename = builder.build(
            metadata,
            "plot_base",
            ["temperature"],
            grouping_variable=["pressure", "volume"],
        )

        assert filename == "plot_base_T300_grouped_by_pressure_and_volume"

    def test_sanitize_values(self, filename_labels):
        """Test that special characters are sanitized in filenames."""
        builder = _PlotFilenameBuilder(filename_labels)
        metadata = {"temperature": 3.14159}

        filename = builder.build(metadata, "plot_base", ["temperature"])

        assert "3p14159" in filename  # Decimal point replaced with 'p'
        assert "." not in filename

    def test_parameter_not_in_metadata_skipped(self, filename_labels):
        """Test that parameters not in metadata are skipped."""
        builder = _PlotFilenameBuilder(filename_labels)
        metadata = {"temperature": 300}  # pressure not included

        filename = builder.build(
            metadata,
            "plot_base",
            ["temperature", "pressure"],  # pressure listed but not in metadata
        )

        assert filename == "plot_base_T300"  # Only temperature included


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
