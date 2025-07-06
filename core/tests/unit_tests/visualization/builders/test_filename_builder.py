import pytest

from library.visualization.builders.filename_builder import PlotFilenameBuilder


# Module-level fixtures
@pytest.fixture
def sample_filename_labels():
    """Create sample filename labels for testing."""
    return {
        "temperature": "T",
        "pressure": "P",
        "lattice_size": "L",
        "beta": "B",
        "mass": "M",
        "energy": "E",
        "time": "t",
        "iterations": "iter",
        "epsilon": "eps",
    }


@pytest.fixture
def filename_builder(sample_filename_labels):
    """Create a filename builder with sample labels."""
    return PlotFilenameBuilder(sample_filename_labels)


@pytest.fixture
def filename_builder_minimal():
    """Create a filename builder with minimal labels."""
    return PlotFilenameBuilder({"temp": "T"})


@pytest.fixture
def simple_metadata():
    """Create simple metadata for basic testing."""
    return {"temperature": 300.0, "pressure": 1.5, "lattice_size": 32}


@pytest.fixture
def complex_metadata():
    """Create complex metadata with special parameters."""
    return {
        "Overlap_operator_method": "Chebyshev",
        "Kernel_operator_type": "Wilson",
        "temperature": 300.5,
        "pressure": 1.25,
        "lattice_size": 32,
        "beta": 0.65,
    }


@pytest.fixture
def kl_metadata():
    """Create metadata with KL method."""
    return {
        "Overlap_operator_method": "KL",
        "Kernel_operator_type": "Brillouin",
        "temperature": 250.0,
        "mass": 0.5,
        "energy": 1.8,
    }


class TestInitialization:
    """Test initialization and configuration."""

    def test_init_with_labels(self, sample_filename_labels):
        """Test initialization with filename labels."""
        builder = PlotFilenameBuilder(sample_filename_labels)
        assert builder.filename_labels == sample_filename_labels
        assert builder._max_filename_length == 250
        assert "Chebyshev" in builder._special_overlap_methods
        assert "Wilson" in builder._special_kernel_types

    def test_init_empty_labels(self):
        """Test initialization with empty labels dictionary."""
        builder = PlotFilenameBuilder({})
        assert builder.filename_labels == {}
        assert builder._max_filename_length == 250


class TestBasicFilenameBuilding:
    """Test basic filename building functionality."""

    def test_build_simple_filename(self, filename_builder, simple_metadata):
        """Test building filename with simple metadata."""
        multivalued_params = ["temperature", "pressure"]

        filename = filename_builder.build(
            simple_metadata, "energy_Vs_time", multivalued_params
        )

        assert filename == "energy_Vs_time_T300p0_P1p5"

    def test_build_with_integer_values(self, filename_builder):
        """Test building filename with integer values."""
        metadata = {"lattice_size": 32, "iterations": 1000}
        multivalued_params = ["lattice_size", "iterations"]

        filename = filename_builder.build(
            metadata, "convergence_Vs_steps", multivalued_params
        )

        assert filename == "convergence_Vs_steps_L32_iter1000"

    def test_build_missing_labels(self, filename_builder_minimal):
        """Test building filename when labels are missing."""
        metadata = {"temp": 300.0, "unknown_param": 42}
        multivalued_params = ["temp", "unknown_param"]

        filename = filename_builder_minimal.build(
            metadata, "test_Vs_param", multivalued_params
        )

        # Should fall back to parameter names
        assert "T300p0" in filename
        assert "unknown_param42" in filename

    def test_build_subset_of_params(self, filename_builder, complex_metadata):
        """Test building filename with subset of available parameters."""
        multivalued_params = ["temperature"]  # Only one param

        filename = filename_builder.build(
            complex_metadata, "energy_Vs_time", multivalued_params
        )

        # Should contain only the specified parameter
        assert "T300p5" in filename
        assert "P" not in filename  # pressure not included
        assert "L" not in filename  # lattice_size not included


class TestSpecialParameterHandling:
    """Test handling of special overlap and kernel parameters."""

    def test_build_chebyshev_filename(self, filename_builder, complex_metadata):
        """Test building filename with Chebyshev method."""
        multivalued_params = ["temperature", "pressure"]

        filename = filename_builder.build(
            complex_metadata, "energy_Vs_time", multivalued_params
        )

        # Should start with Chebyshev, then base name, then Wilson
        assert filename == "Chebyshev_energy_Vs_time_Wilson_T300p5_P1p25"

    def test_build_kl_filename(self, filename_builder, kl_metadata):
        """Test building filename with KL method."""
        multivalued_params = ["temperature", "mass"]

        filename = filename_builder.build(
            kl_metadata, "mass_Vs_temp", multivalued_params
        )

        # Should start with KL, then base name, then Brillouin
        assert filename == "KL_mass_Vs_temp_Brillouin_T250p0_M0p5"

    def test_build_bare_method(self, filename_builder):
        """Test building filename with Bare method."""
        metadata = {
            "Overlap_operator_method": "Bare",
            "Kernel_operator_type": "Wilson",
            "temperature": 300.0,
        }
        multivalued_params = ["temperature"]

        filename = filename_builder.build(
            metadata, "energy_Vs_time", multivalued_params
        )

        assert filename == "Bare_energy_Vs_time_Wilson_T300p0"

    def test_build_overlap_only(self, filename_builder):
        """Test building filename with overlap method but no kernel."""
        metadata = {"Overlap_operator_method": "Chebyshev", "temperature": 300.0}
        multivalued_params = ["temperature"]

        filename = filename_builder.build(
            metadata, "energy_Vs_time", multivalued_params
        )

        assert filename == "Chebyshev_energy_Vs_time_T300p0"

    def test_build_kernel_only(self, filename_builder):
        """Test building filename with kernel type but no overlap method."""
        metadata = {"Kernel_operator_type": "Wilson", "temperature": 300.0}
        multivalued_params = ["temperature"]

        filename = filename_builder.build(
            metadata, "energy_Vs_time", multivalued_params
        )

        assert filename == "energy_Vs_time_Wilson_T300p0"

    def test_build_no_special_params(self, filename_builder, simple_metadata):
        """Test building filename without special parameters."""
        multivalued_params = ["temperature", "pressure"]

        filename = filename_builder.build(
            simple_metadata, "energy_Vs_time", multivalued_params
        )

        assert filename == "energy_Vs_time_T300p0_P1p5"


class TestPrefixHandling:
    """Test prefix functionality."""

    def test_build_with_combined_prefix(self, filename_builder, simple_metadata):
        """Test building filename with Combined prefix."""
        multivalued_params = ["temperature"]

        filename = filename_builder.build(
            simple_metadata,
            "energy_Vs_time",
            multivalued_params,
            include_combined_prefix=True,
        )

        assert filename == "Combined_energy_Vs_time_T300p0"

    def test_build_with_custom_prefix(self, filename_builder, simple_metadata):
        """Test building filename with custom prefix."""
        multivalued_params = ["temperature"]

        filename = filename_builder.build(
            simple_metadata, "energy_Vs_time", multivalued_params, custom_prefix="Test_"
        )

        assert filename == "Test_energy_Vs_time_T300p0"

    def test_custom_prefix_overrides_combined(self, filename_builder, simple_metadata):
        """Test that custom prefix overrides combined prefix."""
        multivalued_params = ["temperature"]

        filename = filename_builder.build(
            simple_metadata,
            "energy_Vs_time",
            multivalued_params,
            custom_prefix="Custom_",
            include_combined_prefix=True,
        )

        assert filename == "Custom_energy_Vs_time_T300p0"
        assert "Combined_" not in filename


class TestGroupingSuffix:
    """Test grouping suffix functionality."""

    def test_build_single_grouping_variable(self, filename_builder, simple_metadata):
        """Test building filename with single grouping variable."""
        multivalued_params = ["temperature"]

        filename = filename_builder.build(
            simple_metadata,
            "energy_Vs_time",
            multivalued_params,
            grouping_variable="pressure",
        )

        assert filename == "energy_Vs_time_T300p0_grouped_by_pressure"

    def test_build_multiple_grouping_variables(self, filename_builder, simple_metadata):
        """Test building filename with multiple grouping variables."""
        multivalued_params = ["temperature"]

        filename = filename_builder.build(
            simple_metadata,
            "energy_Vs_time",
            multivalued_params,
            grouping_variable=["pressure", "lattice_size"],
        )

        assert filename == "energy_Vs_time_T300p0_grouped_by_pressure_and_lattice_size"

    def test_build_too_many_grouping_variables(self, filename_builder, simple_metadata):
        """Test that too many grouping variables raises error."""
        multivalued_params = ["temperature"]

        with pytest.raises(ValueError, match="Maximum of 2 grouping variables"):
            filename_builder.build(
                simple_metadata,
                "energy_Vs_time",
                multivalued_params,
                grouping_variable=["pressure", "lattice_size", "beta"],
            )

    def test_build_invalid_grouping_variable_type(
        self, filename_builder, simple_metadata
    ):
        """Test that invalid grouping variable type raises error."""
        multivalued_params = ["temperature"]

        with pytest.raises(
            TypeError, match="grouping_variable must be str, list, or None"
        ):
            filename_builder.build(
                simple_metadata,
                "energy_Vs_time",
                multivalued_params,
                grouping_variable=123,  # Invalid type
            )

    def test_build_combined_with_grouping(self, filename_builder, complex_metadata):
        """Test building filename with both combined prefix and grouping."""
        multivalued_params = ["temperature"]

        filename = filename_builder.build(
            complex_metadata,
            "energy_Vs_time",
            multivalued_params,
            include_combined_prefix=True,
            grouping_variable="pressure",
        )

        assert (
            filename
            == "Combined_Chebyshev_energy_Vs_time_Wilson_T300p5_grouped_by_pressure"
        )


class TestValueSanitization:
    """Test value sanitization functionality."""

    def test_sanitize_decimal_values(self, filename_builder):
        """Test sanitization of decimal values."""
        assert filename_builder._sanitize_value(3.14) == "3p14"
        assert filename_builder._sanitize_value(0.5) == "0p5"
        assert filename_builder._sanitize_value(123.456) == "123p456"

    def test_sanitize_special_characters(self, filename_builder):
        """Test sanitization of special characters."""
        assert filename_builder._sanitize_value("test(1,2)") == "test12"
        assert (
            filename_builder._sanitize_value("value,with,commas") == "valuewithcommas"
        )
        assert filename_builder._sanitize_value("spaced value") == "spaced_value"
        assert filename_builder._sanitize_value("path/to/file") == "path_to_file"

    def test_sanitize_integer_values(self, filename_builder):
        """Test sanitization of integer values."""
        assert filename_builder._sanitize_value(42) == "42"
        assert filename_builder._sanitize_value(0) == "0"
        assert filename_builder._sanitize_value(-5) == "-5"

    def test_sanitize_string_values(self, filename_builder):
        """Test sanitization of string values."""
        assert filename_builder._sanitize_value("simple") == "simple"
        assert filename_builder._sanitize_value("test_value") == "test_value"
        assert filename_builder._sanitize_value("") == ""

    def test_sanitize_complex_values(self, filename_builder):
        """Test sanitization of complex combinations."""
        result = filename_builder._sanitize_value("test(3.14, 2.71)/result")
        assert result == "test3p14_2p71_result"


class TestFilenameValidation:
    """Test filename validation functionality."""

    def test_validate_normal_filename(self, filename_builder):
        """Test validation of normal filename."""
        filename = "energy_Vs_time_T300p0_P1p5"
        assert filename_builder.validate_filename(filename) == True

    def test_validate_long_filename(self, filename_builder):
        """Test validation of excessively long filename."""
        # Create a very long filename
        long_filename = "a" * 300
        assert filename_builder.validate_filename(long_filename) == False

    def test_validate_filename_with_invalid_chars(self, filename_builder):
        """Test validation of filename with invalid characters."""
        invalid_filenames = [
            "test<file",
            "test>file",
            "test:file",
            'test"file',
            "test|file",
            "test?file",
            "test*file",
        ]

        for filename in invalid_filenames:
            assert filename_builder.validate_filename(filename) == False

    def test_validate_filename_raises_on_long(self, filename_builder):
        """Test that _validate_filename raises on long filenames."""
        long_filename = "a" * 300

        with pytest.raises(ValueError, match="Filename too long"):
            filename_builder._validate_filename(long_filename)

    def test_validate_filename_raises_on_invalid_chars(self, filename_builder):
        """Test that _validate_filename raises on invalid characters."""
        with pytest.raises(ValueError, match="Invalid character"):
            filename_builder._validate_filename("test<file")


class TestUtilityMethods:
    """Test utility methods."""

    def test_set_max_length(self, filename_builder):
        """Test setting maximum filename length."""
        filename_builder.set_max_length(100)
        assert filename_builder._max_filename_length == 100

        # Test that validation uses new length
        medium_filename = "a" * 150
        assert filename_builder.validate_filename(medium_filename) == False

    def test_set_max_length_invalid(self, filename_builder):
        """Test setting invalid maximum length."""
        with pytest.raises(ValueError, match="Maximum length must be positive"):
            filename_builder.set_max_length(0)

        with pytest.raises(ValueError, match="Maximum length must be positive"):
            filename_builder.set_max_length(-10)

    def test_add_filename_labels(self, filename_builder):
        """Test adding new filename labels."""
        new_labels = {"new_param": "NP", "temperature": "TEMP"}  # Override existing

        filename_builder.add_filename_labels(new_labels)

        assert filename_builder.filename_labels["new_param"] == "NP"
        assert filename_builder.filename_labels["temperature"] == "TEMP"

        # Test that new labels are used
        metadata = {"temperature": 300.0, "new_param": 42}
        multivalued_params = ["temperature", "new_param"]

        filename = filename_builder.build(metadata, "test_Vs_param", multivalued_params)

        assert "TEMP300p0" in filename
        assert "NP42" in filename


class TestFilenamePreview:
    """Test filename preview functionality."""

    def test_get_filename_preview_success(self, filename_builder, complex_metadata):
        """Test getting filename preview for valid input."""
        multivalued_params = ["temperature", "pressure"]

        preview = filename_builder.get_filename_preview(
            complex_metadata,
            "energy_Vs_time",
            multivalued_params,
            grouping_variable="lattice_size",
        )

        assert preview["valid"] == True
        assert (
            preview["filename"]
            == "Chebyshev_energy_Vs_time_Wilson_T300p5_P1p25_grouped_by_lattice_size"
        )
        assert preview["length"] == len(preview["filename"])

        # Check components
        components = preview["components"]
        assert components["overlap"] == "Chebyshev"
        assert components["base_name"] == "energy_Vs_time"
        assert components["kernel"] == "Wilson"
        assert "T300p5" in components["parameters"]
        assert "P1p25" in components["parameters"]
        assert components["suffix"] == "_grouped_by_lattice_size"

    def test_get_filename_preview_with_prefix(self, filename_builder, simple_metadata):
        """Test getting filename preview with prefix."""
        multivalued_params = ["temperature"]

        preview = filename_builder.get_filename_preview(
            simple_metadata, "energy_Vs_time", multivalued_params, custom_prefix="Test_"
        )

        assert preview["valid"] == True
        assert preview["components"]["prefix"] == "Test_"

    def test_get_filename_preview_failure(self, filename_builder):
        """Test getting filename preview for invalid input."""
        # Create scenario that would fail validation
        filename_builder.set_max_length(10)  # Very short limit

        metadata = {"temperature": 300.0, "pressure": 1.5}
        multivalued_params = ["temperature", "pressure"]

        preview = filename_builder.get_filename_preview(
            metadata, "very_long_base_name_that_exceeds_limit", multivalued_params
        )

        assert preview["valid"] == False
        assert preview["filename"] is None
        assert "error" in preview
        assert "too long" in preview["error"].lower()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_build_empty_metadata(self, filename_builder):
        """Test building filename with empty metadata."""
        filename = filename_builder.build({}, "energy_Vs_time", [])
        assert filename == "energy_Vs_time"

    def test_build_empty_multivalued_params(self, filename_builder, simple_metadata):
        """Test building filename with empty multivalued params."""
        filename = filename_builder.build(simple_metadata, "energy_Vs_time", [])
        assert filename == "energy_Vs_time"

    def test_build_params_not_in_metadata(self, filename_builder, simple_metadata):
        """Test building filename with params not in metadata."""
        multivalued_params = ["nonexistent_param", "temperature"]

        filename = filename_builder.build(
            simple_metadata, "energy_Vs_time", multivalued_params
        )

        # Should only include existing parameters
        assert filename == "energy_Vs_time_T300p0"

    def test_build_special_params_not_recognized(self, filename_builder):
        """Test building filename with unrecognized special parameters."""
        metadata = {
            "Overlap_operator_method": "UnknownMethod",
            "Kernel_operator_type": "UnknownKernel",
            "temperature": 300.0,
        }
        multivalued_params = ["temperature"]

        filename = filename_builder.build(
            metadata, "energy_Vs_time", multivalued_params
        )

        # Should not include unrecognized special parameters
        assert filename == "energy_Vs_time_T300p0"

    def test_build_none_values_in_metadata(self, filename_builder):
        """Test building filename with None values in metadata."""
        metadata = {"temperature": None, "pressure": 1.5}
        multivalued_params = ["temperature", "pressure"]

        filename = filename_builder.build(
            metadata, "energy_Vs_time", multivalued_params
        )

        # Should handle None gracefully
        assert "TNone" in filename
        assert "P1p5" in filename


# Parametrized tests for comprehensive coverage
class TestParametrized:
    """Parametrized tests for comprehensive scenarios."""

    @pytest.mark.parametrize(
        "method,kernel,expected_start",
        [
            ("Chebyshev", "Wilson", "Chebyshev_base_Wilson"),
            ("KL", "Brillouin", "KL_base_Brillouin"),
            ("Bare", "Wilson", "Bare_base_Wilson"),
            ("Bare", "Brillouin", "Bare_base_Brillouin"),
        ],
    )
    def test_overlap_kernel_combinations(
        self, filename_builder, method, kernel, expected_start
    ):
        """Test various overlap/kernel combinations."""
        metadata = {
            "Overlap_operator_method": method,
            "Kernel_operator_type": kernel,
            "temperature": 300.0,
        }
        multivalued_params = ["temperature"]

        filename = filename_builder.build(metadata, "base", multivalued_params)

        assert filename.startswith(expected_start)
        assert "T300p0" in filename

    @pytest.mark.parametrize(
        "value,expected",
        [
            (3.14, "3p14"),
            (0.5, "0p5"),
            (42, "42"),
            ("test string", "test_string"),
            ("path/to/file", "path_to_file"),
            ("value(1,2)", "value12"),
            ("", ""),
        ],
    )
    def test_value_sanitization_variations(self, filename_builder, value, expected):
        """Test various value sanitization scenarios."""
        result = filename_builder._sanitize_value(value)
        assert result == expected

    @pytest.mark.parametrize(
        "grouping_var,expected_suffix",
        [
            ("pressure", "_grouped_by_pressure"),
            (["pressure"], "_grouped_by_pressure"),
            (["pressure", "temp"], "_grouped_by_pressure_and_temp"),
            (None, ""),
            ("", ""),
        ],
    )
    def test_grouping_suffix_variations(
        self, filename_builder, grouping_var, expected_suffix
    ):
        """Test various grouping suffix scenarios."""
        if grouping_var == "":
            # Empty string case
            result = filename_builder._build_grouping_suffix(
                None
            )  # Treat empty as None
        else:
            result = filename_builder._build_grouping_suffix(grouping_var)
        assert result == expected_suffix


# Running specific test groups
if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
