import pytest

from library.visualization.builders import PlotTitleBuilder


# Module-level fixtures
@pytest.fixture
def sample_title_labels():
    """Create sample title labels for testing."""
    return {
        "temperature": "Temperature (K)",
        "pressure": "Pressure (Pa)",
        "energy": "Energy (eV)",
        "lattice_size": "Lattice Size",
        "beta": "β",
        "mass": "Mass (GeV)",
        "Overlap_operator_method": "Overlap Method",
        "Kernel_operator_type": "Kernel Type",
    }


@pytest.fixture
def title_builder(sample_title_labels):
    """Create a title builder with sample labels."""
    return PlotTitleBuilder(sample_title_labels)


@pytest.fixture
def title_builder_default():
    """Create a title builder with minimal labels."""
    return PlotTitleBuilder({})


@pytest.fixture
def simple_metadata():
    """Create simple metadata for basic testing."""
    return {"temperature": 300.0, "pressure": 1.5, "energy": 2.8}


@pytest.fixture
def complex_metadata():
    """Create complex metadata with special parameters."""
    return {
        "Overlap_operator_method": "Chebyshev",
        "Kernel_operator_type": "Wilson",
        "Number_of_Chebyshev_terms": 50,
        "temperature": 300.0,
        "pressure": 1.5,
        "lattice_size": 32,
        "beta": 0.65,
    }


@pytest.fixture
def kl_metadata():
    """Create metadata with KL method."""
    return {
        "Overlap_operator_method": "KL",
        "Kernel_operator_type": "Brillouin",
        "KL_diagonal_order": 3,
        "temperature": 250.0,
        "mass": 0.5,
    }


class TestInitialization:
    """Test initialization and configuration."""

    def test_init_with_labels(self, sample_title_labels):
        """Test initialization with title labels."""
        builder = PlotTitleBuilder(sample_title_labels)
        assert builder.title_labels == sample_title_labels
        assert builder.title_number_format == ".2f"

    def test_init_custom_format(self, sample_title_labels):
        """Test initialization with custom number format."""
        builder = PlotTitleBuilder(sample_title_labels, title_number_format=".3g")
        assert builder.title_number_format == ".3g"

    def test_init_empty_labels(self):
        """Test initialization with empty labels dictionary."""
        builder = PlotTitleBuilder({})
        assert builder.title_labels == {}
        assert builder.title_number_format == ".2f"


class TestSimpleTitleBuilding:
    """Test simple title building functionality."""

    def test_build_simple_metadata(self, title_builder, simple_metadata):
        """Test building title with simple metadata."""
        tunable_params = ["temperature", "pressure", "energy"]

        title = title_builder.build(simple_metadata, tunable_params)

        assert "Temperature (K)=300.00" in title
        assert "Pressure (Pa)=1.50" in title
        assert "Energy (eV)=2.80" in title

    def test_build_with_excluded_params(self, title_builder, simple_metadata):
        """Test building title with excluded parameters."""
        tunable_params = ["temperature", "pressure", "energy"]
        excluded = {"pressure"}

        title = title_builder.build(simple_metadata, tunable_params, excluded=excluded)

        assert "Temperature (K)=300.00" in title
        assert "Pressure (Pa)" not in title
        assert "Energy (eV)=2.80" in title

    def test_build_with_leading_substring(self, title_builder, simple_metadata):
        """Test building title with leading substring."""
        tunable_params = ["temperature"]
        leading = "Test Run:"

        title = title_builder.build(
            simple_metadata, tunable_params, leading_substring=leading
        )

        assert title.startswith("Test Run:")
        assert "Temperature (K)=300.00" in title

    def test_build_missing_labels(self, title_builder_default, simple_metadata):
        """Test building title when labels are missing."""
        tunable_params = ["temperature", "pressure"]

        title = title_builder_default.build(simple_metadata, tunable_params)

        # Should fall back to parameter names
        assert "temperature=300.00" in title
        assert "pressure=1.50" in title


class TestComplexTitleBuilding:
    """Test complex title building with special parameters."""

    def test_build_chebyshev_title(self, title_builder, complex_metadata):
        """Test building title with Chebyshev method."""
        tunable_params = ["temperature", "pressure", "lattice_size"]

        title = title_builder.build(complex_metadata, tunable_params)

        # Should start with Chebyshev Wilson 50
        assert title.startswith("Chebyshev Wilson 50,")
        assert "Temperature (K)=300.00" in title
        assert "Pressure (Pa)=1.50" in title
        assert "Lattice Size=32" in title

    def test_build_kl_title(self, title_builder, kl_metadata):
        """Test building title with KL method."""
        tunable_params = ["temperature", "mass"]

        title = title_builder.build(kl_metadata, tunable_params)

        # Should start with KL Brillouin 3
        assert title.startswith("KL Brillouin 3,")
        assert "Temperature (K)=250.00" in title
        assert "Mass (GeV)=0.50" in title

    def test_build_bare_method(self, title_builder):
        """Test building title with Bare method."""
        metadata = {
            "Overlap_operator_method": "Bare",
            "Kernel_operator_type": "Wilson",
            "temperature": 300.0,
        }
        tunable_params = ["temperature"]

        title = title_builder.build(metadata, tunable_params)

        # Should start with Bare Wilson (no additional number)
        assert title.startswith("Bare Wilson,")
        assert "Temperature (K)=300.00" in title

    def test_build_exclude_special_params(self, title_builder, complex_metadata):
        """Test excluding special overlap/kernel parameters."""
        tunable_params = ["temperature"]
        excluded = {"Overlap_operator_method", "Kernel_operator_type"}

        title = title_builder.build(complex_metadata, tunable_params, excluded=excluded)

        # Should not contain Chebyshev Wilson
        assert not title.startswith("Chebyshev Wilson")
        assert "Temperature (K)=300.00" in title


class TestTitleFromColumns:
    """Test simple title building from specific columns."""

    def test_title_from_columns_basic(self, title_builder, simple_metadata):
        """Test building title from specific columns only."""
        columns = ["temperature", "pressure"]

        title = title_builder.build(simple_metadata, [], title_from_columns=columns)

        assert title == "Temperature (K)=300.00, Pressure (Pa)=1.50"

    def test_title_from_columns_kernel(self, title_builder):
        """Test title from columns with kernel type."""
        metadata = {"Kernel_operator_type": "Wilson", "temperature": 300.0}
        columns = ["Kernel_operator_type", "temperature"]

        title = title_builder.build(metadata, [], title_from_columns=columns)

        assert "Wilson Kernel" in title
        assert "Temperature (K)=300.00" in title

    def test_title_from_columns_missing_values(self, title_builder):
        """Test title from columns when some values are missing."""
        metadata = {"temperature": 300.0}  # pressure missing
        columns = ["temperature", "pressure", "energy"]

        title = title_builder.build(metadata, [], title_from_columns=columns)

        assert title == "Temperature (K)=300.00"  # Only existing values


class TestNumberFormatting:
    """Test number formatting functionality."""

    def test_default_number_format(self, title_builder):
        """Test default number formatting (.2f)."""
        metadata = {"temperature": 300.123456}
        tunable_params = ["temperature"]

        title = title_builder.build(metadata, tunable_params)

        assert "Temperature (K)=300.12" in title

    def test_scientific_format(self, sample_title_labels):
        """Test scientific number formatting."""
        builder = PlotTitleBuilder(sample_title_labels, title_number_format=".2e")
        metadata = {"temperature": 300.123}
        tunable_params = ["temperature"]

        title = builder.build(metadata, tunable_params)

        assert "Temperature (K)=3.00e+02" in title

    def test_general_format(self, sample_title_labels):
        """Test general number formatting (.3g)."""
        builder = PlotTitleBuilder(sample_title_labels, title_number_format=".3g")
        metadata = {"temperature": 300.0, "pressure": 0.00123}
        tunable_params = ["temperature", "pressure"]

        title = builder.build(metadata, tunable_params)

        assert "Temperature (K)=300" in title
        assert "Pressure (Pa)=0.00123" in title

    def test_integer_values(self, title_builder):
        """Test formatting of integer values."""
        metadata = {"lattice_size": 32, "temperature": 300}
        tunable_params = ["lattice_size", "temperature"]

        title = title_builder.build(metadata, tunable_params)

        assert "Lattice Size=32.00" in title
        assert "Temperature (K)=300.00" in title

    def test_non_numeric_values(self, title_builder):
        """Test handling of non-numeric values."""
        metadata = {"method": "CG", "temperature": 300.0}
        tunable_params = ["method", "temperature"]

        # Add method to labels
        title_builder.title_labels["method"] = "Method"

        title = title_builder.build(metadata, tunable_params)

        assert "Method=CG" in title
        assert "Temperature (K)=300.00" in title


class TestTitleWrapping:
    """Test title wrapping functionality."""

    def test_no_wrapping_short_title(self, title_builder):
        """Test that short titles are not wrapped."""
        metadata = {"temperature": 300.0}
        tunable_params = ["temperature"]

        title = title_builder.build(metadata, tunable_params, wrapping_length=80)

        assert "\n" not in title

    def test_wrapping_long_title(self, title_builder, complex_metadata):
        """Test that long titles are wrapped."""
        tunable_params = ["temperature", "pressure", "lattice_size", "beta"]

        title = title_builder.build(
            complex_metadata, tunable_params, wrapping_length=30
        )

        assert "\n" in title
        # Should wrap at a comma
        parts = title.split("\n")
        assert len(parts) == 2
        assert parts[0].endswith(",")

    def test_wrapping_disabled(self, title_builder, complex_metadata):
        """Test that wrapping can be disabled."""
        tunable_params = ["temperature", "pressure", "lattice_size", "beta"]

        title = title_builder.build(complex_metadata, tunable_params, wrapping_length=0)

        assert "\n" not in title

    def test_wrapping_no_commas(self, title_builder):
        """Test wrapping behavior when no commas are present."""
        metadata = {"temperature": 300.0}
        tunable_params = ["temperature"]

        # Force wrapping with very short length
        title = title_builder.build(metadata, tunable_params, wrapping_length=5)

        # Should not wrap if no commas available
        assert "\n" not in title


class TestUtilityMethods:
    """Test utility methods for configuration updates."""

    def test_set_number_format(self, title_builder):
        """Test updating number format."""
        title_builder.set_number_format(".1f")
        assert title_builder.title_number_format == ".1f"

        metadata = {"temperature": 300.789}
        tunable_params = ["temperature"]

        title = title_builder.build(metadata, tunable_params)
        assert "Temperature (K)=300.8" in title

    def test_update_labels(self, title_builder):
        """Test updating label mappings."""
        new_labels = {
            "new_param": "New Parameter",
            "temperature": "Temp (K)",  # Override existing
        }

        title_builder.update_labels(new_labels)

        assert title_builder.title_labels["new_param"] == "New Parameter"
        assert title_builder.title_labels["temperature"] == "Temp (K)"

        # Test that new labels are used
        metadata = {"temperature": 300.0, "new_param": 42}
        tunable_params = ["temperature", "new_param"]

        title = title_builder.build(metadata, tunable_params)
        assert "Temp (K)=300.00" in title
        assert "New Parameter=42.00" in title


class TestPrivateMethods:
    """Test private method functionality."""

    def test_format_value_numeric(self, title_builder):
        """Test _format_value with numeric inputs."""
        assert title_builder._format_value(3.14159) == "3.14"
        assert title_builder._format_value(42) == "42.00"
        assert title_builder._format_value(1.0) == "1.00"

    def test_format_value_non_numeric(self, title_builder):
        """Test _format_value with non-numeric inputs."""
        assert title_builder._format_value("string") == "string"
        assert title_builder._format_value(None) == "None"
        assert title_builder._format_value([1, 2, 3]) == "[1, 2, 3]"

    def test_overlap_kernel_parts_full(self, title_builder):
        """Test _overlap_kernel_parts with all components."""
        metadata = {
            "Overlap_operator_method": "Chebyshev",
            "Kernel_operator_type": "Wilson",
            "Number_of_Chebyshev_terms": 50,
        }

        parts = title_builder._overlap_kernel_parts(metadata, set())

        assert len(parts) == 1
        assert parts[0] == "Chebyshev Wilson 50,"

    def test_overlap_kernel_parts_excluded(self, title_builder):
        """Test _overlap_kernel_parts with excluded parameters."""
        metadata = {
            "Overlap_operator_method": "KL",
            "Kernel_operator_type": "Brillouin",
            "KL_diagonal_order": 3,
        }
        excluded = {"Kernel_operator_type"}

        parts = title_builder._overlap_kernel_parts(metadata, excluded)

        assert len(parts) == 1
        assert parts[0] == "KL 3,"  # No Brillouin

    def test_parameter_parts_basic(self, title_builder):
        """Test _parameter_parts with regular parameters."""
        metadata = {"temperature": 300.0, "pressure": 1.5}
        tunable_params = ["temperature", "pressure"]

        parts = title_builder._parameter_parts(metadata, tunable_params, set())

        assert len(parts) == 2
        assert "Temperature (K)=300.00," in parts
        assert "Pressure (Pa)=1.50," in parts

    def test_parameter_parts_excluded(self, title_builder):
        """Test _parameter_parts with excluded parameters."""
        metadata = {"temperature": 300.0, "pressure": 1.5}
        tunable_params = ["temperature", "pressure"]
        excluded = {"pressure"}

        parts = title_builder._parameter_parts(metadata, tunable_params, excluded)

        assert len(parts) == 1
        assert "Temperature (K)=300.00," in parts
        assert any("Pressure" in part for part in parts) == False


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_metadata(self, title_builder):
        """Test building title with empty metadata."""
        title = title_builder.build({}, ["temperature", "pressure"])

        assert title == ""  # Should be empty

    def test_empty_tunable_params(self, title_builder, simple_metadata):
        """Test building title with empty tunable params list."""
        title = title_builder.build(simple_metadata, [])

        assert title == ""  # Should be empty

    def test_all_params_excluded(self, title_builder, simple_metadata):
        """Test building title when all params are excluded."""
        tunable_params = ["temperature", "pressure"]
        excluded = {"temperature", "pressure"}

        title = title_builder.build(simple_metadata, tunable_params, excluded=excluded)

        assert title == ""

    def test_special_characters_in_values(self, title_builder):
        """Test handling values with special characters."""
        metadata = {
            "method": "CG-method",
            "config": "test_config.xml",
            "temperature": 300.0,
        }
        title_builder.title_labels.update(
            {"method": "Method", "config": "Configuration"}
        )
        tunable_params = ["method", "config", "temperature"]

        title = title_builder.build(metadata, tunable_params)

        assert "Method=CG-method" in title
        assert "Configuration=test_config.xml" in title


# Parametrized tests for comprehensive coverage
class TestParametrized:
    """Parametrized tests for comprehensive scenarios."""

    @pytest.mark.parametrize(
        "method,kernel,expected_start",
        [
            ("Chebyshev", "Wilson", "Chebyshev Wilson"),
            ("KL", "Brillouin", "KL Brillouin"),
            ("Bare", "Wilson", "Bare Wilson"),
            ("Bare", "Brillouin", "Bare Brillouin"),
        ],
    )
    def test_overlap_kernel_combinations(
        self, title_builder, method, kernel, expected_start
    ):
        """Test various overlap/kernel combinations."""
        metadata = {
            "Overlap_operator_method": method,
            "Kernel_operator_type": kernel,
            "temperature": 300.0,
        }
        tunable_params = ["temperature"]

        title = title_builder.build(metadata, tunable_params)

        assert title.startswith(expected_start)

    @pytest.mark.parametrize(
        "format_str,value,expected",
        [
            (".1f", 3.14159, "3.1"),
            (".3f", 3.14159, "3.142"),
            (".2e", 300.0, "3.00e+02"),
            (".3g", 0.00123, "0.00123"),
            (".0f", 42.7, "43"),
        ],
    )
    def test_number_format_variations(
        self, sample_title_labels, format_str, value, expected
    ):
        """Test various number formatting options."""
        builder = PlotTitleBuilder(sample_title_labels, title_number_format=format_str)
        metadata = {"temperature": value}
        tunable_params = ["temperature"]

        title = builder.build(metadata, tunable_params)

        assert f"Temperature (K)={expected}" in title

    @pytest.mark.parametrize("length", [10, 20, 30, 50, 100])
    def test_wrapping_length_variations(self, title_builder, complex_metadata, length):
        """Test different wrapping lengths."""
        tunable_params = ["temperature", "pressure", "lattice_size", "beta"]

        title = title_builder.build(
            complex_metadata, tunable_params, wrapping_length=length
        )

        if length < 50:  # Short lengths should trigger wrapping
            assert "\n" in title
        else:  # Longer lengths might not need wrapping
            # This depends on the actual title length
            pass


# Running specific test groups
if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
