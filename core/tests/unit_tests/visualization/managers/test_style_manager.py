"""
Unit tests for PlotStyleManager class.

This module provides comprehensive testing for the PlotStyleManager class,
covering marker/color generation, axes configuration, legend handling, and other
styling functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.ticker import MaxNLocator

from library.visualization.managers import PlotStyleManager


class MockConstants:
    """Mock constants module for testing."""

    MARKER_STYLES = ["o", "s", "^", "v", "D", "p", "*", "h"]
    DEFAULT_COLORS = ["blue", "green", "red", "purple", "orange", "brown"]
    PARAMETERS_WITH_EXPONENTIAL_FORMAT = ["pressure", "temperature"]
    PARAMETERS_OF_INTEGER_VALUE = ["iterations", "count"]
    AXES_LABELS_BY_COLUMN_NAME = {
        "temperature": "Temperature (K)",
        "pressure": "Pressure (Pa)",
        "iterations": "Number of Iterations",
    }
    LEGEND_LABELS_BY_COLUMN_NAME = {
        "temperature": "T",
        "pressure": "P",
        "group_var": "Group Variable",
    }


@pytest.fixture
def mock_constants():
    """Provide mock constants module."""
    return MockConstants()


@pytest.fixture
def style_manager(mock_constants):
    """Create a PlotStyleManager instance with mock constants."""
    return PlotStyleManager(mock_constants)


@pytest.fixture
def mock_axes():
    """Create a mock matplotlib axes object."""
    ax = MagicMock(spec=Axes)
    ax.legend = MagicMock(return_value=MagicMock(spec=Legend))
    ax.xaxis = MagicMock()
    ax.yaxis = MagicMock()
    ax.get_xlim = MagicMock(return_value=(0, 10))
    ax.get_ylim = MagicMock(return_value=(0, 100))
    return ax


@pytest.fixture
def mock_figure():
    """Create a mock matplotlib figure object."""
    return MagicMock(spec=Figure)


class TestPlotStyleManager:
    """Test cases for PlotStyleManager class."""

    def test_initialization(self, mock_constants):
        """Test PlotStyleManager initialization."""
        manager = PlotStyleManager(mock_constants)
        assert manager.constants == mock_constants
        assert manager._style_cache == {}

    def test_generate_marker_color_map_basic(self, style_manager):
        """Test basic marker/color map generation."""
        values = ["A", "B", "C"]
        style_map = style_manager.generate_marker_color_map(values)

        assert len(style_map) == 3
        assert "A" in style_map
        assert "B" in style_map
        assert "C" in style_map

        # Check that each value has a (marker, color) tuple
        for value in values:
            marker, color = style_map[value]
            assert marker in style_manager.constants.MARKER_STYLES
            assert color in style_manager.constants.DEFAULT_COLORS

    def test_generate_marker_color_map_with_custom_map(self, style_manager):
        """Test marker/color map generation with custom mapping."""
        values = ["A", "B", "C"]
        custom_map = {"B": ("x", "yellow")}

        style_map = style_manager.generate_marker_color_map(
            values, custom_map=custom_map
        )

        # Custom mapping should be preserved
        assert style_map["B"] == ("x", "yellow")

        # Other values should get auto-assigned
        assert style_map["A"][0] in style_manager.constants.MARKER_STYLES
        assert style_map["C"][0] in style_manager.constants.MARKER_STYLES

    def test_generate_marker_color_map_with_index_shift(self, style_manager):
        """Test marker/color map generation with index shift."""
        values = ["A", "B"]

        # Without shift
        map1 = style_manager.generate_marker_color_map(values)

        # With shift
        map2 = style_manager.generate_marker_color_map(values, index_shift=2)

        # Maps should be different
        assert map1["A"] != map2["A"]
        assert map1["B"] != map2["B"]

    def test_generate_marker_color_map_caching(self, style_manager):
        """Test that marker/color maps are cached."""
        values = ["A", "B", "C"]

        # First call
        map1 = style_manager.generate_marker_color_map(values)

        # Second call with same parameters
        map2 = style_manager.generate_marker_color_map(values)

        # Should return the same object (cached)
        assert map1 is map2

        # Different parameters should not use cache
        map3 = style_manager.generate_marker_color_map(["D", "E"])
        assert map3 is not map1

    def test_configure_axes_style_basic(self, style_manager, mock_axes):
        """Test basic axes configuration."""
        style_manager.configure_axes_style(
            mock_axes, xaxis_variable="temperature", yaxis_variable="pressure"
        )

        # Check grid was enabled
        mock_axes.grid.assert_called_once_with(True, linestyle="--", alpha=0.5)

        # Check labels were set with constants
        mock_axes.set_xlabel.assert_called_once_with("Temperature (K)", fontsize=15)
        mock_axes.set_ylabel.assert_called_once_with("Pressure (Pa)", fontsize=15)

        # Check tick params
        mock_axes.tick_params.assert_called_once_with(axis="both", labelsize=13)

        # Check log scale was set (both variables in PARAMETERS_WITH_EXPONENTIAL_FORMAT)
        mock_axes.set_xscale.assert_called_once_with("log")
        mock_axes.set_yscale.assert_called_once_with("log")

    def test_configure_axes_style_custom_labels(self, style_manager, mock_axes):
        """Test axes configuration with custom labels."""
        style_manager.configure_axes_style(
            mock_axes,
            xaxis_variable="var1",
            yaxis_variable="var2",
            xaxis_label="Custom X",
            yaxis_label="Custom Y",
        )

        mock_axes.set_xlabel.assert_called_once_with("Custom X", fontsize=15)
        mock_axes.set_ylabel.assert_called_once_with("Custom Y", fontsize=15)

    def test_configure_axes_style_integer_locators(self, style_manager, mock_axes):
        """Test that integer locators are set for integer parameters."""
        style_manager.configure_axes_style(
            mock_axes, xaxis_variable="iterations", yaxis_variable="count"
        )

        # Both axes should have MaxNLocator set
        assert mock_axes.xaxis.set_major_locator.called
        assert mock_axes.yaxis.set_major_locator.called

        # Check that MaxNLocator was used
        calls = mock_axes.xaxis.set_major_locator.call_args_list
        assert len(calls) == 1
        locator = calls[0][0][0]
        assert isinstance(locator, MaxNLocator)

    def test_configure_axes_style_inversion(self, style_manager, mock_axes):
        """Test axes inversion."""
        style_manager.configure_axes_style(
            mock_axes,
            xaxis_variable="var1",
            yaxis_variable="var2",
            invert_xaxis=True,
            invert_yaxis=True,
        )

        mock_axes.invert_xaxis.assert_called_once()
        mock_axes.invert_yaxis.assert_called_once()

    def test_configure_axes_style_grid_disabled(self, style_manager, mock_axes):
        """Test disabling grid."""
        style_manager.configure_axes_style(
            mock_axes, xaxis_variable="var1", yaxis_variable="var2", grid_enabled=False
        )

        mock_axes.grid.assert_not_called()

    def test_configure_legend_basic(self, style_manager, mock_axes):
        """Test basic legend configuration."""
        style_manager.configure_legend(mock_axes)

        mock_axes.legend.assert_called_once_with(loc="upper left", fontsize=13, ncol=1)

    def test_configure_legend_with_title(self, style_manager, mock_axes):
        """Test legend configuration with auto-generated title."""
        mock_legend = mock_axes.legend.return_value

        style_manager.configure_legend(mock_axes, grouping_variable="temperature")

        # Legend should be created
        mock_axes.legend.assert_called_once()

        # Title should be set from constants
        mock_legend.set_title.assert_called_once()
        call_args = mock_legend.set_title.call_args
        assert call_args[0][0] == "T"  # From LEGEND_LABELS_BY_COLUMN_NAME

    def test_configure_legend_custom_title(self, style_manager, mock_axes):
        """Test legend configuration with custom title."""
        mock_legend = mock_axes.legend.return_value

        style_manager.configure_legend(mock_axes, legend_title="My Custom Title")

        mock_legend.set_title.assert_called_once()
        call_args = mock_legend.set_title.call_args
        assert call_args[0][0] == "My Custom Title"

    def test_configure_legend_not_included(self, style_manager, mock_axes):
        """Test legend configuration when legend is disabled."""
        style_manager.configure_legend(mock_axes, include_legend=False)

        mock_axes.legend.assert_not_called()

    def test_configure_legend_title_cleanup(self, style_manager, mock_axes):
        """Test that non-LaTeX legend titles have underscores replaced."""
        mock_legend = mock_axes.legend.return_value

        # Use a variable not in LEGEND_LABELS_BY_COLUMN_NAME
        style_manager.configure_legend(
            mock_axes, grouping_variable="some_variable_name"
        )

        mock_legend.set_title.assert_called_once()
        call_args = mock_legend.set_title.call_args
        assert call_args[0][0] == "some variable name"  # Underscores replaced

    def test_set_axis_limits_explicit(self, style_manager, mock_axes):
        """Test setting explicit axis limits."""
        style_manager.set_axis_limits(mock_axes, xlim=(1, 10), ylim=(0, 50))

        mock_axes.set_xlim.assert_called_once_with((1, 10))
        mock_axes.set_ylim.assert_called_once_with((0, 50))

    def test_set_axis_limits_start_at_zero(self, style_manager, mock_axes):
        """Test forcing axes to start at zero."""
        style_manager.set_axis_limits(
            mock_axes, xaxis_start_at_zero=True, yaxis_start_at_zero=True
        )

        # Should query current limits and adjust
        mock_axes.get_xlim.assert_called_once()
        mock_axes.get_ylim.assert_called_once()

        mock_axes.set_xlim.assert_called_once_with(left=0, right=10)
        mock_axes.set_ylim.assert_called_once_with(bottom=0, top=100)

    def test_format_legend_value(self, style_manager):
        """Test legend value formatting."""
        # Test float formatting
        assert style_manager.format_legend_value(3.14159, ".2f") == "3.14"
        assert style_manager.format_legend_value(1000, ".2e") == "1.00e+03"

        # Test integer formatting
        assert style_manager.format_legend_value(42, ".2f") == "42.00"

        # Test string passthrough
        assert style_manager.format_legend_value("text", ".2f") == "text"

    def test_get_marker_properties_filled(self, style_manager):
        """Test getting properties for filled markers."""
        props = style_manager.get_marker_properties(
            marker="o", filled=True, color="red", size=10
        )

        expected = {
            "marker": "o",
            "color": "red",
            "markersize": 10,
            "markerfacecolor": "red",
            "markeredgecolor": "red",
        }
        assert props == expected

    def test_get_marker_properties_empty(self, style_manager):
        """Test getting properties for empty markers."""
        props = style_manager.get_marker_properties(
            marker="s", filled=False, color="blue", size=8
        )

        expected = {
            "marker": "s",
            "markersize": 8,
            "markerfacecolor": "none",
            "markeredgecolor": "blue",
            "color": "blue",
        }
        assert props == expected

    def test_apply_figure_margins(self, style_manager, mock_figure):
        """Test applying figure margins."""
        style_manager.apply_figure_margins(mock_figure)

        mock_figure.subplots_adjust.assert_called_once_with(
            left=0.15, right=0.94, bottom=0.12, top=0.92
        )

    def test_apply_figure_margins_custom(self, style_manager, mock_figure):
        """Test applying custom figure margins."""
        style_manager.apply_figure_margins(
            mock_figure, left=0.1, right=0.9, bottom=0.1, top=0.9
        )

        mock_figure.subplots_adjust.assert_called_once_with(
            left=0.1, right=0.9, bottom=0.1, top=0.9
        )

    def test_clear_cache(self, style_manager):
        """Test clearing the style cache."""
        # Generate some cached entries
        style_manager.generate_marker_color_map(["A", "B"])
        style_manager.generate_marker_color_map(["C", "D"])

        assert len(style_manager._style_cache) > 0

        # Clear cache
        style_manager.clear_cache()

        assert len(style_manager._style_cache) == 0

    def test_sorted_value_consistency(self, style_manager):
        """Test that sorted values produce consistent mappings."""
        # Test that different orderings produce the same mapping
        values1 = ["Z", "A", "M"]
        values2 = ["A", "M", "Z"]

        map1 = style_manager.generate_marker_color_map(values1)
        map2 = style_manager.generate_marker_color_map(values2)

        # Clear cache between calls to ensure fresh computation
        style_manager.clear_cache()
        map3 = style_manager.generate_marker_color_map(values2)

        # All should have the same mappings
        assert map1 == map3
        for key in ["A", "M", "Z"]:
            assert map1[key] == map3[key]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_grouping_values(self, style_manager):
        """Test marker/color map with empty values."""
        style_map = style_manager.generate_marker_color_map([])
        assert style_map == {}

    def test_none_labels(self, style_manager, mock_axes):
        """Test handling of None labels."""
        # When variable is not in AXES_LABELS_BY_COLUMN_NAME
        style_manager.configure_axes_style(
            mock_axes, xaxis_variable="unknown_var", yaxis_variable="another_unknown"
        )

        # Should use the variable name itself
        mock_axes.set_xlabel.assert_called_once_with("unknown_var", fontsize=15)
        mock_axes.set_ylabel.assert_called_once_with("another_unknown", fontsize=15)

    def test_list_grouping_variable(self, style_manager, mock_axes):
        """Test legend with list-type grouping variable."""
        mock_legend = mock_axes.legend.return_value

        style_manager.configure_legend(mock_axes, grouping_variable=["var1", "var2"])

        # Should use first variable from list
        mock_legend.set_title.assert_called_once()
        call_args = mock_legend.set_title.call_args
        assert "var1" in call_args[0][0]

    def test_latex_in_legend_title(self, style_manager, mock_axes):
        """Test that LaTeX titles are not modified."""
        mock_legend = mock_axes.legend.return_value

        style_manager.configure_legend(mock_axes, legend_title="$\\alpha_0$")

        mock_legend.set_title.assert_called_once()
        call_args = mock_legend.set_title.call_args
        assert call_args[0][0] == "$\\alpha_0$"  # Should not be modified


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
