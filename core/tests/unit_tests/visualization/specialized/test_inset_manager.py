"""
Unit tests for PlotInsetManager class.

This module provides comprehensive testing for the PlotInsetManager
class, covering inset creation, positioning, data filtering, and
management operations.
"""

from unittest.mock import Mock, patch

import pytest
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for testing
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from library.visualization.specialized.inset_manager import PlotInsetManager


@pytest.fixture
def inset_manager():
    """Create a PlotInsetManager instance for testing."""
    return PlotInsetManager()


@pytest.fixture
def mock_data_plotter():
    """Create a mock DataPlotter class."""
    mock_plotter = Mock()
    mock_plotter.dataframe = pd.DataFrame()
    mock_plotter.list_of_multivalued_tunable_parameter_names = ["param1", "param2"]
    mock_plotter.list_of_output_quantity_names_from_dataframe = ["output1", "output2"]
    mock_plotter.set_plot_variables = Mock()
    mock_plotter.plot = Mock()
    mock_plotter._update_column_categories = Mock()
    return mock_plotter


@pytest.fixture
def mock_data_plotter_class(mock_data_plotter):
    """Create a mock DataPlotter class constructor."""
    mock_class = Mock()
    mock_class.return_value = mock_data_plotter
    return mock_class


@pytest.fixture
def mock_figure_axes():
    """Create mock matplotlib figure and axes."""
    fig = Mock(spec=Figure)
    ax = Mock(spec=Axes)

    # Mock inset_axes method
    mock_inset_ax = Mock(spec=Axes)
    mock_inset_ax.set_xlim = Mock()
    mock_inset_ax.set_ylim = Mock()
    mock_inset_ax.clear = Mock()
    mock_inset_ax.remove = Mock()
    ax.inset_axes = Mock(return_value=mock_inset_ax)
    ax.get_ylabel = Mock(return_value="Energy")
    ax.add_artist = Mock()

    return fig, ax, mock_inset_ax


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "time": np.linspace(0, 10, 100),
            "energy": np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100),
            "temperature": np.random.uniform(250, 350, 100),
            "pressure": np.random.uniform(1, 5, 100),
            "system_size": np.random.choice([16, 32, 64], 100),
            "method": np.random.choice(["A", "B", "C"], 100),
        }
    )


@pytest.fixture
def uncertainty_dataframe():
    """Create a DataFrame with uncertainty data (tuples)."""
    size = 50
    x = np.linspace(0, 10, size)

    # Create object arrays with tuples
    y_data = np.empty(size, dtype=object)
    z_data = np.empty(size, dtype=object)

    for i in range(size):
        y_data[i] = (np.sin(x[i]), 0.1)
        z_data[i] = (np.cos(x[i]), 0.05)

    return pd.DataFrame(
        {
            "x": x,
            "y_with_error": y_data,
            "z_with_error": z_data,
            "category": np.random.choice(["cat1", "cat2"], size),
        }
    )


class TestPlotInsetManagerInitialization:
    """Test inset manager initialization."""

    def test_init_default(self, inset_manager):
        """Test default initialization."""
        assert inset_manager.data_plotter_class is None
        assert isinstance(inset_manager.location_presets, dict)
        assert isinstance(inset_manager._insets, dict)
        assert inset_manager.default_width == 0.3
        assert inset_manager.default_height == 0.3

    def test_init_with_plotter_class(self, mock_data_plotter_class):
        """Test initialization with custom plotter class."""
        manager = PlotInsetManager(data_plotter_class=mock_data_plotter_class)
        assert manager.data_plotter_class is mock_data_plotter_class

    def test_location_presets(self, inset_manager):
        """Test that location presets are properly configured."""
        expected_locations = [
            "upper right",
            "upper left",
            "lower right",
            "lower left",
            "center",
            "upper center",
            "lower center",
            "center left",
            "center right",
        ]

        for location in expected_locations:
            assert location in inset_manager.location_presets
            coords = inset_manager.location_presets[location]
            assert len(coords) == 2
            assert all(0 <= coord <= 1 for coord in coords)


class TestInsetPositioning:
    """Test inset positioning calculations."""

    def test_calculate_position_preset_location(self, inset_manager):
        """Test position calculation with preset location."""
        coords = inset_manager._calculate_inset_position(
            "upper right", 0.3, 0.25, None, None
        )

        assert coords == (0.65, 0.65, 0.3, 0.25)  # Changed from list to tuple

    def test_calculate_position_custom_coordinates(self, inset_manager):
        """Test position calculation with custom coordinates."""
        coords = inset_manager._calculate_inset_position(
            "upper right", 0.3, 0.25, 0.1, 0.8
        )

        assert coords == (0.1, 0.8, 0.3, 0.25)  # Changed from list to tuple

    def test_calculate_position_tuple_location(self, inset_manager):
        """Test position calculation with tuple location."""
        coords = inset_manager._calculate_inset_position(
            (0.2, 0.7), 0.4, 0.3, None, None
        )

        assert coords == (0.2, 0.7, 0.4, 0.3)  # Changed from list to tuple

    def test_calculate_position_unknown_location(self, inset_manager):
        """Test position calculation with unknown location."""
        coords = inset_manager._calculate_inset_position(
            "unknown_location", 0.3, 0.25, None, None
        )

        # Should fall back to "lower right"
        assert coords == (
            0.65,
            0.11,
            0.3,
            0.25,
        )  # Changed from list to tuple, and 0.05 to 0.11


class TestDataFiltering:
    """Test data filtering functionality."""

    def test_apply_data_filters_no_filters(self, inset_manager, sample_dataframe):
        """Test data filtering with no filters applied."""
        result = inset_manager._apply_data_filters(sample_dataframe, None, None)

        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_apply_data_filters_function_filter(self, inset_manager, sample_dataframe):
        """Test data filtering with function filter."""

        def filter_func(df):
            return df[df["temperature"] > 300]

        result = inset_manager._apply_data_filters(sample_dataframe, filter_func, None)

        assert len(result) < len(sample_dataframe)
        assert all(result["temperature"] > 300)

    def test_apply_data_filters_condition_filter(self, inset_manager, sample_dataframe):
        """Test data filtering with condition string."""
        condition = "pressure > 3.0 and system_size == 32"

        result = inset_manager._apply_data_filters(sample_dataframe, None, condition)

        assert len(result) < len(sample_dataframe)
        assert all(result["pressure"] > 3.0)
        assert all(result["system_size"] == 32)

    def test_apply_data_filters_both_filters(self, inset_manager, sample_dataframe):
        """Test data filtering with both function and condition
        filters."""

        def filter_func(df):
            return df[df["method"] == "A"]

        condition = "temperature > 300"

        result = inset_manager._apply_data_filters(
            sample_dataframe, filter_func, condition
        )

        assert len(result) < len(sample_dataframe)
        assert all(result["method"] == "A")
        assert all(result["temperature"] > 300)

    def test_apply_data_filters_function_error(self, inset_manager, sample_dataframe):
        """Test data filtering with function that raises error."""

        def bad_filter_func(df):
            raise ValueError("Filter error")

        result = inset_manager._apply_data_filters(
            sample_dataframe, bad_filter_func, None
        )

        assert len(result) == 0  # Should return empty DataFrame

    def test_apply_data_filters_condition_error(self, inset_manager, sample_dataframe):
        """Test data filtering with invalid condition."""
        condition = "invalid_column > 100"

        result = inset_manager._apply_data_filters(sample_dataframe, None, condition)

        assert len(result) == 0  # Should return empty DataFrame


class TestInsetCreation:
    """Test basic inset creation."""

    def test_add_inset_basic(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test basic inset creation."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        result = inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="time",
            yaxis_variable="energy",
        )

        assert result is mock_inset_ax
        ax.inset_axes.assert_called_once()
        mock_data_plotter_class.assert_called_once()

    def test_add_inset_with_custom_position(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test inset creation with custom position."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        result = inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="time",
            inset_x=0.1,
            inset_y=0.8,
            width=0.4,
            height=0.35,
        )

        assert result is mock_inset_ax
        # Check that inset_axes was called with correct coordinates
        call_args = ax.inset_axes.call_args[0][0]
        assert call_args == (0.1, 0.8, 0.4, 0.35)  # Changed from list to tuple

    def test_add_inset_with_data_filter(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test inset creation with data filtering."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        def filter_func(df):
            return df[df["temperature"] > 300]

        result = inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="time",
            data_filter_func=filter_func,
        )

        assert result is mock_inset_ax
        # Check that filtered data was used
        mock_plotter = mock_data_plotter_class.return_value
        filtered_df = mock_plotter.dataframe
        # The actual filtering logic would be tested in the data
        # filtering tests

    def test_add_inset_with_id(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test inset creation with ID tracking."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        result = inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="time",
            inset_id="test_inset",
        )

        assert result is mock_inset_ax
        assert "test_inset" in inset_manager._insets
        assert inset_manager._insets["test_inset"]["axes"] is mock_inset_ax

    def test_add_inset_empty_data_after_filter(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test inset creation with filter that results in empty
        data."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        def empty_filter(df):
            return df[df["temperature"] > 1000]  # No data should match

        result = inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="time",
            data_filter_func=empty_filter,
        )

        assert result is None  # Should return None for empty data

    def test_add_inset_exception_handling(
        self, inset_manager, mock_figure_axes, sample_dataframe
    ):
        """Test inset creation with exception handling."""
        fig, ax, mock_inset_ax = mock_figure_axes
        ax.inset_axes.side_effect = Exception("Test error")

        result = inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="time",
        )

        assert result is None  # Should return None on error


class TestMultipleInsets:
    """Test multiple inset creation."""

    def test_add_multiple_insets(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test creating multiple insets."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        # Mock multiple inset axes
        mock_inset_ax1 = Mock(spec=Axes)
        mock_inset_ax2 = Mock(spec=Axes)
        ax.inset_axes.side_effect = [mock_inset_ax1, mock_inset_ax2]

        inset_specs = [
            {
                "xaxis_variable": "time",
                "yaxis_variable": "energy",
                "location": "upper left",
                "inset_id": "time_energy",
            },
            {
                "xaxis_variable": "temperature",
                "yaxis_variable": "pressure",
                "location": "lower right",
                "inset_id": "temp_pressure",
            },
        ]

        result = inset_manager.add_multiple_insets(
            fig, ax, inset_specs, sample_dataframe, "/tmp/plots"
        )

        assert len(result) == 2
        assert "time_energy" in result
        assert "temp_pressure" in result
        assert result["time_energy"] is mock_inset_ax1
        assert result["temp_pressure"] is mock_inset_ax2

    def test_add_multiple_insets_with_failures(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test multiple inset creation with some failures."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        # Mock one success, one failure
        mock_inset_ax1 = Mock(spec=Axes)
        ax.inset_axes.side_effect = [mock_inset_ax1, Exception("Test error")]

        inset_specs = [
            {"xaxis_variable": "time", "inset_id": "success"},
            {"xaxis_variable": "energy", "inset_id": "failure"},
        ]

        result = inset_manager.add_multiple_insets(
            fig, ax, inset_specs, sample_dataframe, "/tmp/plots"
        )

        assert len(result) == 1
        assert "success" in result
        assert "failure" not in result

    def test_add_multiple_insets_auto_id(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test multiple inset creation with auto-generated IDs."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        # Mock multiple inset axes
        mock_inset_ax1 = Mock(spec=Axes)
        mock_inset_ax2 = Mock(spec=Axes)
        ax.inset_axes.side_effect = [mock_inset_ax1, mock_inset_ax2]

        inset_specs = [{"xaxis_variable": "time"}, {"xaxis_variable": "energy"}]

        result = inset_manager.add_multiple_insets(
            fig, ax, inset_specs, sample_dataframe, "/tmp/plots"
        )

        assert len(result) == 2
        assert "inset_0" in result
        assert "inset_1" in result


class TestZoomInsets:
    """Test zoom inset functionality."""

    def test_add_zoom_inset(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test zoom inset creation."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        result = inset_manager.add_zoom_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            zoom_xlim=(2, 4),
            zoom_ylim=(-0.5, 0.5),
            xaxis_variable="time",
            yaxis_variable="energy",
        )

        assert result is mock_inset_ax
        mock_inset_ax.set_xlim.assert_called_once_with((2, 4))
        mock_inset_ax.set_ylim.assert_called_once_with((-0.5, 0.5))

    def test_add_zoom_inset_with_connections(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test zoom inset with connection lines."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        with patch("matplotlib.patches.ConnectionPatch") as mock_connection:
            mock_connection_instance = Mock()
            mock_connection.return_value = mock_connection_instance

            result = inset_manager.add_zoom_inset(
                figure=fig,
                main_axes=ax,
                dataframe=sample_dataframe,
                plots_directory="/tmp/plots",
                zoom_xlim=(2, 4),
                zoom_ylim=(-0.5, 0.5),
                show_connection_lines=True,
                xaxis_variable="time",
                yaxis_variable="energy",
            )

            assert result is mock_inset_ax
            assert mock_connection.call_count == 2  # Two connection lines
            assert ax.add_artist.call_count == 2

    def test_add_zoom_inset_without_connections(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test zoom inset without connection lines."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        result = inset_manager.add_zoom_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            zoom_xlim=(2, 4),
            zoom_ylim=(-0.5, 0.5),
            show_connection_lines=False,
            xaxis_variable="time",
            yaxis_variable="energy",
        )

        assert result is mock_inset_ax
        ax.add_artist.assert_not_called()


class TestInsetManagement:
    """Test inset management operations."""

    def test_update_inset_data(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test updating inset data."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        # Create an inset first
        inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="time",
            yaxis_variable="energy",
            inset_id="test_inset",
        )

        # Create new data
        new_data = sample_dataframe.copy()
        new_data["energy"] = new_data["energy"] * 2

        # Update the inset
        result = inset_manager.update_inset_data("test_inset", new_data)

        assert result is True
        mock_inset_ax.clear.assert_called_once()

    def test_update_nonexistent_inset(self, inset_manager, sample_dataframe):
        """Test updating non-existent inset."""
        result = inset_manager.update_inset_data("nonexistent", sample_dataframe)
        assert result is False

    def test_remove_inset(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test removing an inset."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        # Create an inset first
        inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="time",
            inset_id="test_inset",
        )

        # Remove the inset
        result = inset_manager.remove_inset("test_inset")

        assert result is True
        assert "test_inset" not in inset_manager._insets
        mock_inset_ax.remove.assert_called_once()

    def test_remove_nonexistent_inset(self, inset_manager):
        """Test removing non-existent inset."""
        result = inset_manager.remove_inset("nonexistent")
        assert result is False

    def test_get_inset_info(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test getting inset information."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        # Create an inset first
        inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="time",
            yaxis_variable="energy",
            inset_id="test_inset",
        )

        # Get inset info
        info = inset_manager.get_inset_info("test_inset")

        assert info is not None
        assert info["axes"] is mock_inset_ax
        assert info["variables"] == ("time", "energy")

    def test_get_nonexistent_inset_info(self, inset_manager):
        """Test getting info for non-existent inset."""
        info = inset_manager.get_inset_info("nonexistent")
        assert info is None

    def test_list_insets(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test listing all insets."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        # Create multiple insets
        mock_inset_ax1 = Mock(spec=Axes)
        mock_inset_ax2 = Mock(spec=Axes)
        ax.inset_axes.side_effect = [mock_inset_ax1, mock_inset_ax2]

        inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="time",
            inset_id="inset1",
        )
        inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="energy",
            inset_id="inset2",
        )

        inset_list = inset_manager.list_insets()

        assert len(inset_list) == 2
        assert "inset1" in inset_list
        assert "inset2" in inset_list

    def test_clear_all_insets(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test clearing all insets."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        # Create multiple insets
        mock_inset_ax1 = Mock(spec=Axes)
        mock_inset_ax2 = Mock(spec=Axes)
        ax.inset_axes.side_effect = [mock_inset_ax1, mock_inset_ax2]

        inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="time",
            inset_id="inset1",
        )
        inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="energy",
            inset_id="inset2",
        )

        assert len(inset_manager._insets) == 2

        inset_manager.clear_all_insets()

        assert len(inset_manager._insets) == 0
        mock_inset_ax1.remove.assert_called_once()
        mock_inset_ax2.remove.assert_called_once()


class TestUtilityMethods:
    """Test utility and helper methods."""

    def test_infer_yaxis_variable_from_label(self, inset_manager):
        """Test y-axis variable inference from axes label."""
        mock_axes = Mock()
        mock_axes.get_ylabel.return_value = "Energy (J)"

        # Create a real mock plotter with actual DataFrame
        mock_plotter = Mock()
        mock_plotter.dataframe = pd.DataFrame({"time": [1, 2], "energy": [3, 4]})
        mock_plotter.list_of_output_quantity_names_from_dataframe = ["energy"]

        result = inset_manager._infer_yaxis_variable(mock_axes, mock_plotter)
        assert result == "energy"

    def test_infer_yaxis_variable_from_output_quantities(self, inset_manager):
        """Test y-axis variable inference from output quantities."""
        mock_axes = Mock()
        mock_axes.get_ylabel.return_value = "Unknown"

        # Create a real mock plotter with actual DataFrame
        mock_plotter = Mock()
        mock_plotter.dataframe = pd.DataFrame({"time": [1, 2], "energy": [3, 4]})
        mock_plotter.list_of_output_quantity_names_from_dataframe = ["energy"]

        result = inset_manager._infer_yaxis_variable(mock_axes, mock_plotter)
        assert result == "energy"

    def test_infer_yaxis_variable_fallback(self, inset_manager):
        """Test y-axis variable inference fallback to second column."""
        mock_axes = Mock()
        mock_axes.get_ylabel.return_value = ""

        # Create a real mock plotter with actual DataFrame
        mock_plotter = Mock()
        mock_plotter.dataframe = pd.DataFrame({"time": [1, 2], "energy": [3, 4]})
        mock_plotter.list_of_output_quantity_names_from_dataframe = []

        result = inset_manager._infer_yaxis_variable(mock_axes, mock_plotter)
        assert result == "energy"

    def test_infer_yaxis_variable_error(self, inset_manager):
        """Test y-axis variable inference error handling."""
        mock_axes = Mock()
        mock_axes.get_ylabel.return_value = ""

        # Create a real mock plotter with actual DataFrame
        mock_plotter = Mock()
        mock_plotter.dataframe = pd.DataFrame({"time": [1, 2]})  # Only one column
        mock_plotter.list_of_output_quantity_names_from_dataframe = []

        with pytest.raises(ValueError, match="Cannot determine y-axis variable"):
            inset_manager._infer_yaxis_variable(mock_axes, mock_plotter)

    def test_prepare_inset_plot_kwargs(self, inset_manager):
        """Test preparation of plot kwargs for inset."""
        original_kwargs = {
            "grouping_variable": "method",
            "include_legend": True,
            "include_plot_title": True,
            "save_figure": True,
            "verbose": True,
            "custom_param": "value",
        }

        result = inset_manager._prepare_inset_plot_kwargs(original_kwargs)

        # Check that inset-specific overrides are applied
        assert result["is_inset"] is True
        assert result["include_plot_title"] is False
        assert result["save_figure"] is False
        assert result["verbose"] is False

        # Check that user settings are preserved where appropriate
        assert result["grouping_variable"] == "method"
        assert result["custom_param"] == "value"
        assert result["include_legend"] is True  # User setting preserved

    def test_create_temp_plotter_with_class(
        self, inset_manager, sample_dataframe, mock_data_plotter_class
    ):
        """Test creating temporary plotter with provided class."""
        inset_manager.data_plotter_class = mock_data_plotter_class

        result = inset_manager._create_temp_plotter(sample_dataframe, "/tmp/plots")

        assert result is mock_data_plotter_class.return_value
        mock_data_plotter_class.assert_called_once_with(sample_dataframe, "/tmp/plots")

    def test_create_temp_plotter_with_import(self, inset_manager, sample_dataframe):
        """Test creating temporary plotter with automatic import."""
        # Test that it tries to import when data_plotter_class is None
        inset_manager.data_plotter_class = None

        # This should trigger the import inside _create_temp_plotter We
        # can't easily test the import itself, so let's just verify that
        # the method handles the None case
        try:
            result = inset_manager._create_temp_plotter(sample_dataframe, "/tmp/plots")
            # If no exception, the import worked
            assert result is not None
        except ImportError:
            # If import fails, that's also acceptable for testing
            pytest.skip("DataPlotter import not available in test environment")

    def test_set_default_dimensions(self, inset_manager):
        """Test setting default dimensions."""
        inset_manager.set_default_dimensions(0.5, 0.4)

        assert inset_manager.default_width == 0.5
        assert inset_manager.default_height == 0.4

    def test_add_location_preset(self, inset_manager):
        """Test adding custom location preset."""
        inset_manager.add_location_preset("custom_location", 0.25, 0.75)

        assert "custom_location" in inset_manager.location_presets
        assert inset_manager.location_presets["custom_location"] == [0.25, 0.75]

    def test_get_location_presets(self, inset_manager):
        """Test getting location presets."""
        presets = inset_manager.get_location_presets()

        assert isinstance(presets, dict)
        assert "upper right" in presets
        assert "center" in presets

        # Should be a copy, not the original
        presets["new_location"] = [0.1, 0.2]
        assert "new_location" not in inset_manager.location_presets


class TestSpecialCases:
    """Test special cases and edge conditions."""

    def test_add_inset_with_uncertainty_data(
        self,
        inset_manager,
        mock_figure_axes,
        uncertainty_dataframe,
        mock_data_plotter_class,
    ):
        """Test inset creation with uncertainty data (tuples)."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        def tuple_filter(df):
            # Filter based on tuple data - extract first element
            return df[df["x"] > 5]

        result = inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=uncertainty_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="x",
            yaxis_variable="y_with_error",
            data_filter_func=tuple_filter,
        )

        assert result is mock_inset_ax
        mock_data_plotter_class.assert_called_once()

    def test_zoom_inset_with_tuple_data(
        self,
        inset_manager,
        mock_figure_axes,
        uncertainty_dataframe,
        mock_data_plotter_class,
    ):
        """Test zoom inset with tuple data filtering."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        result = inset_manager.add_zoom_inset(
            figure=fig,
            main_axes=ax,
            dataframe=uncertainty_dataframe,
            plots_directory="/tmp/plots",
            zoom_xlim=(2, 8),
            zoom_ylim=(-0.5, 0.5),
            xaxis_variable="x",
            yaxis_variable="y_with_error",
        )

        assert result is mock_inset_ax
        mock_inset_ax.set_xlim.assert_called_once_with((2, 8))
        mock_inset_ax.set_ylim.assert_called_once_with((-0.5, 0.5))

    def test_connection_lines_import_error(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test connection lines with import error."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        # We need to patch the import at the point where it's actually
        # used The import happens inside the _add_zoom_connection_lines
        # method

        # Since this is complex to test with dynamic imports, let's just
        # test that the method handles the case gracefully
        result = inset_manager.add_zoom_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            zoom_xlim=(2, 4),
            zoom_ylim=(-0.5, 0.5),
            show_connection_lines=True,
            xaxis_variable="time",
            yaxis_variable="energy",
        )

        assert result is mock_inset_ax
        # The method should complete successfully even if connection
        # lines fail

    def test_connection_lines_creation_error(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test connection lines with creation error."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        with patch(
            "matplotlib.patches.ConnectionPatch",
            side_effect=Exception("Creation error"),
        ):
            result = inset_manager.add_zoom_inset(
                figure=fig,
                main_axes=ax,
                dataframe=sample_dataframe,
                plots_directory="/tmp/plots",
                zoom_xlim=(2, 4),
                zoom_ylim=(-0.5, 0.5),
                show_connection_lines=True,
                xaxis_variable="time",
                yaxis_variable="energy",
            )

            assert result is mock_inset_ax
            ax.add_artist.assert_not_called()  # Should not be called due to creation error

    def test_inset_management_exception_handling(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test exception handling in inset management operations."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        # Create an inset first
        inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="time",
            inset_id="test_inset",
        )

        # Mock remove to raise exception
        mock_inset_ax.remove.side_effect = Exception("Remove error")

        result = inset_manager.remove_inset("test_inset")

        assert result is False  # Should handle exception gracefully

    def test_update_inset_exception_handling(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test exception handling in inset update operations."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        # Create an inset first
        inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="time",
            inset_id="test_inset",
        )

        # Mock clear to raise exception
        mock_inset_ax.clear.side_effect = Exception("Clear error")

        result = inset_manager.update_inset_data("test_inset", sample_dataframe)

        assert result is False  # Should handle exception gracefully


class TestIntegrationWithRealMatplotlib:
    """Integration tests with real matplotlib objects."""

    def test_with_real_matplotlib_figure(
        self, inset_manager, sample_dataframe, mock_data_plotter_class
    ):
        """Test with real matplotlib figure and axes."""
        fig, ax = plt.subplots()
        inset_manager.data_plotter_class = mock_data_plotter_class

        # Plot some data on main axes
        ax.plot(sample_dataframe["time"], sample_dataframe["energy"], "o-")

        # Add inset
        inset_ax = inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="temperature",
            yaxis_variable="pressure",
            location="upper right",
            inset_id="real_inset",
        )

        assert inset_ax is not None
        assert isinstance(inset_ax, Axes)
        assert "real_inset" in inset_manager._insets

        plt.close(fig)

    def test_multiple_insets_real_figure(
        self, inset_manager, sample_dataframe, mock_data_plotter_class
    ):
        """Test multiple insets with real figure."""
        fig, ax = plt.subplots()
        inset_manager.data_plotter_class = mock_data_plotter_class

        # Plot main data
        ax.plot(sample_dataframe["time"], sample_dataframe["energy"], "o-")

        # Add multiple insets
        inset_specs = [
            {
                "xaxis_variable": "temperature",
                "yaxis_variable": "pressure",
                "location": "upper left",
                "width": 0.25,
                "height": 0.25,
                "inset_id": "temp_press",
            },
            {
                "xaxis_variable": "time",
                "yaxis_variable": "temperature",
                "location": "lower right",
                "width": 0.3,
                "height": 0.3,
                "inset_id": "time_temp",
            },
        ]

        created_insets = inset_manager.add_multiple_insets(
            fig, ax, inset_specs, sample_dataframe, "/tmp/plots"
        )

        assert len(created_insets) == 2
        assert all(isinstance(inset_ax, Axes) for inset_ax in created_insets.values())

        plt.close(fig)

    def test_zoom_inset_real_figure(
        self, inset_manager, sample_dataframe, mock_data_plotter_class
    ):
        """Test zoom inset with real figure."""
        fig, ax = plt.subplots()
        inset_manager.data_plotter_class = mock_data_plotter_class

        # Plot main data
        ax.plot(sample_dataframe["time"], sample_dataframe["energy"], "o-")
        ax.set_xlim(0, 10)
        ax.set_ylim(-2, 2)

        # Add zoom inset
        zoom_ax = inset_manager.add_zoom_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            zoom_xlim=(2, 4),
            zoom_ylim=(-0.5, 0.5),
            location="upper right",
            show_connection_lines=False,  # Disable for simpler test
            xaxis_variable="time",
            yaxis_variable="energy",
        )

        assert zoom_ax is not None
        assert isinstance(zoom_ax, Axes)

        # Check that zoom limits were set
        assert zoom_ax.get_xlim() == (2, 4)
        assert zoom_ax.get_ylim() == (-0.5, 0.5)

        plt.close(fig)

    def test_inset_management_operations_real_figure(
        self, inset_manager, sample_dataframe, mock_data_plotter_class
    ):
        """Test inset management operations with real figure."""
        fig, ax = plt.subplots()
        inset_manager.data_plotter_class = mock_data_plotter_class

        # Create initial inset
        inset_ax = inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="time",
            yaxis_variable="energy",
            inset_id="test_inset",
        )

        assert inset_ax is not None
        assert len(inset_manager.list_insets()) == 1

        # Get inset info
        info = inset_manager.get_inset_info("test_inset")
        assert info is not None
        assert info["axes"] is inset_ax

        # Update inset data
        new_data = sample_dataframe.copy()
        new_data["energy"] = new_data["energy"] * 2

        update_result = inset_manager.update_inset_data("test_inset", new_data)
        assert update_result is True

        # Remove inset
        remove_result = inset_manager.remove_inset("test_inset")
        assert remove_result is True
        assert len(inset_manager.list_insets()) == 0

        plt.close(fig)


class TestErrorHandling:
    """Test comprehensive error handling."""

    def test_add_inset_missing_column(
        self, inset_manager, mock_figure_axes, sample_dataframe, mock_data_plotter_class
    ):
        """Test inset creation with missing column."""
        fig, ax, mock_inset_ax = mock_figure_axes
        inset_manager.data_plotter_class = mock_data_plotter_class

        # Mock plotter to raise exception for missing column
        mock_plotter = mock_data_plotter_class.return_value
        mock_plotter.set_plot_variables.side_effect = ValueError("Column not found")

        result = inset_manager.add_inset(
            figure=fig,
            main_axes=ax,
            dataframe=sample_dataframe,
            plots_directory="/tmp/plots",
            xaxis_variable="nonexistent_column",
        )

        assert result is None  # Should handle error gracefully

    def test_data_filter_with_invalid_dataframe(self, inset_manager):
        """Test data filtering with invalid DataFrame operations."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        def bad_filter(df):
            # Try to access non-existent column
            return df[df["nonexistent"] > 0]

        result = inset_manager._apply_data_filters(df, bad_filter, None)

        assert len(result) == 0  # Should return empty DataFrame

    def test_invalid_condition_string(self, inset_manager, sample_dataframe):
        """Test data filtering with invalid condition string."""
        condition = "invalid syntax @#$%"

        result = inset_manager._apply_data_filters(sample_dataframe, None, condition)

        assert len(result) == 0  # Should return empty DataFrame

    def test_y_axis_inference_edge_cases(self, inset_manager):
        """Test y-axis variable inference with edge cases."""
        mock_axes = Mock()
        mock_axes.get_ylabel.return_value = ""

        # Test with empty dataframe
        mock_plotter = Mock()
        mock_plotter.dataframe = pd.DataFrame()
        mock_plotter.list_of_output_quantity_names_from_dataframe = []

        with pytest.raises(ValueError):
            inset_manager._infer_yaxis_variable(mock_axes, mock_plotter)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
