import pytest
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from library.visualization.managers.layout_manager import PlotLayoutManager


# Module-level fixtures (accessible to all test classes)
@pytest.fixture
def mock_constants():
    """Create a mock constants module for testing."""
    constants = Mock()
    constants.AXES_LABELS_BY_COLUMN_NAME = {
        "temperature": "Temperature (K)",
        "pressure": "Pressure (Pa)",
        "energy": "Energy (eV)",
    }
    constants.PARAMETERS_WITH_EXPONENTIAL_FORMAT = {"pressure", "energy"}
    constants.PARAMETERS_OF_INTEGER_VALUE = {"count", "iterations"}
    return constants


@pytest.fixture
def layout_manager(mock_constants):
    """Create layout manager with mock constants."""
    return PlotLayoutManager(mock_constants)


@pytest.fixture
def layout_manager_no_constants():
    """Create layout manager without constants."""
    return PlotLayoutManager()


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def mock_axes():
    """Create a properly mocked matplotlib Axes object."""
    mock_ax = Mock(spec=Axes)
    # Add the xaxis and yaxis attributes that real Axes objects have
    mock_ax.xaxis = Mock()
    mock_ax.yaxis = Mock()
    return mock_ax


class TestInitialization:
    """Test initialization and default values."""

    def test_init_with_constants(self, mock_constants):
        """Test initialization with constants module."""
        manager = PlotLayoutManager(mock_constants)
        assert manager.constants == mock_constants
        assert manager.default_figure_size == (7, 5)
        assert manager.default_font_size == 13
        assert manager.default_margins == {
            "left": 0.15,
            "right": 0.94,
            "bottom": 0.12,
            "top": 0.92,
        }

    def test_init_without_constants(self):
        """Test initialization without constants module."""
        manager = PlotLayoutManager()
        assert manager.constants is None
        # Should still have sensible defaults
        assert manager.default_figure_size == (7, 5)
        assert manager.default_font_size == 13


class TestCreateFigure:
    """Test figure creation functionality."""

    @patch("matplotlib.pyplot.subplots")
    def test_create_figure_default_size(self, mock_subplots, layout_manager):
        """Test creating figure with default size."""
        mock_fig, mock_ax = Mock(spec=Figure), Mock(spec=Axes)
        mock_subplots.return_value = (mock_fig, mock_ax)

        fig, ax = layout_manager.create_figure()

        mock_subplots.assert_called_once_with(figsize=(7, 5))
        mock_ax.grid.assert_called_once_with(True, linestyle="--", alpha=0.5)
        assert fig == mock_fig
        assert ax == mock_ax

    @patch("matplotlib.pyplot.subplots")
    def test_create_figure_custom_size(self, mock_subplots, layout_manager):
        """Test creating figure with custom size."""
        mock_fig, mock_ax = Mock(spec=Figure), Mock(spec=Axes)
        mock_subplots.return_value = (mock_fig, mock_ax)

        custom_size = (10, 8)
        fig, ax = layout_manager.create_figure(custom_size)

        mock_subplots.assert_called_once_with(figsize=custom_size)


class TestPrivateSetupAxesLabels:
    """Test private _setup_axes_labels functionality."""

    def test_setup_axes_labels_with_constants_lookup(self, layout_manager, mock_axes):
        """Test label setup using constants for lookup."""
        layout_manager._setup_axes_labels(mock_axes, "temperature", "pressure")

        mock_axes.set_xlabel.assert_called_once_with("Temperature (K)", fontsize=15)
        mock_axes.set_ylabel.assert_called_once_with("Pressure (Pa)", fontsize=15)
        mock_axes.tick_params.assert_called_once_with(axis="both", labelsize=13)

    def test_setup_axes_labels_custom_labels(self, layout_manager, mock_axes):
        """Test label setup with custom labels."""
        layout_manager._setup_axes_labels(
            mock_axes,
            "temperature",
            "pressure",
            xaxis_label="Custom X",
            yaxis_label="Custom Y",
        )

        mock_axes.set_xlabel.assert_called_once_with("Custom X", fontsize=15)
        mock_axes.set_ylabel.assert_called_once_with("Custom Y", fontsize=15)

    def test_setup_axes_labels_no_constants(
        self, layout_manager_no_constants, mock_axes
    ):
        """Test label setup without constants module."""
        layout_manager_no_constants._setup_axes_labels(
            mock_axes, "temperature", "pressure"
        )

        # Should fall back to variable names
        mock_axes.set_xlabel.assert_called_once_with("temperature", fontsize=15)
        mock_axes.set_ylabel.assert_called_once_with("pressure", fontsize=15)

    def test_setup_axes_labels_custom_font_size(self, layout_manager, mock_axes):
        """Test label setup with custom font size."""
        layout_manager._setup_axes_labels(
            mock_axes, "temperature", "pressure", font_size=16
        )

        mock_axes.set_xlabel.assert_called_once_with("Temperature (K)", fontsize=18)
        mock_axes.tick_params.assert_called_once_with(axis="both", labelsize=16)


class TestPrivateSetupAxesScaling:
    """Test private _setup_axes_scaling functionality."""

    def test_setup_axes_scaling_auto_log_from_constants(
        self, layout_manager, mock_axes
    ):
        """Test automatic log scaling based on constants."""
        layout_manager._setup_axes_scaling(
            mock_axes, "pressure", "energy"  # Both in EXPONENTIAL_FORMAT
        )

        # Should call set_xscale and set_yscale with 'log'
        assert mock_axes.set_xscale.call_count == 1
        assert mock_axes.set_yscale.call_count == 1
        mock_axes.set_xscale.assert_called_with("log")
        mock_axes.set_yscale.assert_called_with("log")

    def test_setup_axes_scaling_force_log(self, layout_manager, mock_axes):
        """Test forced log scaling regardless of constants."""
        layout_manager._setup_axes_scaling(
            mock_axes,
            "temperature",
            "count",  # Neither in EXPONENTIAL_FORMAT
            xaxis_log_scale=True,
            yaxis_log_scale=True,
        )

        mock_axes.set_xscale.assert_called_with("log")
        mock_axes.set_yscale.assert_called_with("log")

    def test_setup_axes_scaling_no_log(self, layout_manager, mock_axes):
        """Test no log scaling applied."""
        layout_manager._setup_axes_scaling(
            mock_axes, "temperature", "count"  # Neither in EXPONENTIAL_FORMAT
        )

        # Should not call set_xscale or set_yscale
        mock_axes.set_xscale.assert_not_called()
        mock_axes.set_yscale.assert_not_called()

    def test_setup_axes_scaling_auto_log_disabled(self, layout_manager, mock_axes):
        """Test behavior when auto_log_scale is disabled."""
        layout_manager._setup_axes_scaling(
            mock_axes,
            "pressure",
            "energy",  # Both in EXPONENTIAL_FORMAT
            auto_log_scale=False,  # But auto scaling disabled
        )

        # Should not call set_xscale or set_yscale
        mock_axes.set_xscale.assert_not_called()
        mock_axes.set_yscale.assert_not_called()


class TestPrivateSetupIntegerTicks:
    """Test private _setup_integer_ticks functionality."""

    @patch("matplotlib.ticker.MaxNLocator")
    def test_setup_integer_ticks_both_axes(
        self, mock_locator, layout_manager, mock_axes
    ):
        """Test integer ticks on both axes."""
        mock_locator_instance = Mock()
        mock_locator.return_value = mock_locator_instance

        layout_manager._setup_integer_ticks(
            mock_axes, "count", "iterations"  # Both in INTEGER_VALUE
        )

        # Should set MaxNLocator for both axes
        assert mock_locator.call_count == 2
        mock_locator.assert_called_with(integer=True)
        mock_axes.xaxis.set_major_locator.assert_called_once()
        mock_axes.yaxis.set_major_locator.assert_called_once()

    def test_setup_integer_ticks_no_constants(
        self, layout_manager_no_constants, mock_axes
    ):
        """Test integer ticks without constants module."""
        layout_manager_no_constants._setup_integer_ticks(
            mock_axes, "count", "iterations"
        )

        # Should not call anything without constants
        mock_axes.xaxis.set_major_locator.assert_not_called()
        mock_axes.yaxis.set_major_locator.assert_not_called()

    @patch("matplotlib.ticker.MaxNLocator")
    def test_setup_integer_ticks_single_axis(
        self, mock_locator, layout_manager, mock_axes
    ):
        """Test integer ticks on single axis only."""
        mock_locator_instance = Mock()
        mock_locator.return_value = mock_locator_instance

        layout_manager._setup_integer_ticks(
            mock_axes, "count", "temperature"  # Only count in INTEGER_VALUE
        )

        # Should set MaxNLocator for x-axis only
        assert mock_locator.call_count == 1
        mock_axes.xaxis.set_major_locator.assert_called_once()
        mock_axes.yaxis.set_major_locator.assert_not_called()


class TestPrivateSetupAxesLimitsAndOrientation:
    """Test private _setup_axes_limits_and_orientation functionality."""

    def test_setup_explicit_limits(self, layout_manager, mock_axes):
        """Test setting explicit axis limits."""
        layout_manager._setup_axes_limits_and_orientation(
            mock_axes, xlim=(0, 100), ylim=(-5, 5)
        )

        mock_axes.set_xlim.assert_called_once_with((0, 100))
        mock_axes.set_ylim.assert_called_once_with((-5, 5))

    def test_setup_start_at_zero(self, layout_manager, mock_axes):
        """Test forcing axes to start at zero."""
        mock_axes.get_xlim.return_value = (-10, 100)
        mock_axes.get_ylim.return_value = (-20, 50)

        layout_manager._setup_axes_limits_and_orientation(
            mock_axes, xaxis_start_at_zero=True, yaxis_start_at_zero=True
        )

        mock_axes.set_xlim.assert_called_once_with(left=0, right=100)
        mock_axes.set_ylim.assert_called_once_with(bottom=0, top=50)

    def test_setup_invert_axes(self, layout_manager, mock_axes):
        """Test axis inversion."""
        layout_manager._setup_axes_limits_and_orientation(
            mock_axes, invert_xaxis=True, invert_yaxis=True
        )

        mock_axes.invert_xaxis.assert_called_once()
        mock_axes.invert_yaxis.assert_called_once()

    def test_setup_explicit_limits_override_zero_start(self, layout_manager, mock_axes):
        """Test that explicit limits override start_at_zero."""
        layout_manager._setup_axes_limits_and_orientation(
            mock_axes, xlim=(10, 100), xaxis_start_at_zero=True  # xlim should win
        )

        mock_axes.set_xlim.assert_called_once_with((10, 100))
        mock_axes.get_xlim.assert_not_called()  # Should not check current limits


class TestPrivateAdjustMargins:
    """Test private _adjust_margins functionality."""

    def test_adjust_margins_custom_values(self, layout_manager):
        """Test margin adjustment with custom values."""
        mock_fig = Mock(spec=Figure)

        layout_manager._adjust_margins(
            mock_fig, left=0.2, right=0.9, bottom=0.15, top=0.85
        )

        expected_margins = {"left": 0.2, "right": 0.9, "bottom": 0.15, "top": 0.85}
        mock_fig.subplots_adjust.assert_called_once_with(**expected_margins)

    def test_adjust_margins_partial_custom(self, layout_manager):
        """Test margin adjustment with partial custom values."""
        mock_fig = Mock(spec=Figure)

        layout_manager._adjust_margins(mock_fig, left=0.2)

        # Should use custom left, defaults for others
        expected_margins = {"left": 0.2, "right": 0.94, "bottom": 0.12, "top": 0.92}
        mock_fig.subplots_adjust.assert_called_once_with(**expected_margins)

    def test_adjust_margins_all_defaults(self, layout_manager):
        """Test margin adjustment with all default values."""
        mock_fig = Mock(spec=Figure)

        layout_manager._adjust_margins(mock_fig)

        # Should use all defaults
        expected_margins = layout_manager.default_margins
        mock_fig.subplots_adjust.assert_called_once_with(**expected_margins)


class TestSetupCompleteLayout:
    """Test the public setup_complete_layout functionality."""

    def test_setup_complete_layout_all_methods_called(self, layout_manager):
        """Test that complete layout setup calls all private methods."""
        with patch.object(
            layout_manager, "create_figure"
        ) as mock_create_figure, patch.object(
            layout_manager, "_setup_axes_labels"
        ) as mock_setup_labels, patch.object(
            layout_manager, "_setup_axes_scaling"
        ) as mock_setup_scaling, patch.object(
            layout_manager, "_setup_integer_ticks"
        ) as mock_setup_integer, patch.object(
            layout_manager, "_setup_axes_limits_and_orientation"
        ) as mock_setup_limits, patch.object(
            layout_manager, "_adjust_margins"
        ) as mock_adjust_margins:

            mock_fig, mock_ax = Mock(spec=Figure), Mock(spec=Axes)
            mock_create_figure.return_value = (mock_fig, mock_ax)

            result = layout_manager.setup_complete_layout(
                "temperature", "pressure", figure_size=(8, 6), font_size=14
            )

            # Verify all methods were called
            mock_create_figure.assert_called_once_with((8, 6))
            mock_setup_labels.assert_called_once()
            mock_setup_scaling.assert_called_once()
            mock_setup_integer.assert_called_once()
            mock_setup_limits.assert_called_once()
            mock_adjust_margins.assert_called_once()

            # Verify return value
            assert result == (mock_fig, mock_ax)

    def test_setup_complete_layout_with_custom_function(self, layout_manager):
        """Test complete layout setup with custom function."""
        mock_custom_function = Mock()

        with patch.object(layout_manager, "create_figure") as mock_create:
            mock_fig, mock_ax = Mock(spec=Figure), Mock(spec=Axes)
            mock_create.return_value = (mock_fig, mock_ax)

            layout_manager.setup_complete_layout(
                "temperature", "pressure", apply_custom_function=mock_custom_function
            )

            # Custom function should be called with the axes
            mock_custom_function.assert_called_once_with(mock_ax)

    def test_setup_complete_layout_parameter_passing(self, layout_manager):
        """Test that parameters are correctly passed to private methods."""
        with patch.object(
            layout_manager, "create_figure"
        ) as mock_create_figure, patch.object(
            layout_manager, "_setup_axes_labels"
        ) as mock_setup_labels, patch.object(
            layout_manager, "_setup_axes_scaling"
        ) as mock_setup_scaling, patch.object(
            layout_manager, "_setup_axes_limits_and_orientation"
        ) as mock_setup_limits, patch.object(
            layout_manager, "_adjust_margins"
        ) as mock_adjust_margins:

            mock_fig, mock_ax = Mock(spec=Figure), Mock(spec=Axes)
            mock_create_figure.return_value = (mock_fig, mock_ax)

            layout_manager.setup_complete_layout(
                "temperature",
                "pressure",
                figure_size=(10, 8),
                font_size=16,
                xaxis_label="Custom X",
                yaxis_label="Custom Y",
                xaxis_log_scale=True,
                yaxis_log_scale=True,
                xlim=(0, 100),
                ylim=(-5, 5),
                left_margin=0.2,
            )

            # Verify parameters were passed correctly
            mock_setup_labels.assert_called_once_with(
                mock_ax, "temperature", "pressure", "Custom X", "Custom Y", 16
            )
            mock_setup_scaling.assert_called_once_with(
                mock_ax, "temperature", "pressure", True, True
            )
            mock_setup_limits.assert_called_once_with(
                mock_ax, (0, 100), (-5, 5), False, False, False, False
            )
            mock_adjust_margins.assert_called_once_with(mock_fig, 0.2, None, None, None)


class TestConfigureExistingAxes:
    """Test the public configure_existing_axes functionality."""

    def test_configure_existing_axes_all_methods_called(
        self, layout_manager, mock_axes
    ):
        """Test that configure_existing_axes calls appropriate private methods."""
        with patch.object(
            layout_manager, "_setup_axes_labels"
        ) as mock_setup_labels, patch.object(
            layout_manager, "_setup_axes_scaling"
        ) as mock_setup_scaling, patch.object(
            layout_manager, "_setup_integer_ticks"
        ) as mock_setup_integer, patch.object(
            layout_manager, "_setup_axes_limits_and_orientation"
        ) as mock_setup_limits:

            layout_manager.configure_existing_axes(mock_axes, "temperature", "pressure")

            # Verify all methods were called (but not create_figure or adjust_margins)
            mock_setup_labels.assert_called_once()
            mock_setup_scaling.assert_called_once()
            mock_setup_integer.assert_called_once()
            mock_setup_limits.assert_called_once()

    def test_configure_existing_axes_with_custom_function(
        self, layout_manager, mock_axes
    ):
        """Test configure_existing_axes with custom function."""
        mock_custom_function = Mock()

        layout_manager.configure_existing_axes(
            mock_axes,
            "temperature",
            "pressure",
            apply_custom_function=mock_custom_function,
        )

        # Custom function should be called with the axes
        mock_custom_function.assert_called_once_with(mock_axes)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_strings_for_labels(self, layout_manager_no_constants, mock_axes):
        """Test handling of empty strings for axis labels."""
        layout_manager_no_constants._setup_axes_labels(
            mock_axes, "", "", xaxis_label="", yaxis_label=""
        )

        # Should set empty labels without error
        mock_axes.set_xlabel.assert_called_once_with("", fontsize=15)
        mock_axes.set_ylabel.assert_called_once_with("", fontsize=15)

    def test_none_values_handling(self, layout_manager_no_constants, mock_axes):
        """Test handling of None values in various methods."""
        mock_fig = Mock(spec=Figure)

        # Should not raise errors with None values
        layout_manager_no_constants._setup_axes_labels(mock_axes, "", "")
        layout_manager_no_constants._adjust_margins(mock_fig)  # All None margins


# Parametrized tests for comprehensive coverage
class TestParametrized:
    """Parametrized tests for comprehensive scenarios."""

    @pytest.mark.parametrize(
        "x_var,y_var,expected_x_calls,expected_y_calls",
        [
            ("pressure", "energy", 1, 1),  # Both should get log scale
            ("temperature", "count", 0, 0),  # Neither should get log scale
            ("pressure", "count", 1, 0),  # Only x should get log scale
            ("temperature", "energy", 0, 1),  # Only y should get log scale
        ],
    )
    def test_log_scaling_combinations(
        self, layout_manager, x_var, y_var, expected_x_calls, expected_y_calls
    ):
        """Test various combinations of log scaling."""
        mock_ax = Mock(spec=Axes)

        layout_manager._setup_axes_scaling(mock_ax, x_var, y_var)

        assert mock_ax.set_xscale.call_count == expected_x_calls
        assert mock_ax.set_yscale.call_count == expected_y_calls

    @pytest.mark.parametrize("size", [(7, 5), (10, 8), (12, 9), None])
    def test_figure_size_variations(self, layout_manager, size):
        """Test different figure size inputs."""
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig, mock_ax = Mock(spec=Figure), Mock(spec=Axes)
            mock_subplots.return_value = (mock_fig, mock_ax)

            layout_manager.create_figure(size)

            expected_size = size or (7, 5)
            mock_subplots.assert_called_once_with(figsize=expected_size)

    @pytest.mark.parametrize(
        "margins",
        [
            {"left": 0.1, "right": 0.9, "bottom": 0.1, "top": 0.9},
            {"left": 0.2},  # Only left specified
            {"right": 0.8, "top": 0.85},  # Only right and top specified
            {},  # All defaults
        ],
    )
    def test_margin_combinations(self, layout_manager, margins):
        """Test various margin combinations."""
        mock_fig = Mock(spec=Figure)

        layout_manager._adjust_margins(mock_fig, **margins)

        # Build expected margins with defaults for unspecified values
        expected = layout_manager.default_margins.copy()
        expected.update(margins)

        mock_fig.subplots_adjust.assert_called_once_with(**expected)


# Integration tests that test the full workflow
class TestIntegration:
    """Integration tests for full workflows."""

    @patch("matplotlib.pyplot.subplots")
    def test_complete_workflow_integration(self, mock_subplots, layout_manager):
        """Test complete workflow from start to finish."""
        mock_fig, mock_ax = Mock(spec=Figure), Mock(spec=Axes)
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Test a realistic workflow
        fig, ax = layout_manager.setup_complete_layout(
            "pressure",
            "energy",  # Both require log scaling
            figure_size=(10, 6),
            font_size=14,
            xaxis_label="Custom Pressure",
            xlim=(1, 1000),
            xaxis_log_scale=True,
            left_margin=0.2,
        )

        # Verify figure creation
        mock_subplots.assert_called_once_with(figsize=(10, 6))

        # Verify labels were set
        mock_ax.set_xlabel.assert_called_once_with("Custom Pressure", fontsize=16)
        mock_ax.set_ylabel.assert_called_once()  # Should have energy label

        # Verify log scaling was applied
        assert mock_ax.set_xscale.call_count >= 1  # At least once
        assert mock_ax.set_yscale.call_count >= 1  # At least once

        # Verify limits were set
        mock_ax.set_xlim.assert_called_once_with((1, 1000))

        # Verify margins were adjusted
        mock_fig.subplots_adjust.assert_called_once()

        # Verify return values
        assert fig == mock_fig
        assert ax == mock_ax


# Running specific test groups
if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])

    # Run specific test class
    # pytest.main([f'{__file__}::TestInitialization', '-v'])

    # Run with coverage
    # pytest.main([__file__, '--cov=library.visualization.managers.layout_manager', '-v'])
