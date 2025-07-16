"""
Unit tests for PlotAnnotationManager class.

This module provides comprehensive testing for the PlotAnnotationManager
class, covering annotation creation, styling, positioning, and
management operations.
"""

from unittest.mock import Mock

import pytest
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for testing
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.text import Text

from library.visualization.managers.annotation_manager import PlotAnnotationManager
from library.visualization.managers.data_processor import PlotDataProcessor


@pytest.fixture
def mock_data_processor():
    """Create a mock PlotDataProcessor for testing."""
    processor = Mock(spec=PlotDataProcessor)
    processor.prepare_annotation_data.return_value = [
        (1.0, 10.0, "A"),
        (2.0, 20.0, "B"),
        (3.0, 30.0, "C"),
    ]
    processor.format_annotation_value.side_effect = lambda x: str(x)
    return processor


@pytest.fixture
def annotation_manager(mock_data_processor):
    """Create a PlotAnnotationManager instance for testing."""
    return PlotAnnotationManager(data_processor=mock_data_processor)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
            "labels": ["A", "B", "C", "D", "E"],
            "values": [100, 200, 300, 400, 500],
        }
    )


@pytest.fixture
def mock_axes():
    """Create a mock matplotlib Axes object."""
    ax = Mock(spec=Axes)
    ax.annotate.return_value = Mock(spec=Text)
    ax.texts = []
    ax.collections = []
    ax.lines = []
    ax.get_children.return_value = []
    ax.transData = Mock()
    ax.transAxes = Mock()
    ax.transData.transform.return_value = [100, 200]
    return ax


class TestPlotAnnotationManagerInitialization:
    """Test annotation manager initialization."""

    def test_init_with_data_processor(self, mock_data_processor):
        """Test initialization with provided data processor."""
        manager = PlotAnnotationManager(data_processor=mock_data_processor)
        assert manager.data_processor is mock_data_processor
        assert isinstance(manager.default_style, dict)
        assert isinstance(manager.offset_presets, dict)

    def test_init_without_data_processor(self):
        """Test initialization creates data processor if none provided."""
        # Test that initialization works without providing a data processor
        manager = PlotAnnotationManager()

        # Should have created a data processor
        assert manager.data_processor is not None

        # Should have the expected methods
        assert hasattr(manager.data_processor, "prepare_annotation_data")
        assert hasattr(manager.data_processor, "format_annotation_value")

    def test_default_style_structure(self, annotation_manager):
        """Test that default style contains expected keys."""
        required_keys = [
            "fontsize",
            "boxstyle",
            "facecolor",
            "alpha",
            "edgecolor",
            "offset",
            "arrow_style",
            "arrow_color",
            "arrow_width",
            "arrow_alpha",
            "text_alignment",
        ]

        for key in required_keys:
            assert key in annotation_manager.default_style

    def test_offset_presets_structure(self, annotation_manager):
        """Test that offset presets contain expected directions."""
        expected_presets = [
            "above",
            "below",
            "left",
            "right",
            "above_left",
            "above_right",
            "below_left",
            "below_right",
        ]

        for preset in expected_presets:
            assert preset in annotation_manager.offset_presets
            assert isinstance(annotation_manager.offset_presets[preset], tuple)
            assert len(annotation_manager.offset_presets[preset]) == 2


class TestDataPointAnnotations:
    """Test data point annotation functionality."""

    def test_add_data_point_annotations_basic(
        self, annotation_manager, mock_axes, sample_dataframe
    ):
        """Test basic data point annotation creation."""
        annotations = annotation_manager.add_data_point_annotations(
            mock_axes, sample_dataframe, "x", "y", "labels"
        )

        # Should call data processor
        annotation_manager.data_processor.prepare_annotation_data.assert_called_once_with(
            sample_dataframe, "x", "y", "labels", None
        )

        # Should create annotations
        assert len(annotations) == 3
        assert mock_axes.annotate.call_count == 3

    def test_add_data_point_annotations_with_label_prefix(
        self, annotation_manager, mock_axes, sample_dataframe
    ):
        """Test annotations with label prefix."""
        annotation_manager.add_data_point_annotations(
            mock_axes, sample_dataframe, "x", "y", "labels", annotation_label="Type: "
        )

        # Check that annotate was called with prefixed labels
        calls = mock_axes.annotate.call_args_list
        assert calls[0][0][0] == "Type: A"  # First argument should be 'Type: A'
        assert calls[1][0][0] == "Type: B"
        assert calls[2][0][0] == "Type: C"

    def test_add_data_point_annotations_with_range(
        self, annotation_manager, mock_axes, sample_dataframe
    ):
        """Test annotations with range specification."""
        annotation_range = (0, 3, 2)  # Start=0, end=3, step=2

        annotation_manager.add_data_point_annotations(
            mock_axes,
            sample_dataframe,
            "x",
            "y",
            "labels",
            annotation_range=annotation_range,
        )

        # Should pass range to data processor
        annotation_manager.data_processor.prepare_annotation_data.assert_called_once_with(
            sample_dataframe, "x", "y", "labels", annotation_range
        )

    def test_add_data_point_annotations_empty_data(
        self, annotation_manager, mock_axes, sample_dataframe
    ):
        """Test handling of empty annotation data."""
        annotation_manager.data_processor.prepare_annotation_data.return_value = []

        annotations = annotation_manager.add_data_point_annotations(
            mock_axes, sample_dataframe, "x", "y", "labels"
        )

        assert annotations == []
        assert mock_axes.annotate.call_count == 0

    def test_add_data_point_annotations_with_custom_format(
        self, annotation_manager, mock_axes, sample_dataframe
    ):
        """Test annotations with custom formatting function."""

        def custom_format(value):
            return f"[{value}]"

        annotation_manager.add_data_point_annotations(
            mock_axes, sample_dataframe, "x", "y", "labels", format_func=custom_format
        )

        # Check that custom formatting was applied
        calls = mock_axes.annotate.call_args_list
        assert calls[0][0][0] == "[A]"
        assert calls[1][0][0] == "[B]"
        assert calls[2][0][0] == "[C]"

    def test_add_data_point_annotations_with_style_overrides(
        self, annotation_manager, mock_axes, sample_dataframe
    ):
        """Test annotations with style overrides."""
        style_overrides = {"fontsize": 12, "facecolor": "yellow", "alpha": 0.9}

        annotation_manager.add_data_point_annotations(
            mock_axes,
            sample_dataframe,
            "x",
            "y",
            "labels",
            style_overrides=style_overrides,
        )

        # Check that style overrides were applied
        calls = mock_axes.annotate.call_args_list
        first_call_kwargs = calls[0][1]
        assert first_call_kwargs["fontsize"] == 12
        assert first_call_kwargs["bbox"]["facecolor"] == "yellow"
        assert first_call_kwargs["bbox"]["alpha"] == 0.9


class TestCustomAnnotations:
    """Test custom annotation functionality."""

    def test_add_custom_annotation_basic(self, annotation_manager, mock_axes):
        """Test basic custom annotation creation."""
        annotation = annotation_manager.add_custom_annotation(
            mock_axes, 5.0, 10.0, "Test annotation"
        )

        assert annotation is not None
        mock_axes.annotate.assert_called_once()

        # Check call arguments
        args, kwargs = mock_axes.annotate.call_args
        assert args[0] == "Test annotation"
        assert kwargs["xy"] == (5.0, 10.0)

    def test_add_custom_annotation_with_preset_offset(
        self, annotation_manager, mock_axes
    ):
        """Test custom annotation with preset offset."""
        annotation_manager.add_custom_annotation(
            mock_axes, 5.0, 10.0, "Test", offset="above_right"
        )

        args, kwargs = mock_axes.annotate.call_args
        expected_offset = annotation_manager.offset_presets["above_right"]
        assert kwargs["xytext"] == expected_offset

    def test_add_custom_annotation_with_tuple_offset(
        self, annotation_manager, mock_axes
    ):
        """Test custom annotation with tuple offset."""
        custom_offset = (15, -5)
        annotation_manager.add_custom_annotation(
            mock_axes, 5.0, 10.0, "Test", offset=custom_offset
        )

        args, kwargs = mock_axes.annotate.call_args
        assert kwargs["xytext"] == custom_offset

    def test_add_custom_annotation_axes_coordinates(
        self, annotation_manager, mock_axes
    ):
        """Test custom annotation with axes coordinates."""
        annotation_manager.add_custom_annotation(
            mock_axes, 0.5, 0.5, "Center", coordinate_system="axes"
        )

        args, kwargs = mock_axes.annotate.call_args
        assert kwargs["transform"] == mock_axes.transAxes

    def test_add_custom_annotation_with_style_overrides(
        self, annotation_manager, mock_axes
    ):
        """Test custom annotation with style overrides."""
        style_overrides = {
            "fontsize": 14,
            "boxstyle": "round,pad=0.5",
            "facecolor": "lightblue",
        }

        annotation_manager.add_custom_annotation(
            mock_axes, 5.0, 10.0, "Test", style_overrides=style_overrides
        )

        args, kwargs = mock_axes.annotate.call_args
        assert kwargs["fontsize"] == 14
        assert kwargs["bbox"]["boxstyle"] == "round,pad=0.5"
        assert kwargs["bbox"]["facecolor"] == "lightblue"


class TestBatchAnnotations:
    """Test batch annotation functionality."""

    def test_add_batch_annotations_basic(self, annotation_manager, mock_axes):
        """Test basic batch annotation creation."""
        annotations = [
            {"x": 1, "y": 10, "text": "First"},
            {"x": 2, "y": 20, "text": "Second"},
            {"x": 3, "y": 30, "text": "Third"},
        ]

        created = annotation_manager.add_batch_annotations(mock_axes, annotations)

        assert len(created) == 3
        assert mock_axes.annotate.call_count == 3

    def test_add_batch_annotations_with_individual_styles(
        self, annotation_manager, mock_axes
    ):
        """Test batch annotations with individual style overrides."""
        annotations = [
            {"x": 1, "y": 10, "text": "Red", "color": "red"},
            {"x": 2, "y": 20, "text": "Blue", "color": "blue", "offset": "below"},
            {"x": 3, "y": 30, "text": "Green", "style": {"fontsize": 16}},
        ]

        annotation_manager.add_batch_annotations(mock_axes, annotations)

        # Check that individual styles were applied
        calls = mock_axes.annotate.call_args_list

        # First annotation: red color
        assert calls[0][1]["bbox"]["edgecolor"] == "red"

        # Second annotation: blue color and below offset
        assert calls[1][1]["bbox"]["edgecolor"] == "blue"
        assert calls[1][1]["xytext"] == annotation_manager.offset_presets["below"]

        # Third annotation: custom fontsize
        assert calls[2][1]["fontsize"] == 16

    def test_add_batch_annotations_with_global_style(
        self, annotation_manager, mock_axes
    ):
        """Test batch annotations with global style overrides."""
        annotations = [
            {"x": 1, "y": 10, "text": "First"},
            {"x": 2, "y": 20, "text": "Second"},
        ]

        global_style = {"fontsize": 12, "alpha": 0.8}

        annotation_manager.add_batch_annotations(
            mock_axes, annotations, global_style_overrides=global_style
        )

        # Check that global style was applied to all annotations
        calls = mock_axes.annotate.call_args_list
        for call in calls:
            assert call[1]["fontsize"] == 12
            assert call[1]["bbox"]["alpha"] == 0.8


class TestArrowAnnotations:
    """Test arrow annotation functionality."""

    def test_add_arrow_annotation_basic(self, annotation_manager, mock_axes):
        """Test basic arrow annotation creation."""
        start_point = (1, 10)
        end_point = (2, 20)

        text_obj, arrow_obj = annotation_manager.add_arrow_annotation(
            mock_axes, start_point, end_point, "Arrow text"
        )

        assert text_obj is not None
        assert arrow_obj is not None
        assert mock_axes.annotate.call_count == 2  # One for arrow, one for text

    def test_add_arrow_annotation_text_positions(self, annotation_manager, mock_axes):
        """Test arrow annotation with different text positions."""
        start_point = (1, 10)
        end_point = (3, 30)

        # Test start position
        annotation_manager.add_arrow_annotation(
            mock_axes, start_point, end_point, "Start", text_position="start"
        )

        # Test end position
        annotation_manager.add_arrow_annotation(
            mock_axes, start_point, end_point, "End", text_position="end"
        )

        # Test middle position (default)
        annotation_manager.add_arrow_annotation(
            mock_axes, start_point, end_point, "Middle", text_position="middle"
        )

        # Should have created 6 annotations total (2 per arrow)
        assert mock_axes.annotate.call_count == 6

    def test_add_arrow_annotation_custom_style(self, annotation_manager, mock_axes):
        """Test arrow annotation with custom styling."""
        style_overrides = {"arrow_color": "red", "arrow_width": 2, "arrow_alpha": 0.5}

        annotation_manager.add_arrow_annotation(
            mock_axes, (1, 10), (2, 20), "Custom arrow", style_overrides=style_overrides
        )

        # Check that arrow style was applied
        calls = mock_axes.annotate.call_args_list
        arrow_call = calls[0]  # First call is for the arrow
        arrow_props = arrow_call[1]["arrowprops"]
        assert arrow_props["color"] == "red"
        assert arrow_props["lw"] == 2
        assert arrow_props["alpha"] == 0.5


class TestAnnotationManagement:
    """Test annotation management operations."""

    def test_remove_specific_annotations(self, annotation_manager, mock_axes):
        """Test removing specific annotations."""
        mock_annotation1 = Mock(spec=Text)
        mock_annotation2 = Mock(spec=Text)
        mock_axes.texts = [mock_annotation1, mock_annotation2]

        annotation_manager.remove_annotations(mock_axes, annotations=[mock_annotation1])

        mock_annotation1.remove.assert_called_once()
        mock_annotation2.remove.assert_not_called()

    def test_remove_all_annotations(self, annotation_manager, mock_axes):
        """Test removing all annotations."""
        mock_annotation1 = Mock(spec=Text)
        mock_annotation2 = Mock(spec=Text)
        mock_axes.texts = [mock_annotation1, mock_annotation2]

        annotation_manager.remove_annotations(mock_axes, annotation_type="all")

        mock_annotation1.remove.assert_called_once()
        mock_annotation2.remove.assert_called_once()

    def test_get_annotation_count(self, annotation_manager, mock_axes):
        """Test getting annotation count."""
        mock_axes.texts = [Mock(), Mock(), Mock()]

        count = annotation_manager.get_annotation_count(mock_axes)
        assert count == 3

    def test_set_default_style(self, annotation_manager):
        """Test updating default style."""
        original_fontsize = annotation_manager.default_style["fontsize"]

        annotation_manager.set_default_style({"fontsize": 16, "alpha": 0.9})

        assert annotation_manager.default_style["fontsize"] == 16
        assert annotation_manager.default_style["alpha"] == 0.9
        # Other values should remain unchanged
        assert annotation_manager.default_style["boxstyle"] == "round,pad=0.3"


class TestAnnotationPositioning:
    """Test annotation positioning and adjustment."""

    def test_adjust_annotation_positions_simple_spread(self, annotation_manager):
        """Test simple spread adjustment of annotations."""
        # Create mock annotations with positions
        mock_ann1 = Mock(spec=Text)
        mock_ann1.get_position.return_value = (1, 10)
        mock_ann1.set_position = Mock()
        mock_ann1.axes = Mock()
        mock_ann1.axes.transData.transform.return_value = [100, 200]

        mock_ann2 = Mock(spec=Text)
        mock_ann2.get_position.return_value = (2, 12)  # Close to first annotation
        mock_ann2.set_position = Mock()
        mock_ann2.axes = Mock()
        mock_ann2.axes.transData.transform.return_value = [
            150,
            210,
        ]  # Close in display coords

        annotations = [mock_ann1, mock_ann2]

        annotation_manager.adjust_annotation_positions(
            annotations, method="simple_spread", spacing=50.0
        )

        # Should attempt to adjust positions
        mock_ann1.get_position.assert_called()
        mock_ann2.get_position.assert_called()

    def test_adjust_annotation_positions_empty_list(self, annotation_manager):
        """Test adjustment with empty annotation list."""
        # Should not raise an error
        annotation_manager.adjust_annotation_positions([], method="simple_spread")

    def test_adjust_annotation_positions_single_annotation(self, annotation_manager):
        """Test adjustment with single annotation."""
        mock_ann = Mock(spec=Text)
        mock_ann.get_position.return_value = (1, 10)

        # Should not raise an error
        annotation_manager.adjust_annotation_positions(
            [mock_ann], method="simple_spread"
        )


class TestUtilityMethods:
    """Test utility and helper methods."""

    def test_merge_styles_with_none(self, annotation_manager):
        """Test style merging with None overrides."""
        merged = annotation_manager._merge_styles(None)
        assert merged == annotation_manager.default_style

    def test_merge_styles_with_overrides(self, annotation_manager):
        """Test style merging with actual overrides."""
        overrides = {"fontsize": 16, "alpha": 0.9}
        merged = annotation_manager._merge_styles(overrides)

        assert merged["fontsize"] == 16
        assert merged["alpha"] == 0.9
        assert merged["boxstyle"] == annotation_manager.default_style["boxstyle"]

    def test_get_series_color_from_lines(self, annotation_manager, mock_axes):
        """Test getting series color from line plots."""
        mock_line = Mock()
        mock_line.get_color.return_value = "red"
        mock_axes.lines = [mock_line]

        color = annotation_manager._get_series_color(mock_axes)
        assert color == "red"

    def test_get_series_color_from_collections(self, annotation_manager, mock_axes):
        """Test getting series color from scatter plots."""
        mock_collection = Mock()
        mock_collection.get_facecolors.return_value = ["blue"]
        mock_axes.lines = []
        mock_axes.collections = [mock_collection]

        color = annotation_manager._get_series_color(mock_axes)
        assert color == "blue"

    def test_get_series_color_no_data(self, annotation_manager, mock_axes):
        """Test getting series color when no data exists."""
        mock_axes.lines = []
        mock_axes.collections = []

        color = annotation_manager._get_series_color(mock_axes)
        assert color is None

    def test_clear_annotation_cache(self, annotation_manager, mock_data_processor):
        """Test clearing annotation cache."""
        annotation_manager.clear_annotation_cache()
        mock_data_processor.clear_cache.assert_called_once()


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_create_annotation_with_exception(self, annotation_manager, mock_axes):
        """Test handling of exceptions during annotation creation."""
        mock_axes.annotate.side_effect = Exception("Mock error")

        # Should not raise exception, but return None
        result = annotation_manager._create_single_annotation(
            mock_axes, 1, 2, "Test", "black", annotation_manager.default_style
        )

        assert result is None

    def test_get_series_color_with_exception(self, annotation_manager, mock_axes):
        """Test getting series color when exception occurs."""
        mock_axes.lines = [Mock()]
        mock_axes.lines[0].get_color.side_effect = Exception("Mock error")

        color = annotation_manager._get_series_color(mock_axes)
        assert color is None

    def test_data_processor_without_clear_cache(self, annotation_manager):
        """Test cache clearing when data processor doesn't have clear_cache method."""
        annotation_manager.data_processor = Mock()
        del annotation_manager.data_processor.clear_cache  # Remove the method

        # Should not raise an error
        annotation_manager.clear_annotation_cache()


class TestIntegrationWithRealMatplotlib:
    """Integration tests with real matplotlib objects."""

    def test_with_real_matplotlib_axes(self, annotation_manager):
        """Test with real matplotlib axes object."""
        fig, ax = plt.subplots()

        # Create some sample data
        x = [1, 2, 3]
        y = [10, 20, 30]
        ax.plot(x, y, "o-", color="blue")

        # Add annotation
        annotation = annotation_manager.add_custom_annotation(
            ax, 2, 20, "Test Point", offset="above"
        )

        assert annotation is not None
        assert len(ax.texts) == 1
        assert ax.texts[0].get_text() == "Test Point"

        plt.close(fig)

    def test_with_real_dataframe(self, annotation_manager, mock_data_processor):
        """Test with real DataFrame and axes."""
        fig, ax = plt.subplots()

        df = pd.DataFrame(
            {"x": [1, 2, 3], "y": [10, 20, 30], "labels": ["A", "B", "C"]}
        )

        # Plot some data first
        ax.plot(df["x"], df["y"], "o-")

        # Add annotations
        annotations = annotation_manager.add_data_point_annotations(
            ax, df, "x", "y", "labels", annotation_label="Point: "
        )

        assert len(annotations) == 3
        assert len(ax.texts) == 3

        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
