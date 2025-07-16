"""
Plot annotation management for visualization components.

This module provides the PlotAnnotationManager class for handling all
types of plot annotations including data point annotations, text labels,
arrows, and other visual annotations.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.collections import Collection
from matplotlib.colors import rgb2hex


class PlotAnnotationManager:
    """
    Manages all forms of plot annotations including data points, text,
    and visual elements.

    This class handles:
    - Data point annotations with values from DataFrame columns
    - Custom text annotations at specific coordinates
    - Arrow annotations connecting points
    - Annotation formatting and styling
    - Batch annotation operations
    - Annotation collision detection and adjustment

    The manager integrates with PlotDataProcessor for data preparation
    and supports both automatic and manual annotation placement.
    """

    def __init__(self, data_processor=None):
        """
        Initialize the annotation manager.

        Parameters:
        -----------
        data_processor : PlotDataProcessor, optional
            Data processor for preparing annotation data. If None,
            creates a new one.
        """
        if data_processor is None:
            # Import here to avoid circular imports
            from .data_processor import PlotDataProcessor

            self.data_processor = PlotDataProcessor()
        else:
            self.data_processor = data_processor

        # Default annotation styles
        self.default_style = {
            "fontsize": 8,
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.7,
            "edgecolor": "black",
            "offset": (0, 10),  # (x, y) offset in points
            "arrow_style": "-",
            "arrow_color": "black",
            "arrow_width": 1,
            "arrow_alpha": 0.8,
            "text_alignment": ("center", "bottom"),
        }

        # Annotation positioning presets
        self.offset_presets = {
            "above": (0, 10),
            "below": (0, -10),
            "left": (-10, 0),
            "right": (10, 0),
            "above_left": (-10, 10),
            "above_right": (10, 10),
            "below_left": (-10, -10),
            "below_right": (10, -10),
        }

    def add_data_point_annotations(
        self,
        ax: Axes,
        dataframe: pd.DataFrame,
        x_variable: str,
        y_variable: str,
        annotation_variable: str,
        annotation_label: str = "",
        annotation_range: Optional[Tuple[int, Optional[int], int]] = None,
        style_overrides: Optional[Dict[str, Any]] = None,
        color_by_series: bool = True,
        series_color: str = "black",
        format_func: Optional[Callable[[Any], str]] = None,
    ) -> List[Text]:
        """
        Add annotations to data points using values from a DataFrame
        column.

        Parameters:
        -----------
        - ax : matplotlib.axes.Axes
            The axes to add annotations to
        - dataframe : pd.DataFrame
            DataFrame containing the data
        - x_variable : str
            Column name for x-coordinates
        - y_variable : str
            Column name for y-coordinates
        - annotation_variable : str
            Column name containing values to annotate
        - annotation_label : str, optional
            Prefix text for annotations (e.g., "N=" shows as "N=10")
        - annotation_range : tuple, optional
            Controls which points to annotate: (start, end, step)
        - style_overrides : dict, optional
            Style parameters to override defaults
        - color_by_series : bool, optional
            Whether to match annotation color to data series color
        - series_color : str, optional
            Color to use if color_by_series is False
        - format_func : callable, optional
            Custom function to format annotation values

        Returns:
        --------
        list
            List of matplotlib Text objects created
        """
        # Get prepared annotation data
        annotation_data = self.data_processor.prepare_annotation_data(
            dataframe, x_variable, y_variable, annotation_variable, annotation_range
        )

        if not annotation_data:
            return []

        # Merge styles
        style = self._merge_styles(style_overrides)

        # Format function
        if format_func is None:
            format_func = self.data_processor.format_annotation_value

        created_annotations = []

        for x_coord, y_coord, value in annotation_data:
            # Format the annotation text
            formatted_value = format_func(value)
            annotation_text = f"{annotation_label}{formatted_value}"

            # Determine color
            if color_by_series:
                # Try to get color from the current line/scatter plot
                edge_color = self._get_series_color(ax) or series_color
            else:
                edge_color = series_color

            # Create annotation
            annotation = self._create_single_annotation(
                ax, x_coord, y_coord, annotation_text, edge_color, style
            )

            if annotation:
                created_annotations.append(annotation)

        return created_annotations

    def add_custom_annotation(
        self,
        ax: Axes,
        x: float,
        y: float,
        text: str,
        offset: Union[str, Tuple[float, float], None] = "above",
        style_overrides: Optional[Dict[str, Any]] = None,
        color: str = "black",
        coordinate_system: str = "data",
    ) -> Optional[Text]:
        """
        Add a custom annotation at specified coordinates.

        Parameters:
        -----------
        - ax : matplotlib.axes.Axes
            The axes to add annotation to
        - x, y : float
            Coordinates for the annotation
        - text : str
            Text content of the annotation
        - offset : str or tuple
            Offset direction ('above', 'below', etc.) or (x, y) tuple in
            points
        - style_overrides : dict, optional
            Style parameters to override defaults
        - color : str, optional
            Color for the annotation box edge
        - coordinate_system : str, optional
            'data' for data coordinates, 'axes' for axes coordinates

        Returns:
        --------
        matplotlib.text.Text or None
            The created annotation object
        """
        # Resolve offset
        if isinstance(offset, str):
            offset = self.offset_presets.get(offset, self.default_style["offset"])

        # Merge styles
        style = self._merge_styles(style_overrides)
        style["offset"] = offset

        return self._create_single_annotation(
            ax, x, y, text, color, style, coordinate_system
        )

    def add_batch_annotations(
        self,
        ax: Axes,
        annotations: List[Dict[str, Any]],
        global_style_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[Text]:
        """
        Add multiple annotations from a list of annotation
        specifications.

        Parameters:
        -----------
        - ax : matplotlib.axes.Axes
            The axes to add annotations to
        - annotations : list of dict
            List of annotation specifications, each containing:
            - 'x', 'y': coordinates
            - 'text': annotation text
            - 'offset': optional offset (str or tuple)
            - 'color': optional color
            - 'style': optional style overrides
        - global_style_overrides : dict, optional
            Global style overrides applied to all annotations

        Returns:
        --------
        list
            List of created annotation objects
        """
        created_annotations = []

        for ann_spec in annotations:
            # Extract required parameters
            x = ann_spec["x"]
            y = ann_spec["y"]
            text = ann_spec["text"]

            # Extract optional parameters
            offset = ann_spec.get("offset", "above")
            color = ann_spec.get("color", "black")
            local_style = ann_spec.get("style", {})

            # Merge all style layers
            combined_style = self._merge_styles(global_style_overrides)
            combined_style.update(local_style)

            # Create annotation
            annotation = self.add_custom_annotation(
                ax, x, y, text, offset, combined_style, color
            )

            if annotation:
                created_annotations.append(annotation)

        return created_annotations

    def add_arrow_annotation(
        self,
        ax: Axes,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        text: str,
        text_position: str = "middle",
        arrow_style: str = "->",
        style_overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Text], Any]:
        """
        Add an arrow annotation between two points.

        Parameters:
        -----------
        - ax : matplotlib.axes.Axes
            The axes to add annotation to
        - start_point : tuple
            (x, y) coordinates of arrow start
        - end_point : tuple
            (x, y) coordinates of arrow end
        - text : str
            Text to display along the arrow
        - text_position : str, optional
            Position for text: 'start', 'middle', 'end'
        - arrow_style : str, optional
            Arrow style string for matplotlib
        - style_overrides : dict, optional
            Style parameters to override defaults

        Returns:
        --------
        tuple
            (text_object, arrow_object) - either can be None if creation
            failed
        """
        style = self._merge_styles(style_overrides)

        # Create arrow
        arrow = ax.annotate(
            "",
            xy=end_point,
            xytext=start_point,
            arrowprops=dict(
                arrowstyle=arrow_style,
                color=style.get("arrow_color", "black"),
                lw=style.get("arrow_width", 1),
                alpha=style.get("arrow_alpha", 0.8),
            ),
        )

        # Determine text position
        if text_position == "start":
            text_pos = start_point
        elif text_position == "end":
            text_pos = end_point
        else:  # middle
            text_pos = (
                (start_point[0] + end_point[0]) / 2,
                (start_point[1] + end_point[1]) / 2,
            )

        # Create text annotation
        text_obj = self._create_single_annotation(
            ax, text_pos[0], text_pos[1], text, style.get("arrow_color", "black"), style
        )

        return text_obj, arrow

    def adjust_annotation_positions(
        self,
        annotations: List[Text],
        method: str = "simple_spread",
        spacing: float = 20.0,
    ) -> None:
        """
        Adjust annotation positions to avoid overlaps.

        Parameters:
        -----------
        - annotations : list
            List of matplotlib Text objects to adjust
        - method : str, optional
            Method for adjustment: 'simple_spread', 'smart_offset'
        - spacing : float, optional
            Minimum spacing between annotations in points
        """
        if method == "simple_spread":
            self._simple_spread_adjustment(annotations, spacing)
        elif method == "smart_offset":
            self._smart_offset_adjustment(annotations, spacing)

    def remove_annotations(
        self,
        ax: Axes,
        annotations: Optional[List[Text]] = None,
        annotation_type: str = "all",
    ) -> None:
        """
        Remove annotations from the plot.

        Parameters:
        -----------
        - ax : matplotlib.axes.Axes
            The axes to remove annotations from
        - annotations : list, optional
            Specific annotations to remove. If None, removes based on
            type.
        - annotation_type : str, optional
            Type of annotations to remove: 'all', 'text', 'arrows'
        """
        if annotations:
            # Remove specific annotations
            for ann in annotations:
                if ann in ax.texts:
                    ann.remove()
        else:
            # Remove by type
            if annotation_type in ("all", "text"):
                for text in ax.texts[
                    :
                ]:  # Copy list to avoid modification during iteration
                    text.remove()

            if annotation_type in ("all", "arrows"):
                # Remove arrow annotations (these are stored differently)
                for child in ax.get_children()[:]:
                    if hasattr(child, "arrow_patch"):
                        child.remove()

    def _create_single_annotation(
        self,
        ax: Axes,
        x: float,
        y: float,
        text: str,
        edge_color: str,
        style: Dict[str, Any],
        coordinate_system: str = "data",
    ) -> Optional[Text]:
        """
        Create a single annotation with the specified style.

        Parameters:
        -----------
        - ax : matplotlib.axes.Axes
            The axes to add annotation to
        - x, y : float
            Coordinates for the annotation
        - text : str
            Text content
        - edge_color : str
            Color for the annotation box edge
        - style : dict
            Style parameters
        - coordinate_system : str
            'data' for data coordinates, 'axes' for axes coordinates

        Returns:
        --------
        matplotlib.text.Text or None
            Created annotation object
        """
        try:
            # Determine coordinate transform
            if coordinate_system == "axes":
                transform = ax.transAxes
            else:
                transform = ax.transData

            # Create annotation
            annotation = ax.annotate(
                text,
                xy=(x, y),
                xytext=style["offset"],
                textcoords="offset points",
                transform=transform,
                fontsize=style["fontsize"],
                ha=style["text_alignment"][0],
                va=style["text_alignment"][1],
                bbox=dict(
                    boxstyle=style["boxstyle"],
                    facecolor=style["facecolor"],
                    alpha=style["alpha"],
                    edgecolor=edge_color,
                ),
                arrowprops=(
                    dict(
                        arrowstyle=style["arrow_style"],
                        color=style["arrow_color"],
                        lw=style["arrow_width"],
                        alpha=style["arrow_alpha"],
                    )
                    if style["arrow_style"] != "none"
                    else None
                ),
            )

            return annotation

        except Exception as e:
            print(f"Warning: Failed to create annotation '{text}': {e}")
            return None

    def _merge_styles(self, overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge style overrides with default style.

        Parameters:
        -----------
        overrides : dict or None
            Style overrides to apply

        Returns:
        --------
        dict
            Merged style dictionary
        """
        merged = self.default_style.copy()
        if overrides:
            merged.update(overrides)
        return merged

    def _get_series_color(self, ax: Axes) -> Optional[str]:
        """
        Attempt to get the color of the most recently added data series.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to examine

        Returns:
        --------
        str or None
            Color of the most recent series as hex string, or None if cannot determine
        """
        try:
            # Try to get color from the last line
            if ax.lines:
                color = ax.lines[-1].get_color()
                return rgb2hex(color) if color is not None else None

            # Try to get color from the last scatter plot
            if hasattr(ax, "collections") and ax.collections:
                collection: Collection = ax.collections[-1]
                color = collection.get_facecolor()
                if color.size > 0:  # Check if we got any colors
                    return rgb2hex(color[0])  # Convert first color to hex

            return None

        except Exception:
            return None

    def _simple_spread_adjustment(
        self, annotations: List[Text], spacing: float
    ) -> None:
        """
        Simple vertical spreading of overlapping annotations.

        Parameters:
        -----------
        - annotations : list
            List of annotations to adjust
        - spacing : float
            Minimum spacing in points
        """
        if len(annotations) < 2:
            return

        # Filter out annotations without valid axes
        valid_annotations = []
        for ann in annotations:
            if ann.axes is not None and hasattr(ann.axes, "transData"):
                valid_annotations.append(ann)

        if len(valid_annotations) < 2:
            return

        # Sort the filtered annotations instead of the original list
        sorted_annotations = sorted(
            valid_annotations, key=lambda a: a.get_position()[1]
        )

        # Adjust positions to maintain minimum spacing
        for i in range(1, len(sorted_annotations)):
            prev_ann = sorted_annotations[i - 1]
            curr_ann = sorted_annotations[i]

            prev_pos = prev_ann.get_position()
            curr_pos = curr_ann.get_position()

            try:
                # Convert to display coordinates for spacing calculation
                prev_display = prev_ann.axes.transData.transform(prev_pos)
                curr_display = curr_ann.axes.transData.transform(curr_pos)

                if abs(curr_display[1] - prev_display[1]) < spacing:
                    # Adjust current annotation
                    new_y = prev_pos[1] + (
                        spacing / 72.0
                    )  # Convert points to data units
                    curr_ann.set_position((curr_pos[0], new_y))

            except (AttributeError, ValueError) as e:
                continue

    def _smart_offset_adjustment(self, annotations: List[Text], spacing: float) -> None:
        """
        Smart adjustment that tries to minimize overlaps while keeping
        annotations readable.

        Parameters:
        -----------
        - annotations : list
            List of annotations to adjust
        - spacing : float
            Minimum spacing in points
        """
        # This is a placeholder for a more sophisticated algorithm
        # that could consider:
        # - Annotation box sizes
        # - Data point density
        # - Optimal offset directions
        # - Collision detection

        # For now, fall back to simple spreading
        self._simple_spread_adjustment(annotations, spacing)

    def set_default_style(self, style_updates: Dict[str, Any]) -> None:
        """
        Update the default annotation style.

        Parameters:
        -----------
        style_updates : dict
            Style parameters to update in defaults
        """
        self.default_style.update(style_updates)

    def get_annotation_count(self, ax: Axes) -> int:
        """
        Get the number of annotations on the axes.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to count annotations for

        Returns:
        --------
        int
            Number of text annotations
        """
        return len(ax.texts)

    def clear_annotation_cache(self) -> None:
        """Clear any internal caches used by the annotation manager."""
        if hasattr(self.data_processor, "clear_cache"):
            self.data_processor.clear_cache()
