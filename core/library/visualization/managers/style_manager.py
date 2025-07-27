"""
Plot style management for visualization components.

This module provides the PlotStyleManager class for managing visual styling
aspects of plots including markers, colors, legends, and axes formatting.
"""

from typing import Dict, List, Optional, Tuple, Union, Any

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties


class PlotStyleManager:
    """
    Manages visual styling for plots including markers, colors, legends, and
    axes.

    This class centralizes all styling decisions for plots, providing consistent
    visual appearance across different plot types. It handles:
    - Marker and color assignment for data series
    - Legend configuration and formatting
    - Axes scaling and formatting
    - Grid and tick configuration

    Attributes:
        constants: Module containing styling constants (MARKER_STYLES,
        DEFAULT_COLORS, etc.) _style_cache: Cache for computed style mappings
    """

    def __init__(self, constants_module):
        """
        Initialize the style manager with constants module.

        Args:
            constants_module: Module containing style constants like
            MARKER_STYLES, DEFAULT_COLORS, PARAMETERS_WITH_EXPONENTIAL_FORMAT,
            etc.
        """
        self.constants = constants_module
        self._style_cache = {}

    def generate_marker_color_map(
        self,
        grouping_values: List[Any],
        custom_map: Optional[Dict[Any, Tuple[str, str]]] = None,
        index_shift: int = 0,
    ) -> Dict[Any, Tuple[str, str]]:
        """
        Generate a stable mapping from grouping values to (marker, color) pairs.

        This method ensures consistent visual mapping across plots by sorting
        values and assigning styles in a predictable manner.

        Args:
            - grouping_values: List of unique values of the grouping variable.
            - custom_map: Optional custom mapping from value to (marker, color).
              Values not included will be auto-assigned.
            - index_shift: Integer to shift the style assignment (useful for
              creating different style sets for the same values).

        Returns:
            Dict mapping each value to a (marker, color) tuple.

        Example:
            >>> manager = PlotStyleManager(constants)
            >>> values = ['A', 'B', 'C']
            >>> styles = manager.generate_marker_color_map(values)
            >>> # Returns: {'A': ('o', 'blue'), 'B': ('s', 'green'), ...}
        """
        # Create cache key
        cache_key = (
            tuple(sorted(grouping_values, key=lambda x: str(x))),
            tuple(custom_map.items()) if custom_map else None,
            index_shift,
        )

        # Check cache
        if cache_key in self._style_cache:
            return self._style_cache[cache_key]

        # Generate mapping
        sorted_values = sorted(grouping_values, key=lambda x: str(x))
        num_markers = len(self.constants.MARKER_STYLES)
        num_colors = len(self.constants.DEFAULT_COLORS)

        style_map = {}

        for idx, value in enumerate(sorted_values):
            if custom_map and value in custom_map:
                style_map[value] = custom_map[value]
            else:
                marker = self.constants.MARKER_STYLES[(idx + index_shift) % num_markers]
                color = self.constants.DEFAULT_COLORS[(idx + index_shift) % num_colors]
                style_map[value] = (marker, color)

        # Cache result
        self._style_cache[cache_key] = style_map
        return style_map

    def configure_axes_style(
        self,
        ax: Axes,
        xaxis_variable: str,
        yaxis_variable: str,
        xaxis_label: Optional[str] = None,
        yaxis_label: Optional[str] = None,
        xaxis_log_scale: bool = False,
        yaxis_log_scale: bool = False,
        invert_xaxis: bool = False,
        invert_yaxis: bool = False,
        font_size: int = 13,
        grid_enabled: bool = True,
        grid_style: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Configure axes styling including labels, scales, and grid.

        Args:
            - ax: Matplotlib axes to configure.
            - xaxis_variable: Name of x-axis variable (for determining
              formatting).
            - yaxis_variable: Name of y-axis variable (for determining
              formatting).
            - xaxis_label: Custom x-axis label (uses default if None).
            - yaxis_label: Custom y-axis label (uses default if None).
            - xaxis_log_scale: Force logarithmic scale on x-axis.
            - yaxis_log_scale: Force logarithmic scale on y-axis.
            - invert_xaxis: Whether to invert x-axis.
            - invert_yaxis: Whether to invert y-axis.
            - font_size: Base font size for labels and ticks.
            - grid_enabled: Whether to show grid.
            - grid_style: Custom grid style dict (uses defaults if None).
        """
        # Configure grid
        if grid_enabled:
            grid_params = grid_style or {"linestyle": "--", "alpha": 0.5}
            ax.grid(True, **grid_params)

        # Set axis labels
        if xaxis_label is None:
            xaxis_label = self.constants.AXES_LABELS_BY_COLUMN_NAME.get(
                xaxis_variable, xaxis_variable
            )
        ax.set_xlabel(xaxis_label or "", fontsize=font_size + 2)

        if yaxis_label is None:
            yaxis_label = self.constants.AXES_LABELS_BY_COLUMN_NAME.get(
                yaxis_variable, yaxis_variable
            )
        ax.set_ylabel(yaxis_label or "", fontsize=font_size + 2)

        # Configure tick parameters
        ax.tick_params(axis="both", labelsize=font_size)

        # Set scales (log scale if variable is in exponential format or explicitly requested)
        if (
            xaxis_variable in self.constants.PARAMETERS_WITH_EXPONENTIAL_FORMAT
            or xaxis_log_scale
        ):
            ax.set_xscale("log")

        if (
            yaxis_variable in self.constants.PARAMETERS_WITH_EXPONENTIAL_FORMAT
            or yaxis_log_scale
        ):
            ax.set_yscale("log")

        # Set integer locators for integer-valued parameters
        if xaxis_variable in self.constants.PARAMETERS_OF_INTEGER_VALUE:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if yaxis_variable in self.constants.PARAMETERS_OF_INTEGER_VALUE:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Handle axis inversion
        if invert_xaxis:
            ax.invert_xaxis()
        if invert_yaxis:
            ax.invert_yaxis()

    def configure_legend(
        self,
        ax: Axes,
        include_legend: bool = True,
        legend_location: str = "upper left",
        legend_columns: int = 1,
        include_legend_title: bool = True,
        legend_title: Optional[str] = None,
        font_size: int = 13,
        grouping_variable: Optional[Union[str, List[str]]] = None,
        labeling_variable: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """
        Configure legend appearance and placement.

        Args:
            - ax: Matplotlib axes containing the legend.
            - include_legend: Whether to show legend at all.
            - legend_location: Legend position string or tuple.
            - legend_columns: Number of columns in legend.
            - include_legend_title: Whether to include a title on the legend.
            - legend_title: Custom legend title (auto-generated if None).
            - font_size: Base font size for legend.
            - grouping_variable: Variable used for grouping (for title
              generation).
            - labeling_variable: Variable used for labeling (for title
              generation).
        """
        if not include_legend:
            return

        legend = ax.legend(loc=legend_location, fontsize=font_size, ncol=legend_columns)

        if include_legend_title and legend:
            if legend_title is None:
                # Auto-generate title based on labeling or grouping variable
                title_var = (
                    labeling_variable if labeling_variable else grouping_variable
                )

                if isinstance(title_var, list):
                    # Handle list of variables
                    title_var = title_var[0] if title_var else None

                if title_var:
                    legend_title = (
                        self.constants.LEGEND_LABELS_BY_COLUMN_NAME.get(
                            title_var, title_var
                        )
                        or ""
                    )
                    # Clean up title if not LaTeX
                    if "$" not in legend_title:
                        legend_title = legend_title.replace("_", " ")

            if legend_title:
                font_props = FontProperties(size=font_size + 1)
                legend.set_title(legend_title, prop=font_props)

    def set_axis_limits(
        self,
        ax: Axes,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        xaxis_start_at_zero: bool = False,
        yaxis_start_at_zero: bool = False,
    ) -> None:
        """
        Set axis limits with support for starting at zero.

        Args:
            - ax: Matplotlib axes to configure.
            - xlim: Explicit x-axis limits as (min, max) tuple.
            - ylim: Explicit y-axis limits as (min, max) tuple.
            - xaxis_start_at_zero: Force x-axis to start at 0.
            - yaxis_start_at_zero: Force y-axis to start at 0.
        """
        # Set x-axis limits
        if xlim is not None:
            ax.set_xlim(xlim)
        elif xaxis_start_at_zero:
            current_xlim = ax.get_xlim()
            ax.set_xlim(left=0, right=current_xlim[1])

        # Set y-axis limits
        if ylim is not None:
            ax.set_ylim(ylim)
        elif yaxis_start_at_zero:
            current_ylim = ax.get_ylim()
            ax.set_ylim(bottom=0, top=current_ylim[1])

    def format_legend_value(
        self, value: Union[int, float, str], format_spec: str = ".2f"
    ) -> str:
        """
        Format a value for display in legends.

        Args:
            - value: Value to format.
            - format_spec: Format specification string for numeric values.

        Returns:
            Formatted string representation of the value.
        """
        if isinstance(value, (int, float)):
            return format(value, format_spec)
        return str(value)

    def get_marker_properties(
        self,
        marker: str,
        filled: bool = True,
        color: str = "blue",
        size: int = 8,
        plot_type: str = "errorbar",
    ) -> Dict[str, Any]:
        """
        Get marker properties dict for scatter/errorbar plots with
        consistent visual sizing.

        The key fix: Convert marker sizes so they appear visually
        equivalent between scatter plots (area-based) and errorbar plots
        (diameter-based).

        Args:
            - marker: Marker style string.
            - filled: Whether marker should be filled.
            - color: Marker color.
            - size: Marker size (this will be converted appropriately
              for each plot type).
            - plot_type: "scatter" or "errorbar" to determine correct
              size parameter.

        Returns:
            Dict of marker properties suitable for matplotlib functions.
        """
        # Convert size for visual equivalence
        if plot_type == "scatter":
            # For scatter: convert diameter to area for visual
            # equivalence. Formula: area = π * (diameter/2)². But
            # matplotlib scatter uses a simpler conversion that looks
            # good
            visual_size = self._convert_size_for_scatter(size)
            size_key = "s"
        else:  # errorbar
            # For errorbar: use size directly as diameter
            visual_size = size
            size_key = "markersize"

        # Rest of the method stays the same, but use visual_size
        if plot_type == "scatter":
            # For scatter plots, use 'c' and avoid 'color'
            if filled:
                base_props = {
                    "marker": marker,
                    "c": color,
                    "edgecolors": color,
                    size_key: visual_size,
                }
            else:
                base_props = {
                    "marker": marker,
                    "facecolors": "none",
                    "edgecolors": color,
                    "c": color,
                    size_key: visual_size,
                }
        else:  # errorbar
            # For errorbar plots, use 'color' and marker properties
            if filled:
                base_props = {
                    "marker": marker,
                    "color": color,
                    "markerfacecolor": color,
                    "markeredgecolor": color,
                    size_key: visual_size,
                }
            else:
                base_props = {
                    "marker": marker,
                    "color": color,
                    "markerfacecolor": "none",
                    "markeredgecolor": color,
                    size_key: visual_size,
                }

        return base_props

    def _convert_size_for_scatter(self, errorbar_size: int) -> float:
        """
        Convert errorbar markersize to equivalent scatter size for
        visual consistency.
        
        Args:
            errorbar_size: The size that would be used for errorbar
            markersize
            
        Returns:
            Equivalent size for scatter plot 's' parameter
            
        The conversion aims to make markers appear the same visual size
        between scatter and errorbar plots.
        """
        # Method 1: Mathematical conversion (diameter to area)
        # scatter_area = π * (diameter/2)². But this often looks too big,
        # so we use a calibrated conversion
        
        # Method 2: Empirically calibrated conversion (recommended)
        # Based on visual testing, this gives good equivalence:
        return errorbar_size ** 1.8
        
        # Alternative conversions you could try:
        # return errorbar_size ** 2      # Pure area conversion (often too big)
        # return errorbar_size * 6.25    # Linear approximation
        # return errorbar_size * errorbar_size * 0.7  # Slightly smaller area

    def apply_figure_margins(
        self,
        fig: Figure,
        left: float = 0.15,
        right: float = 0.94,
        bottom: float = 0.12,
        top: float = 0.92,
    ) -> None:
        """
        Apply margin adjustments to a figure.

        Args:
            - fig: Matplotlib figure to adjust.
            - left: Left margin as fraction of figure width.
            - right: Right margin as fraction of figure width.
            - bottom: Bottom margin as fraction of figure height.
            - top: Top margin as fraction of figure height.
        """
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    def clear_cache(self) -> None:
        """Clear the internal style cache."""
        self._style_cache.clear()
