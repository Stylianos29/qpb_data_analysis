from typing import Optional, Tuple, Callable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator


class PlotLayoutManager:
    """
    Manages figure layout, axes properties, and visual formatting for plots.

    This class handles all aspects of plot layout including:
    - Figure creation and sizing
    - Axes labels and scaling
    - Margins and subplot adjustments
    - Axis limits and orientations
    - Tick formatting
    """

    def __init__(self, constants_module=None):
        """
        Initialize the layout manager.

        Parameters:
        -----------
        constants_module : module, optional
            Module containing constants like AXES_LABELS_BY_COLUMN_NAME,
            PARAMETERS_WITH_EXPONENTIAL_FORMAT, etc.
        """
        self.constants = constants_module

        # Default layout settings
        self.default_figure_size = (7, 5)
        self.default_font_size = 13
        self.default_margins = {
            "left": 0.15,
            "right": 0.94,
            "bottom": 0.12,
            "top": 0.92,
        }

    def create_figure(
        self, figure_size: Optional[Tuple[float, float]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Create a new figure and axes with the specified size.

        Parameters:
        -----------
        figure_size : tuple, optional
            (width, height) in inches. Uses default if None.

        Returns:
        --------
        tuple: (figure, axes) objects
        """
        size = figure_size or self.default_figure_size
        fig, ax = plt.subplots(figsize=size)

        # Add grid by default
        ax.grid(True, linestyle="--", alpha=0.5)

        return fig, ax

    def _setup_axes_labels(
        self,
        ax: Axes,
        x_variable: str,
        y_variable: str,
        xaxis_label: Optional[str] = None,
        yaxis_label: Optional[str] = None,
        font_size: Optional[int] = None,
    ) -> None:
        """
        Set up x and y axis labels.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to configure
        x_variable : str
            Name of x-axis variable (for automatic label lookup)
        y_variable : str
            Name of y-axis variable (for automatic label lookup)
        xaxis_label : str, optional
            Custom x-axis label. If None, looks up from constants.
        yaxis_label : str, optional
            Custom y-axis label. If None, looks up from constants.
        font_size : int, optional
            Font size for labels. Uses default if None.
        """
        font_size = font_size or self.default_font_size

        # Determine x-axis label
        if xaxis_label is None and self.constants:
            xaxis_label = getattr(self.constants, "AXES_LABELS_BY_COLUMN_NAME", {}).get(
                x_variable, x_variable
            )
        elif xaxis_label is None:
            xaxis_label = x_variable

        # Determine y-axis label
        if yaxis_label is None and self.constants:
            yaxis_label = getattr(self.constants, "AXES_LABELS_BY_COLUMN_NAME", {}).get(
                y_variable, y_variable
            )
        elif yaxis_label is None:
            yaxis_label = y_variable

        ax.set_xlabel(xaxis_label or "", fontsize=font_size + 2)
        ax.set_ylabel(yaxis_label or "", fontsize=font_size + 2)
        ax.tick_params(axis="both", labelsize=font_size)

    def _setup_axes_scaling(
        self,
        ax: Axes,
        x_variable: str,
        y_variable: str,
        xaxis_log_scale: bool = False,
        yaxis_log_scale: bool = False,
        auto_log_scale: bool = True,
    ) -> None:
        """
        Configure axes scaling (linear vs log).

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to configure
        x_variable : str
            Name of x-axis variable
        y_variable : str
            Name of y-axis variable
        xaxis_log_scale : bool
            Force x-axis to log scale
        yaxis_log_scale : bool
            Force y-axis to log scale
        auto_log_scale : bool
            Automatically use log scale for variables in PARAMETERS_WITH_EXPONENTIAL_FORMAT
        """
        # Check for automatic log scaling
        if auto_log_scale and self.constants:
            exp_params = getattr(
                self.constants, "PARAMETERS_WITH_EXPONENTIAL_FORMAT", set()
            )
            if x_variable in exp_params or xaxis_log_scale:
                ax.set_xscale("log")
            if y_variable in exp_params or yaxis_log_scale:
                ax.set_yscale("log")
        else:
            if xaxis_log_scale:
                ax.set_xscale("log")
            if yaxis_log_scale:
                ax.set_yscale("log")

    def _setup_integer_ticks(self, ax: Axes, x_variable: str, y_variable: str) -> None:
        """
        Configure integer-only ticks for appropriate variables.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to configure
        x_variable : str
            Name of x-axis variable
        y_variable : str
            Name of y-axis variable
        """
        if not self.constants:
            return

        int_params = getattr(self.constants, "PARAMETERS_OF_INTEGER_VALUE", set())

        if x_variable in int_params:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if y_variable in int_params:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def _setup_axes_limits_and_orientation(
        self,
        ax: Axes,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        xaxis_start_at_zero: bool = False,
        yaxis_start_at_zero: bool = False,
        invert_xaxis: bool = False,
        invert_yaxis: bool = False,
    ) -> None:
        """
        Configure axis limits and orientations.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to configure
        xlim : tuple, optional
            (min, max) for x-axis
        ylim : tuple, optional
            (min, max) for y-axis
        xaxis_start_at_zero : bool
            Force x-axis to start at zero
        yaxis_start_at_zero : bool
            Force y-axis to start at zero
        invert_xaxis : bool
            Invert x-axis direction
        invert_yaxis : bool
            Invert y-axis direction
        """
        # Set explicit limits if provided
        if xlim is not None:
            ax.set_xlim(xlim)
        elif xaxis_start_at_zero:
            current_xlim = ax.get_xlim()
            ax.set_xlim(left=0, right=current_xlim[1])

        if ylim is not None:
            ax.set_ylim(ylim)
        elif yaxis_start_at_zero:
            current_ylim = ax.get_ylim()
            ax.set_ylim(bottom=0, top=current_ylim[1])

        # Handle axis inversion
        if invert_xaxis:
            ax.invert_xaxis()
        if invert_yaxis:
            ax.invert_yaxis()

    def _adjust_margins(
        self,
        fig: Figure,
        left: Optional[float] = None,
        right: Optional[float] = None,
        bottom: Optional[float] = None,
        top: Optional[float] = None,
    ) -> None:
        """
        Adjust figure margins.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            The figure to adjust
        left, right, bottom, top : float, optional
            Margin values (0-1). Uses defaults if None.
        """
        margins = {
            "left": left or self.default_margins["left"],
            "right": right or self.default_margins["right"],
            "bottom": bottom or self.default_margins["bottom"],
            "top": top or self.default_margins["top"],
        }

        fig.subplots_adjust(**margins)

    def setup_complete_layout(
        self,
        x_variable: str,
        y_variable: str,
        figure_size: Optional[Tuple[float, float]] = None,
        font_size: Optional[int] = None,
        xaxis_label: Optional[str] = None,
        yaxis_label: Optional[str] = None,
        xaxis_log_scale: bool = False,
        yaxis_log_scale: bool = False,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        xaxis_start_at_zero: bool = False,
        yaxis_start_at_zero: bool = False,
        invert_xaxis: bool = False,
        invert_yaxis: bool = False,
        left_margin: Optional[float] = None,
        right_margin: Optional[float] = None,
        bottom_margin: Optional[float] = None,
        top_margin: Optional[float] = None,
        apply_custom_function: Optional[Callable] = None,
    ) -> Tuple[Figure, Axes]:
        """
        One-stop method to create and fully configure a plot layout.

        This method creates the figure, sets up all axes properties, and returns
        a ready-to-use figure and axes pair.

        Parameters:
        -----------
        x_variable : str
            Name of x-axis variable
        y_variable : str
            Name of y-axis variable
        ... (all other parameters from individual methods)
        apply_custom_function : callable, optional
            Custom function to apply additional modifications to axes

        Returns:
        --------
        tuple: (figure, axes) ready for plotting
        """
        # Create figure
        fig, ax = self.create_figure(figure_size)

        # Setup all axes properties
        self._setup_axes_labels(
            ax, x_variable, y_variable, xaxis_label, yaxis_label, font_size
        )
        self._setup_axes_scaling(
            ax, x_variable, y_variable, xaxis_log_scale, yaxis_log_scale
        )
        self._setup_integer_ticks(ax, x_variable, y_variable)
        self._setup_axes_limits_and_orientation(
            ax,
            xlim,
            ylim,
            xaxis_start_at_zero,
            yaxis_start_at_zero,
            invert_xaxis,
            invert_yaxis,
        )

        # Apply custom modifications if provided
        if apply_custom_function is not None:
            apply_custom_function(ax)

        # Adjust margins
        self._adjust_margins(fig, left_margin, right_margin, bottom_margin, top_margin)

        return fig, ax

    # Example usage and integration pattern:
    def configure_existing_axes(
        self,
        ax: Axes,
        x_variable: str,
        y_variable: str,
        font_size: Optional[int] = None,
        xaxis_label: Optional[str] = None,
        yaxis_label: Optional[str] = None,
        xaxis_log_scale: bool = False,
        yaxis_log_scale: bool = False,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        xaxis_start_at_zero: bool = False,
        yaxis_start_at_zero: bool = False,
        invert_xaxis: bool = False,
        invert_yaxis: bool = False,
        apply_custom_function: Optional[Callable] = None,
    ) -> None:
        """
        Configure an existing axes object without creating a new figure.

        This is useful when you want to apply layout settings to axes that
        already exist (e.g., subplots, insets, or externally created axes).

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The existing axes to configure
        x_variable : str
            Name of x-axis variable
        y_variable : str
            Name of y-axis variable
        ... (other parameters same as setup_complete_layout)
        """
        # Setup all axes properties (same as complete layout but no figure creation/margins)
        self._setup_axes_labels(
            ax, x_variable, y_variable, xaxis_label, yaxis_label, font_size
        )
        self._setup_axes_scaling(
            ax, x_variable, y_variable, xaxis_log_scale, yaxis_log_scale
        )
        self._setup_integer_ticks(ax, x_variable, y_variable)
        self._setup_axes_limits_and_orientation(
            ax,
            xlim,
            ylim,
            xaxis_start_at_zero,
            yaxis_start_at_zero,
            invert_xaxis,
            invert_yaxis,
        )

        # Apply custom modifications if provided
        if apply_custom_function is not None:
            apply_custom_function(ax)

    def configure_inset_axes(
        self,
        ax: Axes,
        x_variable: str,
        y_variable: str,
        font_size: int = 8,
        xaxis_log_scale: bool = False,
        yaxis_log_scale: bool = False,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        xaxis_start_at_zero: bool = False,
        yaxis_start_at_zero: bool = False,
        invert_xaxis: bool = False,
        invert_yaxis: bool = False,
    ) -> None:
        """
        Configure axes specifically for insets with simplified styling.
        """
        # Apply grid with lighter styling for insets
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

        # Setup log scaling
        self._setup_axes_scaling(
            ax, x_variable, y_variable, xaxis_log_scale, yaxis_log_scale
        )

        # Configure limits and orientation
        self._setup_axes_limits_and_orientation(
            ax,
            xlim,
            ylim,
            xaxis_start_at_zero,
            yaxis_start_at_zero,
            invert_xaxis,
            invert_yaxis,
        )

        # Setup simplified labels for insets
        self._setup_inset_labels(ax, x_variable, y_variable)

        # Configure font sizes for inset
        ax.tick_params(labelsize=font_size)

    def _setup_inset_labels(self, ax: Axes, x_variable: str, y_variable: str) -> None:
        """Setup simplified labels for inset axes."""
        # Create abbreviated labels for insets
        x_label = self._create_abbreviated_label(x_variable)
        y_label = self._create_abbreviated_label(y_variable)

        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)

    def _create_abbreviated_label(self, variable_name: str) -> str:
        """Create abbreviated labels for inset axes."""
        # Create mapping for common variables to short labels
        abbreviations = {
            "Number_of_Chebyshev_terms": "N_Cheb",
            "Average_sign_squared_violation_values": "Sign²",
            "Bare_mass": "m₀",
            "QCD_beta_value": "β",
            "Plaquette": "P",
            "Condition_number": "κ",
            "Delta_Min": "Δ_min",
            "Delta_Max": "Δ_max",
        }

        # Return abbreviation if available, otherwise create a generic one
        if variable_name in abbreviations:
            return abbreviations[variable_name]
        
        # Generic abbreviation: take first few characters and remove underscores
        abbreviated = variable_name.replace("_", " ").title()
        if len(abbreviated) > 10:
            # Take first word or first 8 characters
            words = abbreviated.split()
            if len(words) > 1 and len(words[0]) <= 8:
                return words[0]
            else:
                return abbreviated[:8] + "..."
        
        return abbreviated
