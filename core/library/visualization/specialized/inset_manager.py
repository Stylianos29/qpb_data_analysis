"""
Inset management for visualization components.

This module provides the PlotInsetManager class for adding and managing
insets (smaller plots within plots) with different data, variables, and
styling options.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class PlotInsetManager:
    """
    Manages inset plots within existing matplotlib figures.

    This class handles:
    - Creating insets with different variables than the main plot
    - Positioning insets using predefined locations or custom coordinates
    - Applying data filters to inset content
    - Managing inset styling independently from main plots
    - Coordinating with existing plotting infrastructure

    Insets are useful for:
    - Showing zoomed-in regions of data
    - Displaying different variable relationships
    - Providing additional context or detail
    - Creating multi-scale visualizations
    """

    def __init__(self, data_plotter_class=None):
        """
        Initialize the inset manager.

        Parameters:
        -----------
        data_plotter_class : class, optional
            Reference to the DataPlotter class for creating temporary plotters.
            If None, will be imported when needed.
        """
        self.data_plotter_class = data_plotter_class

        # Predefined inset locations
        self.location_presets = {
            "upper right": [0.65, 0.65],
            "upper left": [0.05, 0.65],
            "lower right": [0.65, 0.05],
            "lower left": [0.05, 0.05],
            "center": [0.5, 0.5],
            "upper center": [0.5, 0.75],
            "lower center": [0.5, 0.15],
            "center left": [0.15, 0.5],
            "center right": [0.75, 0.5],
        }

        # Default inset dimensions
        self.default_width = 0.3
        self.default_height = 0.3

        # Storage for created insets
        self._insets = {}

    def add_inset(
        self,
        figure: Figure,
        main_axes: Axes,
        dataframe: pd.DataFrame,
        plots_directory: str,
        xaxis_variable: str,
        yaxis_variable: Optional[str] = None,
        location: Union[str, Tuple[float, float]] = "lower right",
        width: float = 0.3,
        height: float = 0.3,
        inset_x: Optional[float] = None,
        inset_y: Optional[float] = None,
        data_filter_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        data_condition: Optional[str] = None,
        inset_id: Optional[str] = None,
        **plot_kwargs,
    ) -> Optional[Axes]:
        """
        Add an inset to the main plot with different variables or filtered data.

        Parameters:
        -----------
        figure : matplotlib.figure.Figure
            The main figure to add the inset to
        main_axes : matplotlib.axes.Axes
            The main axes object
        dataframe : pd.DataFrame
            DataFrame containing the data for the inset
        plots_directory : str
            Directory for the plotting system
        xaxis_variable : str
            Variable to use for the x-axis of the inset
        yaxis_variable : str, optional
            Variable to use for the y-axis of the inset. If None, uses the same as main plot.
        location : str or tuple, optional
            Predefined location name or (x, y) coordinates for the inset position
        width : float, optional
            Width of the inset as a fraction of the main axes
        height : float, optional
            Height of the inset as a fraction of the main axes
        inset_x : float, optional
            Custom x-coordinate for the inset (overrides location)
        inset_y : float, optional
            Custom y-coordinate for the inset (overrides location)
        data_filter_func : callable, optional
            Function to filter the DataFrame for the inset
        data_condition : str, optional
            String condition to filter the DataFrame (pandas query syntax)
        inset_id : str, optional
            Unique identifier for this inset (for later reference)
        **plot_kwargs : dict
            Additional keyword arguments passed to the plot method

        Returns:
        --------
        matplotlib.axes.Axes or None
            The created inset axes, or None if creation failed
        """
        try:
            # Determine inset position
            bbox_coords = self._calculate_inset_position(
                location, width, height, inset_x, inset_y
            )

            # Create the inset axes
            inset_ax = main_axes.inset_axes(bbox_coords)

            # Apply data filtering if specified
            filtered_df = self._apply_data_filters(
                dataframe, data_filter_func, data_condition
            )

            if len(filtered_df) == 0:
                print("Warning: No data remaining after filtering for inset")
                return None

            # Create temporary plotter for the inset
            temp_plotter = self._create_temp_plotter(filtered_df, plots_directory)

            # Determine y-axis variable
            if yaxis_variable is None:
                # Try to infer from main axes or use same as main plot
                yaxis_variable = self._infer_yaxis_variable(main_axes, temp_plotter)

            # Set plot variables
            temp_plotter.set_plot_variables(
                xaxis_variable, yaxis_variable, clear_existing=False
            )

            # Check if we have multivalued parameters for grouping
            if not temp_plotter.list_of_multivalued_tunable_parameter_names:
                grouping_variable = plot_kwargs.get("grouping_variable")
                excluded_list = plot_kwargs.get("excluded_from_grouping_list")
                if grouping_variable or excluded_list:
                    print(
                        "Warning: No multivalued parameters available for grouping in inset"
                    )

            # Configure inset-specific plot parameters
            inset_plot_kwargs = self._prepare_inset_plot_kwargs(plot_kwargs)

            # Create the inset plot
            temp_plotter.plot(**inset_plot_kwargs)

            # Store inset reference
            if inset_id:
                self._insets[inset_id] = {
                    "axes": inset_ax,
                    "plotter": temp_plotter,
                    "dataframe": filtered_df,
                    "variables": (xaxis_variable, yaxis_variable),
                }

            return inset_ax

        except Exception as e:
            print(f"Error creating inset: {e}")
            return None

    def add_multiple_insets(
        self,
        figure: Figure,
        main_axes: Axes,
        inset_specs: List[Dict[str, Any]],
        base_dataframe: pd.DataFrame,
        plots_directory: str,
    ) -> Dict[str, Axes]:
        """
        Add multiple insets to a plot based on specifications.

        Parameters:
        -----------
        figure : matplotlib.figure.Figure
            The main figure
        main_axes : matplotlib.axes.Axes
            The main axes
        inset_specs : list of dict
            List of inset specifications, each containing parameters for add_inset
        base_dataframe : pd.DataFrame
            Base DataFrame for all insets
        plots_directory : str
            Directory for the plotting system

        Returns:
        --------
        dict
            Dictionary mapping inset IDs to created axes objects
        """
        created_insets = {}

        for i, spec in enumerate(inset_specs):
            # Generate ID if not provided
            inset_id = spec.get("inset_id", f"inset_{i}")
            spec["inset_id"] = inset_id

            # Create the inset
            inset_ax = self.add_inset(
                figure, main_axes, base_dataframe, plots_directory, **spec
            )

            if inset_ax is not None:
                created_insets[inset_id] = inset_ax

        return created_insets

    def add_zoom_inset(
        self,
        figure: Figure,
        main_axes: Axes,
        dataframe: pd.DataFrame,
        plots_directory: str,
        zoom_xlim: Tuple[float, float],
        zoom_ylim: Tuple[float, float],
        location: Union[str, Tuple[float, float]] = "upper right",
        width: float = 0.4,
        height: float = 0.4,
        show_connection_lines: bool = True,
        connection_line_style: Optional[Dict[str, Any]] = None,
        **plot_kwargs,
    ) -> Optional[Axes]:
        """
        Add a zoom inset showing a magnified region of the main plot.

        Parameters:
        -----------
        figure : matplotlib.figure.Figure
            The main figure
        main_axes : matplotlib.axes.Axes
            The main axes
        dataframe : pd.DataFrame
            DataFrame containing the data
        plots_directory : str
            Directory for the plotting system
        zoom_xlim : tuple
            (min, max) x-axis limits for the zoom region
        zoom_ylim : tuple
            (min, max) y-axis limits for the zoom region
        location : str or tuple, optional
            Position for the zoom inset
        width : float, optional
            Width of the zoom inset
        height : float, optional
            Height of the zoom inset
        show_connection_lines : bool, optional
            Whether to show lines connecting the zoom region to the inset
        connection_line_style : dict, optional
            Style parameters for connection lines
        **plot_kwargs : dict
            Additional plotting parameters

        Returns:
        --------
        matplotlib.axes.Axes or None
            The created zoom inset axes
        """

        # Filter data to zoom region
        def zoom_filter(df):
            x_var = plot_kwargs.get("xaxis_variable", df.columns[0])
            y_var = plot_kwargs.get("yaxis_variable", df.columns[1])

            # Handle tuple data by extracting values
            x_vals = df[x_var].apply(lambda x: x[0] if isinstance(x, tuple) else x)
            y_vals = df[y_var].apply(lambda x: x[0] if isinstance(x, tuple) else x)

            return df[
                (x_vals >= zoom_xlim[0])
                & (x_vals <= zoom_xlim[1])
                & (y_vals >= zoom_ylim[0])
                & (y_vals <= zoom_ylim[1])
            ]

        # Create the zoom inset
        zoom_ax = self.add_inset(
            figure,
            main_axes,
            dataframe,
            plots_directory,
            data_filter_func=zoom_filter,
            location=location,
            width=width,
            height=height,
            **plot_kwargs,
        )

        if zoom_ax is not None:
            # Set zoom limits
            zoom_ax.set_xlim(zoom_xlim)
            zoom_ax.set_ylim(zoom_ylim)

            # Add connection lines if requested
            if show_connection_lines:
                self._add_zoom_connection_lines(
                    main_axes, zoom_ax, zoom_xlim, zoom_ylim, connection_line_style
                )

        return zoom_ax

    def update_inset_data(
        self, inset_id: str, new_dataframe: pd.DataFrame, **plot_kwargs
    ) -> bool:
        """
        Update the data in an existing inset.

        Parameters:
        -----------
        inset_id : str
            ID of the inset to update
        new_dataframe : pd.DataFrame
            New data for the inset
        **plot_kwargs : dict
            Additional plotting parameters

        Returns:
        --------
        bool
            True if update was successful, False otherwise
        """
        if inset_id not in self._insets:
            print(f"Inset '{inset_id}' not found")
            return False

        try:
            inset_info = self._insets[inset_id]
            inset_ax = inset_info["axes"]
            x_var, y_var = inset_info["variables"]

            # Clear existing content
            inset_ax.clear()

            # Update plotter with new data
            temp_plotter = inset_info["plotter"]
            temp_plotter.dataframe = new_dataframe
            temp_plotter._update_column_categories()

            # Re-plot with new data
            temp_plotter.set_plot_variables(x_var, y_var, clear_existing=False)

            # Configure for inset plotting
            inset_plot_kwargs = self._prepare_inset_plot_kwargs(plot_kwargs)
            temp_plotter.plot(**inset_plot_kwargs)

            return True

        except Exception as e:
            print(f"Error updating inset '{inset_id}': {e}")
            return False

    def remove_inset(self, inset_id: str) -> bool:
        """
        Remove an inset from the plot.

        Parameters:
        -----------
        inset_id : str
            ID of the inset to remove

        Returns:
        --------
        bool
            True if removal was successful, False otherwise
        """
        if inset_id not in self._insets:
            print(f"Inset '{inset_id}' not found")
            return False

        try:
            inset_info = self._insets[inset_id]
            inset_ax = inset_info["axes"]

            # Remove the axes
            inset_ax.remove()

            # Clean up stored reference
            del self._insets[inset_id]

            return True

        except Exception as e:
            print(f"Error removing inset '{inset_id}': {e}")
            return False

    def get_inset_info(self, inset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific inset.

        Parameters:
        -----------
        inset_id : str
            ID of the inset

        Returns:
        --------
        dict or None
            Dictionary containing inset information, or None if not found
        """
        return self._insets.get(inset_id)

    def list_insets(self) -> List[str]:
        """
        Get a list of all inset IDs.

        Returns:
        --------
        list
            List of inset IDs
        """
        return list(self._insets.keys())

    def _calculate_inset_position(
        self,
        location: Union[str, Tuple[float, float]],
        width: float,
        height: float,
        inset_x: Optional[float] = None,
        inset_y: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Calculate the position coordinates for an inset.

        Returns:
        --------
        list
            [x, y, width, height] coordinates for the inset
        """
        # Use explicit coordinates if provided
        if inset_x is not None and inset_y is not None:
            return (inset_x, inset_y, width, height)

        # Use predefined location
        if isinstance(location, str):
            if location in self.location_presets:
                x, y = self.location_presets[location]
            else:
                print(f"Warning: Unknown location '{location}', using 'lower right'")
                x, y = self.location_presets["lower right"]
        else:
            # Assume it's a tuple of coordinates
            x, y = location

        return (x, y, width, height)

    def _apply_data_filters(
        self,
        dataframe: pd.DataFrame,
        filter_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
        condition: Optional[str],
    ) -> pd.DataFrame:
        """
        Apply data filtering to the DataFrame.

        Returns:
        --------
        pd.DataFrame
            Filtered DataFrame
        """
        filtered_df = dataframe.copy()

        # Apply function filter
        if filter_func is not None:
            try:
                filtered_df = filter_func(filtered_df)
            except Exception as e:
                print(f"Error applying filter function: {e}")
                return pd.DataFrame()  # Return empty DataFrame on error

        # Apply condition filter
        if condition is not None:
            try:
                filtered_df = filtered_df.query(condition)
            except Exception as e:
                print(f"Error applying condition filter: {e}")
                return pd.DataFrame()  # Return empty DataFrame on error

        return filtered_df

    def _create_temp_plotter(self, dataframe: pd.DataFrame, plots_directory: str):
        """
        Create a temporary plotter instance for the inset.

        Returns:
        --------
        DataPlotter
            Temporary plotter instance
        """
        if self.data_plotter_class is None:
            # Import here to avoid circular imports
            from ..plotters.data_plotter import DataPlotter  # type: ignore

            self.data_plotter_class = DataPlotter

        return self.data_plotter_class(dataframe, plots_directory)

    def _infer_yaxis_variable(self, main_axes: Axes, temp_plotter) -> str:
        """
        Infer the y-axis variable from the main plot or plotter.

        Returns:
        --------
        str
            Y-axis variable name
        """
        # Try to get from main axes ylabel
        ylabel = main_axes.get_ylabel()
        if ylabel:
            # Look for a matching column in the plotter
            for col in temp_plotter.dataframe.columns:
                if col in ylabel or ylabel in col:
                    return col

        # Fallback to first non-tunable parameter column
        if temp_plotter.list_of_output_quantity_names_from_dataframe:
            return temp_plotter.list_of_output_quantity_names_from_dataframe[0]

        # Last resort: second column
        if len(temp_plotter.dataframe.columns) > 1:
            return temp_plotter.dataframe.columns[1]

        raise ValueError("Cannot determine y-axis variable for inset")

    def _prepare_inset_plot_kwargs(self, plot_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare plot kwargs specifically for inset plotting.

        Returns:
        --------
        dict
            Modified plot kwargs for inset
        """
        inset_kwargs = plot_kwargs.copy()

        # Override settings for inset behavior
        inset_kwargs.update(
            {
                "target_ax": inset_kwargs.get("target_ax"),  # Will be set by caller
                "is_inset": True,
                "include_plot_title": False,
                "save_figure": False,
                "verbose": False,
                "include_legend": inset_kwargs.get("include_legend", False),
            }
        )

        return inset_kwargs

    def _add_zoom_connection_lines(
        self,
        main_axes: Axes,
        zoom_axes: Axes,
        zoom_xlim: Tuple[float, float],
        zoom_ylim: Tuple[float, float],
        line_style: Optional[Dict[str, Any]],
    ) -> None:
        """
        Add connection lines between zoom region and inset.

        Parameters:
        -----------
        main_axes : matplotlib.axes.Axes
            Main plot axes
        zoom_axes : matplotlib.axes.Axes
            Zoom inset axes
        zoom_xlim : tuple
            X-axis limits of zoom region
        zoom_ylim : tuple
            Y-axis limits of zoom region
        line_style : dict, optional
            Style parameters for connection lines
        """
        try:
            from matplotlib.patches import ConnectionPatch

            # Default line style
            if line_style is None:
                line_style = {
                    "color": "gray",
                    "linestyle": "--",
                    "linewidth": 1,
                    "alpha": 0.7,
                }

            # Create connection lines from zoom region corners to inset
            zoom_corners = [
                (zoom_xlim[0], zoom_ylim[0]),  # bottom-left
                (zoom_xlim[1], zoom_ylim[1]),  # top-right
            ]

            inset_corners = [
                (0, 0),  # bottom-left of inset
                (1, 1),  # top-right of inset
            ]

            for zoom_corner, inset_corner in zip(zoom_corners, inset_corners):
                connection = ConnectionPatch(
                    zoom_corner,
                    inset_corner,
                    "data",
                    "axes fraction",
                    axesA=main_axes,
                    axesB=zoom_axes,
                    **line_style,
                )
                main_axes.add_artist(connection)

        except ImportError:
            print("ConnectionPatch not available, skipping connection lines")
        except Exception as e:
            print(f"Error adding connection lines: {e}")

    def set_default_dimensions(self, width: float, height: float) -> None:
        """
        Set default dimensions for new insets.

        Parameters:
        -----------
        width : float
            Default width as fraction of main axes
        height : float
            Default height as fraction of main axes
        """
        self.default_width = width
        self.default_height = height

    def add_location_preset(self, name: str, x: float, y: float) -> None:
        """
        Add a custom location preset.

        Parameters:
        -----------
        name : str
            Name of the location preset
        x : float
            X-coordinate (0-1)
        y : float
            Y-coordinate (0-1)
        """
        self.location_presets[name] = [x, y]

    def get_location_presets(self) -> Dict[str, List[float]]:
        """
        Get all available location presets.

        Returns:
        --------
        dict
            Dictionary of location presets
        """
        return self.location_presets.copy()

    def clear_all_insets(self) -> None:
        """Remove all tracked insets."""
        for inset_id in list(self._insets.keys()):
            self.remove_inset(inset_id)
