"""
Simplified HDF5Plotter implementation with a clean, focused API that
closely mirrors DataPlotter.

This enhanced design provides both plot_dataset() for convenience and
plot() for DataPlotter-like workflow where you set variables first, then
plot with different options.
"""

from typing import Optional, Union
from pathlib import Path

import pandas as pd

from ...data.hdf5_analyzer import HDF5Analyzer
from .data_plotter import DataPlotter

# Import all the same components that DataPlotter uses
from ..builders.title_builder import PlotTitleBuilder
from ..builders.filename_builder import PlotFilenameBuilder
from ..managers.file_manager import PlotFileManager
from ..managers.layout_manager import PlotLayoutManager
from ..managers.style_manager import PlotStyleManager
from ..managers.data_processor import PlotDataProcessor
from ..managers.annotation_manager import PlotAnnotationManager
from ..specialized.curve_fitter import CurveFitter
from ..specialized.inset_manager import PlotInsetManager
from library import constants


class HDF5Plotter(HDF5Analyzer):
    """
    A simplified plotting interface for HDF5 data that mirrors
    DataPlotter's API.

    This class provides a clean, dataset-focused API with the same
    workflow as DataPlotter: 1. set_plot_variables() - Set which
    datasets to plot 2. plot() - Create plots with grouping, styling,
    and fitting options

    Key Design Principles:
    ---------------------
    - Only datasets can be plotted (not parameters - use parameters for
      grouping)
    - Convenient syntax: single argument assumes dataset vs time
    - Full reuse of DataPlotter infrastructure and options
    - Automatic gvar handling for error bar plots

    Key Features:
    -----------
    - Dataset-only plotting (parameters used for grouping)
    - Automatic gvar handling via use_gvar parameter
    - Smart defaults: single dataset argument plots vs time
    - Full DataPlotter compatibility for all plot options
    - Clean, consistent API

    Usage Examples:
    ---------------
    >>> plotter = HDF5Plotter('data.h5', 'plots/')
    >>>
    >>> # Most common: dataset vs time (single argument)
    >>> plotter.set_plot_variables('energy')  # energy vs time_index
    >>> plotter.plot()
    >>> plotter.plot(grouping_variable='temperature')
    >>>
    >>> # Dataset vs dataset (two arguments)
    >>> plotter.set_plot_variables('PCAC_mass', 'Number_of_Chebyshev_terms')
    >>> plotter.plot(grouping_variable='Kernel_operator_type', use_gvar=True)
    >>>
    >>> # All DataPlotter options work
    >>> plotter.plot(
    ...     grouping_variable='temperature',
    ...     fit_function='exponential',
    ...     figure_size=(10, 6)
    ... )
    """

    def __init__(self, hdf5_file_path: Union[str, Path], plots_directory: str):
        """
        Initialize the HDF5Plotter.

        Parameters:
        -----------
        hdf5_file_path : str or Path
            Path to the HDF5 file to analyze and plot
        plots_directory : str
            Base directory where plots will be saved
        """
        # Initialize HDF5 data access
        super().__init__(hdf5_file_path)

        # Initialize all the same managers as DataPlotter
        self.file_manager = PlotFileManager(plots_directory)
        self.layout_manager = PlotLayoutManager(constants)
        self.style_manager = PlotStyleManager(constants)
        self.data_processor = PlotDataProcessor()
        self.annotation_manager = PlotAnnotationManager(self.data_processor)
        self.title_builder = PlotTitleBuilder(
            constants.TITLE_LABELS_BY_COLUMN_NAME,
            title_number_format=".2f",
            title_exponential_format=".0e",
        )
        self.filename_builder = PlotFilenameBuilder(
            constants.FILENAME_LABELS_BY_COLUMN_NAME
        )
        self.curve_fitter = CurveFitter()
        self.inset_manager = PlotInsetManager(data_plotter_class=DataPlotter)

        # Store directory paths
        self.plots_directory = plots_directory
        self.individual_plots_subdirectory = plots_directory
        self.combined_plots_subdirectory = plots_directory

        # Plot variables (like DataPlotter)
        self.xaxis_variable_name = None
        self.yaxis_variable_name = None
        self._plots_base_name = None

        # Storage for recent plots (for inset functionality)
        self._last_plot_figures = {}
        self._last_plot_paths = {}

        # Cache for converted DataFrames
        self._dataframe_cache = {}

    def set_plot_variables(
        self,
        y_variable: str,
        x_variable: str = "time_index",
        *,
        clear_existing: bool = False,
    ) -> None:
        """
        Set the x- and y-axis variables for plotting (matches
        DataPlotter API).

        This method only accepts dataset names (not parameters).
        Parameters should be used for grouping via the grouping_variable
        argument in plot().

        Parameters:
        -----------
        y_variable : str
            Dataset name for y-axis (must be a dataset in the HDF5 file)
        x_variable : str, optional
            Dataset name for x-axis or 'time_index'. Default is
            'time_index'.
        clear_existing : bool, optional
            Whether to clear existing plot subdirectory

        Raises:
        -------
        ValueError
            If variables are not valid datasets

        Examples:
        ---------
        set_plot_variables("energy")  # energy vs time_index
        set_plot_variables("PCAC_mass", "Number_of_Chebyshev_terms")  #
        PCAC_mass vs Number_of_Chebyshev_terms
        """
        # Validate that non-time_index variables are datasets
        if y_variable != "time_index":
            self._validate_dataset_variable(y_variable)
        if x_variable != "time_index":
            self._validate_dataset_variable(x_variable)

        # Special case: can't have time_index vs time_index
        if x_variable == "time_index" and y_variable == "time_index":
            raise ValueError("Cannot plot time_index vs time_index")

        # Set the variables
        self.xaxis_variable_name = x_variable
        self.yaxis_variable_name = y_variable
        self.plots_base_name = f"{y_variable}_Vs_{x_variable}"

        # Prepare subdirectory using file manager
        self.individual_plots_subdirectory = self.file_manager.prepare_subdirectory(
            self.plots_base_name, clear_existing=clear_existing
        )

    def plot(
        self,
        use_gvar: bool = False,
        use_dataframe_cache: bool = True,
        time_offset: int = 0,
        **plot_kwargs,
    ) -> "HDF5Plotter":
        """
        Plot using the previously set plot variables (DataPlotter-like
        API).

        This method works similarly to DataPlotter.plot() - you first
        call set_plot_variables() to set the x and y variables, then
        call plot() with grouping and styling options.

        Parameters:
        -----------
        use_gvar : bool, optional
            Whether to treat the y-variable as a gvar dataset
            (mean/error pairs). If True, looks for
            y_variable_mean_values and y_variable_error_values and
            creates (mean, error) tuples for error bar plotting.
        use_dataframe_cache : bool, optional
            Whether to cache converted DataFrames for performance
        time_offset : int, optional
            Offset to add to time_index values. Useful for shifting
            time series data (e.g., time_offset=1 changes indices from
            0,1,2,... to 1,2,3,...). Default is 0.
        **plot_kwargs
            All the same plotting arguments as DataPlotter.plot(),
            including:

            - grouping_variable, labeling_variable
            - figure_size, font_size, margins
            - xaxis_log_scale, yaxis_log_scale, xlim, ylim
            - marker_size, empty_markers, capsize
            - include_legend, legend_location
            - fit_function, annotation_variable
            - And all other DataPlotter options

        Returns:
        --------
        HDF5Plotter
            Self for method chaining

        Raises:
        -------
        ValueError
            If plot variables haven't been set via set_plot_variables()

        Examples:
        ---------
        # Basic workflow like DataPlotter
        >>> plotter.set_plot_variables('energy')  # energy vs time_index
        >>> plotter.plot()

        # Plot with grouping
        >>> plotter.set_plot_variables('PCAC_mass', 'Number_of_Chebyshev_terms')
        >>> plotter.plot(grouping_variable='Kernel_operator_type', use_gvar=True)

        # Change grouping without changing variables
        >>> plotter.plot(grouping_variable='Configuration_label')

        # All DataPlotter options work
        >>> plotter.plot(
        ...     grouping_variable='temperature',
        ...     fit_function='exponential',
        ...     figure_size=(10, 6),
        ...     include_legend=True
        ... )
        """
        if self.xaxis_variable_name is None or self.yaxis_variable_name is None:
            raise ValueError(
                "Plot variables not set. Call "
                "'set_plot_variables(y_var, x_var)' before plotting."
            )

        # Convert HDF5 data to DataFrame using the set variables
        df = self._convert_to_dataframe(
            use_gvar=use_gvar, use_cache=use_dataframe_cache, time_offset=time_offset
        )

        # Plot using the converted DataFrame (variables already set)
        return self._plot_via_dataframer(df, **plot_kwargs)

    def add_inset(self, **kwargs) -> "HDF5Plotter":
        """
        Add an inset to the most recently created plots.

        This delegates to the same inset manager used by DataPlotter.
        All the same inset options are available.

        Parameters:
        -----------
        **kwargs
            Same arguments as DataPlotter.add_inset(), including:
            - xaxis_variable, yaxis_variable
            - location, width, height
            - data_filter_func, data_condition
            - All plotting options for the inset

        Returns:
        --------
        HDF5Plotter
            Self for method chaining
        """
        if not hasattr(self, "_last_plot_figures") or not self._last_plot_figures:
            raise ValueError("Call plot() before add_inset()")

        # Use the same inset manager
        for group_keys, (fig, ax, group_df) in self._last_plot_figures.items():
            try:
                inset_ax = self.inset_manager.add_inset(
                    figure=fig,
                    main_axes=ax,
                    dataframe=group_df,
                    plots_directory=self.plots_directory,
                    **kwargs,
                )

                if inset_ax is not None and group_keys in self._last_plot_paths:
                    save_path = self._last_plot_paths[group_keys]
                    fig.savefig(save_path)
            except Exception as e:
                print(f"Warning: Failed to add inset to group {group_keys}: {e}")

        return self

    def _validate_dataset_variable(self, variable_name: str) -> None:
        """
        Validate that a variable name corresponds to a dataset in the
        HDF5 file.

        For gvar datasets, accepts either the base name or the full
        _mean_values name.
        """
        # Check if it's a regular dataset
        if variable_name in self.list_of_output_quantity_names_from_hdf5:
            return

        # Check if it's a gvar base name (has corresponding _mean_values
        # and _error_values)
        mean_dataset = f"{variable_name}_mean_values"
        error_dataset = f"{variable_name}_error_values"

        if (
            mean_dataset in self.list_of_output_quantity_names_from_hdf5
            and error_dataset in self.list_of_output_quantity_names_from_hdf5
        ):
            return

        # If neither worked, raise an error with helpful message
        available_datasets = self.list_of_output_quantity_names_from_hdf5[
            :10
        ]  # Show first 10
        raise ValueError(
            f"'{variable_name}' is not a valid dataset in the HDF5 file.\n"
            f"Available datasets (first 10): {available_datasets}\n"
            f"For gvar datasets, use the base name "
            "(without _mean_values/_error_values suffix)"
        )

    def _validate_x_variable(self, variable_name: str) -> None:
        """
        Validate that x_variable is either 'time_index', a parameter, or
        a dataset.

        This is more permissive than dataset validation since x can be
        various things.
        """
        if variable_name == "time_index":
            return

        # Check if it's a dataset
        if variable_name in self.list_of_output_quantity_names_from_hdf5:
            return

        # Check if it's a parameter (we'll do this by trying to create a
        # sample DataFrame and seeing if the variable appears as a
        # column)
        try:
            # Get a sample dataset to check parameter columns
            if self.list_of_output_quantity_names_from_hdf5:
                sample_dataset = self.list_of_output_quantity_names_from_hdf5[0]
                sample_df = self.create_dataset_dataframe(
                    sample_dataset, add_time_column=False, flatten_arrays=True
                )

                if variable_name in sample_df.columns:
                    return

        except Exception:
            pass  # If sampling fails, we'll give a generic error below

        # If we get here, the variable wasn't found
        print(
            f"Warning: '{variable_name}' not found as a dataset or parameter. "
            f"This may cause issues during plotting."
        )

    def _convert_to_dataframe(
        self, use_gvar: bool, use_cache: bool = True, time_offset: int = 0
    ) -> pd.DataFrame:
        """
        Convert HDF5 dataset(s) to DataFrame format compatible with
        DataPlotter.

        Uses the currently set xaxis_variable_name and
        yaxis_variable_name. Handles cases where either could be
        "time_index" or actual datasets.

        This method handles:
        - Regular datasets
        - Gvar datasets (creating value/error tuples)
        - Time series data (with time_index)
        - Dataset vs dataset plots

        The key insight: we create (value, error) tuples that
        DataPlotter already knows how to handle, rather than using gvar
        objects.
        """
        cache_key = (
            self.xaxis_variable_name,
            self.yaxis_variable_name,
            use_gvar,
            time_offset,
        )

        if use_cache and cache_key in self._dataframe_cache:
            return self._dataframe_cache[cache_key]

        # Determine which variables are datasets vs time_index
        x_is_time = self.xaxis_variable_name == "time_index"
        y_is_time = self.yaxis_variable_name == "time_index"

        if x_is_time and y_is_time:
            raise ValueError("Cannot have both x and y as time_index")

        # Determine the dataset(s) we need to load
        datasets_to_load = []

        if not y_is_time:
            if use_gvar:
                # For gvar, we need both mean and error datasets
                datasets_to_load.extend(
                    [
                        f"{self.yaxis_variable_name}_mean_values",
                        f"{self.yaxis_variable_name}_error_values",
                    ]
                )
            else:
                datasets_to_load.append(self.yaxis_variable_name)

        if not x_is_time:
            datasets_to_load.append(self.xaxis_variable_name)

        # Create the DataFrame
        if x_is_time or y_is_time:
            # One axis is time_index
            df = self.to_dataframe(
                datasets=datasets_to_load,
                add_time_column=True,
                flatten_arrays=True,
            )
        else:
            # Both axes are datasets
            df = self.to_dataframe(
                datasets=datasets_to_load,
                add_time_column=False,
                flatten_arrays=True,
            )

        # Handle gvar processing if needed
        if use_gvar and not y_is_time:
            mean_col = f"{self.yaxis_variable_name}_mean_values"
            error_col = f"{self.yaxis_variable_name}_error_values"

            if mean_col in df.columns and error_col in df.columns:
                # Create (mean, error) tuples
                mean_values = df[mean_col].values
                error_values = df[error_col].values

                df[self.yaxis_variable_name] = [
                    (mean_val, error_val)
                    for mean_val, error_val in zip(mean_values, error_values)
                ]

                # Remove the original mean/error columns
                df = df.drop(columns=[mean_col, error_col])

        # Apply time offset if specified and time_index column exists
        if time_offset != 0 and "time_index" in df.columns:
            df["time_index"] = df["time_index"] + time_offset

        if use_cache:
            self._dataframe_cache[cache_key] = df

        return df

    def _plot_via_dataframer(self, df: pd.DataFrame, **plot_kwargs) -> "HDF5Plotter":
        """
        Plot by creating a temporary DataPlotter instance.

        This approach maximizes code reuse by delegating to the full
        DataPlotter implementation.
        """
        # Check if we have valid plot variables
        if self.xaxis_variable_name is None or self.yaxis_variable_name is None:
            raise ValueError("Both x and y axis variables must be set before plotting")

        # Create temporary DataPlotter
        temp_plotter = DataPlotter(df, self.plots_directory)

        # Set the same plot variables
        temp_plotter.set_plot_variables(
            self.xaxis_variable_name, self.yaxis_variable_name, clear_existing=False
        )

        # Delegate plotting
        temp_plotter.plot(**plot_kwargs)

        # Transfer last plot info for inset support
        self._last_plot_figures = temp_plotter._last_plot_figures
        self._last_plot_paths = temp_plotter._last_plot_paths

        return self

    def generate_column_uniqueness_report(
        self, max_width: int = 80, separate_by_type: bool = True
    ) -> str:
        """
        Generate a report similar to DataPlotter's version but for HDF5
        data.
        """
        return self.generate_uniqueness_report(
            max_width=max_width, separate_by_type=separate_by_type
        )

    # Convenience methods that mirror DataPlotter API
    def set_default_file_format(self, format: str) -> None:
        """Set the default file format for saved plots."""
        self.file_manager.set_default_format(format)

    def clear_style_cache(self) -> None:
        """Clear cached style mappings."""
        self.style_manager.clear_cache()

    def get_file_manager(self) -> PlotFileManager:
        """Get the file manager instance."""
        return self.file_manager

    def get_style_manager(self) -> PlotStyleManager:
        """Get the style manager instance."""
        return self.style_manager

    def get_layout_manager(self) -> PlotLayoutManager:
        """Get the layout manager instance."""
        return self.layout_manager

    def get_curve_fitter(self) -> CurveFitter:
        """Get the curve fitter instance."""
        return self.curve_fitter

    def get_inset_manager(self) -> PlotInsetManager:
        """Get the inset manager instance."""
        return self.inset_manager

    def get_annotation_manager(self) -> PlotAnnotationManager:
        """Get the annotation manager instance."""
        return self.annotation_manager

    # Property to maintain compatibility
    @property
    def plots_base_name(self) -> str:
        """Get the base name for plots."""
        if self._plots_base_name is None:
            raise ValueError(
                "plots_base_name not set. Call set_plot_variables() first."
            )
        return self._plots_base_name

    @plots_base_name.setter
    def plots_base_name(self, value: Optional[str]) -> None:
        """Set the base name for plots."""
        self._plots_base_name = value

    def __repr__(self) -> str:
        """String representation of the HDF5Plotter."""
        return (
            f"HDF5Plotter("
            f"file='{self._original_file_path}', "
            f"plots_directory='{self.plots_directory}', "
            f"x_var='{self.xaxis_variable_name}', "
            f"y_var='{self.yaxis_variable_name}')"
        )
