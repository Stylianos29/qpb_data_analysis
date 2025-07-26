"""
Simplified HDF5Plotter implementation with a clean, focused API.

This design uses a single plot_dataset() method that can handle both regular 
datasets and gvar datasets through parameters, similar to how DataPlotter 
has a single plot() method with many options.
"""

from typing import Optional, Union, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np

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
    A simplified plotting interface for HDF5 data that maximally reuses DataPlotter components.

    This class provides a clean, focused API with a single plotting method that can handle
    both regular datasets and gvar datasets (mean/error pairs). The design philosophy
    mirrors DataPlotter's single plot() method with comprehensive options.

    Key Features:
    -----------
    - Single plot_dataset() method for all plotting needs
    - Automatic gvar handling via use_gvar parameter
    - Full reuse of DataPlotter infrastructure
    - Simple delegation approach for maximum code reuse
    - Consistent API with DataPlotter where possible

    Example Usage:
    --------------
    >>> plotter = HDF5Plotter('data.h5', 'plots/')
    >>> 
    >>> # Plot regular dataset vs time indices
    >>> plotter.plot_dataset('energy')
    >>> 
    >>> # Plot with grouping by parameters  
    >>> plotter.plot_dataset('correlator', grouping_variable='temperature')
    >>> 
    >>> # Plot gvar data with error bars (automatic mean/error merging)
    >>> plotter.plot_dataset('PCAC_mass', use_gvar=True)
    >>> 
    >>> # Plot vs another parameter instead of time
    >>> plotter.plot_dataset('plaquette', x_variable='beta')
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
        self, x_variable: str, y_variable: str, clear_existing: bool = False
    ) -> None:
        """
        Set the x- and y-axis variables for plotting.

        For HDF5 data, x_variable can be:
        - 'time_index' for array indices (default)
        - Name of a parameter column  
        - Name of another dataset

        Parameters:
        -----------
        x_variable : str
            Variable name for x-axis
        y_variable : str
            Dataset name for y-axis
        clear_existing : bool, optional
            Whether to clear existing plot subdirectory
        """
        # For HDF5, y_variable should be a dataset or processed dataset name
        self.xaxis_variable_name = x_variable
        self.yaxis_variable_name = y_variable
        self.plots_base_name = f"{y_variable}_Vs_{x_variable}"

        # Prepare subdirectory using file manager
        self.individual_plots_subdirectory = self.file_manager.prepare_subdirectory(
            self.plots_base_name, clear_existing=clear_existing
        )

    def plot_dataset(
        self,
        dataset_name: str,
        x_variable: str = "time_index",
        use_gvar: bool = False,
        use_dataframe_cache: bool = True,
        **plot_kwargs,
    ) -> "HDF5Plotter":
        """
        Plot a dataset from the HDF5 file with comprehensive options.

        This is the main plotting method that handles all HDF5 plotting scenarios
        through parameters, similar to how DataPlotter.plot() works.

        Parameters:
        -----------
        dataset_name : str
            Name of the dataset to plot. For gvar datasets, this should be the
            base name (without _mean_values/_error_values suffix)
        x_variable : str, optional
            Variable for x-axis. Options:
            - 'time_index': Use array indices (default)
            - Parameter name: Use parameter values
            - Dataset name: Use another dataset values
        use_gvar : bool, optional
            Whether to treat this as a gvar dataset (mean/error pairs).
            If True, looks for dataset_name_mean_values and dataset_name_error_values
            and creates (mean, error) tuples for error bar plotting.
        use_dataframe_cache : bool, optional
            Whether to cache converted DataFrames for performance
        **plot_kwargs
            All the same plotting arguments as DataPlotter.plot(), including:
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

        Examples:
        ---------
        # Basic time series plot
        plotter.plot_dataset('energy')

        # Plot with error bars from gvar data  
        plotter.plot_dataset('PCAC_mass', use_gvar=True)

        # Plot vs parameter with grouping
        plotter.plot_dataset('plaquette', x_variable='beta', 
                           grouping_variable='lattice_size')

        # All DataPlotter options work
        plotter.plot_dataset('correlator', use_gvar=True,
                           fit_function='exponential',
                           figure_size=(10, 6),
                           include_legend=True)
        """
        # Convert HDF5 data to DataFrame
        df = self._convert_to_dataframe(
            dataset_name, x_variable, use_gvar, use_cache=use_dataframe_cache
        )

        # Set plot variables based on conversion result
        y_column = self._determine_y_column_name(dataset_name, use_gvar)
        self.set_plot_variables(x_variable, y_column)

        # Delegate to DataPlotter
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
            raise ValueError("Call plot_dataset() before add_inset()")

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

    def _convert_to_dataframe(
        self, dataset_name: str, x_variable: str, use_gvar: bool, use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Convert HDF5 dataset to DataFrame format compatible with DataPlotter.

        This method handles:
        - Regular datasets  
        - Gvar datasets (creating value/error tuples)
        - Time series data (with time_index)
        - Parameter combinations

        The key insight: we create (value, error) tuples that DataPlotter 
        already knows how to handle, rather than using gvar objects.
        """
        cache_key = (dataset_name, x_variable, use_gvar)

        if use_cache and cache_key in self._dataframe_cache:
            return self._dataframe_cache[cache_key]

        if use_gvar:
            # Handle gvar datasets by creating (mean, error) tuples
            df = self._create_gvar_dataframe(dataset_name, x_variable)
        else:
            # Handle regular datasets
            df = self._create_regular_dataframe(dataset_name, x_variable)

        if use_cache:
            self._dataframe_cache[cache_key] = df

        return df

    def _create_regular_dataframe(self, dataset_name: str, x_variable: str) -> pd.DataFrame:
        """Create DataFrame for regular (non-gvar) datasets."""
        # Validate that dataset exists
        if dataset_name not in self.list_of_output_quantity_names_from_hdf5:
            raise ValueError(f"'{dataset_name}' is not a dataset in the HDF5 file.")

        # Use HDF5Analyzer's standard DataFrame creation
        df = self.create_dataset_dataframe(
            dataset_name,
            add_time_column=(x_variable == "time_index"),
            flatten_arrays=True,
        )

        return df

    def _create_gvar_dataframe(self, base_name: str, x_variable: str) -> pd.DataFrame:
        """
        Create DataFrame for gvar datasets using (mean, error) tuples.
        
        This creates tuples that DataPlotter can handle directly with errorbar(),
        avoiding the complexity of gvar objects in DataFrames.
        """
        # Validate that the gvar datasets exist
        mean_dataset = f"{base_name}_mean_values"
        error_dataset = f"{base_name}_error_values"

        if mean_dataset not in self.list_of_output_quantity_names_from_hdf5:
            raise ValueError(f"'{mean_dataset}' is not a dataset in the HDF5 file.")

        if error_dataset not in self.list_of_output_quantity_names_from_hdf5:
            raise ValueError(f"'{error_dataset}' is not a dataset in the HDF5 file.")

        # Get separate DataFrames for mean and error
        mean_df = self.create_dataset_dataframe(
            mean_dataset,
            add_time_column=(x_variable == "time_index"),
            flatten_arrays=True,
        )

        error_df = self.create_dataset_dataframe(
            error_dataset,
            add_time_column=(x_variable == "time_index"),
            flatten_arrays=True,
        )

        # Create combined DataFrame with (mean, error) tuples
        # Start with mean_df structure
        combined_df = mean_df.copy()

        # Replace the dataset column with (mean, error) tuples
        mean_values = mean_df[mean_dataset].values
        error_values = error_df[error_dataset].values

        # Create tuples - this is what DataPlotter expects for error bars
        combined_df[base_name] = [
            (mean_val, error_val) for mean_val, error_val in zip(mean_values, error_values)
        ]

        # Remove the original mean/error columns
        combined_df = combined_df.drop(columns=[mean_dataset])

        return combined_df

    def _determine_y_column_name(self, dataset_name: str, use_gvar: bool) -> str:
        """Determine what the y-column will be called in the DataFrame."""
        if use_gvar:
            return dataset_name  # Base name for gvar
        else:
            return dataset_name  # Same for regular datasets

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
        Generate a report similar to DataPlotter's version but for HDF5 data.
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
