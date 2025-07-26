"""
HDF5Plotter implementation that maximally reuses DataPlotter components.

This design uses composition to leverage all the managers, builders, and
specialized components from the DataPlotter refactoring.
"""

from typing import Optional, List, Union, Dict, Any
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
    A plotting interface for HDF5 data that leverages the modular
    DataPlotter architecture.

    This class acts as a bridge between HDF5 data and the plotting
    infrastructure, converting HDF5 datasets into DataFrames and then
    using either DataPlotter instances or the same component managers
    for visualization.

    Key Features:
    -----------
    - Inherits HDF5 data access from HDF5Analyzer
    - Reuses all plotting components from DataPlotter
    - Supports both delegation to DataPlotter and direct component usage
    - Handles HDF5-specific data types (gvar arrays, time series, etc.)
    - Maintains consistent API with DataPlotter where possible
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
        self.plots_base_name = None

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

        For HDF5 data, x_variable can be: - 'time_index' for array
        indices - Name of a parameter column - Name of another dataset

        Parameters:
        -----------
        x_variable : str
            Variable name for x-axis
        y_variable : str
            Dataset name for y-axis
        clear_existing : bool, optional
            Whether to clear existing plot subdirectory
        """
        # Validate that y_variable exists as a dataset
        if y_variable not in self.list_of_output_quantity_names_from_hdf5:
            raise ValueError(f"'{y_variable}' is not a dataset in the HDF5 file.")

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
        conversion_method: str = "auto",
        use_dataframe_cache: bool = True,
        **plot_kwargs,
    ) -> "HDF5Plotter":
        """
        Plot a dataset from the HDF5 file using the DataPlotter
        infrastructure.

        Parameters:
        -----------
        dataset_name : str
            Name of the dataset to plot
        x_variable : str, optional
            Variable for x-axis ('time_index' for array indices, or
            parameter name)
        conversion_method : str, optional
            How to convert HDF5 data to DataFrame: - 'auto':
            Automatically choose based on data - 'delegate': Create
            DataPlotter instance - 'direct': Use components directly
        use_dataframe_cache : bool, optional
            Whether to cache converted DataFrames
        **plot_kwargs
            All the same plotting arguments as DataPlotter.plot()

        Returns:
        --------
        HDF5Plotter
            Self for method chaining
        """
        # Set plot variables
        self.set_plot_variables(x_variable, dataset_name)

        # Convert HDF5 data to DataFrame
        df = self._convert_to_dataframe(
            dataset_name, x_variable, use_cache=use_dataframe_cache
        )

        if conversion_method == "auto":
            # Choose method based on data complexity
            conversion_method = self._choose_conversion_method(df, plot_kwargs)

        if conversion_method == "delegate":
            return self._plot_via_dataframer(df, **plot_kwargs)
        else:
            return self._plot_direct(df, **plot_kwargs)

    def plot_multiple_datasets(
        self,
        dataset_names: List[str],
        x_variable: str = "time_index",
        time_aligned: bool = True,
        **plot_kwargs,
    ) -> "HDF5Plotter":
        """
        Plot multiple datasets on the same axes.

        Parameters:
        -----------
        dataset_names : list
            List of dataset names to plot
        x_variable : str, optional
            Variable for x-axis
        time_aligned : bool, optional
            Whether datasets should be time-aligned
        **plot_kwargs
            Plotting arguments
        """
        # Create multi-dataset DataFrame
        df = self._create_multi_dataset_dataframe(
            dataset_names, x_variable, time_aligned
        )

        # For multiple datasets, we'll typically want to group by
        # dataset name
        if "grouping_variable" not in plot_kwargs:
            plot_kwargs["grouping_variable"] = "dataset_name"

        # Use first dataset for plot variable setup
        self.set_plot_variables(x_variable, "value")

        return self._plot_via_dataframer(df, **plot_kwargs)

    def plot_gvar_dataset(
        self, base_name: str, x_variable: str = "time_index", **plot_kwargs
    ) -> "HDF5Plotter":
        """
        Plot a gvar dataset (automatically merging mean/error values).

        Parameters:
        -----------
        base_name : str
            Base name of the gvar dataset (without
            _mean_values/_error_values)
        x_variable : str, optional
            Variable for x-axis
        **plot_kwargs
            Plotting arguments
        """

        # Fix: Check for the actual datasets that will be used
        mean_dataset = f"{base_name}_mean_values"
        error_dataset = f"{base_name}_error_values"

        if mean_dataset not in self.list_of_output_quantity_names_from_hdf5:
            raise ValueError(f"'{mean_dataset}' is not a dataset in the HDF5 file.")
        
        if error_dataset not in self.list_of_output_quantity_names_from_hdf5:
            raise ValueError(f"'{error_dataset}' is not a dataset in the HDF5 file.")
        
        # Use the HDF5Analyzer's gvar DataFrame creation
        df = self.create_merged_value_error_dataframe(
            base_name, add_time_column=(x_variable == "time_index")
        )

        self.set_plot_variables(x_variable, mean_dataset)
        return self._plot_via_dataframer(df, **plot_kwargs)

    def add_inset(self, **kwargs) -> "HDF5Plotter":
        """
        Add an inset to the most recently created plots.

        This delegates to the same inset manager used by DataPlotter.
        """
        if not hasattr(self, "_last_plot_figures") or not self._last_plot_figures:
            raise ValueError("Call a plot method before add_inset()")

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
        self, dataset_name: str, x_variable: str, use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Convert HDF5 dataset to DataFrame format compatible with
        DataPlotter.

        This method handles: - Time series data (with time_index) -
        Parameter combinations - Gvar data (automatic detection) -
        Multi-valued vs single-valued parameters
        """
        cache_key = (dataset_name, x_variable)

        if use_cache and cache_key in self._dataframe_cache:
            return self._dataframe_cache[cache_key]

        # Check if this is a gvar dataset
        if dataset_name in self._gvar_dataset_pairs:
            df = self.create_merged_value_error_dataframe(
                dataset_name, add_time_column=(x_variable == "time_index")
            )
        else:
            # Use standard dataset DataFrame creation
            df = self.create_dataset_dataframe(
                dataset_name,
                add_time_column=(x_variable == "time_index"),
                flatten_arrays=True,
            )

        if use_cache:
            self._dataframe_cache[cache_key] = df

        return df

    def _create_multi_dataset_dataframe(
        self, dataset_names: List[str], x_variable: str, time_aligned: bool
    ) -> pd.DataFrame:
        """
        Create a DataFrame containing multiple datasets.

        This creates a 'long format' DataFrame where multiple datasets
        are stacked with a 'dataset_name' column for grouping.
        """
        all_dfs = []

        for dataset_name in dataset_names:
            df = self._convert_to_dataframe(dataset_name, x_variable, use_cache=True)
            df["dataset_name"] = dataset_name
            df["value"] = df[dataset_name]  # Standardized value column
            all_dfs.append(df)

        # Combine all DataFrames
        combined_df = pd.concat(all_dfs, ignore_index=True)

        return combined_df

    def _choose_conversion_method(
        self, df: pd.DataFrame, plot_kwargs: Dict[str, Any]
    ) -> str:
        """
        Automatically choose between delegation and direct plotting.

        Factors to consider: - DataFrame size and complexity - Requested
        plot features - Performance considerations
        """
        # For now, prefer delegation for simplicity Could add logic
        # based on:
        # - df.shape
        # - presence of advanced features in plot_kwargs
        # - memory constraints, etc.

        return "delegate"

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

    def _plot_direct(self, df: pd.DataFrame, **plot_kwargs) -> "HDF5Plotter":
        """
        Plot directly using the component managers.

        This approach gives more control but requires implementing the
        orchestration logic here.
        """
        # This would essentially replicate DataPlotter.plot() logic but
        # using self.layout_manager, self.style_manager, etc.
        #
        # For now, fall back to delegation approach
        return self._plot_via_dataframer(df, **plot_kwargs)

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
