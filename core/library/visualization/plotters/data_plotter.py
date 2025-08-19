import os
from typing import Optional, Callable, Dict, List, Union, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from library.data.analyzer import DataFrameAnalyzer
from library.data.table_generator import TableGenerator
from library import constants

# Import all the specialized components
from ..builders.title_builder import PlotTitleBuilder
from ..builders.filename_builder import PlotFilenameBuilder
from ..managers.file_manager import PlotFileManager
from ..managers.layout_manager import PlotLayoutManager
from ..managers.style_manager import PlotStyleManager
from ..managers.data_processor import PlotDataProcessor
from ..managers.annotation_manager import PlotAnnotationManager
from ..specialized.curve_fitter import CurveFitter
from ..specialized.inset_manager import PlotInsetManager


class DataPlotter(DataFrameAnalyzer):
    """
    A modular plotting interface for visualizing grouped data from a
    DataFrame.

    This refactored class extends DataFrameAnalyzer to provide plotting
    functionality using a composition-based architecture with
    specialized managers and builders. Each component handles specific
    aspects of the plotting process:
    - FileManager: File operations and path construction
    - LayoutManager: Figure layout and axes configuration
    - StyleManager: Visual styling and color/marker assignment
    - DataProcessor: Data validation and transformation
    - AnnotationManager: Text annotations and labels
    - TitleBuilder: Plot title construction
    - FilenameBuilder: Plot filename generation
    - CurveFitter: Curve fitting operations
    - InsetManager: Inset plot management

    This design provides better separation of concerns, easier testing,
    and improved maintainability compared to the monolithic original.
    """

    def __init__(self, dataframe: pd.DataFrame, plots_directory: str):
        """
        Initialize the DataPlotter with a DataFrame and output
        directory.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input data for plotting
        plots_directory : str
            Base directory where plots will be saved
        """
        super().__init__(dataframe)

        # Initialize all managers and builders
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

        # Plot variables
        self.xaxis_variable_name = None
        self.yaxis_variable_name = None
        self.plots_base_name = None

        # Storage for recent plots (for inset functionality)
        self._last_plot_figures = {}
        self._last_plot_paths = {}

        # Store fit results by group
        self.stored_fit_results = {}

    def generate_column_uniqueness_report(
        self, max_width=80, separate_by_type=True
    ) -> str:
        """Generate a report of column uniqueness using
        TableGenerator."""
        table_generator = TableGenerator(self.dataframe)
        return table_generator.generate_column_uniqueness_report(
            max_width=max_width,
            separate_by_type=separate_by_type,
            export_to_file=False,
        )

    def set_plot_variables(
        self, x_variable: str, y_variable: str, clear_existing: bool = False
    ) -> None:
        """
        Set the x- and y-axis variables for plotting.

        Parameters:
        -----------
        x_variable : str
            Column name for x-axis variable
        y_variable : str
            Column name for y-axis variable
        clear_existing : bool, optional
            Whether to clear existing plot subdirectory
        """
        if x_variable not in self.dataframe.columns:
            raise ValueError(f"'{x_variable}' is not a column in the DataFrame.")
        if y_variable not in self.dataframe.columns:
            raise ValueError(f"'{y_variable}' is not a column in the DataFrame.")

        self.xaxis_variable_name = x_variable
        self.yaxis_variable_name = y_variable
        self.plots_base_name = f"{y_variable}_Vs_{x_variable}"

        # Prepare subdirectory using file manager
        self.individual_plots_subdirectory = self.file_manager.prepare_subdirectory(
            self.plots_base_name, clear_existing=clear_existing
        )

    def plot(
        self,
        # Grouping and data organization
        grouping_variable: Optional[Union[str, List[str]]] = None,
        excluded_from_grouping_list: Optional[List[str]] = None,
        labeling_variable: Optional[Union[str, List[str]]] = None,
        sorting_variable: Optional[str] = None,
        sort_ascending: Optional[bool] = None,
        # Figure and layout
        figure_size: Tuple[float, float] = (7, 5),
        font_size: int = 13,
        left_margin_adjustment: float = 0.15,
        right_margin_adjustment: float = 0.94,
        bottom_margin_adjustment: float = 0.12,
        top_margin_adjustment: float = 0.92,
        # Axes configuration
        xaxis_label: Optional[str] = None,
        yaxis_label: Optional[str] = None,
        show_xaxis_label: bool = True,
        show_yaxis_label: bool = True,
        xaxis_log_scale: bool = False,
        yaxis_log_scale: bool = False,
        invert_xaxis: bool = False,
        invert_yaxis: bool = False,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        xaxis_start_at_zero: bool = False,
        yaxis_start_at_zero: bool = False,
        # Styling
        styling_variable: Optional[str] = None,
        marker_color_map: Optional[Dict[Any, Tuple[str, str]]] = None,
        color_index_shift: int = 0,
        marker_size: int = 8,
        empty_markers: bool = False,
        alternate_filled_markers: bool = False,
        alternate_filled_markers_reversed: bool = False,
        capsize: float = 5,
        # Legend
        include_legend: bool = True,
        legend_location: str = "upper left",
        legend_columns: int = 1,
        include_legend_title: bool = True,
        legend_number_format: str = ".2f",
        # Titles
        include_plot_title: bool = False,
        custom_plot_title: Optional[str] = None,
        title_from_columns: Optional[List[str]] = None,
        custom_plot_titles_dict: Optional[Dict[Any, str]] = None,
        title_size: int = 15,
        bold_title: bool = False,
        leading_plot_substring: Optional[str] = None,
        excluded_from_title_list: Optional[List[str]] = None,
        title_number_format: str = ".2f",
        title_exponential_format: str = ".0e",
        title_wrapping_length: int = 80,
        # Advanced features
        customization_function: Optional[Callable[[Axes], None]] = None,
        post_plot_customization_function: Optional[Callable[..., None]] = None,
        include_interpolation: bool = False,
        # Curve fitting
        fit_function: Optional[str] = None,
        fit_label_format: str = ".2f",
        show_fit_parameters_on_plot: bool = True,
        fit_curve_style: Optional[Dict[str, Any]] = None,
        fit_label_location: str = "top left",
        fit_index_range: Optional[Tuple[int, Optional[int]]] = None,
        fit_on_values: Optional[Union[Any, Tuple[Any, ...]]] = None,
        fit_label_in_legend: bool = False,
        fit_curve_range: Optional[Tuple[float, float]] = None,
        fit_min_data_points: Optional[int] = None,
        # Annotations
        annotation_variable: Optional[str] = None,
        annotation_label: str = "",
        annotation_range: Optional[Tuple[int, Optional[int], int]] = None,
        annotation_fontsize: int = 8,
        annotation_boxstyle: str = "round,pad=0.3",
        annotation_alpha: float = 0.7,
        annotation_offset: Tuple[float, float] = (0, 10),
        # Output control
        target_ax: Optional[Axes] = None,
        is_inset: bool = False,
        save_figure: bool = True,
        file_format: Optional[str] = None,
        verbose: bool = True,
    ) -> "DataPlotter":
        """
        Create plots from the DataFrame with comprehensive customization
        options.

        This method orchestrates the entire plotting process using the
        specialized managers and builders. It supports both individual
        and grouped plots with extensive customization capabilities.

        Key Parameters:
        ---------------
        file_format : str, optional
            Output file format for saved plots. If None, uses the default format
            set in the file manager (default: "png"). Supported formats include:
            "png", "pdf", "svg", "eps", "jpg", "tiff", "ps".
            Examples: "pdf", "svg", "png"

        target_ax : matplotlib.axes.Axes, optional
            If provided, plot on this existing axes instead of creating new figure.

        save_figure : bool, optional
            Whether to save the figure to disk. Default is True.

        show_xaxis_label : bool, optional
            Whether to display the x-axis label. Default is True.
            Useful for insets where axis labels might be redundant.
        show_yaxis_label : bool, optional
            Whether to display the y-axis label. Default is True.
            Useful for insets where axis labels might be redundant.
        fit_min_data_points : int, optional
            Minimum number of data points required for curve fitting.
            If None, uses function-specific defaults from CurveFitter.

        Returns:
        --------
        DataPlotter
            Self for method chaining
        """
        if self.xaxis_variable_name is None or self.yaxis_variable_name is None:
            raise ValueError("Call 'set_plot_variables()' before plotting.")

        # Handle alternate filled markers
        if alternate_filled_markers:
            empty_markers = False  # Override user setting

        # Convert fit_index_range to slice for internal use
        fit_slice: Optional[slice] = None
        if fit_index_range:
            if not isinstance(fit_index_range, tuple) or len(fit_index_range) != 2:
                raise ValueError("fit_index_range must be a tuple like (start, stop)")
            # fit_index_range = slice(*fit_index_range)
            fit_slice = slice(*fit_index_range)

        # Prepare grouping configuration
        grouping_config = self._prepare_grouping_configuration(
            grouping_variable, excluded_from_grouping_list, styling_variable
        )

        # Get grouped data
        grouped = self.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=grouping_config["excluded"],
            verbose=verbose,
        )

        # Initialize storage for last plot info
        self._last_plot_figures = {}
        self._last_plot_paths = {}

        # Process each group
        for group_keys, group_df in grouped:
            # Create or use provided figure/axes
            if target_ax is None:
                fig, ax = self.layout_manager.create_figure(figure_size)
                if not is_inset:
                    self._last_plot_figures[group_keys] = (fig, ax, group_df)
            else:
                ax = target_ax
                fig = ax.figure

            # Configure axes layout
            if not is_inset:
                self.layout_manager.configure_existing_axes(
                    ax=ax,
                    x_variable=self.xaxis_variable_name,
                    y_variable=self.yaxis_variable_name,
                    font_size=font_size,
                    xaxis_label=xaxis_label,
                    yaxis_label=yaxis_label,
                    show_xaxis_label=show_xaxis_label,
                    show_yaxis_label=show_yaxis_label,
                    xaxis_log_scale=xaxis_log_scale,
                    yaxis_log_scale=yaxis_log_scale,
                    xlim=xlim,
                    ylim=ylim,
                    xaxis_start_at_zero=xaxis_start_at_zero,
                    yaxis_start_at_zero=yaxis_start_at_zero,
                    invert_xaxis=invert_xaxis,
                    invert_yaxis=invert_yaxis,
                    apply_custom_function=customization_function,
                )
            else:
                # For insets, apply essential configurations but with inset-specific styling
                self.layout_manager.configure_inset_axes(
                    ax=ax,
                    x_variable=self.xaxis_variable_name,
                    y_variable=self.yaxis_variable_name,
                    show_xaxis_label=show_xaxis_label,
                    show_yaxis_label=show_yaxis_label,
                    font_size=max(8, font_size - 4),  # Smaller font for insets
                    xaxis_log_scale=xaxis_log_scale,
                    yaxis_log_scale=yaxis_log_scale,
                    xlim=xlim,
                    ylim=ylim,
                    xaxis_start_at_zero=xaxis_start_at_zero,
                    yaxis_start_at_zero=yaxis_start_at_zero,
                    invert_xaxis=invert_xaxis,
                    invert_yaxis=invert_yaxis,
                )

            # Prepare metadata for this group
            metadata = self._extract_group_metadata(group_keys, group_df)

            if grouping_variable:
                # Handle grouped plotting
                self._create_grouped_plot(
                    ax=ax,
                    group_df=group_df,
                    metadata=metadata,
                    grouping_variable=grouping_variable,
                    labeling_variable=labeling_variable,
                    sorting_variable=sorting_variable,
                    sort_ascending=sort_ascending,
                    marker_color_map=marker_color_map,
                    color_index_shift=color_index_shift,
                    marker_size=marker_size,
                    empty_markers=empty_markers,
                    alternate_filled_markers=alternate_filled_markers,
                    alternate_filled_markers_reversed=alternate_filled_markers_reversed,
                    capsize=capsize,
                    include_interpolation=include_interpolation,
                    annotation_variable=annotation_variable,
                    annotation_label=annotation_label,
                    annotation_range=annotation_range,
                    annotation_fontsize=annotation_fontsize,
                    annotation_boxstyle=annotation_boxstyle,
                    annotation_alpha=annotation_alpha,
                    annotation_offset=annotation_offset,
                    fit_function=fit_function,
                    fit_label_format=fit_label_format,
                    show_fit_parameters_on_plot=show_fit_parameters_on_plot,
                    fit_curve_style=fit_curve_style,
                    fit_label_location=fit_label_location,
                    fit_index_range=fit_slice,
                    fit_on_values=fit_on_values,
                    fit_label_in_legend=fit_label_in_legend,
                    fit_curve_range=fit_curve_range,
                    fit_min_data_points=fit_min_data_points,
                    legend_number_format=legend_number_format,
                )
            else:
                # Handle individual plotting
                self._create_individual_plot(
                    ax=ax,
                    group_df=group_df,
                    styling_variable=styling_variable,
                    marker_color_map=marker_color_map,
                    color_index_shift=color_index_shift,
                    marker_size=marker_size,
                    empty_markers=empty_markers,
                    capsize=capsize,
                    include_interpolation=include_interpolation,
                    annotation_variable=annotation_variable,
                    annotation_label=annotation_label,
                    annotation_range=annotation_range,
                    annotation_fontsize=annotation_fontsize,
                    annotation_boxstyle=annotation_boxstyle,
                    annotation_alpha=annotation_alpha,
                    annotation_offset=annotation_offset,
                    fit_function=fit_function,
                    fit_label_format=fit_label_format,
                    show_fit_parameters_on_plot=show_fit_parameters_on_plot,
                    fit_curve_style=fit_curve_style,
                    fit_label_location=fit_label_location,
                    fit_index_range=fit_index_range,
                    fit_curve_range=fit_curve_range,
                    fit_min_data_points=fit_min_data_points,
                )

            # Configure legend
            if grouping_variable and include_legend:
                self.style_manager.configure_legend(
                    ax=ax,
                    include_legend=include_legend,
                    legend_location=legend_location,
                    legend_columns=legend_columns,
                    include_legend_title=include_legend_title,
                    font_size=font_size,
                    grouping_variable=grouping_variable,
                    labeling_variable=labeling_variable,
                )

            # Add plot title
            if include_plot_title and not is_inset:
                title = self._construct_plot_title(
                    metadata=metadata,
                    custom_plot_title=custom_plot_title,
                    custom_plot_titles_dict=custom_plot_titles_dict,
                    title_from_columns=title_from_columns,
                    group_keys=group_keys,
                    grouping_variable=grouping_variable,
                    labeling_variable=labeling_variable,
                    leading_plot_substring=leading_plot_substring,
                    excluded_from_title_list=excluded_from_title_list,
                    title_number_format=title_number_format,
                    title_exponential_format=title_exponential_format,
                    title_wrapping_length=title_wrapping_length,
                )
                ax.set_title(
                    title,
                    fontsize=title_size,
                    weight="bold" if bold_title else "normal",
                )

            # Adjust figure margins - only for main figures, not
            # subfigures
            if not is_inset and isinstance(fig, Figure):
                self.style_manager.apply_figure_margins(
                    fig,
                    left=left_margin_adjustment,
                    right=right_margin_adjustment,
                    bottom=bottom_margin_adjustment,
                    top=top_margin_adjustment,
                )

            # Apply post-plot customization function if provided
            if post_plot_customization_function is not None:
                self._apply_post_plot_customization(
                    post_plot_customization_function=post_plot_customization_function,
                    ax=ax,
                    group_df=group_df,
                    grouping_variable=grouping_variable,
                    metadata=metadata,
                    group_keys=group_keys,
                )

            # Save figure - only save main figures
            if save_figure and not is_inset and isinstance(fig, Figure):
                self._save_plot(
                    fig=fig,
                    metadata=metadata,
                    group_keys=group_keys,
                    grouping_variable=grouping_variable,
                    file_format=file_format,
                )

        return self

    def add_inset(self, **kwargs) -> "DataPlotter":
        """
        Add an inset to the most recently created plots.

        This method delegates to the InsetManager for all inset
        functionality. All parameters are passed through to
        InsetManager.add_inset().

        Returns:
        --------
        DataPlotter
            Self for method chaining
        """
        if not hasattr(self, "_last_plot_figures") or not self._last_plot_figures:
            raise ValueError("Call plot() before add_inset()")

        # For each stored figure, add an inset
        for group_keys, (fig, ax, group_df) in self._last_plot_figures.items():
            try:
                # Use inset manager to add the inset
                inset_ax = self.inset_manager.add_inset(
                    figure=fig,
                    main_axes=ax,
                    dataframe=group_df,
                    plots_directory=self.plots_directory,
                    **kwargs,
                )

                if inset_ax is not None and group_keys in self._last_plot_paths:
                    # Save the modified figure
                    save_path = self._last_plot_paths[group_keys]
                    fig.savefig(save_path)
                    plt.close(fig)

            except Exception as e:
                print(f"Warning: Failed to add inset to group {group_keys}: {e}")
                import traceback

                traceback.print_exc()

        return self

    def get_fit_results(self) -> Dict[Any, Optional[Dict[str, Any]]]:
        """
        Retrieve stored fit results from the most recent plot() call.

        Returns:
        --------
        Dict[Any, Optional[Dict[str, Any]]]
            Dictionary mapping group values to fit result dictionaries.
            Fit result contains: 'function', 'parameters', 'covariance',
            'r_squared', 'method', etc.
        """
        return self.stored_fit_results.copy()

    def _apply_post_plot_customization(
        self,
        post_plot_customization_function: Callable[..., None],
        ax: Axes,
        group_df: pd.DataFrame,
        grouping_variable: Optional[Union[str, List[str]]],
        metadata: Dict[str, Any],
        group_keys: Tuple[Any, ...],
    ) -> None:
        """
        Apply post-plot customization function with appropriate context.

        Parameters:
        -----------
        post_plot_customization_function : Callable
            User-provided customization function
        ax : Axes
            The matplotlib axes object
        group_df : pd.DataFrame
            DataFrame containing the plotted data
        grouping_variable : str or List[str], optional
            Variable(s) used for grouping
        metadata : dict
            Plot metadata
        group_keys : tuple
            Keys identifying the current group
        """
        if grouping_variable:
            # Grouped plot case - call once with all data
            all_plot_data = {}
            all_fit_results = {}

            for group_value in self.stored_fit_results.keys():
                # Get group data
                if isinstance(grouping_variable, list):
                    group_mask = (group_df[grouping_variable] == list(group_value)).all(
                        axis=1
                    )
                else:
                    group_mask = group_df[grouping_variable] == group_value

                group_subset = group_df[group_mask]

                all_plot_data[group_value] = {
                    "x_data": group_subset[self.xaxis_variable_name].to_numpy(),
                    "y_data": group_subset[self.yaxis_variable_name].to_numpy(),
                    "group_value": group_value,
                }
                all_fit_results[group_value] = self.stored_fit_results.get(group_value)

            # Call with comprehensive context
            post_plot_customization_function(
                ax=ax,
                plot_data=all_plot_data,  # Dict of {group_value: data}
                fit_results=all_fit_results,  # Dict of {group_value: fit_result}
                group_info={
                    "grouping_variable": grouping_variable,
                    "metadata": metadata,
                    "num_groups": len(all_plot_data),
                },
                plot_type="grouped",
            )
        else:
            # Individual plot case - simpler
            plot_data = {
                "x_data": group_df[self.xaxis_variable_name].to_numpy(),
                "y_data": group_df[self.yaxis_variable_name].to_numpy(),
            }

            # Get fit results if available
            fit_result = (
                self.stored_fit_results.get(list(self.stored_fit_results.keys())[0])
                if self.stored_fit_results
                else None
            )

            post_plot_customization_function(
                ax=ax,
                plot_data=plot_data,
                fit_results=fit_result,
                group_info={"metadata": metadata, "group_keys": group_keys},
                plot_type="individual",
            )

    def _prepare_grouping_configuration(
        self,
        grouping_variable: Optional[Union[str, List[str]]],
        excluded_from_grouping_list: Optional[List[str]],
        styling_variable: Optional[str],
    ) -> Dict[str, Any]:
        """Prepare grouping configuration by determining excluded
        parameters."""
        excluded = set(excluded_from_grouping_list or [])

        # Exclude axis variables from grouping
        for axis_variable in [self.xaxis_variable_name, self.yaxis_variable_name]:
            if axis_variable in self.list_of_multivalued_tunable_parameter_names:
                excluded.add(axis_variable)

        # Exclude grouping variables from grouping
        if grouping_variable:
            grouping_columns = (
                [grouping_variable]
                if isinstance(grouping_variable, str)
                else grouping_variable
            )
            for col in grouping_columns:
                if col not in self.list_of_multivalued_tunable_parameter_names:
                    raise ValueError(f"'{col}' is not a multivalued tunable parameter.")
                excluded.add(col)

        # Validate styling variable
        if styling_variable:
            if (
                styling_variable
                not in self.list_of_tunable_parameter_names_from_dataframe
            ):
                raise ValueError("'styling_variable' must be tunable parameter.")

        return {
            "excluded": list(excluded),
            "grouping_columns": grouping_columns if grouping_variable else None,
        }

    def _extract_group_metadata(
        self, group_keys: Tuple[Any, ...], group_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract metadata dictionary for a plot group."""
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        metadata = {
            name: group_key
            for name, group_key in zip(
                self.reduced_multivalued_tunable_parameter_names_list, group_keys
            )
        }

        # Add method and kernel type if constant in the group
        for special in ["Overlap_operator_method", "Kernel_operator_type"]:
            if special in group_df.columns:
                unique_vals = group_df[special].unique()
                if len(unique_vals) == 1:
                    metadata[special] = unique_vals[0]

        return metadata

    def _create_grouped_plot(self, ax: Axes, group_df: pd.DataFrame, **kwargs) -> None:
        """Create a plot with grouped data series."""
        grouping_variable = kwargs["grouping_variable"]

        # Get unique group values with sorting
        unique_group_values = self._get_sorted_group_values(
            group_df,
            grouping_variable,
            kwargs.get("sorting_variable"),
            kwargs.get("sort_ascending"),
        )

        # Generate style map
        style_map = self.style_manager.generate_marker_color_map(
            unique_group_values,
            custom_map=kwargs.get("marker_color_map"),
            index_shift=kwargs.get("color_index_shift", 0),
        )

        # Plot each group
        for curve_index, value in enumerate(unique_group_values):
            subgroup = self._extract_subgroup(group_df, grouping_variable, value)

            # Generate label
            if kwargs.get("labeling_variable"):
                label = self._generate_data_group_label(
                    subgroup,
                    kwargs["labeling_variable"],
                    kwargs.get("legend_number_format", ".2f"),
                    constants,
                )
            else:
                label = self._format_group_value_as_label(value)

            # Get style
            marker, color = style_map[value]

            # Handle alternating markers
            empty_marker = self._should_use_empty_marker(
                curve_index,
                kwargs.get("empty_markers", False),
                kwargs.get("alternate_filled_markers", False),
                kwargs.get("alternate_filled_markers_reversed", False),
            )

            # Apply curve fitting if requested
            fit_label_suffix = ""
            if kwargs.get("fit_function") and self._should_apply_fit(
                kwargs.get("fit_on_values"), value
            ):
                fit_result = self._apply_curve_fitting(ax, subgroup, color, kwargs)
                self.stored_fit_results[value] = fit_result  # Store results by group
                if fit_result and kwargs.get("fit_label_in_legend"):
                    fit_label_suffix = self._format_fit_label_suffix(fit_result, kwargs)

            # Plot the group data
            self._plot_single_group(
                ax=ax,
                group_df=subgroup,
                label=label + fit_label_suffix,
                color=color,
                marker=marker,
                marker_size=kwargs.get("marker_size", 8),
                capsize=kwargs.get("capsize", 5),
                empty_markers=empty_marker,
                include_interpolation=kwargs.get("include_interpolation", False),
                annotation_params=self._extract_annotation_params(kwargs),
            )

    def _create_individual_plot(
        self, ax: Axes, group_df: pd.DataFrame, **kwargs
    ) -> None:
        """Create a plot with a single data series."""
        # Determine style
        if kwargs.get("styling_variable"):
            styling_values = self.unique_values(kwargs["styling_variable"])
            style_map = self.style_manager.generate_marker_color_map(
                styling_values,
                custom_map=kwargs.get("marker_color_map"),
                index_shift=kwargs.get("color_index_shift", 0),
            )
            style_key = group_df[kwargs["styling_variable"]].iloc[0]
            marker, color = style_map.get(style_key, ("o", "blue"))
        else:
            marker, color = ("o", "blue")

        # Apply curve fitting if requested
        if kwargs.get("fit_function"):
            fit_result = self._apply_curve_fitting(ax, group_df, color, kwargs)
            if fit_result:
                self.stored_fit_results['individual'] = fit_result

        # Plot the data
        self._plot_single_group(
            ax=ax,
            group_df=group_df,
            label=None,
            color=color,
            marker=marker,
            marker_size=kwargs.get("marker_size", 8),
            capsize=kwargs.get("capsize", 5),
            empty_markers=kwargs.get("empty_markers", False),
            include_interpolation=kwargs.get("include_interpolation", False),
            annotation_params=self._extract_annotation_params(kwargs),
        )

    def _plot_single_group(
        self,
        ax: Axes,
        group_df: pd.DataFrame,
        label: Optional[str],
        color: str,
        marker: str,
        marker_size: int,
        capsize: float,
        empty_markers: bool,
        include_interpolation: bool,
        annotation_params: Dict[str, Any],
    ) -> None:
        """Plot a single group of data using the data processor."""
        x_data = group_df[self.xaxis_variable_name].to_numpy()
        y_data = group_df[self.yaxis_variable_name].to_numpy()

        # Validate data
        is_valid, error_msg = self.data_processor.validate_plot_data(
            x_data, y_data, label
        )
        if not is_valid:
            print(f"Warning: {error_msg}")
            return

        # Filter valid data
        x_filtered, y_filtered, _ = self.data_processor.filter_valid_data(
            x_data, y_data
        )

        if len(x_filtered) == 0:
            print(f"Warning: No valid data points to plot for {label}")
            return

        # Determine plot method based on data type
        x_is_tuple = self.data_processor.is_tuple_array(x_filtered)
        y_is_tuple = self.data_processor.is_tuple_array(y_filtered)

        # Plot based on data types
        if not x_is_tuple and not y_is_tuple:
            # Scatter plot - use plot_type="scatter"
            marker_props = self.style_manager.get_marker_properties(
                marker, not empty_markers, color, marker_size, plot_type="scatter"
            )
            ax.scatter(x_filtered, y_filtered, label=label, **marker_props)
        elif not x_is_tuple and y_is_tuple:
            # Error bars in y - use plot_type="errorbar"
            marker_props = self.style_manager.get_marker_properties(
                marker, not empty_markers, color, marker_size, plot_type="errorbar"
            )
            y_vals, y_errs = self.data_processor.extract_values_and_errors(y_filtered)
            ax.errorbar(
                x_filtered,
                y_vals,
                yerr=y_errs,
                capsize=capsize,
                label=label,
                linestyle="none",  # No connecting lines, just markers
                **marker_props,
            )
        elif x_is_tuple and not y_is_tuple:
            # Error bars in x - use plot_type="errorbar"
            marker_props = self.style_manager.get_marker_properties(
                marker, not empty_markers, color, marker_size, plot_type="errorbar"
            )
            x_vals, x_errs = self.data_processor.extract_values_and_errors(x_filtered)
            ax.errorbar(
                x_vals,
                y_filtered,
                xerr=x_errs,
                capsize=capsize,
                label=label,
                linestyle="none",  # No connecting lines, just markers
                **marker_props,
            )
        else:
            # Error bars in both x and y - use plot_type="errorbar"
            marker_props = self.style_manager.get_marker_properties(
                marker, not empty_markers, color, marker_size, plot_type="errorbar"
            )
            x_vals, x_errs = self.data_processor.extract_values_and_errors(x_filtered)
            y_vals, y_errs = self.data_processor.extract_values_and_errors(y_filtered)
            ax.errorbar(
                x_vals,
                y_vals,
                xerr=x_errs,
                yerr=y_errs,
                capsize=capsize,
                label=label,
                linestyle="none",  # No connecting lines, just markers
                **marker_props,
            )

        # Add interpolation if requested
        if include_interpolation:
            self._add_interpolation(ax, x_filtered, y_filtered, color)

        # Add annotations if requested
        if annotation_params["variable"]:
            self._add_data_annotations(ax, group_df, annotation_params, color)

    def _add_interpolation(
        self, ax: Axes, x_data: np.ndarray, y_data: np.ndarray, color: str
    ) -> None:
        """Add interpolated smooth curve to the plot."""
        x_smooth, y_smooth = self.data_processor.create_interpolation(
            x_data, y_data, num_points=100, kind="cubic"
        )

        if x_smooth is not None and y_smooth is not None:
            ax.plot(x_smooth, y_smooth, ":", color=color, alpha=0.7)

    def _add_data_annotations(
        self,
        ax: Axes,
        group_df: pd.DataFrame,
        annotation_params: Dict[str, Any],
        color: str,
    ) -> None:
        """Add data point annotations using the annotation manager."""

        # Validate that plot variables are set
        if self.xaxis_variable_name is None or self.yaxis_variable_name is None:
            raise ValueError("Plot variables must be set before adding annotations")

        self.annotation_manager.add_data_point_annotations(
            ax=ax,
            dataframe=group_df,
            x_variable=self.xaxis_variable_name,
            y_variable=self.yaxis_variable_name,
            annotation_variable=annotation_params["variable"],
            annotation_label=annotation_params["label"],
            annotation_range=annotation_params["range"],
            style_overrides={
                "fontsize": annotation_params["fontsize"],
                "boxstyle": annotation_params["boxstyle"],
                "alpha": annotation_params["alpha"],
                "offset": annotation_params["offset"],
            },
            series_color=color,
        )

    def _apply_curve_fitting(
        self, ax: Axes, group_df: pd.DataFrame, color: str, kwargs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Apply curve fitting using the curve fitter."""
        x_data = group_df[self.xaxis_variable_name].to_numpy()
        y_data = group_df[self.yaxis_variable_name].to_numpy()

        return self.curve_fitter.apply_fit(
            ax=ax,
            x_data=x_data,
            y_data=y_data,
            fit_function=kwargs["fit_function"],
            show_parameters=kwargs.get("show_fit_parameters_on_plot", True),
            curve_style=kwargs.get("fit_curve_style"),
            parameter_format=kwargs.get("fit_label_format", ".2f"),
            label_location=kwargs.get("fit_label_location", "top left"),
            index_range=kwargs.get("fit_index_range"),
            curve_range=kwargs.get("fit_curve_range"),
            series_color=color,
            min_data_points=kwargs.get("fit_min_data_points"),
        )

    def _construct_plot_title(
        self,
        metadata: Dict[str, Any],
        custom_plot_title: Optional[str],
        custom_plot_titles_dict: Optional[Dict[Any, str]],
        title_from_columns: Optional[List[str]],
        group_keys: Tuple[Any, ...],
        grouping_variable: Optional[Union[str, List[str]]],
        labeling_variable: Optional[Union[str, List[str]]],
        leading_plot_substring: Optional[str],
        excluded_from_title_list: Optional[List[str]],
        title_number_format: str,
        title_exponential_format: str,
        title_wrapping_length: int,
    ) -> str:
        """Construct plot title using the title builder."""
        if custom_plot_title:
            return custom_plot_title

        if custom_plot_titles_dict is not None:
            # Handle empty or single-element group_keys safely
            if len(group_keys) == 0:
                title_key = None  # or some default value
            elif len(group_keys) == 1:
                title_key = group_keys[0]
            else:
                title_key = group_keys

            # Only look up in dict if we have a valid key
            if title_key is not None and title_key in custom_plot_titles_dict:
                return custom_plot_titles_dict[title_key]

        # Prepare exclusion list
        excluded = set(excluded_from_title_list or [])
        excluded.update(
            [
                "Main_program_type",
                "Maximum_Lanczos_iterations",
                "MPI_geometry",
                "Number_of_spinors",
                "Number_of_vectors",
                "Outer_solver_epsilon",
                "Threads_per_process",
            ]
        )
        if grouping_variable:
            if isinstance(grouping_variable, str):
                excluded.add(grouping_variable)
            else:
                excluded.update(grouping_variable)
        if labeling_variable:
            if isinstance(labeling_variable, str):
                excluded.add(labeling_variable)
            else:
                excluded.update(labeling_variable)

        # Combine metadata with unique value columns
        full_metadata = {**metadata, **self.unique_value_columns_dictionary}

        # Temporarily update title builder formatting
        original_number_format = self.title_builder.title_number_format
        original_exponential_format = self.title_builder.title_exponential_format

        self.title_builder.title_number_format = title_number_format
        self.title_builder.title_exponential_format = title_exponential_format

        try:
            result = self.title_builder.build(
                metadata_dict=full_metadata,
                tunable_params=self.list_of_tunable_parameter_names_from_dataframe,
                excluded=excluded,
                leading_substring=leading_plot_substring,
                title_from_columns=title_from_columns,
                wrapping_length=title_wrapping_length,
            )
        finally:
            # Restore original formatting
            self.title_builder.title_number_format = original_number_format
            self.title_builder.title_exponential_format = original_exponential_format

        return result

    def _save_plot(
        self,
        fig: Figure,
        metadata: Dict[str, Any],
        group_keys: Tuple[Any, ...],
        grouping_variable: Optional[Union[str, List[str]]],
        file_format: Optional[str] = None,
    ) -> None:
        """Save the plot using file manager and filename builder."""

        # Validate that plots_base_name is set
        if self.plots_base_name is None:
            raise ValueError(
                "plots_base_name not set. Call set_plot_variables() first."
            )

        # Construct filename
        filename = self.filename_builder.build(
            metadata_dict=metadata,
            base_name=self.plots_base_name,
            multivalued_params=self.reduced_multivalued_tunable_parameter_names_list,
            grouping_variable=grouping_variable,
            include_combined_prefix=(grouping_variable is not None),
        )

        # Determine save directory
        if grouping_variable:
            if isinstance(grouping_variable, str):
                nested_dirname = f"Grouped_by_{grouping_variable}"
            else:
                nested_dirname = "Grouped_by_" + "_and_".join(grouping_variable)

            self.combined_plots_subdirectory = self.file_manager.prepare_subdirectory(
                os.path.join(self.plots_base_name, nested_dirname)
            )
            save_directory = self.combined_plots_subdirectory
        else:
            save_directory = self.individual_plots_subdirectory

        # Get full path and save
        full_path = self.file_manager.plot_path(
            save_directory, filename, format=file_format
        )
        self._last_plot_paths[group_keys] = full_path

        fig.savefig(full_path)
        plt.close(fig)

    # Helper methods for grouped plotting
    def _get_sorted_group_values(
        self,
        group_df: pd.DataFrame,
        grouping_variable: Union[str, List[str]],
        sorting_variable: Optional[str],
        sort_ascending: Optional[bool],
    ) -> List[Tuple[Any, ...]]:
        """Get unique group values with proper sorting."""
        grouping_columns = (
            [grouping_variable]
            if isinstance(grouping_variable, str)
            else grouping_variable
        )

        if sorting_variable:
            sorted_df = group_df.sort_values(
                by=sorting_variable, ascending=(sort_ascending is not False)
            )
            unique_groups = (
                sorted_df[grouping_columns]
                .drop_duplicates()
                .apply(tuple, axis=1)
                .tolist()
            )
        else:
            unique_groups = (
                group_df[grouping_columns]
                .drop_duplicates()
                .apply(tuple, axis=1)
                .tolist()
            )

            if sort_ascending is True:
                unique_groups = sorted(unique_groups)
            elif sort_ascending is False:
                unique_groups = sorted(unique_groups, reverse=True)

        return unique_groups

    def _extract_subgroup(
        self,
        group_df: pd.DataFrame,
        grouping_variable: Union[str, List[str]],
        value: Tuple[Any, ...],
    ) -> pd.DataFrame:
        """Extract subgroup based on grouping variable and value."""
        if isinstance(grouping_variable, str):
            actual_value = (
                value[0] if isinstance(value, tuple) and len(value) == 1 else value
            )
            return group_df[group_df[grouping_variable] == actual_value]
        else:
            mask = group_df[grouping_variable].apply(tuple, axis=1) == value
            return group_df[mask]

    def _generate_data_group_label(
        self,
        subgroup: pd.DataFrame,
        labeling_variable: Union[str, List[str]],
        number_format: str,
        constants,
    ) -> str:
        """Generate label for a data group."""
        if isinstance(labeling_variable, list) and len(labeling_variable) == 2:
            var1, var2 = labeling_variable
            val1 = subgroup[var1].unique()
            val2 = subgroup[var2].unique()

            # Extract single values or handle multiple values
            if len(val1) == 1:
                val1 = val1[0]
            else:
                val1 = f"[{', '.join(map(str, val1))}]"  # Handle multiple values

            if len(val2) == 1:
                val2_single = val2[0]
                if isinstance(val2_single, (int, float)):
                    # Use f-string formatting instead of format()
                    # function
                    val2_formatted = f"{val2_single:{number_format}}"
                else:
                    val2_formatted = str(val2_single)
            else:
                # Handle multiple values
                val2_formatted = f"[{', '.join(str(v) for v in val2)}]"

            var2_label = constants.LEGEND_LABELS_BY_COLUMN_NAME.get(var2, var2)
            return f"{val1} ({var2_label}{val2_formatted})"
        else:
            # Handle single labeling variable
            if isinstance(labeling_variable, str):
                label_value = subgroup[labeling_variable].unique()
            else:
                # This shouldn't happen based on the if/else logic, but
                # handle it
                label_value = subgroup[labeling_variable[0]].unique()

            if len(label_value) == 1:
                single_value = label_value[0]
                if isinstance(single_value, (int, float)):
                    # Use f-string formatting instead of format()
                    # function
                    formatted_value = f"{single_value:{number_format}}"
                else:
                    formatted_value = str(single_value)
                return formatted_value
            else:
                # Handle multiple values
                return f"[{', '.join(str(v) for v in label_value)}]"

    def _format_group_value_as_label(self, value: Tuple[Any, ...]) -> str:
        """Format group value as a readable label."""
        if isinstance(value, tuple):
            return " ".join(str(v) for v in value)
        return str(value)

    def _should_use_empty_marker(
        self,
        curve_index: int,
        empty_markers: bool,
        alternate_filled_markers: bool,
        alternate_filled_markers_reversed: bool,
    ) -> bool:
        """Determine if this curve should use empty markers."""
        if alternate_filled_markers:
            empty_marker = curve_index % 2 == 1  # odd indices → empty
            if alternate_filled_markers_reversed:
                empty_marker = not empty_marker
            return empty_marker
        return empty_markers

    def _should_apply_fit(
        self, fit_on_values: Optional[Union[Any, Tuple[Any, ...]]], value: Any
    ) -> bool:
        """Determine if fitting should be applied to this group."""
        if fit_on_values is None:
            return True

        if not isinstance(fit_on_values, tuple):
            fit_on_values = (fit_on_values,)

        return value == fit_on_values

    def _format_fit_label_suffix(
        self, fit_result: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> str:
        """Format fit results for inclusion in legend labels."""
        fit_function = kwargs.get("fit_function")
        fit_format = kwargs.get("fit_label_format", ".2f")

        if fit_result["method"] == "scipy":
            params = fit_result["parameters"]

            if fit_function == "exponential" and len(params) >= 3:
                c_fmt = format(params[2], fit_format)
                return f" (a$m^{{n\\to\\infty}}_{{\\text{{PCAC}}}}$={c_fmt})"
            else:
                a_fmt = format(params[0], fit_format)
                b_fmt = format(params[1], fit_format) if len(params) > 1 else ""
                if b_fmt:
                    return f" (a={a_fmt}, b={b_fmt})"
                else:
                    return f" (a={a_fmt})"
        elif fit_result["method"] == "gvar":
            # Handle gvar results
            try:
                import gvar

                params = gvar.mean(fit_result["parameters"])
                a_fmt = format(params[0], fit_format)
                b_fmt = format(params[1], fit_format) if len(params) > 1 else ""
                if b_fmt:
                    return f" (a={a_fmt}, b={b_fmt})"
                else:
                    return f" (a={a_fmt})"
            except ImportError:
                return " (fit applied)"

        return " (fit applied)"

    def _extract_annotation_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract annotation parameters from kwargs."""
        return {
            "variable": kwargs.get("annotation_variable"),
            "label": kwargs.get("annotation_label", ""),
            "range": kwargs.get("annotation_range"),
            "fontsize": kwargs.get("annotation_fontsize", 8),
            "boxstyle": kwargs.get("annotation_boxstyle", "round,pad=0.3"),
            "alpha": kwargs.get("annotation_alpha", 0.7),
            "offset": kwargs.get("annotation_offset", (0, 10)),
        }

    # Convenience methods for accessing managers
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

    # Additional utility methods that were in the original class
    def add_custom_annotation(
        self, ax: Axes, x: float, y: float, text: str, **kwargs
    ) -> None:
        """Add a custom annotation using the annotation manager."""
        self.annotation_manager.add_custom_annotation(ax, x, y, text, **kwargs)

    def apply_curve_fit(
        self, ax: Axes, x_data: np.ndarray, y_data: np.ndarray, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Apply curve fitting using the curve fitter."""
        return self.curve_fitter.apply_fit(ax, x_data, y_data, **kwargs)

    def set_default_file_format(self, format: str) -> None:
        """Set the default file format for saved plots."""
        self.file_manager.set_default_format(format)

    def clear_style_cache(self) -> None:
        """Clear cached style mappings."""
        self.style_manager.clear_cache()

    # Migration helpers for backward compatibility
    def _prepare_plot_subdirectory(
        self, subdir_name: str, clear_existing: bool = False
    ) -> str:
        """Legacy method - delegates to file manager."""
        return self.file_manager.prepare_subdirectory(subdir_name, clear_existing)

    def _generate_marker_color_map(
        self,
        grouping_values: List[Any],
        custom_map: Optional[Dict[Any, Tuple[str, str]]] = None,
        index_shift: int = 0,
    ) -> Dict[Any, Tuple[str, str]]:
        """Legacy method - delegates to style manager."""
        return self.style_manager.generate_marker_color_map(
            grouping_values, custom_map, index_shift
        )

    def _construct_plot_filename(self, metadata_dict: Dict[str, Any], **kwargs) -> str:
        """Legacy method - delegates to filename builder."""
        return self.filename_builder.build(
            metadata_dict=metadata_dict,
            base_name=self.plots_base_name,
            multivalued_params=self.reduced_multivalued_tunable_parameter_names_list,
            **kwargs,
        )

    # Method to maintain compatibility with original API
    @property
    def plots_base_name(self) -> str:
        """Get the base name for plots (guaranteed to be set)."""
        if self._plots_base_name is None:
            raise ValueError(
                "plots_base_name not set. Call set_plot_variables() first."
            )
        return self._plots_base_name

    @plots_base_name.setter
    def plots_base_name(self, value: Optional[str]) -> None:
        """Set the base name for plots."""
        self._plots_base_name = value

    def set_default_title_formats(
        self,
        number_format: Optional[str] = None,
        exponential_format: Optional[str] = None,
    ) -> None:
        """
        Set default title formatting for numbers and exponential values.

        Parameters:
        -----------
        number_format : str, optional
            Format string for regular numeric values (e.g., ".2f", ".3g")
        exponential_format : str, optional
            Format string for exponential values (e.g., ".0e", ".1e")
        """
        if number_format is not None:
            self.title_builder.set_number_format(number_format)
        if exponential_format is not None:
            self.title_builder.set_exponential_format(exponential_format)

    def __repr__(self) -> str:
        """String representation of the DataPlotter."""
        return (
            f"DataPlotter("
            f"dataframe={self.dataframe.shape}, "
            f"plots_directory='{self.plots_directory}', "
            f"x_var='{self.xaxis_variable_name}', "
            f"y_var='{self.yaxis_variable_name}')"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"DataPlotter with {self.dataframe.shape[0]} rows, "
            f"{self.dataframe.shape[1]} columns\n"
            f"Output directory: {self.plots_directory}\n"
            f"Plot variables: {self.xaxis_variable_name} vs {self.yaxis_variable_name}"
        )
