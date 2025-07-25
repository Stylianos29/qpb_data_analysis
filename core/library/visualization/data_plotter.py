import os, shutil
from typing import Optional, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
import gvar
import lsqfit
from matplotlib.colors import to_rgba
from scipy.interpolate import make_interp_spline

from library.data.analyzer import DataFrameAnalyzer
from library.data.table_generator import TableGenerator
from library import constants

FIT_LABEL_POSITIONS = {
    "top left": ((0.05, 0.95), ("left", "top")),
    "top right": ((0.95, 0.95), ("right", "top")),
    "bottom left": ((0.05, 0.05), ("left", "bottom")),
    "bottom right": ((0.95, 0.05), ("right", "bottom")),
    "center": ((0.5, 0.5), ("center", "center")),
}

from .base_plotter import _PlotFileManager, _PlotTitleBuilder, _PlotFilenameBuilder


class DataPlotter(DataFrameAnalyzer):
    """
    A plotting interface for visualizing grouped data from a DataFrame using
    matplotlib.

    This class extends `DataFrameAnalyzer` to provide plotting functionality
    tailored to data that varies across subsets of tunable parameters. It
    supports generating visual summaries of results for each combination of
    multivalued tunable parameters, and is designed to produce
    publication-quality figures in a structured directory hierarchy.

    Attributes:
    -----------
    dataframe : pd.DataFrame
        The main DataFrame containing raw or processed data to be visualized.
    plots_directory : str
        The base output directory where all plots will be saved.
    individual_plots_subdirectory : str
        Subdirectory within `plots_directory` used for individual plots.
    combined_plots_subdirectory : str
        Subdirectory within `plots_directory` used for combined plots.
    xaxis_variable_name : Optional[str]
        Name of the variable to be used as x-axis in plots.
    yaxis_variable_name : Optional[str]
        Name of the variable to be used as y-axis in plots.
    plots_base_name : Optional[str]
        Base string used in naming plot files and directories.
    """

    def __init__(self, dataframe: pd.DataFrame, plots_directory: str):
        """
        Initialize the DataPlotter with a DataFrame and an output directory for
        plots.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input data for plotting. This is expected to include tunable
            parameters and output quantities as defined in the analysis logic.
        plots_directory : str
            The base path where plots will be saved. Subdirectories will be
            created inside this path for individual and grouped plots.

        Raises:
        -------
        TypeError:
            If the input is not a Pandas DataFrame.
        ValueError:
            If the provided `plots_directory` does not point to a valid
            directory.
        """
        super().__init__(dataframe)

        # Initialize file manager
        self._file_manager = _PlotFileManager(plots_directory)
        self.plots_directory = plots_directory

        # Initialize builders
        self._title_builder = _PlotTitleBuilder(constants.TITLE_LABELS_BY_COLUMN_NAME)
        self._filename_builder = _PlotFilenameBuilder(
            constants.FILENAME_LABELS_BY_COLUMN_NAME
        )

        # if not os.path.isdir(plots_directory):
        #     raise ValueError(f"Invalid plots directory: '{plots_directory}'")

        # self.plots_directory = plots_directory
        self.individual_plots_subdirectory = plots_directory
        self.combined_plots_subdirectory = plots_directory

        self.xaxis_variable_name = None
        self.yaxis_variable_name = None
        self.plots_base_name = None

    def generate_column_uniqueness_report(
        self, max_width=80, separate_by_type=True
    ) -> str:
        assert isinstance(
            self.dataframe, pd.DataFrame
        ), "self.dataframe must be a DataFrame"
        table_generator = TableGenerator(self.dataframe)
        return table_generator.generate_column_uniqueness_report(
            max_width=max_width,
            separate_by_type=separate_by_type,
            export_to_file=False,
        )

    def _prepare_plot_subdirectory(
        self, subdir_name: str, clear_existing: bool = False
    ) -> str:
        """Delegate to file manager."""
        return self._file_manager.prepare_subdirectory(subdir_name, clear_existing)

    def _plot_group(
        self,
        ax,
        group_df: pd.DataFrame,
        label: Optional[str] = None,
        color: str = "blue",
        marker: str = "o",
        marker_size: int = 6,
        capsize: float = 5,
        empty_markers: bool = False,
        include_interpolation: bool = False,
        annotation_variable: Optional[str] = None,
        annotation_label: str = "",
        annotation_range: Optional[tuple] = None,
        annotation_fontsize: int = 8,
        annotation_boxstyle: str = "round,pad=0.3",
        annotation_alpha: float = 0.7,
        annotation_offset: tuple = (0, 10),  # (x_offset, y_offset) in points
    ):
        """
        Plot a single data group on the provided Matplotlib Axes object with optional annotations.

        Parameters:
        -----------
        annotation_variable : str, optional
            DataFrame column name to use for annotation values
        annotation_label : str, optional
            Prefix text for annotations (e.g., "N=" will show as "N=10")
        annotation_range : tuple, optional
            Controls which points to annotate: (start, end, step)
            - start: index to start annotations (0 for first point)
            - end: index to end annotations (None for all points)
            - step: annotate every nth point (1 for all, 2 for every other, etc.)
        annotation_fontsize : int, optional
            Font size for annotation text
        annotation_boxstyle : str, optional
            Matplotlib boxstyle for annotation boxes
        annotation_alpha : float, optional
            Transparency level for annotation boxes
        annotation_offset : tuple, optional
            (x, y) offset in points for annotation placement
        """
        x_raw = group_df[self.xaxis_variable_name].to_numpy()
        y_raw = group_df[self.yaxis_variable_name].to_numpy()

        # is_tuple_array = lambda arr: (isinstance(arr[0], tuple) and len(arr[0]) == 2)

        def is_tuple_array(arr):
            # Filter out any NaNs or Nones
            for element in arr:
                if isinstance(element, tuple) and len(element) == 2:
                    return True
            return False

        x_is_tuple = is_tuple_array(x_raw)
        y_is_tuple = is_tuple_array(y_raw)

        # if not (np.issubdtype(x_raw.dtype, np.number) or x_is_tuple):
        #     raise TypeError(f"x-axis data has unsupported type: {x_raw.dtype}")
        # if not (np.issubdtype(y_raw.dtype, np.number) or y_is_tuple):
        #     raise TypeError(f"y-axis data has unsupported type: {y_raw.dtype}")

        # if not (np.issubdtype(x_raw.dtype, np.number) or x_is_tuple):
        #     example = x_raw[0]
        #     raise TypeError(
        #         f"x-axis data has unsupported type: {x_raw.dtype}. "
        #         f"Example value: {example} (type: {type(example).__name__})"
        #     )

        # def is_numeric_array(arr):
        #     return all(
        #         isinstance(x, (int, float, np.integer, np.floating))
        #         or (np.isscalar(x) and np.isnan(x))
        #         for x in arr
        #     )

        def is_numeric_array(arr):
            # Check if the array contains data that can be plotted
            try:
                # Convert to numpy array and check dtype
                clean_arr = np.array([x for x in arr if x is not None], dtype=float)
                return True
            except (ValueError, TypeError):
                return False

        # # Then use:
        # if not (is_numeric_array(y_raw) or y_is_tuple):
        #     example = y_raw[0]
        #     raise TypeError(
        #         f"y-axis data has unsupported type: {y_raw.dtype}. "
        #         f"Example value: {example} (type: {type(example).__name__})"
        #     )

        if not (is_numeric_array(x_raw) or x_is_tuple):
            example = x_raw[0]
            raise TypeError(
                f"x-axis data has unsupported type: {x_raw.dtype}. "
                f"Example value: {example} (type: {type(example).__name__})"
            )

        if not (is_numeric_array(y_raw) or y_is_tuple):
            example = y_raw[0]
            raise TypeError(
                f"y-axis data has unsupported type: {y_raw.dtype}. "
                f"Example value: {example} (type: {type(example).__name__})"
            )

        # if not (np.issubdtype(y_raw.dtype, np.number) or y_is_tuple):
        #     example = y_raw[0]
        #     raise TypeError(
        #         f"y-axis data has unsupported type: {y_raw.dtype}. "
        #         f"Example value: {example} (type: {type(example).__name__})"
        #     )

        # Filter out NaNs and Infs
        if x_is_tuple and y_is_tuple:
            # Both are tuple arrays (value, error)
            valid_indices = []
            for i, ((x_val, x_err), (y_val, y_err)) in enumerate(zip(x_raw, y_raw)):
                if (
                    np.isfinite(x_val)
                    and np.isfinite(x_err)
                    and np.isfinite(y_val)
                    and np.isfinite(y_err)
                ):
                    valid_indices.append(i)

            if not valid_indices:
                print(f"Warning: No valid data points to plot for {label}")
                return

            x_filtered = np.array([x_raw[i] for i in valid_indices])
            y_filtered = np.array([y_raw[i] for i in valid_indices])

        elif x_is_tuple and not y_is_tuple:
            # x is tuple, y is scalar
            valid_indices = []
            for i, ((x_val, x_err), y_val) in enumerate(zip(x_raw, y_raw)):
                if np.isfinite(x_val) and np.isfinite(x_err) and np.isfinite(y_val):
                    valid_indices.append(i)

            if not valid_indices:
                print(f"Warning: No valid data points to plot for {label}")
                return

            x_filtered = np.array([x_raw[i] for i in valid_indices])
            y_filtered = np.array([y_raw[i] for i in valid_indices])

        elif not x_is_tuple and y_is_tuple:
            # x is scalar, y is tuple
            valid_indices = []
            for i, (x_val, (y_val, y_err)) in enumerate(zip(x_raw, y_raw)):
                if np.isfinite(x_val) and np.isfinite(y_val) and np.isfinite(y_err):
                    valid_indices.append(i)

            if not valid_indices:
                print(f"Warning: No valid data points to plot for {label}")
                return

            x_filtered = np.array([x_raw[i] for i in valid_indices])
            y_filtered = np.array([y_raw[i] for i in valid_indices])

        else:
            # Both are scalar arrays
            valid_mask = np.isfinite(x_raw) & np.isfinite(y_raw)
            if not np.any(valid_mask):
                print(f"Warning: No valid data points to plot for {label}")
                return

            x_filtered = x_raw[valid_mask]
            y_filtered = y_raw[valid_mask]

        x_raw = x_filtered
        y_raw = y_filtered

        # Case 1: both scalar
        if not x_is_tuple and not y_is_tuple:
            # ax.scatter(
            #     x_raw, y_raw, color=color, marker=marker, label=label, s=marker_size**2
            # )
            if empty_markers:
                # ax.scatter(
                #     x_raw,
                #     y_raw,
                #     marker=marker,
                #     s=marker_size**2,
                #     markerfacecolor="none",
                #     markeredgecolor=color,
                #     label=label,
                # )
                ax.scatter(
                    x_raw,
                    y_raw,
                    marker=marker,
                    s=marker_size**2,
                    facecolors="none",  # hollow marker
                    edgecolors=color,  # border color
                    label=label,
                )
            else:
                ax.scatter(
                    x_raw,
                    y_raw,
                    marker=marker,
                    s=marker_size**2,
                    color=color,
                    label=label,
                )

        # Case 2: scalar x, tuple y
        elif not x_is_tuple and y_is_tuple:
            y_val = np.array([val for val, _ in y_raw])
            y_err = np.array([err for _, err in y_raw])

            # In _plot_group method, where you handle interpolation
            if include_interpolation:
                # Ensure x values are strictly increasing by sorting and removing duplicates
                if x_is_tuple:
                    x_values = np.array([val for val, _ in x_filtered])
                    if y_is_tuple:
                        y_values = np.array([val for val, _ in y_filtered])
                    else:
                        y_values = y_filtered
                else:
                    x_values = x_filtered
                    if y_is_tuple:
                        y_values = np.array([val for val, _ in y_filtered])
                    else:
                        y_values = y_filtered

                # Sort the data and remove duplicates
                if len(x_values) > 1:
                    # Get indices that would sort x_values
                    sort_indices = np.argsort(x_values)
                    x_sorted = x_values[sort_indices]
                    y_sorted = y_values[sort_indices]

                    # Remove duplicates by finding unique x values
                    _, unique_indices = np.unique(x_sorted, return_index=True)
                    unique_indices = np.sort(
                        unique_indices
                    )  # Sort indices to maintain order

                    x_unique = x_sorted[unique_indices]
                    y_unique = y_sorted[unique_indices]

                    # Only attempt interpolation if we have enough unique points
                    if len(x_unique) > 3:  # Need at least 4 points for cubic spline
                        try:
                            x_smooth = np.linspace(min(x_unique), max(x_unique), 100)
                            spl = make_interp_spline(
                                x_unique, y_unique, k=3
                            )  # k=3 for cubic spline
                            y_smooth = spl(x_smooth)
                            ax.plot(x_smooth, y_smooth, ":", color=color, alpha=0.7)
                        except Exception as e:
                            print(f"Interpolation failed: {e}")
                            # Fall back to simple interpolation if cubic spline fails
                            try:
                                x_smooth = np.linspace(
                                    min(x_unique), max(x_unique), 100
                                )
                                y_smooth = np.interp(x_smooth, x_unique, y_unique)
                                ax.plot(x_smooth, y_smooth, ":", color=color, alpha=0.7)
                            except Exception as e2:
                                print(f"Simple interpolation also failed: {e2}")

            # ax.errorbar(
            #     x_raw,
            #     y_val,
            #     yerr=y_err,
            #     fmt=marker,
            #     capsize=capsize,
            #     color=color,
            #     label=label,
            #     markersize=marker_size,
            # )
            if empty_markers:
                ax.errorbar(
                    x_raw,
                    y_val,
                    yerr=y_err,
                    fmt=marker,
                    markersize=marker_size,
                    capsize=capsize,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    color=color,
                    label=label,
                )
            else:
                ax.errorbar(
                    x_raw,
                    y_val,
                    yerr=y_err,
                    fmt=marker,
                    markersize=marker_size,
                    capsize=capsize,
                    color=color,
                    label=label,
                )

        # Case 3: tuple x, scalar y
        elif x_is_tuple and not y_is_tuple:
            x_val = np.array([val for val, _ in x_raw])
            # ax.scatter(x_val, y_raw, color=color, marker=marker, label=label)
            if empty_markers:
                # ax.scatter(
                #     x_raw,
                #     y_raw,
                #     marker=marker,
                #     s=marker_size**2,
                #     markerfacecolor="none",
                #     markeredgecolor=color,
                #     label=label,
                # )
                ax.scatter(
                    x_raw,
                    y_raw,
                    marker=marker,
                    s=marker_size**2,
                    facecolors="none",  # hollow marker
                    edgecolors=color,  # border color
                    label=label,
                )
            else:
                ax.scatter(
                    x_raw,
                    y_raw,
                    marker=marker,
                    s=marker_size**2,
                    color=color,
                    label=label,
                )

        # Case 4: tuple x and tuple y
        elif x_is_tuple and y_is_tuple:
            x_val = np.array([val for val, _ in x_raw])
            y_val = np.array([val for val, _ in y_raw])
            y_err = np.array([err for _, err in y_raw])
            # ax.errorbar(
            #     x_val,
            #     y_val,
            #     yerr=y_err,
            #     fmt=marker,
            #     capsize=capsize,
            #     color=color,
            #     label=label,
            #     markersize=marker_size,
            # )
            if empty_markers:
                ax.errorbar(
                    x_val,
                    y_val,
                    yerr=y_err,
                    fmt=marker,
                    markersize=marker_size,
                    capsize=capsize,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    color=color,
                    label=label,
                )
            else:
                ax.errorbar(
                    x_val,
                    y_val,
                    yerr=y_err,
                    fmt=marker,
                    markersize=marker_size,
                    capsize=capsize,
                    color=color,
                    label=label,
                )

        # Add annotations if requested
        if annotation_variable is not None and annotation_variable in group_df.columns:
            # Get annotation values
            annotation_values = np.sort(group_df[annotation_variable].to_numpy())

            # Determine which points to annotate
            if annotation_range is None:
                # Default: annotate all points
                annotation_indices = range(len(x_filtered))
            else:
                # Parse annotation range
                start = annotation_range[0] if len(annotation_range) > 0 else 0
                end = annotation_range[1] if len(annotation_range) > 1 else None
                step = annotation_range[2] if len(annotation_range) > 2 else 1

                # Create indices list
                if end is None:
                    annotation_indices = range(start, len(x_filtered), step)
                else:
                    annotation_indices = range(start, min(end, len(x_filtered)), step)

            # Get x and y coordinates for annotations
            if x_is_tuple:
                x_coords = np.array([val for val, _ in x_filtered])
            else:
                x_coords = x_filtered

            if y_is_tuple:
                y_coords = np.array([val for val, _ in y_filtered])
            else:
                y_coords = y_filtered

            # Sort based on x_coords
            sort_indices = np.argsort(x_coords)
            x_coords = x_coords[sort_indices]
            y_coords = y_coords[sort_indices]
            annotation_values = annotation_values[sort_indices]

            # Add annotations
            for idx in annotation_indices:
                if (
                    idx < len(annotation_values)
                    and idx < len(x_coords)
                    and idx < len(y_coords)
                ):
                    # Skip NaN or Inf values
                    ann_value = annotation_values[idx]
                    if isinstance(ann_value, (int, float)) and not np.isfinite(
                        ann_value
                    ):
                        continue  # Skip this annotation if value is NaN or Inf

                    if isinstance(ann_value, (int, float)):
                        if ann_value.is_integer():  # Check if it's a whole number
                            # Format as integer by removing decimal point and zeros
                            formatted_value = int(ann_value)
                        else:
                            # Keep as float
                            formatted_value = ann_value
                    else:
                        formatted_value = ann_value

                    # Format the annotation text
                    # ann_text = f"{annotation_label}{annotation_values[idx]}"
                    ann_text = f"{annotation_label}{formatted_value}"

                    # Add the annotation
                    ax.annotate(
                        ann_text,
                        (x_coords[idx], y_coords[idx]),
                        xytext=annotation_offset,
                        textcoords="offset points",
                        fontsize=annotation_fontsize,
                        bbox=dict(
                            boxstyle=annotation_boxstyle,
                            facecolor="white",
                            alpha=annotation_alpha,
                            edgecolor=color,  # Use same color as markers
                        ),
                        arrowprops=dict(
                            arrowstyle="-",  # Simple line (no arrow head)
                            color="black",  # Use same color as markers
                            lw=1,  # Line width
                            alpha=0.8,  # Slightly transparent
                        ),
                        ha="center",
                        va="bottom",
                    )

    def _generate_marker_color_map(
        self,
        grouping_values: list,
        custom_map: Optional[dict] = None,
        index_shift: int = 0,
    ) -> dict:
        """
        Generate a stable mapping from grouping values to (marker, color) pairs.

        Parameters:
        -----------
        grouping_values : list
            List of unique values of the grouping variable.
        custom_map : dict, optional
            Custom mapping from value to (marker, color). Values not included will be auto-assigned.

        Returns:
        --------
        dict:
            Complete mapping from value → (marker, color)
        """
        sorted_values = sorted(grouping_values, key=lambda x: str(x))
        num_markers = len(constants.MARKER_STYLES)
        num_colors = len(constants.DEFAULT_COLORS)

        style_map = {}

        for idx, value in enumerate(sorted_values):
            if custom_map and value in custom_map:
                style_map[value] = custom_map[value]
            else:
                marker = constants.MARKER_STYLES[(idx + index_shift) % num_markers]
                color = constants.DEFAULT_COLORS[(idx + index_shift) % num_colors]
                style_map[value] = (marker, color)

        return style_map

    def _construct_plot_title(self, metadata_dict: dict, **kwargs) -> str:
        """Delegate to title builder."""
        # Extract the excluded set from various sources
        excluded = set(kwargs.get("excluded_from_title_list") or [])
        if kwargs.get("grouping_variable"):
            excluded.add(kwargs["grouping_variable"])
        if kwargs.get("labeling_variable"):
            excluded.add(kwargs["labeling_variable"])

        return self._title_builder.build(
            metadata_dict,
            self.list_of_tunable_parameter_names_from_dataframe,
            excluded=excluded,
            leading_substring=kwargs.get("leading_plot_substring"),
            title_from_columns=kwargs.get("title_from_columns"),
            wrapping_length=kwargs.get("title_wrapping_length", 80),
        )

    # def _construct_plot_title(
    #     self,
    #     metadata_dict: dict,
    #     grouping_variable: Optional[str] = None,
    #     labeling_variable: Optional[str] = None,
    #     leading_plot_substring: Optional[str] = None,
    #     excluded_from_title_list: Optional[list] = None,
    #     title_number_format: str = ".2f",  # ".4g" for scientific/float hybrid format
    #     title_wrapping_length: int = 90,
    #     title_from_columns: Optional[list] = None,
    # ) -> str:
    #     """
    #     Construct an informative plot title based on metadata and user
    #     preferences.

    #     Parameters:
    #     -----------
    #     metadata_dict : dict
    #         Dictionary containing key-value pairs for this plot group.
    #     grouping_variable : str, optional
    #         If provided, exclude this variable from appearing in the title.
    #     labeling_variable : str, optional
    #         If provided, exclude this variable from appearing in the title.
    #     leading_plot_substring : str, optional
    #         Optional leading string to prepend to the title.
    #     excluded_from_title_list : list, optional
    #         List of additional variable names to exclude from the title.

    #     Returns:
    #     --------
    #     str
    #         The constructed plot title.
    #     """
    #     title_parts = []

    #     # 1. Include leading substring if given
    #     if leading_plot_substring:
    #         title_parts.append(leading_plot_substring)

    #     # 1.5. Shortcut override: simple title from selected columns
    #     if title_from_columns:
    #         simple_parts = []
    #         for col in title_from_columns:
    #             value = metadata_dict.get(col)
    #             if value is None:
    #                 continue
    #             label = constants.TITLE_LABELS_BY_COLUMN_NAME.get(col, col)
    #             if isinstance(value, (int, float)):
    #                 formatted_value = format(value, title_number_format)
    #             else:
    #                 formatted_value = str(value)

    #             if "Kernel_operator_type" in col:
    #                 simple_parts.append(f"{formatted_value} Kernel")
    #             else:
    #                 simple_parts.append(f"{label}={formatted_value}")
    #         return ", ".join(simple_parts)

    #     # 2. Handle Overlap/Kernel/KL_or_Chebyshev_Order special logic
    #     excluded = set(excluded_from_title_list or [])
    #     if grouping_variable:
    #         excluded.add(grouping_variable)
    #     if labeling_variable:
    #         excluded.add(labeling_variable)

    #     overlap_method = metadata_dict.get("Overlap_operator_method")
    #     kernel_type = metadata_dict.get("Kernel_operator_type")
    #     chebyshev_terms = metadata_dict.get("Number_of_Chebyshev_terms")
    #     kl_order = metadata_dict.get("KL_diagonal_order")

    #     if overlap_method and "Overlap_operator_method" not in excluded:
    #         if overlap_method == "Bare":
    #             # Only Overlap_operator_method + Kernel_operator_type
    #             temp = []
    #             temp.append(str(overlap_method))
    #             if kernel_type and "Kernel_operator_type" not in excluded:
    #                 temp.append(str(kernel_type))
    #             title_parts.append(" ".join(temp) + ",")
    #         else:
    #             # Chebyshev or KL
    #             temp = []
    #             temp.append(str(overlap_method))
    #             if kernel_type and "Kernel_operator_type" not in excluded:
    #                 temp.append(str(kernel_type))
    #             if (
    #                 overlap_method == "Chebyshev"
    #                 and "Number_of_Chebyshev_terms" not in excluded
    #                 and chebyshev_terms is not None
    #             ):
    #                 temp.append(str(chebyshev_terms))
    #             if (
    #                 overlap_method == "KL"
    #                 and "KL_diagonal_order" not in excluded
    #                 and kl_order is not None
    #             ):
    #                 temp.append(str(kl_order))
    #             title_parts.append(" ".join(temp) + ",")

    #     # 3. Handle all remaining tunable parameters
    #     for param_name in self.list_of_tunable_parameter_names_from_dataframe:
    #         if param_name in excluded:
    #             continue
    #         if param_name not in metadata_dict:
    #             continue

    #         value = metadata_dict[param_name]
    #         label = constants.TITLE_LABELS_BY_COLUMN_NAME.get(param_name, param_name)

    #         # Format number values cleanly
    #         if isinstance(value, (int, float)):
    #             formatted_value = format(value, title_number_format)
    #         else:
    #             formatted_value = str(value)

    #         title_parts.append(f"{label} {formatted_value},")

    #     # 4. Merge and clean up final title
    #     full_title = " ".join(title_parts).strip()

    #     # Remove trailing comma if any
    #     if full_title.endswith(","):
    #         full_title = full_title[:-1]

    #     if title_wrapping_length and len(full_title) > title_wrapping_length:
    #         # Find a good place (comma) to insert newline
    #         comma_positions = [
    #             pos for pos, char in enumerate(full_title) if char == ","
    #         ]

    #         if comma_positions:
    #             # Find best comma to split around the middle
    #             split_pos = min(
    #                 comma_positions, key=lambda x: abs(x - len(full_title) // 2)
    #             )
    #             full_title = (
    #                 full_title[: split_pos + 1] + "\n" + full_title[split_pos + 1 :]
    #             )

    #     return full_title

    def _construct_plot_filename(
        self,
        metadata_dict: dict,
        include_combined_prefix: bool = False,
        custom_leading_substring: Optional[str] = None,
        grouping_variable: Optional[str] = None,
    ) -> str:
        """Delegate to filename builder."""
        return self._filename_builder.build(
            metadata_dict,
            self.plots_base_name,
            self.reduced_multivalued_tunable_parameter_names_list,
            grouping_variable=grouping_variable or "",
            include_combined_prefix=include_combined_prefix,
            custom_prefix=custom_leading_substring or "",
        )

    def _apply_curve_fit(
        self,
        ax,
        x_raw,
        y_raw,
        fit_function: str,
        show_fit_parameters_on_plot: bool = True,
        fit_curve_style: Optional[dict] = None,
        fit_label_format: str = ".2e",
        fit_label_location: str = "top left",
        fit_index_range: slice = slice(None),
        fit_curve_range: Optional[tuple] = None,
    ):
        try:

            if fit_index_range:
                # Apply the fit only to a specific range of indices
                x_raw = x_raw[fit_index_range]
                y_raw = y_raw[fit_index_range]

            position, alignment = FIT_LABEL_POSITIONS.get(
                fit_label_location, ((0.05, 0.95), ("left", "top"))
            )
            if isinstance(y_raw[0], tuple) and len(y_raw[0]) == 2:

                # gvar-based fit (with uncertainty)
                y_gv = gvar.gvar([t[0] for t in y_raw], [t[1] for t in y_raw])
                x_raw = np.asarray(x_raw, dtype=float)

                def linear_gvar(x, p):
                    return p[0] * x + p[1]

                def exponential_gvar(x, p):
                    return p[0] * np.exp(-p[1] * x) + p[2]

                def power_law_gvar(x, p):
                    return p[0] * x ** p[1]

                def shifted_power_law(x, p):
                    return p[0] / (x - p[1]) + p[2]

                fit_func_map = {
                    "linear": linear,
                    "exponential": exponential,
                    "power_law": power_law,
                }

                p0_map = {
                    "linear": [1, 1],
                    "exponential": [1, 1, 1],
                    "power_law": [1, 1],
                    "shifted_power_law": [1, 1, 1],
                }

                fcn = fit_func_map.get(fit_function)
                if fcn is None:
                    raise ValueError(f"Unsupported fit function: '{fit_function}'")

                fit = lsqfit.nonlinear_fit(
                    data=(x_raw, y_gv),
                    fcn=fcn,
                    p0=p0_map.get(fit_function),
                    debug=False,
                )

                if fit_curve_range is not None:
                    x_min, x_max = fit_curve_range
                else:
                    x_min, x_max = min(x_raw), max(x_raw)

                fit_params = gvar.mean(fit.p)
                x_smooth = np.linspace(x_min, x_max, 200)
                # x_smooth = np.linspace(min(x_raw), max(x_raw), 200)
                y_smooth = fcn(x_smooth, fit_params)

                if fit_curve_style is None:
                    rgba = to_rgba(self.color)  # Use the marker color
                    lighter_rgba = (
                        *rgba[:3],
                        0.5,
                    )  # Keep RGB, reduce alpha to 0.5 for transparency
                    style = {"color": lighter_rgba, "linestyle": "--"}
                else:
                    style = fit_curve_style
                ax.plot(x_smooth, gvar.mean(y_smooth), **style)

                if show_fit_parameters_on_plot:
                    a_fmt = format(fit_params[0], fit_label_format)
                    b_fmt = format(fit_params[1], fit_label_format)
                    param_text = f"a = {a_fmt}, b = {b_fmt}"
                    ax.text(
                        *position,
                        param_text,
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment=alignment[1],
                        horizontalalignment=alignment[0],
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                    )
                else:
                    return fit_params

            else:
                # Regular fit with scipy
                def linear(x, a, b):
                    return a * x + b

                def exponential(x, a, b):
                    return a * np.exp(b * x)

                def power_law(x, a, b):
                    return a * x**b

                def shifted_power_law(x, a, b, c):
                    return a / (x - b) + c

                fit_func_map = {
                    "linear": linear,
                    "exponential": exponential,
                    "power_law": power_law,
                    "shifted_power_law": shifted_power_law,
                }

                fit_func = fit_func_map.get(fit_function)
                if fit_func is None:
                    raise ValueError(f"Unsupported fit function: '{fit_function}'")

                # Remove NaNs
                valid = ~np.isnan(x_raw) & ~np.isnan(y_raw)
                x_fit = x_raw[valid]
                y_fit = y_raw[valid]

                if fit_function == "power_law" and np.any(x_fit <= 0):
                    raise ValueError(
                        "Power-law fit requires strictly positive x values."
                    )

                fit_params, _ = curve_fit(fit_func, x_fit, y_fit)

                x_smooth = np.linspace(min(x_fit), max(x_fit), 200)
                y_smooth = fit_func(x_smooth, *fit_params)

                if fit_curve_style is None:
                    rgba = to_rgba(self.color)  # Use the marker color
                    lighter_rgba = (
                        *rgba[:3],
                        0.5,
                    )  # Keep RGB, reduce alpha to 0.5 for transparency
                    style = {"color": lighter_rgba, "linestyle": "--"}
                else:
                    style = fit_curve_style
                ax.plot(x_smooth, y_smooth, **style)

                if show_fit_parameters_on_plot:
                    param_names = ["a", "b"]
                    param_text = ", ".join(
                        f"{name} = {val:{fit_label_format}}"
                        for name, val in zip(param_names, fit_params)
                    )
                    ax.text(
                        *position,
                        param_text,
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment=alignment[1],
                        horizontalalignment=alignment[0],
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                    )
                else:
                    return fit_params

        except Exception as e:
            print(f"Fit failed: {e}")

    def set_plot_variables(
        self, x_variable: str, y_variable: str, clear_existing: bool = False
    ) -> None:
        """
        Set the x- and y-axis variables for plotting and prepare the
        corresponding subdirectory.

        Parameters:
        -----------
        x_variable : str
            The name of the DataFrame column to use as the x-axis variable.
        y_variable : str
            The name of the DataFrame column to use as the y-axis variable.
        clear_existing : bool, optional
            If True and the plot subdirectory already exists, clear its
            contents. Default is False.

        Raises:
        -------
        ValueError:
            If either `x_variable` or `y_variable` is not a column in the
            DataFrame.
        """
        if x_variable not in self.dataframe.columns:
            raise ValueError(f"'{x_variable}' is not a column in the DataFrame.")
        if y_variable not in self.dataframe.columns:
            raise ValueError(f"'{y_variable}' is not a column in the DataFrame.")

        self.xaxis_variable_name = x_variable
        self.yaxis_variable_name = y_variable
        self.plots_base_name = f"{y_variable}_Vs_{x_variable}"

        self.individual_plots_subdirectory = self._prepare_plot_subdirectory(
            self.plots_base_name, clear_existing=clear_existing
        )

    def plot(
        self,
        grouping_variable: Optional[str] = None,
        excluded_from_grouping_list: Optional[list] = None,
        labeling_variable: Optional[str] = None,
        legend_number_format: str = ".2f",
        include_legend_title: bool = True,
        include_legend: bool = True,
        legend_location: str = "upper left",
        legend_columns: int = 1,
        sorting_variable: Optional[str] = None,
        sort_ascending: Optional[bool] = None,
        figure_size=(7, 5),
        font_size: int = 13,
        xaxis_label: Optional[str] = None,
        yaxis_label: Optional[str] = None,
        xaxis_log_scale: bool = False,
        yaxis_log_scale: bool = False,
        invert_xaxis: bool = False,
        invert_yaxis: bool = False,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        xaxis_start_at_zero: bool = False,
        yaxis_start_at_zero: bool = False,
        left_margin_adjustment: float = 0.15,
        right_margin_adjustment: float = 0.94,
        bottom_margin_adjustment: float = 0.12,
        top_margin_adjustment: float = 0.92,
        styling_variable: Optional[str] = None,
        marker_color_map: Optional[dict] = None,
        color_index_shift: int = 0,
        marker_size: int = 8,
        empty_markers: bool = False,
        alternate_filled_markers: bool = False,
        alternate_filled_markers_reversed: bool = False,
        capsize: float = 5,
        include_plot_title: bool = False,
        custom_plot_title: Optional[str] = None,
        title_from_columns: Optional[list] = None,
        custom_plot_titles_dict: Optional[dict] = None,
        title_size: int = 15,
        bold_title: bool = False,
        leading_plot_substring: Optional[str] = None,
        excluded_from_title_list: Optional[list] = None,
        title_number_format: str = ".2f",
        title_wrapping_length: int = 80,
        customization_function: Optional[Callable] = None,
        verbose: bool = True,
        fit_function: Optional[str] = None,  # e.g. "linear"
        fit_label_format: str = ".2f",
        show_fit_parameters_on_plot: bool = True,
        fit_curve_style: Optional[dict] = None,
        fit_label_location: str = "top left",
        fit_index_range: Optional[tuple] = None,  # default: include all
        fit_on_values: Optional[list] = None,
        fit_label_in_legend: bool = False,
        fit_curve_range: Optional[tuple] = None,
        target_ax=None,
        is_inset=False,
        save_figure=True,
        include_interpolation: bool = False,
        annotation_variable=None,
        annotation_label="",
        annotation_range=None,
        annotation_fontsize=8,
        annotation_boxstyle="round,pad=0.3",
        annotation_alpha=0.7,
        annotation_offset=(0, 10),
    ):
        """
        Plot data from the DataFrame, optionally grouped by a specific
        multivalued parameter.

        Parameters:
        -----------
        grouping_variable : str, optional
            If provided, combine plots grouped by this variable into single
            plots.
        excluded_from_grouping_list : list, optional
            Additional multivalued parameters to exclude from grouping.
        figure_size : tuple, optional
            Size of each plot figure.
        xaxis_log_scale : bool, optional
            Use log scale for x-axis. Defaults to True if parameter is in
            EXPONENTIAL_FORMAT.
        yaxis_log_scale : bool, optional
            Use log scale for y-axis. Defaults to True if parameter is in
            EXPONENTIAL_FORMAT.

            TODO: Refer to this example of slicing for fit_index_range parameter:
            plot(..., fit_index_range=slice(3, None))      # From index 3 onward
            plot(..., fit_index_range=slice(None, 10))     # Up to index 10
            plot(..., fit_index_range=slice(3, 10))        # From 3 to 10 (exclusive)

        target_ax : matplotlib.axes.Axes, optional
            If provided, plot to this axes instead of creating a new figure.
        is_inset : bool, optional
            If True, this plot is being used as an inset, so skip some setup
            like figure creation, title, axes labels, etc.

        """
        if self.xaxis_variable_name is None or self.yaxis_variable_name is None:
            raise ValueError("Call 'set_plot_variables()' before plotting.")

        if alternate_filled_markers:
            empty_markers = (
                False  # ignore user's empty_markers setting when alternating
            )

        if fit_index_range:
            if not isinstance(fit_index_range, tuple) or len(fit_index_range) != 2:
                raise ValueError("fit_index_range must be a tuple like (start, stop)")
            fit_index_range = slice(*fit_index_range)

        # Use grouping values unless styling_variable is provided
        if styling_variable:
            if (
                styling_variable
                not in self.list_of_tunable_parameter_names_from_dataframe
            ):
                raise ValueError("'styling_variable' must be tunable parameter.")

            styling_variable_unique_values = self.unique_values(styling_variable)
            style_lookup = self._generate_marker_color_map(
                styling_variable_unique_values,
                custom_map=marker_color_map,
                index_shift=color_index_shift,
            )

        # Determine which tunable parameters to exclude from grouping
        excluded = set(excluded_from_grouping_list or [])
        for axis_variable in [self.xaxis_variable_name, self.yaxis_variable_name]:
            if axis_variable in self.list_of_multivalued_tunable_parameter_names:
                excluded.add(axis_variable)

        if grouping_variable:
            grouping_columns = (
                [grouping_variable]
                if isinstance(grouping_variable, str)
                else grouping_variable
            )
            for grouping_column in grouping_columns:
                if (
                    grouping_column
                    not in self.list_of_multivalued_tunable_parameter_names
                ):
                    raise ValueError(
                        f"'{grouping_column}' is not a multivalued tunable parameter."
                    )
                excluded.add(grouping_column)

        # Get the grouped DataFrame
        grouped = self.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=list(excluded),
            verbose=verbose,
        )

        # Initialize marker and color
        marker, color = (".", "blue")
        self.color = color

        # At the end, store the most recent plot information
        self._last_plot_grouping = grouped
        self._last_plot_figures = (
            {}
        )  # Dictionary mapping group keys to (fig, ax) tuples
        self._last_plot_paths = {}  # Initialize paths dictionary outside the loop

        for group_keys, group_df in grouped:

            if target_ax is None:
                fig, ax = plt.subplots(figsize=figure_size)

                # Store for later use by add_inset
                if not is_inset:
                    if not hasattr(self, "_last_plot_figures"):
                        self._last_plot_figures = {}
                    self._last_plot_figures[group_keys] = (fig, ax, group_df)
            else:
                # New behavior: use the provided axes
                ax = target_ax
                fig = ax.figure

            ax.grid(True, linestyle="--", alpha=0.5)

            # Reconstruct metadata dict
            if not isinstance(group_keys, tuple):
                # group_keys = [group_keys]
                group_keys = (group_keys,)
            metadata = dict(
                zip(self.reduced_multivalued_tunable_parameter_names_list, group_keys)
            )

            # Add method and kernel type if constant
            for special in ["Overlap_operator_method", "Kernel_operator_type"]:
                if special in group_df.columns:
                    unique_vals = group_df[special].unique()
                    if len(unique_vals) == 1:
                        metadata[special] = unique_vals[0]

            if not grouping_variable and styling_variable:
                # Get the style key from the whole group_df
                style_key = group_df[styling_variable].iloc[0]
                marker, color = style_lookup.get(style_key, ("o", "blue"))

            # if not is_inset:
            # Determine axes labels
            if xaxis_label is None:
                xaxis_label = constants.AXES_LABELS_BY_COLUMN_NAME.get(
                    self.xaxis_variable_name, ""
                )
            ax.set_xlabel(xaxis_label, fontsize=font_size + 2)

            if yaxis_label is None:
                yaxis_label = constants.AXES_LABELS_BY_COLUMN_NAME.get(
                    self.yaxis_variable_name, ""
                )
            ax.set_ylabel(yaxis_label, fontsize=font_size + 2)

            ax.tick_params(axis="both", labelsize=font_size)

            # Axes scaling
            if (
                self.xaxis_variable_name in constants.PARAMETERS_WITH_EXPONENTIAL_FORMAT
                or xaxis_log_scale
            ):
                ax.set_xscale("log")
            if (
                self.yaxis_variable_name in constants.PARAMETERS_WITH_EXPONENTIAL_FORMAT
                or yaxis_log_scale
            ):
                ax.set_yscale("log")

            # Integer ticks
            if self.xaxis_variable_name in constants.PARAMETERS_OF_INTEGER_VALUE:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if self.yaxis_variable_name in constants.PARAMETERS_OF_INTEGER_VALUE:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            if invert_xaxis:
                ax.invert_xaxis()
            if invert_yaxis:
                ax.invert_yaxis()

            # Apply any custom user modifications
            if customization_function is not None:
                customization_function(ax)

            if grouping_variable:
                if sorting_variable:
                    unique_group_values = (
                        group_df.sort_values(
                            by=sorting_variable, ascending=(sort_ascending is not False)
                        )[grouping_columns]
                        .drop_duplicates()
                        .apply(tuple, axis=1)
                        .tolist()
                    )
                else:
                    unique_group_values = (
                        group_df[grouping_columns]
                        .drop_duplicates()
                        .apply(tuple, axis=1)
                        .tolist()
                    )

                if sort_ascending is True and not sorting_variable:
                    unique_group_values = sorted(unique_group_values)
                elif sort_ascending is False and not sorting_variable:
                    unique_group_values = sorted(unique_group_values, reverse=True)
                # else sort_ascending=None => preserve order

                # Build color/marker map once per plot
                style_map = self._generate_marker_color_map(
                    unique_group_values,
                    custom_map=marker_color_map,
                    index_shift=color_index_shift,
                )

                for curve_index, value in enumerate(unique_group_values):
                    if labeling_variable:
                        if (
                            isinstance(labeling_variable, list)
                            and len(labeling_variable) == 2
                        ):
                            var1, var2 = labeling_variable

                            if isinstance(grouping_variable, str):
                                actual_value = (
                                    value[0]
                                    if isinstance(value, tuple) and len(value) == 1
                                    else value
                                )
                                label_rows = group_df[
                                    group_df[grouping_variable] == actual_value
                                ]
                            else:
                                mask = (
                                    group_df[grouping_variable].apply(tuple, axis=1)
                                    == value
                                )
                                label_rows = group_df[mask]

                            val1 = label_rows[var1].unique()
                            val2 = label_rows[var2].unique()

                            if len(val1) == 1:
                                val1 = val1[0]
                            if len(val2) == 1:
                                val2 = val2[0]

                            if isinstance(val2, (int, float)):
                                val2 = format(val2, legend_number_format)
                            label = f"{val1} ({constants.LEGEND_LABELS_BY_COLUMN_NAME.get(var2, var2)}{val2})"

                        else:
                            # Old logic for string-valued labeling_variable
                            if isinstance(grouping_variable, str):
                                actual_value = (
                                    value[0]
                                    if isinstance(value, tuple) and len(value) == 1
                                    else value
                                )
                                label_rows = group_df[
                                    group_df[grouping_variable] == actual_value
                                ]
                            else:
                                mask = (
                                    group_df[grouping_variable].apply(tuple, axis=1)
                                    == value
                                )
                                label_rows = group_df[mask]

                            label_value = label_rows[labeling_variable].unique()

                            if len(label_value) == 1:
                                label_value = label_value[0]
                                if isinstance(label_value, (int, float)):
                                    label_value = format(
                                        label_value, legend_number_format
                                    )
                            label = str(label_value)

                    else:
                        # label = str(value)
                        if isinstance(value, tuple):
                            label = " ".join(str(v) for v in value)
                        else:
                            label = str(value)

                    if isinstance(grouping_variable, str):
                        # Value is a 1-tuple, extract the scalar
                        actual_value = (
                            value[0]
                            if isinstance(value, tuple) and len(value) == 1
                            else value
                        )
                        subgroup = group_df[group_df[grouping_variable] == actual_value]
                    else:
                        # grouping_variable is a list of column names
                        mask = group_df[grouping_variable].apply(tuple, axis=1) == value
                        subgroup = group_df[mask]

                    if grouping_variable and not styling_variable:
                        marker, color = style_map[value]

                    if grouping_variable and styling_variable:
                        style_key = group_df[styling_variable].iloc[0]
                        marker, color = style_lookup.get(style_key, ("o", "blue"))

                    # Alternate filling
                    if alternate_filled_markers:
                        empty_marker = curve_index % 2 == 1  # odd indices → empty
                        if alternate_filled_markers_reversed:
                            # empty_marker = not empty_marker
                            if empty_marker:
                                empty_marker = 0
                            else:
                                empty_marker = 1
                    else:
                        empty_marker = (
                            empty_markers  # regular user setting (could be False)
                        )

                    fit_label_suffix = ""
                    if fit_function is not None:

                        self.color = color

                        if fit_on_values is not None and not isinstance(
                            fit_on_values, tuple
                        ):
                            fit_on_values = (fit_on_values,)

                        if fit_on_values is None or value == fit_on_values:

                            x_raw = subgroup[self.xaxis_variable_name].to_numpy()
                            y_raw = subgroup[self.yaxis_variable_name].to_numpy()
                            try:
                                if fit_label_in_legend:
                                    show_fit_parameters_on_plot = False

                                params = self._apply_curve_fit(
                                    ax,
                                    x_raw,
                                    y_raw,
                                    fit_function=fit_function,
                                    show_fit_parameters_on_plot=show_fit_parameters_on_plot,
                                    fit_curve_style=fit_curve_style,
                                    fit_label_format=fit_label_format,
                                    fit_label_location=fit_label_location,
                                    fit_index_range=fit_index_range,
                                    fit_curve_range=fit_curve_range,
                                )

                                if fit_label_in_legend:
                                    if fit_function == "exponential":
                                        c_fmt = format(params[2], fit_label_format)
                                        fit_label_suffix = f" (a$m^{{n\\to\\infty}}_{{\\text{{PCAC}}}}$={c_fmt})"
                                    else:
                                        a_fmt = format(params[0], fit_label_format)
                                        b_fmt = format(params[1], fit_label_format)
                                        fit_label_suffix = f" (a={a_fmt}, b={b_fmt})"
                            except Exception as e:
                                print(f"Fit failed for {value}: {e}")

                    self._plot_group(
                        ax,
                        subgroup,
                        label=label + fit_label_suffix,
                        color=color,
                        marker=marker,
                        marker_size=marker_size,
                        capsize=capsize,
                        empty_markers=empty_marker,
                        include_interpolation=include_interpolation,
                        annotation_variable=annotation_variable,
                        annotation_label=annotation_label,
                        annotation_range=annotation_range,
                        annotation_fontsize=annotation_fontsize,
                        annotation_boxstyle=annotation_boxstyle,
                        annotation_alpha=annotation_alpha,
                        annotation_offset=annotation_offset,
                    )
                if include_legend:
                    legend = ax.legend(
                        loc=legend_location, fontsize=font_size, ncol=legend_columns
                    )
                    if include_legend_title:
                        legend_title = constants.LEGEND_LABELS_BY_COLUMN_NAME.get(
                            (
                                labeling_variable
                                if labeling_variable
                                else grouping_variable
                            ),
                            (
                                labeling_variable
                                if labeling_variable
                                else grouping_variable
                            ),
                        )
                        # If the title is not a LaTeX string (no $ symbols), replace
                        # underscores with spaces
                        if "$" not in legend_title:
                            legend_title = legend_title.replace("_", " ")
                        legend.set_title(legend_title, prop={"size": font_size + 1})

            else:
                # Individual plot
                self._plot_group(
                    ax,
                    group_df,
                    color=color,
                    marker=marker,
                    marker_size=marker_size,
                    capsize=capsize,
                    empty_markers=empty_markers,
                    include_interpolation=include_interpolation,
                    annotation_variable=annotation_variable,
                    annotation_label=annotation_label,
                    annotation_range=annotation_range,
                    annotation_fontsize=annotation_fontsize,
                    annotation_boxstyle=annotation_boxstyle,
                    annotation_alpha=annotation_alpha,
                    annotation_offset=annotation_offset,
                )

                if fit_function is not None:
                    x_raw = group_df[self.xaxis_variable_name].to_numpy()
                    y_raw = group_df[self.yaxis_variable_name].to_numpy()
                    self._apply_curve_fit(
                        ax,
                        x_raw,
                        y_raw,
                        fit_function=fit_function,
                        show_fit_parameters_on_plot=show_fit_parameters_on_plot,
                        fit_curve_style=fit_curve_style,
                        fit_label_format=fit_label_format,
                        fit_label_location=fit_label_location,
                        fit_index_range=fit_index_range,
                        fit_curve_range=fit_curve_range,
                    )

            if not is_inset:
                fig.subplots_adjust(
                    left=left_margin_adjustment,
                    right=right_margin_adjustment,
                    bottom=bottom_margin_adjustment,
                    top=top_margin_adjustment,
                )

            if include_plot_title:
                if custom_plot_title:
                    plot_title = custom_plot_title
                elif custom_plot_titles_dict is not None:
                    title_key = group_keys if len(group_keys) > 1 else group_keys[0]
                    plot_title = custom_plot_titles_dict.get(title_key, "")
                else:
                    if excluded_from_title_list is None:
                        excluded_from_title_list = []
                    excluded_from_title_list.extend(
                        [
                            "Main_program_type",
                            "MPI_geometry",
                            # "Threads_per_process",
                            "Maximum_Lanczos_iterations",
                        ]
                    )

                    plot_title = self._construct_plot_title(
                        metadata_dict={
                            **metadata,
                            **self.unique_value_columns_dictionary,
                        },
                        grouping_variable=grouping_variable,
                        labeling_variable=labeling_variable,
                        leading_plot_substring=leading_plot_substring,
                        excluded_from_title_list=excluded_from_title_list,
                        title_number_format=title_number_format,
                        title_wrapping_length=title_wrapping_length,
                        title_from_columns=title_from_columns,
                    )
                if not is_inset:
                    ax.set_title(
                        plot_title,
                        fontsize=title_size,
                        weight="bold" if bold_title else "normal",
                    )

            # Set axis limits if specified
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

            # Construct filename and save
            filename = self._construct_plot_filename(
                metadata_dict=metadata,
                include_combined_prefix=(
                    grouping_variable is not None  # and leading_plot_substring is None
                ),
                custom_leading_substring="",
                # leading_plot_substring,
                grouping_variable=grouping_variable,
            )
            if grouping_variable:
                # nested_dirname = f"Grouped_by_{grouping_variable}"
                if isinstance(grouping_variable, str):
                    nested_dirname = f"Grouped_by_{grouping_variable}"
                else:
                    nested_dirname = "Grouped_by_" + "_and_".join(grouping_variable)
                self.combined_plots_subdirectory = self._prepare_plot_subdirectory(
                    os.path.join(self.plots_base_name, nested_dirname)
                )
                full_path = os.path.join(
                    self.combined_plots_subdirectory, f"{filename}.png"
                )
            else:
                full_path = os.path.join(
                    self.individual_plots_subdirectory, f"{filename}.png"
                )

            if not is_inset:
                self._last_plot_paths[group_keys] = full_path

            # Only save if requested
            if save_figure:
                fig.savefig(full_path)
                plt.close(fig)
            # fig.savefig(full_path)
            # plt.close(fig)

        return self  # Return self to enable method chaining

    def add_inset(
        self,
        xaxis_variable,
        yaxis_variable=None,
        location="lower right",
        width=0.3,
        height=0.3,
        inset_x=None,
        inset_y=None,
        df_condition=None,
        inset_filter_func=None,
        **inset_kwargs,
    ):
        """
        Add an inset to the previously created plots with different x/y variables.

        Parameters:
        -----------
        xaxis_variable : str
            The variable to use for the x-axis of the inset.
        yaxis_variable : str, optional
            The variable to use for the y-axis of the inset. If None, uses the same as the main plot.
        location : str, optional
            Predefined location: "upper right", "upper left", "lower right", "lower left".
            Used only if inset_x and inset_y are not provided.
        width : float, optional
            Width of the inset as a fraction of the main axes. Default is 0.3.
        height : float, optional
            Height of the inset as a fraction of the main axes. Default is 0.3.
        inset_x : float, optional
            The x-coordinate for the lower-left corner of the inset (as a fraction of the main axes).
            If provided, overrides the 'location' parameter.
        inset_y : float, optional
            The y-coordinate for the lower-left corner of the inset (as a fraction of the main axes).
            If provided along with inset_x, overrides the 'location' parameter.
        inset_filter_func : callable, optional
            A function that takes a DataFrame and returns a filtered DataFrame.
            Use this to filter the data for the inset plot.
        **inset_kwargs :
            Additional keyword arguments passed to the plot() method for the inset.
        """
        if not hasattr(self, "_last_plot_figures") or not self._last_plot_figures:
            raise ValueError("Call plot() before add_inset()")

        # Save current plot variables and directories
        original_x = self.xaxis_variable_name
        original_y = self.yaxis_variable_name
        original_plots_base_name = self.plots_base_name
        original_individual_plots_subdirectory = self.individual_plots_subdirectory
        original_combined_plots_subdirectory = self.combined_plots_subdirectory

        # Determine the inset position
        if inset_x is not None and inset_y is not None:
            # Use explicit coordinates if provided
            bbox_to_anchor = [inset_x, inset_y, width, height]
        else:
            # Use predefined location map
            location_map = {
                "upper right": [0.65, 0.65, width, height],
                "upper left": [0.05, 0.65, width, height],
                "lower right": [0.65, 0.05, width, height],
                "lower left": [0.05, 0.05, width, height],
                "center": [0.5 - width / 2, 0.5 - height / 2, width, height],
            }
            bbox_to_anchor = location_map.get(location, [0.65, 0.05, width, height])

        # For each stored figure, add an inset
        for group_keys, (fig, ax, group_df) in self._last_plot_figures.items():
            # Create inset axes
            inset_ax = ax.inset_axes(bbox_to_anchor)

            # # Filter the DataFrame to keep rows where Number_of_Chebyshev_terms > 20 or is NaN
            # group_df = group_df[
            #     (((group_df['Number_of_Chebyshev_terms'] > 230) |
            #     (group_df['Number_of_Chebyshev_terms'].isna())) &
            #     (group_df['Kernel_operator_type'] == 'Wilson')) |
            #     (((group_df['Number_of_Chebyshev_terms'] > 50) |
            #     (group_df['Number_of_Chebyshev_terms'].isna())) &
            #     (group_df['Kernel_operator_type'] == 'Brillouin'))
            # ]

            # Apply the filter function if provided
            filtered_df = group_df
            if inset_filter_func is not None:
                try:
                    filtered_df = inset_filter_func(group_df)
                    # Skip this group if the filtered DataFrame is empty
                    if len(filtered_df) == 0:
                        continue
                except Exception as e:
                    print(f"Error applying filter function: {e}")
                    # If the filter fails, use the original DataFrame
                    filtered_df = group_df

            # Skip this group if the filtered DataFrame is empty
            if len(filtered_df) == 0:
                continue

            # Create a new plotter using the filtered DataFrame
            temp_plotter = DataPlotter(filtered_df, self.plots_directory)

            # Create a new plotter using the FULL original DataFrame
            # temp_plotter = DataPlotter(group_df, self.plots_directory)
            # self.dataframe
            # Set plot variables
            temp_plotter.set_plot_variables(
                xaxis_variable,
                yaxis_variable if yaxis_variable else self.yaxis_variable_name,
                clear_existing=False,
            )

            if df_condition is not None:
                print(df_condition)
                temp_plotter.restrict_dataframe(condition=df_condition)

            inset_grouping_var = inset_kwargs.get("grouping_variable")
            excluded_from_grouping_list = inset_kwargs.get(
                "excluded_from_grouping_list"
            )

            if (
                inset_grouping_var is not None
                or excluded_from_grouping_list is not None
            ):
                if temp_plotter.list_of_multivalued_tunable_parameter_names == []:
                    continue

            # Pass most of the inset_kwargs to plot, but with some overrides
            plot_kwargs = inset_kwargs.copy()
            plot_kwargs.update(
                {
                    "target_ax": inset_ax,
                    "is_inset": True,
                    "include_plot_title": False,
                    "save_figure": False,
                    "verbose": False,
                    "include_legend": False,
                }
            )

            # Plot with the filtered data
            temp_plotter.plot(**plot_kwargs)

            # Save the modified figure to the original path
            if group_keys in self._last_plot_paths:
                save_path = self._last_plot_paths[group_keys]
                fig.savefig(save_path)
                plt.close(fig)
            else:
                print(f"Warning: Path not found for group: {group_keys}")

            temp_plotter.restore_original_dataframe()

        # Restore original plot variables and directories
        self.xaxis_variable_name = original_x
        self.yaxis_variable_name = original_y
        self.plots_base_name = original_plots_base_name
        self.individual_plots_subdirectory = original_individual_plots_subdirectory
        self.combined_plots_subdirectory = original_combined_plots_subdirectory

        return self
