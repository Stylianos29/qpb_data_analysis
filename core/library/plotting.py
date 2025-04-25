import os, shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from library import DataFrameAnalyzer, TableGenerator, constants


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

        if not os.path.isdir(plots_directory):
            raise ValueError(f"Invalid plots directory: '{plots_directory}'")

        self.plots_directory = plots_directory
        self.individual_plots_subdirectory = plots_directory
        self.combined_plots_subdirectory = plots_directory

        self.xaxis_variable_name = None
        self.yaxis_variable_name = None
        self.plots_base_name = None

    def generate_column_uniqueness_report(
        self, max_width=80, separate_by_type=True
    ) -> str:
        table_generator = TableGenerator(self.dataframe)
        return table_generator.generate_column_uniqueness_report(
            max_width=max_width,
            separate_by_type=separate_by_type,
            export_to_file=False,
        )

    def _prepare_plot_subdirectory(
        self, subdir_name: str, clear_existing: bool = False
    ) -> str:
        """
        Create or clean a subdirectory for storing plots.

        Parameters:
        -----------
        subdir_name : str
            The name of the subdirectory to create inside the main plots
            directory.
        clear_existing : bool, optional
            If True, delete all contents of the subdirectory if it already
            exists.

        Returns:
        --------
        str:
            The full path to the prepared subdirectory.
        """
        full_path = os.path.join(self.plots_directory, subdir_name)
        os.makedirs(full_path, exist_ok=True)

        if clear_existing:
            for item in os.listdir(full_path):
                item_path = os.path.join(full_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

        return full_path

    def _plot_group(
        self,
        ax,
        group_df: pd.DataFrame,
        label: str = None,
        color: str = "blue",
        marker: str = "o",
    ):
        """
        Plot a single data group on the provided Matplotlib Axes object.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes on which to plot.
        group_df : pd.DataFrame
            The DataFrame containing the data for one group.
        label : str, optional
            Label to use in the legend.
        color : str, optional
            Color of the plotted line or points.
        marker : str, optional
            Marker style for scatter/errorbar.
        """
        x_raw = group_df[self.xaxis_variable_name].to_numpy()
        y_raw = group_df[self.yaxis_variable_name].to_numpy()

        is_tuple_array = lambda arr: (isinstance(arr[0], tuple) and len(arr[0]) == 2)

        x_is_tuple = is_tuple_array(x_raw)
        y_is_tuple = is_tuple_array(y_raw)

        if not (np.issubdtype(x_raw.dtype, np.number) or x_is_tuple):
            raise TypeError(f"x-axis data has unsupported type: {x_raw.dtype}")
        if not (np.issubdtype(y_raw.dtype, np.number) or y_is_tuple):
            raise TypeError(f"y-axis data has unsupported type: {y_raw.dtype}")

        # Case 1: both scalar
        if not x_is_tuple and not y_is_tuple:
            ax.scatter(x_raw, y_raw, color=color, marker=marker, label=label)

        # Case 2: scalar x, tuple y
        elif not x_is_tuple and y_is_tuple:
            y_val = np.array([val for val, _ in y_raw])
            y_err = np.array([err for _, err in y_raw])
            ax.errorbar(
                x_raw,
                y_val,
                yerr=y_err,
                fmt=marker,
                capsize=5,
                color=color,
                label=label,
            )

        # Case 3: tuple x, scalar y
        elif x_is_tuple and not y_is_tuple:
            x_val = np.array([val for val, _ in x_raw])
            ax.scatter(x_val, y_raw, color=color, marker=marker, label=label)

        # Case 4: tuple x and tuple y
        elif x_is_tuple and y_is_tuple:
            x_val = np.array([val for val, _ in x_raw])
            y_val = np.array([val for val, _ in y_raw])
            y_err = np.array([err for _, err in y_raw])
            ax.errorbar(
                x_val,
                y_val,
                yerr=y_err,
                fmt=marker,
                capsize=5,
                color=color,
                label=label,
            )

    def _construct_plot_filename(
        self,
        metadata_dict: dict,
        include_combined_prefix: bool = False,
        custom_leading_substring: str = None,
        grouping_variable: str = None,
    ) -> str:
        """
        Construct a plot filename based on metadata and class configuration.

        Parameters:
        -----------
        metadata_dict : dict
            Dictionary containing values of tunable parameters for this plot
            group.
        include_combined_prefix : bool, optional
            Whether to prepend "Combined_" to the filename (used when
            grouping_variable is defined).
        custom_leading_substring : str, optional
            An optional custom prefix that overrides "Combined_".
        grouping_variable : str, optional
            If provided, appends "_grouped_by_{grouping_variable}" to the
            filename.

        Returns:
        --------
        str:
            A string to use as the plot filename (without extension).
        """

        def sanitize(value):
            return (
                str(value)
                .replace(".", "p")
                .replace(",", "")
                .replace("(", "")
                .replace(")", "")
            )

        # -- Build filename in parts
        filename_parts = []

        # 1. Overlap_operator_method
        overlap_method = metadata_dict.get("Overlap_operator_method")
        if overlap_method in {"KL", "Chebyshev", "Bare"}:
            filename_parts.append(overlap_method)
            metadata_dict.pop("Overlap_operator_method", None)

        # 2. plots_base_name (y_Vs_x)
        filename_parts.append(self.plots_base_name)

        # 3. Kernel_operator_type
        kernel_type = metadata_dict.get("Kernel_operator_type")
        if kernel_type in {"Brillouin", "Wilson"}:
            filename_parts.append(kernel_type)
            metadata_dict.pop("Kernel_operator_type", None)

        # 4. Parameters from reduced tunable parameter list
        for param in self.reduced_multivalued_tunable_parameter_names_list:
            if param in metadata_dict:
                label = constants.FILENAME_LABELS_BY_COLUMN_NAME.get(param, param)
                value = sanitize(metadata_dict[param])
                filename_parts.append(f"{label}{value}")

        # 5. Optional prefix override
        if custom_leading_substring is not None:
            prefix = custom_leading_substring
        elif include_combined_prefix:
            prefix = "Combined_"
        else:
            prefix = ""

        # 6. Optional grouping variable suffix
        if grouping_variable:
            suffix = f"_grouped_by_{grouping_variable}"
        else:
            suffix = ""

        return prefix + "_".join(filename_parts) + suffix

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
        grouping_variable: str = None,
        excluded_from_grouping_list: list = None,
        figure_size=(7, 5),
        xaxis_log_scale: bool = False,
        yaxis_log_scale: bool = False,
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
        """
        if self.xaxis_variable_name is None or self.yaxis_variable_name is None:
            raise ValueError("Call 'set_plot_variables()' before plotting.")

        # Determine which tunable parameters to exclude from grouping
        excluded = set(excluded_from_grouping_list or [])
        for axis_variable in [self.xaxis_variable_name, self.yaxis_variable_name]:
            if axis_variable in self.list_of_multivalued_tunable_parameter_names:
                excluded.add(axis_variable)
        if grouping_variable:
            if (
                grouping_variable
                not in self.list_of_multivalued_tunable_parameter_names
            ):
                raise ValueError(
                    f"'{grouping_variable}' is not a multivalued tunable parameter."
                )
            excluded.add(grouping_variable)

        # Get the grouped DataFrame
        grouped = self.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=list(excluded)
        )

        for group_keys, group_df in grouped:
            fig, ax = plt.subplots(figsize=figure_size)
            ax.grid(True, linestyle="--", alpha=0.5)

            # Reconstruct metadata dict
            if not isinstance(group_keys, tuple):
                group_keys = [group_keys]
            metadata = dict(
                zip(self.reduced_multivalued_tunable_parameter_names_list, group_keys)
            )

            # Add method and kernel type if constant
            for special in ["Overlap_operator_method", "Kernel_operator_type"]:
                if special in group_df.columns:
                    unique_vals = group_df[special].unique()
                    if len(unique_vals) == 1:
                        metadata[special] = unique_vals[0]

            # Determine axes labels
            x_label = constants.AXES_LABELS_BY_COLUMN_NAME.get(
                self.xaxis_variable_name, ""
            )
            ax.set_xlabel(x_label)
            y_label = constants.AXES_LABELS_BY_COLUMN_NAME.get(
                self.yaxis_variable_name, ""
            )
            ax.set_ylabel(y_label)

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

            if grouping_variable:
                # Combine subgroups in one plot
                for value, subgroup in group_df.groupby(
                    grouping_variable, observed=True, sort=False
                ):
                    label = f"{value}"  # Can enhance with LEGEND_LABELS_BY_COLUMN_NAME if needed
                    self._plot_group(ax, subgroup, label=label)
                ax.legend(title=grouping_variable.replace("_", " "))
            else:
                # Individual plot
                self._plot_group(ax, group_df)

            # Construct filename and save
            filename = self._construct_plot_filename(
                metadata_dict=metadata,
                include_combined_prefix=(grouping_variable is not None),
                grouping_variable=grouping_variable,
            )
            full_path = os.path.join(
                self.individual_plots_subdirectory, f"{filename}.png"
            )
            fig.savefig(full_path)
            plt.close(fig)
