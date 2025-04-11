import os
import shutil

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib import cm
import textwrap
import gvar as gv
import numpy as np
import pandas as pd
import copy
import hashlib

from matplotlib.markers import MarkerStyle

from library import constants, filesystem_utilities
from library import DataFrameAnalyzer
import inspect


def stable_int_hash(value):
    value_str = str(value)  # Convert to string to handle different types
    return int(hashlib.md5(value_str.encode()).hexdigest(), 16)  # Convert hex to int


def stable_index_mapping(
    grouping_field_values, total_number_of_grouping_field_values, all_unique_values
):
    """
    Maps 'grouping_field_values' to a unique integer in [0,
    total_number_of_grouping_field_values - 1] based on a deterministic ordering
    of all unique values.

    Parameters:
        grouping_field_values (any type): The value to hash uniquely.
        total_number_of_grouping_field_values (int): The total number of unique
        values. all_unique_values (iterable): A list or set containing all
        possible unique values.

    Returns:
        int: A unique integer in the range [0,
        total_number_of_grouping_field_values - 1].
    """
    # Sort the unique values to ensure deterministic ordering
    ordered_values = sorted(
        all_unique_values, key=str
    )  # Sorting ensures stability across runs

    # Create a mapping from values to unique indices
    value_to_index = {value: idx for idx, value in enumerate(ordered_values)}

    # Return the unique index for the given grouping_field_values
    return value_to_index[grouping_field_values]


class DataPlotter(DataFrameAnalyzer):
    def __init__(self, dataframe: pd.DataFrame, base_plots_directory: str):
        """
        Initialize the DataPlotter with a DataFrame.
        """
        # Initialize the DataFrameAnalyzer
        super().__init__(dataframe)

        if not filesystem_utilities.is_valid_directory(base_plots_directory):
            raise ValueError(f"Invalid base plots directory: '{base_plots_directory}'.")
        self.base_plots_directory = base_plots_directory

        # Initialize plots subdirectories
        self.individual_plots_subdirectory = base_plots_directory
        self.combined_plots_subdirectory = base_plots_directory

        # Initialize pair of variable for plotting
        self.xaxis_variable_name = None
        self.yaxis_variable = None

        # # Use provided DataFrameAnalyzer object to extract constant attributes
        # self = analyzer

        # self.list_of_dataframe_fields = self.list_of_dataframe_fields

        # self.single_valued_fields_dictionary = (
        #     self.single_valued_fields_dictionary
        # )

        self.list_of_multivalued_fields = list(
            self.multivalued_fields_dictionary.keys()
        )

    def set_pair_of_variables(
        self,
        xaxis_variable: str,
        yaxis_variable: str,
        plots_base_name: str = None,
    ):
        # Validate input
        if xaxis_variable not in self.list_of_dataframe_fields:
            raise ValueError(f"Invalid x-axis variable name '{xaxis_variable}'.")
        self.xaxis_variable_name = xaxis_variable

        # For the y-axis anticipate the case of a histogram being requested
        if (
            yaxis_variable not in self.list_of_dataframe_fields
            and not yaxis_variable == "Frequency"
        ):
            raise ValueError(f"Invalid y-axis variable name '{yaxis_variable}'.")
        self.yaxis_variable = yaxis_variable

        # Construct plots base name
        if plots_base_name is not None:
            self.plots_base_name = plots_base_name
        else:
            self.plots_base_name = (
                self.yaxis_variable + "_Vs_" + self.xaxis_variable_name
            )

    def _prepare_plots_subdirectory(
        self, plots_base_subdirectory=None, plots_base_name=None, clear_existing=False
    ):

        if plots_base_subdirectory is None:
            plots_base_subdirectory = self.base_plots_directory
        elif not filesystem_utilities.is_valid_directory(plots_base_subdirectory):
            raise ValueError(
                f"Invalid plots base subdirectory: '{plots_base_subdirectory}'."
            )

        if plots_base_name is None:
            plots_base_name = self.plots_base_name

        # Create plots subdirectory if it does not exist
        plots_subdirectory = os.path.join(plots_base_subdirectory, plots_base_name)
        os.makedirs(plots_subdirectory, exist_ok=True)

        # Ensure the directory exists before clearing or using it.
        if clear_existing:
            for item in os.listdir(plots_subdirectory):
                item_path = os.path.join(plots_subdirectory, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

        return plots_subdirectory

    def _construct_plot_title(
        self,
        leading_substring: str,
        metadata_dictionary: dict,
        title_width: int,
        exclude_prioritized_fields_substring: bool = False,
        excluded_title_fields: list = None,
        fields_unique_value_dictionary: dict = None,
        additional_excluded_fields: list = None,
    ):
        """
        Constructs a subtitle string for plots based on unique field values.

        Parameters:
        -----------
        fields_unique_value_dictionary : dict
            A dictionary where keys are field names and values are their unique
            values.

        Returns:
        --------
        str
            A formatted subtitle string.
        """

        # Define the fields to appear first and their conditional additions
        list_of_fields_to_appear_first = [
            "Overlap_operator_method",
            "Kernel_operator_type",
        ]
        additional_fields = {
            "KL": "KL_diagonal_order",
            "Chebyshev": "Number_of_Chebyshev_terms",
        }

        if fields_unique_value_dictionary is None:
            fields_unique_value_dictionary = (
                self.single_valued_fields_dictionary | metadata_dictionary
            )

            # Exclude specified fields from the dictionary if provided
            if excluded_title_fields is not None:
                fields_unique_value_dictionary = {
                    field: value
                    for field, value in fields_unique_value_dictionary.items()
                    if field not in excluded_title_fields
                }

        # Start building the subtitle with the prioritized fields
        prioritized_fields_substring = "".join(
            f"{fields_unique_value_dictionary[field]} "
            for field in list_of_fields_to_appear_first
            if field in fields_unique_value_dictionary
        )

        # Dynamically add an additional field based on Overlap_operator_method
        overlap_operator_method = fields_unique_value_dictionary.get(
            "Overlap_operator_method"
        )
        if overlap_operator_method in additional_fields:
            additional_field = additional_fields[overlap_operator_method]
            if additional_field in fields_unique_value_dictionary:
                prioritized_fields_substring += (
                    f"{constants.TITLE_LABELS_DICTIONARY[additional_field]}="
                    f"{fields_unique_value_dictionary[additional_field]}"
                )

        # Initialize additional_excluded_fields to empty list if None
        if additional_excluded_fields is None:
            additional_excluded_fields = []

        # Filter out prioritized fields from the dictionary
        excluded_fields = set(
            list_of_fields_to_appear_first
            + list(additional_fields.values())
            + additional_excluded_fields
        )
        remaining_fields = {
            key: value
            for key, value in fields_unique_value_dictionary.items()
            if key not in excluded_fields
        }

        # Add remaining fields to the subtitle
        remaining_fields_substring = ", ".join(
            f"{constants.TITLE_LABELS_DICTIONARY[key]}={value}"
            for key, value in remaining_fields.items()
            if key in constants.TITLE_LABELS_DICTIONARY
        )

        # Construct the combined title
        plot_title = leading_substring + prioritized_fields_substring.strip()
        if remaining_fields_substring:
            plot_title += ", " + remaining_fields_substring

        if exclude_prioritized_fields_substring:
            plot_title = remaining_fields_substring

        # Wrap
        wrapper = textwrap.TextWrapper(width=title_width, initial_indent="   ")
        wrapped_plot_title = wrapper.fill(plot_title)

        return wrapped_plot_title

    def _generate_plot_path(
        self,
        plots_subdirectory,
        plots_base_name,
        metadata_dictionary,
        single_valued_fields_dictionary: dict = None,
        excluded_fields: list = None,
    ):
        """
        Generate a file path for a plot based on characteristic substring and
        metadata.

        Parameters:
            plots_subdirectory (str): The directory where the plot will be
            saved. plots_base_name (str): The base name for the plot file.
            fields_unique_value_dictionary (dict): A dictionary
            containing unique values for specific fields. metadata_dictionary
            (dict): A dictionary containing metadata key-value pairs.
            constants.PARAMETERS_PRINTED_LABELS_DICTIONARY (dict): A dictionary
            mapping metadata keys to printed labels.

        Returns:
            str: The full path to the plot file, including the constructed
            characteristic substring.
        """

        if single_valued_fields_dictionary is None:
            single_valued_fields_dictionary = copy.deepcopy(
                self.single_valued_fields_dictionary
            )
            # self.single_valued_fields_dictionary

        # print(type(self.single_valued_fields_dictionary))
        # print(type(metadata_dictionary))

        fields_unique_value_dictionary = {
            **single_valued_fields_dictionary,
            **metadata_dictionary,
        }

        # fields_unique_value_dictionary = (
        #     single_valued_fields_dictionary | metadata_dictionary
        # )
        # Exclude specified fields from the dictionary if provided
        if excluded_fields is not None:
            fields_unique_value_dictionary = {
                field: value
                for field, value in fields_unique_value_dictionary.items()
                if field not in excluded_fields
            }

        # Initialize characteristic substring
        if "Kernel_operator_type" in fields_unique_value_dictionary:
            plots_characteristic_fields_values_string = (
                "_" + fields_unique_value_dictionary["Kernel_operator_type"]
            )
        else:
            plots_characteristic_fields_values_string = ""

        overlap_operator_method = fields_unique_value_dictionary.get(
            "Overlap_operator_method", ""
        ).replace(" operator", "")

        if overlap_operator_method != "":
            if plots_base_name.startswith("Combined_"):
                plots_base_name = (
                    "Combined_"
                    + overlap_operator_method
                    + "_"
                    + plots_base_name[len("Combined_") :]
                )
            else:
                plots_base_name = overlap_operator_method + "_" + plots_base_name

        # Append metadata information
        for key, value in metadata_dictionary.items():
            if (
                key in constants.PARAMETERS_PRINTED_LABELS_DICTIONARY
                and key != "Kernel_operator_type"
            ):
                plots_characteristic_fields_values_string += (
                    "_" + constants.PARAMETERS_PRINTED_LABELS_DICTIONARY[key]
                )
                plots_characteristic_fields_values_string += str(value)

        plots_characteristic_fields_values_string = (
            plots_characteristic_fields_values_string.replace(".", "p")
        )
        plots_characteristic_fields_values_string = (
            plots_characteristic_fields_values_string.replace(" ", "")
        )
        plots_characteristic_fields_values_string = (
            plots_characteristic_fields_values_string.replace("(", "")
        )
        plots_characteristic_fields_values_string = (
            plots_characteristic_fields_values_string.replace(")", "")
        )
        plots_characteristic_fields_values_string = (
            plots_characteristic_fields_values_string.replace(",", "")
        )

        # Construct the plot path
        plot_path = os.path.join(
            plots_subdirectory,
            f"{plots_base_name}{plots_characteristic_fields_values_string}.png",
        )

        return plot_path

    def _plot_group(
        self,
        ax,
        dataframe_group,
        is_histogram: bool = False,
        label_string: str = None,
        marker_style: str = ".",
        marker_facecolors: str = "full",
        marker_color=None,
        fitting_fn=None,
    ):
        """
        Helper method to plot a single group on the given Axes object.

        Parameters:
            ax (matplotlib.axes.Axes): The Axes object to plot on.
            dataframe_group (pd.DataFrame): The DataFrame group to plot.
        """

        # Vectorize isinstance check
        is_tuple = np.vectorize(lambda x: isinstance(x, tuple))

        # TODO: Set ability the user can request to only the mean

        self.xaxis_data = dataframe_group[self.xaxis_variable_name].to_numpy()
        if any(is_tuple(self.xaxis_data)):
            self.xaxis_data = gv.gvar(self.xaxis_data)
            # TODO: Set ability the user can request to plot the error
            self.xaxis_data = gv.mean(self.xaxis_data)

        if is_histogram:
            # Plot the histogram
            ax.hist(
                self.xaxis_data, bins=30, color="blue", alpha=0.7, edgecolor="black"
            )
        else:
            self.yaxis_data = dataframe_group[self.yaxis_variable].to_numpy()

            if marker_facecolors == "full":
                marker_facecolors = marker_color

            if any(is_tuple(self.yaxis_data)):
                self.yaxis_data = gv.gvar(self.yaxis_data)
                # else:

                ax.errorbar(
                    self.xaxis_data,
                    gv.mean(self.yaxis_data),
                    yerr=gv.sdev(self.yaxis_data),
                    fmt=marker_style,
                    markerfacecolor=marker_facecolors,
                    # markersize=8,
                    capsize=10,
                    label=label_string,
                    color=marker_color,
                )

            else:
                # Plot the data points
                ax.scatter(
                    self.xaxis_data,
                    self.yaxis_data,
                    marker=marker_style,
                    facecolors=marker_facecolors,
                    # marker="x",
                    label=label_string,
                    color=marker_color,
                )

            if fitting_fn:
                # # Perform linear fit
                # fit_params = np.polyfit(self.xaxis_data, self.yaxis_data, 1)
                # fit_fn = np.poly1d(fit_params)

                # # Plot the linear fit
                # ax.plot(
                #     self.xaxis_data,
                #     fit_fn(self.xaxis_data),
                #     color="red",
                #     linestyle="--",
                #     label=f"Fit: y={fit_params[0]:.2f}x + {fit_params[1]:.2f}",
                # )

                # ax.legend()
                fitting_fn(self)

    def plot_data(
        self,
        grouping_field: str = None,
        sorting_field: str = None,
        sort_ascending: bool = False,
        labeling_field=None,
        excluded_fields: list = None,
        excluded_title_fields: list = None,
        exclude_prioritized_fields_substring: bool = False,
        dedicated_subdirectory: bool = True,
        clear_existing_plots: bool = False,
        xaxis_log_scale: bool = False,
        yaxis_log_scale: bool = False,
        invert_xaxis: bool = False,
        invert_yaxis: bool = False,
        is_histogram: bool = False,
        include_plot_title: bool = True,
        include_legend: bool = True,
        include_legend_title: bool = True,
        enable_hashed_enumeration: bool = False,
        varying_marker_style: bool = False,
        custom_marker_style: str = None,
        custom_marker_color: str = None,
        alternating_marker_fillings: bool = False,
        marker_facecolors: str = "full",
        plot_title_width: int = 105,
        left_margin_adjustment: float = 0.14,
        right_margin_adjustment: float = 0.9,
        bottom_margin_adjustment: float = 0.12,
        top_margin_adjustment: float = 0.88,
        legend_location: str = "upper left",
        plot_title_leading_substring: str = None,
        customize_fn=None,
        fitting_fn=None,
    ):
        # Check first if pair of variables has been set
        if self.xaxis_variable_name is None or self.yaxis_variable is None:
            raise ValueError("Pair of plotting variables not set yet.")
        # Initialize the grouping fields list
        grouping_fields_list = [
            self.xaxis_variable_name,
            self.yaxis_variable,
        ]

        if excluded_fields is not None:
            # Check if input is list
            if type(excluded_fields) is not list:
                raise TypeError("Expected a list for 'excluded_fields' argument.")
            # Check if provided list elements are valid
            invalid_input = list(
                set(excluded_fields) - set(self.list_of_dataframe_fields)
            )
            if invalid_input:
                raise ValueError(
                    f"Invalid element(s) {invalid_input} passed to 'excluded_fields'"
                )
            # If validated then append to grouping field list
            grouping_fields_list += excluded_fields

        if grouping_field is not None:
            # TODO: Revisit the list that is checked against for validity
            if grouping_field not in self.list_of_multivalued_fields:
                raise ValueError(f"Invalid grouping variable {grouping_field}.")
            # If validated then append to grouping field list
            grouping_fields_list.append(grouping_field)

        for argument_name, argument_value in {
            "dedicated_subdirectory": dedicated_subdirectory,
            "clear_existing_plots": clear_existing_plots,
            "xaxis_log_scale": xaxis_log_scale,
            "yaxis_log_scale": yaxis_log_scale,
            "is_histogram": is_histogram,
        }.items():
            if not isinstance(argument_value, bool):
                raise TypeError(
                    f"{argument_name} must be a boolean,"
                    f" but got '{type(argument_value).__name__}'."
                )

        # Initialize current plots directory
        current_plots_subdirectory = self.base_plots_directory
        current_plots_base_name = self.plots_base_name
        # Create a dedicated subdirectory for individual plots if no grouping
        # field has been provided
        if dedicated_subdirectory:
            if grouping_field is None or is_histogram:
                self.individual_plots_subdirectory = self._prepare_plots_subdirectory(
                    clear_existing=clear_existing_plots
                )
                current_plots_subdirectory = self.individual_plots_subdirectory
            else:
                current_plots_base_name = (
                    "Combined_"
                    + self.plots_base_name
                    + "_grouped_by_"
                    + constants.PARAMETERS_PRINTED_LABELS_DICTIONARY.get(
                        grouping_field, ""
                    )
                )
                self.combined_plots_subdirectory = self._prepare_plots_subdirectory(
                    plots_base_subdirectory=self.individual_plots_subdirectory,
                    plots_base_name=current_plots_base_name,
                    clear_existing=clear_existing_plots,
                )
                current_plots_subdirectory = self.combined_plots_subdirectory

        if not isinstance(plot_title_width, int):
            raise TypeError(
                f"Expected an integer for 'plot_title_width',"
                f" but got '{type(plot_title_width).__name__}'."
            )

        # print(grouping_fields_list)

        dataframe_group = self.group_by_reduced_tunable_parameters_list(
            grouping_fields_list
        )

        reduced_tunable_parameters_list = self.reduced_tunable_parameters_list

        #  Initialize metadata dictionary
        metadata_dictionary = {}
        for values_combination, group in dataframe_group:

            if sorting_field is not None:
                # Sort the group DataFrame by "sorting_field" in ascending order
                group = group.sort_values(by=sorting_field, ascending=sort_ascending)

            # # Fill in the metadata dictionary properly
            # if (
            #     type(values_combination) is not tuple
            #     and len(reduced_tunable_parameters_list) == 1
            # ):
            #     metadata_dictionary[reduced_tunable_parameters_list[0]] = (
            #         values_combination
            #     )
            # elif len(reduced_tunable_parameters_list) > 1:
            #     metadata_dictionary = dict(
            #         zip(reduced_tunable_parameters_list, list(values_combination))
            #     )

            if not isinstance(values_combination, tuple):
                values_combination = [values_combination]
            metadata_dictionary = dict(
                zip(reduced_tunable_parameters_list, values_combination)
            )

            fig, ax = plt.subplots()
            ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

            if include_plot_title:
                if plot_title_leading_substring is None:
                    plot_title_leading_substring = ""
                    if is_histogram:
                        plot_title_leading_substring += "Histogram "
                    if grouping_field:
                        plot_title_leading_substring += "Combined "

                plot_title = self._construct_plot_title(
                    plot_title_leading_substring,
                    metadata_dictionary,
                    exclude_prioritized_fields_substring=exclude_prioritized_fields_substring,
                    title_width=plot_title_width,
                    excluded_title_fields=excluded_title_fields,
                    additional_excluded_fields=excluded_fields,
                )
                ax.set_title(f"{plot_title}", pad=8)

            # Set axes scale
            if (
                self.xaxis_variable_name in constants.PARAMETERS_WITH_EXPONENTIAL_FORMAT
            ) or xaxis_log_scale:
                ax.set_xscale("log")
            if (
                self.yaxis_variable in constants.PARAMETERS_WITH_EXPONENTIAL_FORMAT
            ) or yaxis_log_scale:
                ax.set_yscale("log")

            # Invert axes
            if (
                self.xaxis_variable_name in constants.PARAMETERS_WITH_EXPONENTIAL_FORMAT
            ) or invert_xaxis:
                fig.gca().invert_xaxis()
            if (
                self.yaxis_variable in constants.PARAMETERS_WITH_EXPONENTIAL_FORMAT
            ) or invert_yaxis:
                fig.gca().invert_yaxis()

            # ax.set(
            #     xlabel=constants.AXES_LABELS_DICTIONARY[self.xaxis_variable_name],
            #     ylabel=constants.AXES_LABELS_DICTIONARY[self.yaxis_variable],
            # )

            ax.set_xlabel(
                # xlabel=constants.AXES_LABELS_DICTIONARY[self.xaxis_variable_name]
                xlabel=constants.AXES_LABELS_DICTIONARY.get(
                    self.xaxis_variable_name, ""
                )
            )
            # , fontsize=14)

            if is_histogram:
                ax.set_ylabel("Frequency", fontsize=14)
            else:
                ax.set_ylabel(
                    ylabel=constants.AXES_LABELS_DICTIONARY.get(
                        self.yaxis_variable, None
                    )
                )
                #   , fontsize=14)
            # Adjust left margin
            fig.subplots_adjust(
                left=left_margin_adjustment,
                right=right_margin_adjustment,
                bottom=bottom_margin_adjustment,
                top=top_margin_adjustment,
            )

            # Set axes ticks to integer values only
            if self.xaxis_variable_name in constants.PARAMETERS_OF_INTEGER_VALUE:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if self.yaxis_variable in constants.PARAMETERS_OF_INTEGER_VALUE:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            if grouping_field:

                total_number_of_grouping_field_values = group[grouping_field].nunique()

                # Generate a stable unique ID mapping
                config_label_to_id_dict = {
                    label: idx for idx, label in enumerate(sorted(group[grouping_field].unique()))
                }

                for count, (grouping_field_values, sub_group) in enumerate(
                    group.groupby(
                        grouping_field,
                        observed=True,
                        sort=False,
                    )
                ):

                    if labeling_field is not None:
                        if isinstance(labeling_field, list):
                            label_value = (
                                sub_group[labeling_field[0]].values[0],
                                sub_group[labeling_field[1]].values[0],
                            )
                        else:
                            label_value = sub_group[labeling_field].values[0]
                    else:
                        label_value = grouping_field_values
                        # labeling_field = grouping_field

                    if isinstance(label_value, tuple):
                        label_string = f"{label_value[0]} ($\\kappa_{{\\mathbb{{X}}^2}}$={label_value[1]:.2f})"
                    elif isinstance(label_value, float):
                        if "e" in f"{label_value:.2e}":
                            label_string = f"{label_value:.0e}"  # Exponential format
                        else:
                            label_string = f"{label_value:.2f}"  # Decimal format
                        # label_string = f"{label_value:.2f}"
                    else:
                        label_string = str(label_value)

                    if varying_marker_style:
                        if enable_hashed_enumeration:
                            unique_value = config_label_to_id_dict[grouping_field_values]
                            # (
                            #     stable_int_hash(grouping_field_values)
                            #     % total_number_of_grouping_field_values
                            # )
                        else:
                            unique_value = count

                        marker_index = unique_value % len(constants.MARKER_STYLES)
                        marker_style = constants.MARKER_STYLES[marker_index]

                        color_index = unique_value % len(constants.DEFAULT_COLORS)
                        # print(grouping_field_values, unique_value, color_index, config_label_to_id_dict[grouping_field_values])
                        marker_color = constants.DEFAULT_COLORS[color_index]

                    else:
                        marker_style = "."
                        # TODO: I prefer "x" for not gvar values
                        marker_color = "blue"

                    if custom_marker_color is not None:
                        marker_color = custom_marker_color

                    if custom_marker_style is not None:
                        marker_style = custom_marker_style

                    if marker_facecolors != "none":
                        marker_facecolors = marker_color

                    # TODO: Introduce an optional shift to "count" to allow for
                    # switching the order the  marker fillings alternate
                    if alternating_marker_fillings:
                        if count % 2 == 0:
                            marker_facecolors = marker_color
                        else:
                            marker_facecolors = "none"

                    self._plot_group(
                        ax,
                        sub_group,
                        is_histogram=False,
                        marker_style=marker_style,
                        label_string=label_string,
                        marker_color=marker_color,
                        marker_facecolors=marker_facecolors,
                        fitting_fn=fitting_fn,
                    )
            else:
                self._plot_group(
                    ax, group, is_histogram=is_histogram, fitting_fn=fitting_fn
                )

            if grouping_field:
                if labeling_field:
                    if isinstance(labeling_field, list):
                        if labeling_field[0] == "Kernel_operator_type":
                            legend_title = f"Kernel ({labeling_field[1]})"
                        else:
                            legend_title = f"{labeling_field[0]} ({labeling_field[1]})"
                    else:
                        legend_title = labeling_field
                else:
                    legend_title = grouping_field

                legend_title = legend_title.replace("_", " ") + ":"

                legend_title = legend_title.replace(" operator", "")

                if not include_legend_title:
                    legend_title = ""

                if include_legend:
                    ax.legend(loc=legend_location, title=legend_title, framealpha=1.0)
                    # ax.legend(loc=legend_location, title=legend_title, framealpha=1.0).set_title("")

            # Apply additional customizations if provided

            if customize_fn is not None:
                customize_fn_args = inspect.signature(customize_fn).parameters
                if len(customize_fn_args) == 3:
                    customize_fn(
                        ax, metadata_dictionary=metadata_dictionary, group=group
                    )
                elif len(customize_fn_args) == 2:
                    customize_fn(ax, metadata_dictionary=metadata_dictionary)
                elif len(customize_fn_args) == 1:
                    customize_fn(ax)
                else:
                    raise ValueError("No arguments passed to customize_fn.")

            plot_path = self._generate_plot_path(
                current_plots_subdirectory,
                current_plots_base_name,
                metadata_dictionary,
                excluded_fields=excluded_fields,
            )
            fig.savefig(plot_path)
            plt.close()

    def plot_histogram(
        self,
        histogram_variable: str,
        reference_field: str = None,
        excluded_fields_list: list = None,
        number_of_bins: int = 30,
        color: str = "blue",
        dedicated_subdirectory: bool = True,
        clear_existing_plots: bool = False,
        xaxis_log_scale: bool = False,
        yaxis_log_scale: bool = False,
        invert_xaxis: bool = False,
        invert_yaxis: bool = False,
        plot_title_width: int = 105,
        legend_location: str = "upper left",
    ):
        # Construct a specific histogram base, different from the default format
        histograms_base_name = "Histogram_of_" + histogram_variable + "_values"
        if reference_field is not None:
            histograms_base_name += "_by_" + reference_field

        # Input is validated by the "set_pair_of_variables()" method
        self.set_pair_of_variables(
            histogram_variable, "Frequency", plots_base_name=histograms_base_name
        )

        if reference_field is not None:
            if excluded_fields_list is None:
                excluded_fields_list = [reference_field]
            else:
                if type(excluded_fields_list) is not list:
                    raise TypeError(
                        "Expected a list for 'excluded_fields_list' argument."
                    )
                excluded_fields_list.append(reference_field)

        # Input is validated by the "plot_data()" method
        self.plot_data(
            excluded_fields=excluded_fields_list,
            dedicated_subdirectory=dedicated_subdirectory,
            clear_existing_plots=clear_existing_plots,
            xaxis_log_scale=xaxis_log_scale,
            yaxis_log_scale=yaxis_log_scale,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            is_histogram=True,
            plot_title_width=plot_title_width,
            legend_location=legend_location,
        )


#################


def plot_correlator(
    y_values,
    xlabel,
    ylabel,
    base_name,
    subdirectory,
    metadata_dict,
    tunable_parameters_dict,
    starting_time: int = 0,
    yaxis_log_scale: bool = False,
):
    """Generalized function to plot correlators."""
    y = y_values[starting_time:]
    x = np.arange(starting_time, len(y) + starting_time)

    fig, ax = plt.subplots()
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    plot_title = DataPlotter._construct_plot_title(
        None,
        leading_substring="",
        metadata_dictionary=metadata_dict,
        title_width=100,
        fields_unique_value_dictionary=tunable_parameters_dict,
        excluded_title_fields=[
            "Number_of_vectors",
            "Delta_Min",
            "Delta_Max",
            "Lanczos_epsilon",
        ],
    )
    ax.set_title(f"{plot_title}", pad=8)

    ax.set(xlabel=xlabel, ylabel=ylabel)
    fig.subplots_adjust(left=0.15)  # Adjust left margin

    # Force scientific notation when values are below 1e-4
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits(
        (-4, 4)
    )  # Scientific notation for |values| < 1e-4 or > 1e4
    ax.yaxis.set_major_formatter(formatter)

    # Adjust the position of the offset text (scientific notation)
    # TODO: Make the offset text bold
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_x(-0.15)  # Move left (negative values move it further left)
    offset_text.set_y(-0.05)  # Move down (negative values move it further down)

    if yaxis_log_scale:
        ax.set_yscale("log")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.errorbar(x, gv.mean(y), yerr=gv.sdev(y), fmt=".", markersize=8, capsize=10)

    plot_path = DataPlotter._generate_plot_path(
        None,
        subdirectory,
        base_name,
        metadata_dict,
        single_valued_fields_dictionary=tunable_parameters_dict,
    )

    fig.savefig(plot_path)
    plt.close()


def generic_plot(
    x_values,
    y_values,
    xlabel,
    ylabel,
    base_name,
    base_plots_directory,
    metadata_dict,
    tunable_parameters_dict,
    yaxis_log_scale: bool = False,
):
    x = x_values
    y = y_values

    fig, ax = plt.subplots()
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    plot_title = DataPlotter._construct_plot_title(
        None,
        leading_substring="",
        metadata_dictionary=metadata_dict,
        title_width=100,
        fields_unique_value_dictionary=tunable_parameters_dict,
    )
    ax.set_title(f"{plot_title}", pad=8)

    ax.set(xlabel=xlabel, ylabel=ylabel)
    # fig.subplots_adjust(left=0.15)  # Adjust left margin

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.scatter(x, y, marker=".")  # , markersize=8, capsize=10)
    # ax.errorbar(x, gv.mean(y), yerr=gv.sdev(y), fmt=".", markersize=8, capsize=10)

    plots_subdirectory = DataPlotter._prepare_plots_subdirectory(
        None, base_plots_directory, base_name
    )

    plot_path = DataPlotter._generate_plot_path(
        None,
        plots_subdirectory,
        base_name,
        metadata_dict,
        single_valued_fields_dictionary=tunable_parameters_dict,
    )

    fig.savefig(plot_path)
    plt.close()
