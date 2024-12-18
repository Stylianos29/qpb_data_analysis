import os
import shutil

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import textwrap
import gvar as gv

from library import constants, filesystem_utilities, data_processing


class DataPlotter:
    def __init__(
        self,
        analyzer: data_processing.DataFrameAnalyzer,
        base_plots_directory: str,
    ):

        if not filesystem_utilities.is_valid_directory(base_plots_directory):
            raise ValueError(f"Invalid base plots directory: '{base_plots_directory}'.")
        self.base_plots_directory = base_plots_directory

        # Initialize plots subdirectories
        self.individual_plots_subdirectory = base_plots_directory
        self.combined_plots_subdirectory = base_plots_directory

        # Initialize pair of variable for plotting
        self.independent_variable_name = None
        self.dependent_variable_name = None

        # Use provided DataFrameAnalyzer object to extract constant attributes
        self.analyzer = analyzer

        self.single_valued_fields_dictionary = (
            self.analyzer.single_valued_fields_dictionary
        )

        self.list_of_dataframe_fields = self.analyzer.list_of_dataframe_fields

    def set_pair_of_variables(
        self,
        independent_variable: str,
        dependent_variable: str,
    ):
        # Validate input
        if independent_variable not in self.list_of_dataframe_fields:
            raise ValueError(f"Invalid x-axis variable name {independent_variable}.")
        self.independent_variable_name = independent_variable

        if dependent_variable not in self.list_of_dataframe_fields:
            raise ValueError(f"Invalid y-axis variable name {independent_variable}.")
        self.dependent_variable_name = dependent_variable

        self.plots_base_name = (
            self.dependent_variable_name + "_Vs_" + self.independent_variable_name
        )

    def prepare_plots_subdirectory(
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

    def construct_plot_title(
        self, leading_substring, metadata_dictionary, title_width=105
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

        fields_unique_value_dictionary = (
            self.single_valued_fields_dictionary | metadata_dictionary
        )

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
                    f"{constants.AXES_LABELS_DICTIONARY[additional_field]}="
                    f"{fields_unique_value_dictionary[additional_field]}"
                )

        # Filter out prioritized fields from the dictionary
        excluded_fields = set(
            list_of_fields_to_appear_first + list(additional_fields.values())
        )
        remaining_fields = {
            key: value
            for key, value in fields_unique_value_dictionary.items()
            if key not in excluded_fields
        }

        # Add remaining fields to the subtitle
        remaining_fields_substring = ", ".join(
            f"{constants.AXES_LABELS_DICTIONARY[key]}={value}"
            for key, value in remaining_fields.items()
            if key in constants.AXES_LABELS_DICTIONARY
        )

        # Construct the combined title
        plot_title = leading_substring + prioritized_fields_substring.strip()
        if remaining_fields_substring:
            plot_title += ", " + remaining_fields_substring

        # Wrap
        wrapper = textwrap.TextWrapper(width=title_width, initial_indent="   ")
        wrapped_plot_title = wrapper.fill(plot_title)

        return wrapped_plot_title

    def generate_plot_path(
        self,
        plots_subdirectory,
        plots_base_name,
        metadata_dictionary,
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

        fields_unique_value_dictionary = (
            self.single_valued_fields_dictionary | metadata_dictionary
        )

        # Initialize characteristic substring
        if "Kernel_operator_type" in fields_unique_value_dictionary:
            plots_characteristic_fields_values_string = fields_unique_value_dictionary[
                "Kernel_operator_type"
            ]
        else:
            plots_characteristic_fields_values_string = ""

        # Append metadata information
        for key, value in metadata_dictionary.items():
            if key in constants.PARAMETERS_PRINTED_LABELS_DICTIONARY:
                plots_characteristic_fields_values_string += (
                    "_" + constants.PARAMETERS_PRINTED_LABELS_DICTIONARY[key]
                )
                plots_characteristic_fields_values_string += str(value)

        # Construct the plot path
        plot_path = os.path.join(
            plots_subdirectory,
            f"{plots_base_name}_{plots_characteristic_fields_values_string}.png",
        )

        return plot_path

    def _plot_group(self, ax, dataframe_group):
        """
        Helper method to plot a single group on the given Axes object.

        Parameters:
            ax (matplotlib.axes.Axes): The Axes object to plot on.
            dataframe_group (pd.DataFrame): The DataFrame group to plot.
        """
        independent_data = dataframe_group[self.independent_variable_name].to_numpy()
        dependent_data = gv.gvar(
            dataframe_group[self.dependent_variable_name].to_numpy()
        )

        ax.errorbar(
            independent_data,
            gv.mean(dependent_data),
            yerr=gv.sdev(dependent_data),
            fmt=".",
            markersize=8,
            capsize=10,
        )

    def plot_data(
        self,
        grouping_field: str = None,
        excluded_fields: list = None,
        dedicated_subdirectory: bool = True,
        clear_existing_plots: bool = False,
    ):
        # Check first if pair of variables has been set
        if (
            self.independent_variable_name is None
            or self.dependent_variable_name is None
        ):
            raise ValueError("Pair of plotting variables not set yet.")
        # Initialize the grouping fields list
        grouping_fields_list = [
            self.independent_variable_name,
            self.dependent_variable_name,
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
            if grouping_field not in self.list_of_dataframe_fields:
                raise ValueError(f"Invalid grouping variable {grouping_field}.")
            # If validated then append to grouping field list
            grouping_fields_list.append(grouping_field)

        # Initialize current plots directory
        current_plots_subdirectory = self.base_plots_directory
        current_plots_base_name = self.plots_base_name
        # Create a dedicated subdirectory for individual plots if no grouping
        # field has been provided
        if dedicated_subdirectory:
            if grouping_field is None:
                self.individual_plots_subdirectory = self.prepare_plots_subdirectory(
                    clear_existing=clear_existing_plots
                )
                current_plots_subdirectory = self.individual_plots_subdirectory
            else:
                current_plots_base_name = (
                    "Combined_" + self.plots_base_name + "_grouped_by_" + constants.PARAMETERS_PRINTED_LABELS_DICTIONARY[grouping_field]
                )
                self.combined_plots_subdirectory = self.prepare_plots_subdirectory(
                    plots_base_subdirectory=self.individual_plots_subdirectory,
                    plots_base_name=current_plots_base_name,
                    clear_existing=clear_existing_plots,
                )
                current_plots_subdirectory = self.combined_plots_subdirectory

        dataframe_group = self.analyzer.group_by_reduced_tunable_parameters_list(
            grouping_fields_list
        )

        reduced_tunable_parameters_list = self.analyzer.reduced_tunable_parameters_list

        #  Initialize metadata dictionary
        metadata_dictionary = {}
        for values_combination, group in dataframe_group:

            if (
                type(values_combination) is not tuple
                and len(reduced_tunable_parameters_list) == 1
            ):
                metadata_dictionary[reduced_tunable_parameters_list[0]] = (
                    values_combination
                )
            else:
                metadata_dictionary = dict(
                    zip(reduced_tunable_parameters_list, list(values_combination))
                )

            fig, ax = plt.subplots()
            ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

            plot_title_leading_substring = ""
            if grouping_field:
                plot_title_leading_substring = "Combined "
            plot_title = self.construct_plot_title(
                plot_title_leading_substring,
                metadata_dictionary,
            )
            ax.set_title(f"{plot_title}", pad=8)

            ax.set_yscale("log")

            ax.set(xlabel="N", ylabel="||sgn$^2$(X) - I||$^2$")
            fig.subplots_adjust(left=0.13)  # Adjust left margin

            # Set x-axis ticks to integer values only
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            if grouping_field:
                for _, sub_group in group.groupby(grouping_field):
                    self._plot_group(ax, sub_group)
            else:
                self._plot_group(ax, group)

            plot_path = self.generate_plot_path(
                current_plots_subdirectory,
                current_plots_base_name,
                metadata_dictionary,
            )
            fig.savefig(plot_path)
            plt.close()
