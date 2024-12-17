import os
import shutil

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import textwrap
import gvar as gv

from library import constants, data_processing


class DataPlotter:
    def __init__(self, dataframe, base_plot_directory):
        """
        Initialize the DataPlotter with a pandas DataFrame.

        Parameters: dataframe (pd.DataFrame): The data to be visualized.
        """
        self.dataframe = dataframe
        self.base_plot_directory = base_plot_directory
        self.current_plots_subdirectory = base_plot_directory

        # Create an instance of DataFrameAnalyzer
        self.analyzer = data_processing.DataFrameAnalyzer(self.dataframe)

        self.fields_with_single_unique_values_dictionary = (
            data_processing.get_fields_with_unique_values(self.dataframe)
        )

    def set_pair_of_variables_for_plotting(self, independent_var, dependent_var):

        # TODO: Check input
        self.independent_variable_name = independent_var
        self.dependent_variable_name = dependent_var

        self.plots_base_name = (
            self.dependent_variable_name + "_Vs_" + self.independent_variable_name
        )

    def prepare_plots_directory(
        self, base_directory=None, plots_base_name=None, clear_existing=False
    ):
        """
        Prepare the directory for saving plots.

        Parameters:
            clear_existing (bool): Whether to delete existing files and
            subdirectories.
                - If True, clears all contents of the directory.
                - If False, leaves existing files intact.

        Returns:
            str: The full path to the prepared directory.
        """

        if base_directory is None:
            base_directory = self.base_plot_directory

        if plots_base_name is None:
            plots_base_name = self.plots_base_name

        plots_subdirectory = os.path.join(base_directory, plots_base_name)

        # Ensure the directory exists before clearing or using it.
        os.makedirs(plots_subdirectory, exist_ok=True)

        if clear_existing:
            for item in os.listdir(plots_subdirectory):
                item_path = os.path.join(plots_subdirectory, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

        self.current_plots_subdirectory = plots_subdirectory

    def construct_plot_title(
        self, leading_substring, metadata_dictionary, title_width=100
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
            self.fields_with_single_unique_values_dictionary | metadata_dictionary
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
            self.fields_with_single_unique_values_dictionary | metadata_dictionary
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

    def plot_individuals(self, TEST_list=list()):

        excluded_fields = {
            "Filename",
            *constants.OUTPUT_VALUES_LIST,
            self.independent_variable_name,
            self.dependent_variable_name,
            *TEST_list,
        }
        self.analyzer.set_excluded_fields(excluded_fields)

        self.valid_groups_with_metadata = (
            self.analyzer.get_valid_dataframe_groups_with_metadata()
        )

        for group_data in self.valid_groups_with_metadata:
            dataframe_group = group_data["dataframe_group"]
            metadata_dictionary = group_data["metadata"]

            independent_variable_dataset = dataframe_group[
                self.independent_variable_name
            ].to_numpy()
            dependent_variable_dataset = gv.gvar(
                dataframe_group[self.dependent_variable_name].to_numpy()
            )

            fig, ax = plt.subplots()
            ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

            plot_title_leading_substring = ""
            plot_title = self.construct_plot_title(
                plot_title_leading_substring,
                metadata_dictionary,
            )
            ax.set_title(f"{plot_title}", pad=6)

            ax.set_yscale("log")

            ax.set(xlabel="N", ylabel="||sgn$^2$(X) - I||$^2$")
            fig.subplots_adjust(left=0.14)  # Adjust left margin

            # Set x-axis ticks to integer values only
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            ax.errorbar(
                independent_variable_dataset,
                gv.mean(dependent_variable_dataset),
                yerr=gv.sdev(dependent_variable_dataset),
                fmt=".",
                markersize=8,
                capsize=10,
            )

            plot_path = self.generate_plot_path(
                self.current_plots_subdirectory,
                self.plots_base_name,
                metadata_dictionary,
            )
            fig.savefig(plot_path)
            plt.close()

    def plot_grouped(self, group_var, TEST_list=list(), plots_directory=None):

        self.group_variable_name = group_var

        if plots_directory is None:
            plots_directory = self.current_plots_subdirectory

        plots_base_name = "Combined_" + self.plots_base_name

        self.prepare_plots_directory(plots_directory, plots_base_name)
        plots_subdirectory = self.current_plots_subdirectory

        excluded_fields = {
            "Filename",
            *constants.OUTPUT_VALUES_LIST,
            self.independent_variable_name,
            self.dependent_variable_name,
            self.group_variable_name,
            *TEST_list,
        }
        self.analyzer.set_excluded_fields(excluded_fields)

        self.valid_groups_with_metadata = (
            self.analyzer.get_valid_dataframe_groups_with_metadata()
        )

        for group_data in self.valid_groups_with_metadata:
            dataframe_group = group_data["dataframe_group"]
            metadata_dictionary = group_data["metadata"]

            fig, ax = plt.subplots()
            ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

            plot_title_leading_substring = ""
            plot_title = self.construct_plot_title(
                plot_title_leading_substring, metadata_dictionary
            )
            ax.set_title(f"{plot_title}", pad=6)

            ax.set_yscale("log")

            ax.set(xlabel="N", ylabel="||sgn$^2$(X) - I||$^2$")
            fig.subplots_adjust(left=0.14)  # Adjust left margin

            # Set x-axis ticks to integer values only
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            for (
                group_variable,
                group_variable_group,
            ) in dataframe_group.groupby(self.group_variable_name):

                independent_variable_dataset = group_variable_group[
                    self.independent_variable_name
                ].to_numpy()
                dependent_variable_dataset = gv.gvar(
                    group_variable_group[self.dependent_variable_name].to_numpy()
                )

                ax.errorbar(
                    independent_variable_dataset,
                    gv.mean(dependent_variable_dataset),
                    yerr=gv.sdev(dependent_variable_dataset),
                    fmt=".",
                    markersize=8,
                    capsize=10,
                )

            plot_path = self.generate_plot_path(
                plots_subdirectory,
                self.plots_base_name,
                metadata_dictionary,
            )
            fig.savefig(plot_path)
            plt.close()


def print_dictionaries_side_by_side(left_dictionary, right_dictionary, line_width=80, left_column_title=None, right_column_title=None):
    """
    Print two dictionaries side by side, with the second dictionary
    starting at the middle of the line width.

    Parameters:
    - left_dictionary (dict): The first dictionary to print.
    - right_dictionary (dict): The second dictionary to print.
    - line_width (int, optional): The total width of the line. Default is 80.
    """
    # Calculate the middle position of the line
    middle_position = line_width // 2

    # Prepare keys and values as formatted strings
    left_dictionary_items = [f"{k}: {v}" for k, v in left_dictionary.items()]
    right_dictionary_items = [f"{k}: {v}" for k, v in right_dictionary.items()]

    # Determine the maximum number of lines to print
    max_lines = max(len(left_dictionary_items), len(right_dictionary_items))

    # Print titles if provided, aligned with the key-value pairs
    if left_column_title and right_column_title:
        # Format and align the two column titles
        # Format the first title and add the separator
        title_output = f"{left_column_title:<{middle_position-3}} | {right_column_title}"
        print(title_output)
        # print(f"{left_column_title:<{middle_position}}{right_column_title}")
        print("-" * (line_width))

    # Print dictionaries side by side
    for i in range(max_lines):
        # Get the current item from each dictionary, if it exists
        left = left_dictionary_items[i] if i < len(left_dictionary_items) else ""
        right = right_dictionary_items[i] if i < len(right_dictionary_items) else ""
        
        # Format and align the two outputs
        output = f"{left:<{middle_position}}{right}"
        print(output)
