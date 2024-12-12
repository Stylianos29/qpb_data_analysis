import os
import shutil

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import textwrap
import gvar as gv

from library import constants
from library import data_processing


def construct_plot_subtitle(fields_unique_value_dictionary):
    """
    Constructs a subtitle string for plots based on unique field values.

    Parameters:
    -----------
    fields_unique_value_dictionary : dict
        A dictionary where keys are field names and values are their unique values.

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

    # Return the combined subtitle
    if remaining_fields_substring:
        return prioritized_fields_substring.strip() + ", " + remaining_fields_substring
    return prioritized_fields_substring.strip()


def prepare_plots_directory(base_directory, plot_name):
    """
    Prepare the directory for saving plots by clearing its contents.

    Parameters:
        base_directory (str): Base path for the plots.
        plot_name (str): Subdirectory name for this set of plots.

    Returns:
        str: The full path to the prepared directory.
    """

    plots_subdirectory = os.path.join(base_directory, plot_name)

    if os.path.exists(plots_subdirectory):
        for item in os.listdir(plots_subdirectory):
            item_path = os.path.join(plots_subdirectory, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    os.makedirs(plots_subdirectory, exist_ok=True)
    return plots_subdirectory


def save_plot(fig, plots_directory, base_name, metadata):
    """
    Save the plot to a file with a descriptive name based on metadata.

    Parameters:
        fig (matplotlib.figure.Figure): The figure object to save.
        plots_directory (str): The directory to save the plot.
        base_name (str): Base name for the plot file.
        metadata (dict): Metadata to construct a characteristic name.
    """

    characteristic_string = "_".join(
        f"{constants.PARAMETERS_PRINTED_LABELS_DICTIONARY.get(key, key)}{value}"
        for key, value in metadata.items()
        if key in constants.PARAMETERS_PRINTED_LABELS_DICTIONARY
    )
    plot_path = os.path.join(
        plots_directory, f"{base_name}_{characteristic_string}.png"
    )
    fig.savefig(plot_path)
    plt.close(fig)


def generate_plot(dataframe_group, metadata, x_variable, y_variable, plots_directory):
    """
    Generate and save a plot for given variables.

    Parameters:
        dataframe_group (pd.DataFrame): Data for the plot.
        metadata (dict): Metadata for constructing subtitles and file names.
        x_variable (str): The independent variable.
        y_variable (str): The dependent variable.
        plots_directory (str): Directory to save the plots.
    """

    # analyzer = data_processing.DataFrameAnalyzer(dataframe_group)

    x_values = dataframe_group[x_variable].to_numpy()
    y_values = gv.gvar(dataframe_group[y_variable].to_numpy())

    fig, ax = plt.subplots()
    ax.grid()

    field_unique_values_combined = {
        **analyzer.fields_with_unique_values_dictionary,
        **metadata,
    }
    subtitle = construct_plot_subtitle(field_unique_values_combined)
    wrapped_subtitle = textwrap.fill(subtitle, width=110, initial_indent="   ")

    ax.set_title(f"\n{wrapped_subtitle}", pad=6)
    ax.set(
        xlabel=constants.AXES_LABELS_DICTIONARY[x_variable],
        ylabel=constants.AXES_LABELS_DICTIONARY[y_variable],
    )
    plt.subplots_adjust(left=0.17)  # Adjust left margin
    plt.subplots_adjust(bottom=0.12)  # Adjust left margin

    if x_variable in constants.PARAMETERS_WITH_EXPONENTIAL_FORMAT:
        ax.set_xscale("log")
        plt.gca().invert_xaxis()
    elif x_variable in constants.PARAMETERS_OF_INTEGER_VALUE:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.scatter(x_values, gv.mean(y_values))

    plots_base_name = f"{y_variable}_Vs_{x_variable}"
    if "Kernel_operator_type" in field_unique_values_combined:
        plots_base_name += "_" + field_unique_values_combined["Kernel_operator_type"]

    save_plot(fig, plots_directory, plots_base_name, metadata)


def process_variable_pair(analyzer, base_directory, x_variable, y_variable):
    """
    Process a pair of variables and generate plots for all valid groups.

    Parameters:
        analyzer (DataFrameAnalyzer): Analyzer object to extract groups and metadata.
        base_directory (str): Base directory for plots.
        x_variable (str): Independent variable.
        y_variable (str): Dependent variable.
    """

    plots_directory = prepare_plots_directory(
        base_directory, f"{y_variable}_Vs_{x_variable}"
    )

    excluded_fields = {
        "Filename",
        *constants.OUTPUT_VALUES_LIST,
        x_variable,
    }
    analyzer.set_excluded_fields(excluded_fields)

    for group_data in analyzer.get_valid_dataframe_groups_with_metadata():
        generate_plot(
            group_data["dataframe_group"],
            group_data["metadata"],
            x_variable,
            y_variable,
            plots_directory,
        )

#################################################

# def prepare_directory(directory_path):
#     """Prepare a directory by clearing its contents or creating it if it doesn't exist."""
#     if os.path.exists(directory_path):
#         for item in os.listdir(directory_path):
#             item_path = os.path.join(directory_path, item)
#             if os.path.isfile(item_path):
#                 os.remove(item_path)
#             elif os.path.isdir(item_path):
#                 shutil.rmtree(item_path)
#     os.makedirs(directory_path, exist_ok=True)


# def save_plot(fig, directory, base_name, characteristic_string):
#     """Save a plot figure to a specified directory with a formatted name."""
#     file_name = f"{base_name}_{characteristic_string}.png"
#     plot_path = os.path.join(directory, file_name)
#     fig.savefig(plot_path)
#     plt.close()


# def configure_axes(ax, x_label, y_label, x_log=True, invert_x=False):
#     """Configure the axes with labels, scales, and optional inversion."""
#     ax.set(xlabel=x_label, ylabel=y_label)
#     if x_log:
#         ax.set_xscale("log")
#     if invert_x:
#         ax.invert_xaxis()

# def construct_characteristic_string(metadata, keys_order):
#     """Construct a characteristic string for the plot based on metadata."""
#     return "_".join(
#         f"{constants.PARAMETERS_PRINTED_LABELS_DICTIONARY.get(key, key)}{metadata[key]}"
#         for key in keys_order if key in metadata
#     )


# def plot_grouped_scatter(
#     dataframe_group,
#     independent_variable_name,
#     dependent_variable_name,
#     category_variable_name,
#     x_label,
#     y_label,
#     metadata,
#     directory,
#     base_name,
# ):
#     """Create a grouped scatter plot for a given dataframe group."""
#     fig, ax = plt.subplots()
#     ax.grid()

#     # Construct plot subtitle
#     combined_metadata = analyzer.fields_with_unique_values_dictionary | metadata
#     subtitle = construct_plot_subtitle(combined_metadata)
#     wrapped_subtitle = textwrap.fill(subtitle, width=110, initial_indent="   ")
#     ax.set_title(f"\n{wrapped_subtitle}", pad=6)

#     configure_axes(ax, x_label, y_label, x_log=True, invert_x=True)

#     for category_value, group in dataframe_group.groupby(category_variable_name):
#         x_values = group[independent_variable_name].to_numpy()
#         y_values = gv.gvar(group[dependent_variable_name].to_numpy())

#         ax.scatter(
#             x_values,
#             gv.mean(y_values),
#             label=f"{constants.PARAMETERS_PRINTED_LABELS_DICTIONARY[category_variable_name]}={category_value}",
#         )

#     ax.legend(loc="upper left")

#     characteristic_string = construct_characteristic_string(metadata, keys_order=metadata.keys())
#     save_plot(fig, directory, base_name, characteristic_string)


# def process_grouped_plots(
#     analyzer,
#     independent_variable_name,
#     dependent_variable_name,
#     category_variable_name,
#     plots_base_name,
#     plots_subdirectory,
# ):
#     """Process grouped plots for valid dataframe groups."""
#     prepare_directory(plots_subdirectory)

#     excluded_fields = {
#         "Filename",
#         *constants.OUTPUT_VALUES_LIST,
#         independent_variable_name,
#         category_variable_name,
#     }
#     analyzer.set_excluded_fields(excluded_fields)

#     valid_groups_with_metadata = analyzer.get_valid_dataframe_groups_with_metadata()

#     for group_data in valid_groups_with_metadata:
#         dataframe_group = group_data["dataframe_group"]
#         metadata_dictionary = group_data["metadata"]

#         plot_grouped_scatter(
#             dataframe_group,
#             independent_variable_name,
#             dependent_variable_name,
#             category_variable_name,
#             x_label=constants.AXES_LABELS_DICTIONARY[independent_variable_name],
#             y_label=constants.AXES_LABELS_DICTIONARY[dependent_variable_name],
#             metadata=metadata_dictionary,
#             directory=plots_subdirectory,
#             base_name=plots_base_name,
#         )


import os

def generate_plot_path(
    plots_subdirectory,
    plots_base_name,
    field_unique_values_combined_dictionary,
    metadata_dictionary
):
    """
    Generate a file path for a plot based on characteristic substring and metadata.

    Parameters:
        plots_subdirectory (str): The directory where the plot will be saved.
        plots_base_name (str): The base name for the plot file.
        field_unique_values_combined_dictionary (dict): A dictionary containing unique values for specific fields.
        metadata_dictionary (dict): A dictionary containing metadata key-value pairs.
        constants.PARAMETERS_PRINTED_LABELS_DICTIONARY (dict): A dictionary mapping metadata keys to printed labels.

    Returns:
        str: The full path to the plot file, including the constructed characteristic substring.
    """
    # Initialize characteristic substring
    if "Kernel_operator_type" in field_unique_values_combined_dictionary:
        plots_characteristic_fields_values_string = (
            field_unique_values_combined_dictionary["Kernel_operator_type"]
        )
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
