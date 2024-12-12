import os
import shutil

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import textwrap
import gvar as gv

from library import data_processing
from library import constants
from library import plotting


INPUT_CSV_FILE_PATH = "/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/invert/Chebyshev_several_m_varying_EpsLanczos/PCAC_mass_estimates.csv"
PLOTS_DIRECTORY = "/nvme/h/cy22sg1/qpb_data_analysis/output/plots/invert/Chebyshev_several_m_varying_EpsLanczos"

# ANALYZE .CSV FILE

qpb_log_files_dataframe = data_processing.load_csv(INPUT_CSV_FILE_PATH)

# Create an instance of DataFrameAnalyzer
analyzer = data_processing.DataFrameAnalyzer(qpb_log_files_dataframe)


# PCAC_mass_estimate VS Lanczos_epsilon

independent_variable_name = "Lanczos_epsilon"
dependent_variable_name = "PCAC_mass_estimate"

plots_base_name = dependent_variable_name + "_Vs_" + independent_variable_name
plots_subdirectory = os.path.join(PLOTS_DIRECTORY, plots_base_name)

# Check if the directory exists
if os.path.exists(plots_subdirectory):
    # If it exists, remove all files and subdirectories inside it
    for item in os.listdir(plots_subdirectory):
        item_path = os.path.join(plots_subdirectory, item)
        if os.path.isfile(item_path):
            os.remove(item_path)  # Remove file
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove directory

# Create "plots_subdirectory" if it does not exist
os.makedirs(plots_subdirectory, exist_ok=True)

excluded_fields = {
    "Filename",
    *constants.OUTPUT_VALUES_LIST,
    independent_variable_name,
}
analyzer.set_excluded_fields(excluded_fields)

valid_groups_with_metadata = analyzer.get_valid_dataframe_groups_with_metadata()

for analysis_index, group_data in enumerate(valid_groups_with_metadata, start=1):
    dataframe_group = group_data["dataframe_group"]
    metadata_dictionary = group_data["metadata"]

    independent_variable_values_array = dataframe_group[
        independent_variable_name
    ].to_numpy()
    dependent_variable_values_array = gv.gvar(
        dataframe_group[dependent_variable_name].to_numpy()
    )

    fig, ax = plt.subplots()
    ax.grid()

    plot_title = ""
    field_unique_values_combined_dictionary = (
        analyzer.fields_with_unique_values_dictionary | metadata_dictionary
    )
    parameter_values_subtitle = plotting.construct_plot_subtitle(
        field_unique_values_combined_dictionary
    )
    wrapper = textwrap.TextWrapper(width=110, initial_indent="   ")
    wrapped_parameter_values_subtitle = wrapper.fill(parameter_values_subtitle)

    ax.set_title(f"{plot_title}\n{wrapped_parameter_values_subtitle}", pad=6)

    ax.set(
        xlabel=constants.AXES_LABELS_DICTIONARY[independent_variable_name],
        ylabel=constants.AXES_LABELS_DICTIONARY[dependent_variable_name],
    )
    plt.subplots_adjust(left=0.17)  # Adjust left margin
    plt.subplots_adjust(bottom=0.12)  # Adjust left margin

    ax.set_xscale("log")

    # Invert the x-axis
    plt.gca().invert_xaxis()

    ax.scatter(
        independent_variable_values_array,
        gv.mean(dependent_variable_values_array),
    )

    # Initialize characteristic substring
    if "Kernel_operator_type" in field_unique_values_combined_dictionary:
        plots_characteristic_fields_values_string = (
            field_unique_values_combined_dictionary["Kernel_operator_type"]
        )
    else:
        plots_characteristic_fields_values_string = ""
    for key, value in metadata_dictionary.items():
        if key in constants.PARAMETERS_PRINTED_LABELS_DICTIONARY:
            plots_characteristic_fields_values_string += (
                "_" + constants.PARAMETERS_PRINTED_LABELS_DICTIONARY[key]
            )
            plots_characteristic_fields_values_string += str(value)

    plot_path = os.path.join(
        plots_subdirectory,
        f"{plots_base_name}_{plots_characteristic_fields_values_string}" + ".png",
    )
    fig.savefig(plot_path)
    plt.close()


# PCAC_mass_estimate VS Number_of_Chebyshev_terms

independent_variable_name = "Number_of_Chebyshev_terms"
dependent_variable_name = "PCAC_mass_estimate"

plots_base_name = dependent_variable_name + "_Vs_" + independent_variable_name
plots_subdirectory = os.path.join(PLOTS_DIRECTORY, plots_base_name)

# Check if the directory exists
if os.path.exists(plots_subdirectory):
    # If it exists, remove all files and subdirectories inside it
    for item in os.listdir(plots_subdirectory):
        item_path = os.path.join(plots_subdirectory, item)
        if os.path.isfile(item_path):
            os.remove(item_path)  # Remove file
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove directory

# Create "plots_subdirectory" if it does not exist
os.makedirs(plots_subdirectory, exist_ok=True)

excluded_fields = {
    "Filename",
    *constants.OUTPUT_VALUES_LIST,
    independent_variable_name,
}
analyzer.set_excluded_fields(excluded_fields)

valid_groups_with_metadata = analyzer.get_valid_dataframe_groups_with_metadata()

for analysis_index, group_data in enumerate(valid_groups_with_metadata, start=1):
    dataframe_group = group_data["dataframe_group"]
    metadata_dictionary = group_data["metadata"]

    independent_variable_values_array = dataframe_group[
        independent_variable_name
    ].to_numpy()
    dependent_variable_values_array = gv.gvar(
        dataframe_group[dependent_variable_name].to_numpy()
    )

    fig, ax = plt.subplots()
    ax.grid()

    plot_title = ""
    field_unique_values_combined_dictionary = (
        analyzer.fields_with_unique_values_dictionary | metadata_dictionary
    )
    parameter_values_subtitle = plotting.construct_plot_subtitle(
        field_unique_values_combined_dictionary
    )
    wrapper = textwrap.TextWrapper(width=110, initial_indent="   ")
    wrapped_parameter_values_subtitle = wrapper.fill(parameter_values_subtitle)

    ax.set_title(f"{plot_title}\n{wrapped_parameter_values_subtitle}", pad=6)

    ax.set(
        xlabel=constants.AXES_LABELS_DICTIONARY[independent_variable_name],
        ylabel=constants.AXES_LABELS_DICTIONARY[dependent_variable_name],
    )
    plt.subplots_adjust(left=0.17)  # Adjust left margin
    plt.subplots_adjust(bottom=0.12)  # Adjust left margin

    # ax.set_xscale("log")

    # # Invert the x-axis
    # plt.gca().invert_xaxis()

    # Set x-axis ticks to integer values only
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.scatter(
        independent_variable_values_array,
        gv.mean(dependent_variable_values_array),
    )

    # Initialize characteristic substring
    if "Kernel_operator_type" in field_unique_values_combined_dictionary:
        plots_characteristic_fields_values_string = (
            field_unique_values_combined_dictionary["Kernel_operator_type"]
        )
    else:
        plots_characteristic_fields_values_string = ""
    for key, value in metadata_dictionary.items():
        if key in constants.PARAMETERS_PRINTED_LABELS_DICTIONARY:
            plots_characteristic_fields_values_string += (
                "_" + constants.PARAMETERS_PRINTED_LABELS_DICTIONARY[key]
            )
            plots_characteristic_fields_values_string += str(value)

    plot_path = os.path.join(
        plots_subdirectory,
        f"{plots_base_name}_{plots_characteristic_fields_values_string}" + ".png",
    )
    fig.savefig(plot_path)
    plt.close()


# Collective PCAC_mass_estimate VS Lanczos_epsilon

independent_variable_name = "Lanczos_epsilon"
dependent_variable_name = "PCAC_mass_estimate"

category_variable_name = "Number_of_Chebyshev_terms"

plots_base_name = (
    dependent_variable_name
    + "_Vs_"
    + independent_variable_name
    + "_grouped_by_"
    + category_variable_name
)
plots_subdirectory = os.path.join(PLOTS_DIRECTORY, plots_base_name)

# Check if the directory exists
if os.path.exists(plots_subdirectory):
    # If it exists, remove all files and subdirectories inside it
    for item in os.listdir(plots_subdirectory):
        item_path = os.path.join(plots_subdirectory, item)
        if os.path.isfile(item_path):
            os.remove(item_path)  # Remove file
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove directory

# Create "plots_subdirectory" if it does not exist
os.makedirs(plots_subdirectory, exist_ok=True)

excluded_fields = {
    "Filename",
    *constants.OUTPUT_VALUES_LIST,
    *[independent_variable_name, category_variable_name],
}
analyzer.set_excluded_fields(excluded_fields)

valid_groups_with_metadata = analyzer.get_valid_dataframe_groups_with_metadata()

for analysis_index, group_data in enumerate(valid_groups_with_metadata, start=1):
    dataframe_group = group_data["dataframe_group"]
    metadata_dictionary = group_data["metadata"]

    fig, ax = plt.subplots()
    ax.grid()

    plot_title = ""
    field_unique_values_combined_dictionary = (
        analyzer.fields_with_unique_values_dictionary | metadata_dictionary
    )
    parameter_values_subtitle = plotting.construct_plot_subtitle(
        field_unique_values_combined_dictionary
    )
    wrapper = textwrap.TextWrapper(width=110, initial_indent="   ")
    wrapped_parameter_values_subtitle = wrapper.fill(parameter_values_subtitle)

    ax.set_title(f"{plot_title}\n{wrapped_parameter_values_subtitle}", pad=6)

    ax.set(
        xlabel=constants.AXES_LABELS_DICTIONARY[independent_variable_name],
        ylabel=constants.AXES_LABELS_DICTIONARY[dependent_variable_name],
    )
    plt.subplots_adjust(left=0.17)  # Adjust left margin
    plt.subplots_adjust(bottom=0.12)  # Adjust left margin

    ax.set_xscale("log")

    # Invert the x-axis
    plt.gca().invert_xaxis()

    for category_variable, category_variable_group in dataframe_group.groupby(
        category_variable_name
    ):
        independent_variable_values_array = category_variable_group[
            independent_variable_name
        ].to_numpy()
        dependent_variable_values_array = gv.gvar(
            category_variable_group[dependent_variable_name].to_numpy()
        )

        ax.scatter(
            independent_variable_values_array,
            gv.mean(dependent_variable_values_array),
            label=constants.PARAMETERS_PRINTED_LABELS_DICTIONARY[category_variable_name]
            + "="
            + str(category_variable),
        )

    ax.legend(loc="upper left")

    # Initialize characteristic substring
    if "Kernel_operator_type" in field_unique_values_combined_dictionary:
        plots_characteristic_fields_values_string = (
            field_unique_values_combined_dictionary["Kernel_operator_type"]
        )
    else:
        plots_characteristic_fields_values_string = ""
    for key, value in metadata_dictionary.items():
        if key in constants.PARAMETERS_PRINTED_LABELS_DICTIONARY:
            plots_characteristic_fields_values_string += (
                "_" + constants.PARAMETERS_PRINTED_LABELS_DICTIONARY[key]
            )
            plots_characteristic_fields_values_string += str(value)

    plot_path = os.path.join(
        plots_subdirectory,
        f"{plots_base_name}_{plots_characteristic_fields_values_string}" + ".png",
    )
    fig.savefig(plot_path)
    plt.close()
