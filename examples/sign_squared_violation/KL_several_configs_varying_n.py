import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import ast
import textwrap
import gvar as gv

from library import data_processing, plotting


WORKSPACE_DIRECTORY_FULL_PATH = "/nvme/h/cy22sg1/qpb_data_analysis"
CSV_FILE_FULL_PATH = os.path.join(
    WORKSPACE_DIRECTORY_FULL_PATH,
    "data_files/processed/sign_squared_violation/KL_several_configs_varying_n/qpb_log_files_single_valued_parameters.csv",
)
HDF5_FILE_FULL_PATH = os.path.join(
    WORKSPACE_DIRECTORY_FULL_PATH,
    "data_files/processed/sign_squared_violation/KL_several_configs_varying_n/qpb_log_files_multivalued_parameters.h5",
)
PLOTTING_DIRECTORY = os.path.join(
    WORKSPACE_DIRECTORY_FULL_PATH,
    "output/plots/sign_squared_violation/KL_several_configs_varying_n",
)
os.makedirs(PLOTTING_DIRECTORY, exist_ok=True)

# Check if given .csv file full path is valid
if not (os.path.exists(CSV_FILE_FULL_PATH) and os.path.isfile(CSV_FILE_FULL_PATH)):
    print("Invalid .csv file directory.\nExiting")
    sys.exit()

# Check if given HDF5 file full path is valid
if not (os.path.exists(HDF5_FILE_FULL_PATH) and os.path.isfile(HDF5_FILE_FULL_PATH)):
    print("Invalid HDF5 file directory.\nExiting")
    sys.exit()

# Check if given plots directories is valid
if not (os.path.exists(PLOTTING_DIRECTORY) and os.path.isdir(PLOTTING_DIRECTORY)):
    print("Invalid plots directory.\nExiting...")
    sys.exit()

# Import .csv file
# KL_several_configs_varying_n_dataframe = pd.read_csv(CSV_FILE_FULL_PATH)
KL_several_configs_varying_n_dataframe = data_processing.load_csv(CSV_FILE_FULL_PATH)

# Get a list of columns with more than one unique value
columns_with_multiple_unique_values = {
    col: KL_several_configs_varying_n_dataframe[col].nunique()
    for col in KL_several_configs_varying_n_dataframe.columns
    if KL_several_configs_varying_n_dataframe[col].nunique() > 1
}
print("\nFields with multiple unique values:")
print(columns_with_multiple_unique_values)
# Get a list of columns with more than one unique value
columns_with_a_unique_value = [
    col
    for col in KL_several_configs_varying_n_dataframe.columns
    if KL_several_configs_varying_n_dataframe[col].nunique() == 1
]
print("\nFields with a single unique value:")
print(columns_with_a_unique_value)

# Extract unique values
unique_MPI_geometry = KL_several_configs_varying_n_dataframe["MPI_geometry"].unique()[0]
unique_threads_per_process = KL_several_configs_varying_n_dataframe[
    "Threads_per_process"
].unique()[0]
unique_QCD_beta_value = KL_several_configs_varying_n_dataframe[
    "QCD_beta_value"
].unique()[0]
unique_APE_alpha = KL_several_configs_varying_n_dataframe["APE_alpha"].unique()[0]
unique_APE_iterations = KL_several_configs_varying_n_dataframe[
    "APE_iterations"
].unique()[0]
unique_rho_value = KL_several_configs_varying_n_dataframe["Rho_value"].unique()[0]
unique_bare_mass = KL_several_configs_varying_n_dataframe["Bare_mass"].unique()[0]
unique_clover_coefficient = KL_several_configs_varying_n_dataframe[
    "Clover_coefficient"
].unique()[0]
unique_plaquette = KL_several_configs_varying_n_dataframe["Plaquette"].unique()[0]
unique_maximum_solver_iterations = KL_several_configs_varying_n_dataframe[
    "Maximum_solver_iterations"
].unique()[0]
unique_KL_scaling_factor = KL_several_configs_varying_n_dataframe[
    "KL_scaling_factor"
].unique()[0]
unique_overlap_operator_method = KL_several_configs_varying_n_dataframe[
    "Overlap_operator_method"
].unique()[0]
unique_number_of_vectors = KL_several_configs_varying_n_dataframe[
    "Number_of_vectors"
].unique()[0]
unique_MSCG_epsilon = KL_several_configs_varying_n_dataframe["MSCG_epsilon"].unique()[0]
unique_temporal_lattice_size = KL_several_configs_varying_n_dataframe[
    "Temporal_lattice_size"
].unique()[0]
unique_spatial_lattice_size = KL_several_configs_varying_n_dataframe[
    "Spatial_lattice_size"
].unique()[0]


# Calculate number of MV multiplications
KL_several_configs_varying_n_dataframe["Number_of_MV_multiplications"] = (
    2 * KL_several_configs_varying_n_dataframe["Average_number_of_MSCG_iterations"] * 5
    + 1
)

fields_with_single_unique_values_dictionary = (
    data_processing.get_fields_with_unique_values(
        KL_several_configs_varying_n_dataframe
    )
)
fields_with_single_unique_values_dictionary.pop("Bare_mass")

# Average calculation result Vs Number KL diagonal order

plots_base_name = "Average_calculation_result_Vs_KL_diagonal_order"
plotting_subdirectory = plotting.prepare_plots_directory(
    PLOTTING_DIRECTORY, plots_base_name
)

for (
    kernel_operator_type,
    kernel_operator_type_group,
) in KL_several_configs_varying_n_dataframe.groupby("Kernel_operator_type"):

    for (
        configuration_label,
        configuration_label_group,
    ) in kernel_operator_type_group.groupby("Configuration_label"):

        metadata_dictionary = {
            "Kernel_operator_type": kernel_operator_type,
            "Configuration_label": configuration_label,
        }

        fig, ax = plt.subplots()
        plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

        plot_title = ""
        field_unique_values_combined_dictionary = (
            fields_with_single_unique_values_dictionary | metadata_dictionary
        )
        parameter_values_subtitle = plotting.construct_plot_subtitle(
            field_unique_values_combined_dictionary
        )
        wrapper = textwrap.TextWrapper(width=110, initial_indent="   ")
        wrapped_parameter_values_subtitle = wrapper.fill(parameter_values_subtitle)

        ax.set_title(f"{plot_title}\n{wrapped_parameter_values_subtitle}", pad=6)

        plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

        ax.set_yscale("log")

        ax.set(xlabel="n", ylabel="$||sgn(X) - 1||^2$")

        # Set x-axis ticks to integer values only
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        KL_diagonal_order_values_array = configuration_label_group[
            "KL_diagonal_order"
        ].to_numpy()
        Average_calculation_results_array = gv.gvar(
            configuration_label_group["Average_calculation_result"]
            .apply(ast.literal_eval)
            .to_numpy()
        )

        ax.errorbar(
            KL_diagonal_order_values_array,
            gv.mean(Average_calculation_results_array),
            yerr=gv.sdev(Average_calculation_results_array),
            fmt=".",
            markersize=8,
            capsize=10,
        )

        plot_path = plotting.generate_plot_path(
            plotting_subdirectory,
            plots_base_name,
            field_unique_values_combined_dictionary,
            metadata_dictionary,
        )
        fig.savefig(plot_path)
        plt.close()

# Average calculation result Vs KL diagonal order Grouped by Kernel

plots_base_name = (
    "Average_calculation_result_Vs_KL_diagonal_order_Grouped_by_Kernel_operator_type"
)
plotting_subdirectory = plotting.prepare_plots_directory(
    PLOTTING_DIRECTORY, plots_base_name
)

for (
    kernel_operator_type,
    kernel_operator_type_group,
) in KL_several_configs_varying_n_dataframe.groupby("Kernel_operator_type"):

    metadata_dictionary = {
        "Kernel_operator_type": kernel_operator_type,
    }

    fig, ax = plt.subplots()
    plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    plot_title = ""
    field_unique_values_combined_dictionary = (
        fields_with_single_unique_values_dictionary | metadata_dictionary
    )
    parameter_values_subtitle = plotting.construct_plot_subtitle(
        field_unique_values_combined_dictionary
    )
    wrapper = textwrap.TextWrapper(width=110, initial_indent="   ")
    wrapped_parameter_values_subtitle = wrapper.fill(parameter_values_subtitle)

    ax.set_title(f"{plot_title}\n{wrapped_parameter_values_subtitle}", pad=6)

    plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_yscale("log")

    ax.set(xlabel="N", ylabel="$||sgn^2(X) - 1||^2$")
    plt.subplots_adjust(left=0.13)  # Adjust left margin

    # Set x-axis ticks to integer values only
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for (
        configuration_label,
        configuration_label_group,
    ) in kernel_operator_type_group.groupby("Configuration_label"):

        KL_diagonal_order_array = configuration_label_group[
            "KL_diagonal_order"
        ].to_numpy()
        Average_calculation_results_array = gv.gvar(
            configuration_label_group["Average_calculation_result"]
            .apply(ast.literal_eval)
            .to_numpy()
        )

        ax.errorbar(
            KL_diagonal_order_array,
            gv.mean(Average_calculation_results_array),
            yerr=gv.sdev(Average_calculation_results_array),
            fmt=".",
            markersize=8,
            capsize=10,
            label=f"config: {configuration_label}",
        )

    ax.legend(loc="upper right")

    plot_path = plotting.generate_plot_path(
        plotting_subdirectory,
        plots_base_name,
        field_unique_values_combined_dictionary,
        metadata_dictionary,
    )
    fig.savefig(plot_path)
    plt.close()


# Average calculation result Vs Number of MV multiplications

plots_base_name = "Average_calculation_result_Vs_Number_of_MV_multiplications"
plotting_subdirectory = plotting.prepare_plots_directory(
    PLOTTING_DIRECTORY, plots_base_name
)

for (
    kernel_operator_type,
    kernel_operator_type_group,
) in KL_several_configs_varying_n_dataframe.groupby("Kernel_operator_type"):

    for (
        configuration_label,
        configuration_label_group,
    ) in kernel_operator_type_group.groupby("Configuration_label"):

        metadata_dictionary = {
            "Kernel_operator_type": kernel_operator_type,
            "Configuration_label": configuration_label,
        }

        fig, ax = plt.subplots()
        plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

        plot_title = ""
        field_unique_values_combined_dictionary = (
            fields_with_single_unique_values_dictionary | metadata_dictionary
        )
        parameter_values_subtitle = plotting.construct_plot_subtitle(
            field_unique_values_combined_dictionary
        )
        wrapper = textwrap.TextWrapper(width=110, initial_indent="   ")
        wrapped_parameter_values_subtitle = wrapper.fill(parameter_values_subtitle)

        ax.set_title(f"{plot_title}\n{wrapped_parameter_values_subtitle}", pad=6)

        plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

        ax.set_yscale("log")

        ax.set(xlabel="Number of MV multiplications", ylabel="$||sgn^2(X) - 1||^2$")

        # Set x-axis ticks to integer values only
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        number_of_MV_multiplications_array = configuration_label_group[
            "Number_of_MV_multiplications"
        ].to_numpy()
        Average_calculation_results_array = gv.gvar(
            configuration_label_group["Average_calculation_result"]
            .apply(ast.literal_eval)
            .to_numpy()
        )

        ax.errorbar(
            number_of_MV_multiplications_array,
            gv.mean(Average_calculation_results_array),
            yerr=gv.sdev(Average_calculation_results_array),
            fmt=".",
            markersize=8,
            capsize=10,
        )

        plot_path = plotting.generate_plot_path(
            plotting_subdirectory,
            plots_base_name,
            field_unique_values_combined_dictionary,
            metadata_dictionary,
        )
        fig.savefig(plot_path)
        plt.close()


# Average calculation result Vs Number of MV multiplications Grouped by Kernel
# operator type

plots_base_name = "Average_calculation_result_Vs_Number_of_MV_multiplications_Grouped_by_Kernel_operator_type"
plotting_subdirectory = plotting.prepare_plots_directory(
    PLOTTING_DIRECTORY, plots_base_name
)

for (
    kernel_operator_type,
    kernel_operator_type_group,
) in KL_several_configs_varying_n_dataframe.groupby("Kernel_operator_type"):

    metadata_dictionary = {
        "Kernel_operator_type": kernel_operator_type,
    }

    fig, ax = plt.subplots()
    plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    plot_title = ""
    field_unique_values_combined_dictionary = (
        fields_with_single_unique_values_dictionary | metadata_dictionary
    )
    parameter_values_subtitle = plotting.construct_plot_subtitle(
        field_unique_values_combined_dictionary
    )
    wrapper = textwrap.TextWrapper(width=110, initial_indent="   ")
    wrapped_parameter_values_subtitle = wrapper.fill(parameter_values_subtitle)

    ax.set_title(f"{plot_title}\n{wrapped_parameter_values_subtitle}")  # , pad=6)

    plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_yscale("log")

    # ax.set(xlabel="log$_{10}$(Number of MV multiplications)", ylabel="$||sgn^2(X) - 1||^2$")
    ax.set(xlabel="Number of MV multiplications", ylabel="$||sgn^2(X) - 1||^2$")
    plt.subplots_adjust(left=0.13)  # Adjust left margin

    # Set x-axis ticks to integer values only
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for (
        configuration_label,
        configuration_label_group,
    ) in kernel_operator_type_group.groupby("Configuration_label"):

        # number_of_MV_multiplications_array = np.log10(configuration_label_group[
        #     "Number_of_MV_multiplications"
        # ].to_numpy())
        number_of_MV_multiplications_array = configuration_label_group[
            "Number_of_MV_multiplications"
        ].to_numpy()
        Average_calculation_results_array = gv.gvar(
            configuration_label_group["Average_calculation_result"]
            .apply(ast.literal_eval)
            .to_numpy()
        )

        ax.errorbar(
            number_of_MV_multiplications_array,
            gv.mean(Average_calculation_results_array),
            yerr=gv.sdev(Average_calculation_results_array),
            fmt=".",
            markersize=8,
            capsize=10,
            label=f"config: {configuration_label}",
        )

    ax.legend(loc="upper right")

    plot_path = plotting.generate_plot_path(
        plotting_subdirectory,
        plots_base_name,
        field_unique_values_combined_dictionary,
        metadata_dictionary,
    )
    fig.savefig(plot_path)
    plt.close()


# Average calculation result Vs Number of MV multiplications Grouped by
# Configuration label

plots_base_name = "Average_calculation_result_Vs_Number_of_MV_multiplications_Grouped_by_Configuration_label"
plotting_subdirectory = plotting.prepare_plots_directory(
    PLOTTING_DIRECTORY, plots_base_name
)

for (
    configuration_label,
    configuration_label_group,
) in KL_several_configs_varying_n_dataframe.groupby("Configuration_label"):

    metadata_dictionary = {
        "Configuration_label": configuration_label,
    }

    fig, ax = plt.subplots()
    plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    plot_title = ""
    field_unique_values_combined_dictionary = (
        fields_with_single_unique_values_dictionary | metadata_dictionary
    )
    parameter_values_subtitle = plotting.construct_plot_subtitle(
        field_unique_values_combined_dictionary
    )
    wrapper = textwrap.TextWrapper(width=110, initial_indent="   ")
    wrapped_parameter_values_subtitle = wrapper.fill(parameter_values_subtitle)

    ax.set_title(f"{plot_title}\n{wrapped_parameter_values_subtitle}", pad=6)

    plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_yscale("log")

    ax.set(xlabel="Number of MV multiplications", ylabel="$||sgn^2(X) - 1||^2$")

    # Set x-axis ticks to integer values only
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for (
        kernel_operator_type,
        kernel_operator_type_group,
    ) in configuration_label_group.groupby("Kernel_operator_type", sort=False):

        number_of_MV_multiplications_array = kernel_operator_type_group[
            "Number_of_MV_multiplications"
        ].to_numpy()
        Average_calculation_results_array = gv.gvar(
            kernel_operator_type_group["Average_calculation_result"]
            .apply(ast.literal_eval)
            .to_numpy()
        )

        ax.errorbar(
            number_of_MV_multiplications_array,
            gv.mean(Average_calculation_results_array),
            yerr=gv.sdev(Average_calculation_results_array),
            fmt=".",
            markersize=8,
            capsize=10,
            label=f"{kernel_operator_type}",
        )

    ax.legend(loc="upper right")

    plot_path = plotting.generate_plot_path(
        plotting_subdirectory,
        plots_base_name,
        field_unique_values_combined_dictionary,
        metadata_dictionary,
    )
    fig.savefig(plot_path)
    plt.close()
