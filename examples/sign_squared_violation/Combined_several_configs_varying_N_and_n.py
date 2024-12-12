import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import ast
import textwrap
import gvar as gv

from library import data_processing, plotting


# CHEBYSHEV

WORKSPACE_DIRECTORY_FULL_PATH = "/nvme/h/cy22sg1/qpb_data_analysis"
CSV_FILE_FULL_PATH = os.path.join(
    WORKSPACE_DIRECTORY_FULL_PATH,
    "data_files/processed/sign_squared_violation/Chebyshev_several_configs_varying_N/qpb_log_files_single_valued_parameters.csv",
)
PLOTTING_DIRECTORY = os.path.join(
    WORKSPACE_DIRECTORY_FULL_PATH,
    "output/plots/sign_squared_violation/Combined_several_configs_varying_N",
)
os.makedirs(PLOTTING_DIRECTORY, exist_ok=True)

# Check if given .csv file full path is valid
if not (os.path.exists(CSV_FILE_FULL_PATH) and os.path.isfile(CSV_FILE_FULL_PATH)):
    print("Invalid .csv file directory.\nExiting")
    sys.exit()

# Check if given plots directories is valid
if not (os.path.exists(PLOTTING_DIRECTORY) and os.path.isdir(PLOTTING_DIRECTORY)):
    print("Invalid plots directory.\nExiting...")
    sys.exit()

# Import .csv file
# Chebyshev_several_configs_varying_N_dataframe = pd.read_csv(CSV_FILE_FULL_PATH)
Chebyshev_several_configs_varying_N_dataframe = data_processing.load_csv(
    CSV_FILE_FULL_PATH
)

# Calculate number of MV multiplications
Chebyshev_several_configs_varying_N_dataframe["Number_of_MV_multiplications"] = 2 * (
    Chebyshev_several_configs_varying_N_dataframe["Total_number_of_Lanczos_iterations"]
    + 1
) + 2 * Chebyshev_several_configs_varying_N_dataframe["Number_of_Chebyshev_terms"] - 1

Chebyshev_fields_with_single_unique_values_dictionary = data_processing.get_fields_with_unique_values(Chebyshev_several_configs_varying_N_dataframe)
Chebyshev_fields_with_single_unique_values_dictionary.pop("Bare_mass")
Chebyshev_fields_with_single_unique_values_dictionary.pop("Overlap_operator_method")

# KL

WORKSPACE_DIRECTORY_FULL_PATH = "/nvme/h/cy22sg1/qpb_data_analysis"
CSV_FILE_FULL_PATH = os.path.join(
    WORKSPACE_DIRECTORY_FULL_PATH,
    "data_files/processed/sign_squared_violation/KL_several_configs_varying_n/qpb_log_files_single_valued_parameters.csv",
)

# Check if given .csv file full path is valid
if not (os.path.exists(CSV_FILE_FULL_PATH) and os.path.isfile(CSV_FILE_FULL_PATH)):
    print("Invalid .csv file directory.\nExiting")
    sys.exit()

# Import .csv file
# KL_several_configs_varying_n_dataframe = pd.read_csv(CSV_FILE_FULL_PATH)
KL_several_configs_varying_n_dataframe = data_processing.load_csv(CSV_FILE_FULL_PATH)

# Calculate number of MV multiplications
KL_several_configs_varying_n_dataframe["Number_of_MV_multiplications"] = (
    2 * KL_several_configs_varying_n_dataframe["Average_number_of_MSCG_iterations"]
    + 1
)

KL_fields_with_single_unique_values_dictionary = data_processing.get_fields_with_unique_values(KL_several_configs_varying_n_dataframe)
KL_fields_with_single_unique_values_dictionary.pop("Bare_mass")
KL_fields_with_single_unique_values_dictionary.pop("Overlap_operator_method")

# PLOTS

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

        metadata_dictionary = {"Configuration_label": configuration_label}

        fig, ax = plt.subplots()
        plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

        plot_title = "Combined "
        field_unique_values_combined_dictionary = (
            KL_fields_with_single_unique_values_dictionary 
            | Chebyshev_fields_with_single_unique_values_dictionary 
            | metadata_dictionary
        )
        parameter_values_subtitle = plotting.construct_plot_subtitle(
            field_unique_values_combined_dictionary
        )
        wrapper = textwrap.TextWrapper(width=110, initial_indent="   ")
        wrapped_parameter_values_subtitle = wrapper.fill(plot_title + parameter_values_subtitle)

        ax.set_title(f"{wrapped_parameter_values_subtitle}", pad=6)

        plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

        ax.set_yscale("log")

        ax.set(xlabel="Number of MV multiplications", ylabel="$||sgn^2(X) - 1||^2$")

        # Set x-axis ticks to integer values only
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # KL

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
            label='KL'
        )

        # Chebyshev

        filtered_dataframe = Chebyshev_several_configs_varying_N_dataframe[
            (Chebyshev_several_configs_varying_N_dataframe["Kernel_operator_type"] == kernel_operator_type) & 
            (Chebyshev_several_configs_varying_N_dataframe["Configuration_label"] == configuration_label)
        ]

        number_of_MV_multiplications_array = filtered_dataframe[
            "Number_of_MV_multiplications"
        ].to_numpy()
        Average_calculation_results_array = gv.gvar(
            filtered_dataframe["Average_calculation_result"]
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
            label='Chebyshev'
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
