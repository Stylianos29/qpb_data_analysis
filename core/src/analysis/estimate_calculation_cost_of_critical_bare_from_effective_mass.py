# TODO: Write a detailed introductory commentary
"""
post_processing_analysis/process_qpb_log_files_extracted_values.py

Summary:

Input:

Output:

Functionality:

Usage:
"""

import os
import sys

import click
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
import lsqfit
import logging
import pandas as pd
import copy
from scipy.optimize import curve_fit

from library import (
    filesystem_utilities,
    custom_plotting,
    data_processing,
    fit_functions,
    constants,
    validate_input_directory,
    validate_input_script_log_filename,
)

REFERENCE_BARE_MASS = 0.1
REFERENCE_EFFECTIVE_MASS = 0.15


@click.command()
@click.option(
    "--input_pion_effective_mass_estimates_csv_file_path",
    "input_pion_effective_mass_estimates_csv_file_path",
    "-eff_csv",
    # default="/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/invert/KL_several_config_varying_n/Pion_effective_mass_estimates.csv",
    required=True,
    help="Path to .csv file containing pion effective mass estimates.",
)
@click.option(
    "--output_files_directory",
    "output_files_directory",
    "-out_dir",
    default=None,
    help="Path to directory where all output files will be stored.",
)
@click.option(
    "--plots_directory",
    "plots_directory",
    "-plots_dir",
    default="../../../output/plots",
    # default="/nvme/h/cy22sg1/qpb_data_analysis/output/plots/invert/KL_several_config_varying_n",
    help="Path to the output directory for storing plots.",
)
@click.option(
    "-plot_critical",
    "--plot_critical_bare_mass",
    "plot_critical_bare_mass",
    is_flag=True,
    default=False,
    help="Enable plotting critical bare mass.",
)
@click.option(
    "-plot_cost",
    "--plot_calculation_cost",
    "plot_calculation_cost",
    is_flag=True,
    default=False,
    help="Enable plotting critical bare mass.",
)
@click.option(
    "-annotate",
    "--annotate_data_points",
    "annotate_data_points",
    is_flag=True,
    default=False,
    help="Enable annotating the data points.",
)
@click.option(
    "--output_calculation_cost_csv_filename",
    "output_calculation_cost_csv_filename",
    "-out_csv",
    default="calculation_cost_of_critical_bare_mass_from_effective_mass.csv",
    help="Specific name for the output .csv files.",
)
@click.option(
    "--log_file_directory",
    "log_file_directory",
    "-log_file_dir",
    default=None,
    help="Directory where the script's log file will be stored.",
)
@click.option(
    "--log_filename",
    "log_filename",
    "-log",
    default=None,
    help="Specific name for the script's log file.",
)
def main(
    input_pion_effective_mass_estimates_csv_file_path,
    output_files_directory,
    plots_directory,
    plot_critical_bare_mass,
    plot_calculation_cost,
    annotate_data_points,
    output_calculation_cost_csv_filename,
    log_file_directory,
    log_filename,
):
    # VALIDATE INPUT ARGUMENTS

    if not filesystem_utilities.is_valid_file(
        input_pion_effective_mass_estimates_csv_file_path
    ):
        error_message = "Passed correlator values HDF5 file path is invalid."
        print("ERROR:", error_message)
        sys.exit(1)

    # If no output directory is provided, use the directory of the input file
    if output_files_directory is None:
        output_files_directory = os.path.dirname(
            input_pion_effective_mass_estimates_csv_file_path
        )
    # Check validity if the provided
    elif not filesystem_utilities.is_valid_file(output_files_directory):
        error_message = (
            "Passed output files directory path is invalid " "or not a directory."
        )
        print("ERROR:", error_message)
        print("Exiting...")
        sys.exit(1)

    if not filesystem_utilities.is_valid_directory(plots_directory):
        error_message = "The specified plots directory path is invalid."
        print("ERROR:", error_message)
        sys.exit(1)

    # Specify current script's log file directory
    if log_file_directory is None:
        log_file_directory = output_files_directory
    elif not filesystem_utilities.is_valid_directory(log_file_directory):
        error_message = (
            "Passed directory path to store script's log file is "
            "invalid or not a directory."
        )
        print("ERROR:", error_message)
        print("Exiting...")
        sys.exit(1)

    # Get the script's filename
    script_name = os.path.basename(__file__)

    if log_filename is None:
        log_filename = script_name.replace(".py", "_python_script.log")

    # Check for proper extensions in provided output filenames
    if not output_calculation_cost_csv_filename.endswith(".csv"):
        output_calculation_cost_csv_filename = (
            output_calculation_cost_csv_filename + ".csv"
        )
    if not log_filename.endswith(".log"):
        log_filename = log_filename + ".log"

    # INITIATE LOGGING

    filesystem_utilities.setup_logging(log_file_directory, log_filename)

    # # Create a logger instance for the current script using the script's name.
    # logger = logging.getLogger(__name__)

    # Get the script's filename
    script_name = os.path.basename(__file__)

    # Initiate logging
    logging.info(f"Script '{script_name}' execution initiated.")

    # CREATE PLOTS SUBDIRECTORIES

    if plot_critical_bare_mass or plot_calculation_cost:
        # Create main plots directory if it does not exist
        plots_main_subdirectory = filesystem_utilities.create_subdirectory(
            plots_directory,
            "calculation_cost_of_critical_bare_mass",
        )

    if plot_critical_bare_mass:
        # Create deeper-level subdirectories if requested
        critical_bare_mass_plots_base_name = "critical_bare_mass"
        critical_bare_mass_plots_subdirectory = (
            filesystem_utilities.create_subdirectory(
                plots_main_subdirectory,
                critical_bare_mass_plots_base_name + "_from_pion_effective_mass",
                clear_contents=True,
            )
        )

    if plot_calculation_cost:
        # Create deeper-level subdirectories if requested
        number_of_MV_multiplications_plots_base_name = "number_MV_multiplications"
        number_of_MV_multiplications_plots_subdirectory = (
            filesystem_utilities.create_subdirectory(
                plots_main_subdirectory,
                number_of_MV_multiplications_plots_base_name + "_per_effective_mass",
                clear_contents=True,
            )
        )

    # IMPORT DATASETS AND METADATA

    effective_mass_estimates_dataframe = data_processing.load_csv(
        input_pion_effective_mass_estimates_csv_file_path
    )

    analyzer = data_processing.DataFrameAnalyzer(effective_mass_estimates_dataframe)

    single_valued_fields_dictionary = analyzer.unique_value_columns_dictionary

    tunable_multivalued_parameters_list = (
        analyzer.list_of_multivalued_tunable_parameter_names
    )

    # TODO: Rethink this strategy of excluding "MPI_geometry" manually
    tunable_multivalued_parameters_list = [
        item for item in tunable_multivalued_parameters_list if item != "MPI_geometry"
    ]

    # Remove "Bare_mass" from the list of tunable multivalued parameters
    if "Bare_mass" not in tunable_multivalued_parameters_list:
        error_message = (
            "Critical bare mass analysis cannot be performed without a range "
            "of bare mass values data."
        )
        print("ERROR:", error_message)
        sys.exit(1)
    tunable_multivalued_parameters_list.remove("Bare_mass")

    critical_bare_mass_values_list = []
    for value, group in effective_mass_estimates_dataframe.groupby(
        tunable_multivalued_parameters_list,
        observed=True,
    ):
        # Check for a minimum amount of data points
        if group["Bare_mass"].nunique() < 3:
            # TODO: Log warning
            continue

        # Initialize the parameters values dictionary
        parameters_value_dictionary = copy.deepcopy(single_valued_fields_dictionary)

        # Store for later use
        if not isinstance(value, tuple):
            value = [value]
        metadata_dictionary = dict(zip(tunable_multivalued_parameters_list, value))

        # Append metadata dictionary
        parameters_value_dictionary.update(metadata_dictionary)

        bare_mass_values_array = group["Bare_mass"].to_numpy()

        effective_mass_estimates_array = gv.gvar(
            group["Pion_effective_mass_estimate"].to_numpy()
        )

        number_of_gauge_configurations_array = group[
            "Number_of_gauge_configurations"
        ].to_numpy()

        # average_calculation_time_array = group[
        #     "Average_calculation_time_per_spinor_per_configuration"
        # ].to_numpy()

        average_number_of_MV_multiplications_array = group[
            "Average_number_of_MV_multiplications_per_spinor_per_configuration"
        ].to_numpy()

        # FIT ON PION EFFECTIVE MASS SQUARED VS BARE MASS DATA

        x = bare_mass_values_array
        y = np.square(effective_mass_estimates_array)

        # Guess parameters
        min_index = np.argmin(x)
        max_index = np.argmax(x)

        x_min = np.min(x)
        y_min = y[min_index]
        x_max = np.max(x)
        y_max = y[max_index]

        slope = (y_max - y_min) / (x_max - x_min)
        x_intercept = y_min / slope + x_min
        linear_fit_p0 = gv.mean([slope, x_intercept])

        # Linear fit
        linear_fit = lsqfit.nonlinear_fit(
            data=(x, y),
            p0=linear_fit_p0,
            fcn=fit_functions.linear_function,
            debug=True,
        )

        # Calculate critical bare mass
        critical_bare_mass_value = linear_fit.p[1]

        # Calculate corresponding values to the reference levels set by the user
        effective_mass_reference_value = fit_functions.linear_function(
            REFERENCE_BARE_MASS, linear_fit.p
        )
        slope, x_intercept = gv.mean(linear_fit.p)
        bare_mass_reference_value = REFERENCE_EFFECTIVE_MASS / slope + x_intercept

        # FIT ON # OF MV MULTIPLICATIONS VS BARE MASS DATA

        x = bare_mass_values_array
        y = average_number_of_MV_multiplications_array

        # Check for a minimum amount of data points
        if len(y) < 4:
            # TODO: Log warning
            continue

        # Guess parameters
        a = np.min(y)
        b = np.max(y) - np.min(y)
        index = len(y) // 2
        c = (np.log(y[index] - a) - np.log(y[0] - a)) / (-x[index] + x[0])
        d = np.min(x)
        shifted_exponential_fit_p0 = [a, b, c, d]

        # Shifted exponential fit
        shifted_exponential_coefficients, _ = curve_fit(
            fit_functions.shifted_exponential, x, y, p0=shifted_exponential_fit_p0
        )

        # Calculate corresponding values to the reference levels set by the user
        number_of_MV_multiplications_reference_value_for_constant_bare_mass = (
            fit_functions.shifted_exponential(
                REFERENCE_BARE_MASS, *shifted_exponential_coefficients
            )
        )
        number_of_MV_multiplications_reference_value_for_constant_effective_mass = (
            fit_functions.shifted_exponential(
                bare_mass_reference_value, *shifted_exponential_coefficients
            )
        )

        # PLOT EFFECTIVE MASS VS BARE MASS DATA

        x = bare_mass_values_array
        y = np.square(effective_mass_estimates_array)

        if plot_calculation_cost:
            fig, ax = plt.subplots()
            ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

            plot_title = custom_plotting.DataPlotter._construct_plot_title(
                None,
                leading_substring="",
                metadata_dictionary=metadata_dictionary,
                title_width=100,
                fields_unique_value_dictionary=parameters_value_dictionary,
            )
            ax.set_title(f"{plot_title}", pad=8)

            x_axis_label = constants.AXES_LABELS_DICTIONARY["Bare_mass"]
            y_axis_label = constants.AXES_LABELS_DICTIONARY[
                "Pion_effective_mass_estimate"
            ]

            ax.set(xlabel=x_axis_label, ylabel=y_axis_label + "$^2$")

            ax.axvline(0, color="black")  # x = 0
            ax.axhline(0, color="black")  # y = 0

            ax.errorbar(
                x, gv.mean(y), yerr=gv.sdev(y), fmt=".", markersize=8, capsize=10
            )

            # Plot linear fit
            if gv.mean(critical_bare_mass_value) > 0:
                margin = 0.06
            else:
                margin = -0.06
            x_data = np.linspace(
                gv.mean(critical_bare_mass_value) * (1 - margin),
                max(gv.mean(x)) * (1 + np.abs(margin)),
                100,
            )
            y_data = fit_functions.linear_function(x_data, linear_fit.p)
            label_string = (
                f"Linear fit:\n"
                f"- $\\chi^2$/dof={linear_fit.chi2:.2f}/{linear_fit.dof}="
                f"{linear_fit.chi2/linear_fit.dof:.4f}\n"
                f"- a$m^{{critical}}_{{bare}}$={critical_bare_mass_value:.5f}"
            )
            plt.plot(
                x_data,
                gv.mean(y_data),
                "r--",
                label=label_string,
            )
            ax.fill_between(
                x_data,
                gv.mean(y_data) - gv.sdev(critical_bare_mass_value),
                gv.mean(y_data) + gv.sdev(critical_bare_mass_value),
                color="r",
                alpha=0.2,
            )

            # Plot reference levels for constant bare mass
            ax.axvline(REFERENCE_BARE_MASS, color="blue", linestyle="--")
            ax.axhline(
                effective_mass_reference_value.mean,
                color="brown",
                linestyle="--",
                label=(f"{y_axis_label} for {x_axis_label}={REFERENCE_BARE_MASS:.2f}"),
            )

            # Plot reference levels for constant PCAC mass
            ax.axhline(REFERENCE_EFFECTIVE_MASS, color="brown", linestyle="--")
            ax.axvline(
                bare_mass_reference_value,
                color="blue",
                linestyle="--",
                label=(
                    f"{x_axis_label} for {y_axis_label}={REFERENCE_EFFECTIVE_MASS:.2f}"
                ),
            )

            ax.legend(loc="lower right")

            if annotate_data_points:
                for index, sample_size in enumerate(
                    number_of_gauge_configurations_array
                ):
                    ax.annotate(
                        f"{sample_size}",
                        (x[index], gv.mean(y[index])),
                        xytext=(-40, 10),
                        textcoords="offset pixels",
                        bbox=dict(facecolor="none", edgecolor="black"),
                        arrowprops=dict(arrowstyle="->"),
                        # connectionstyle="arc,angleA=0,armA=50,rad=10")
                    )

            current_plots_base_name = critical_bare_mass_plots_base_name
            plot_path = custom_plotting.DataPlotter._generate_plot_path(
                None,
                critical_bare_mass_plots_subdirectory,
                current_plots_base_name,
                metadata_dictionary,
                single_valued_fields_dictionary=single_valued_fields_dictionary,
            )

            fig.savefig(plot_path)
            plt.close()

            # PLOT NUMBER OF MV MULTIPLICATIONS VS BARE MASS DATA

            x = bare_mass_values_array
            y = average_number_of_MV_multiplications_array

            fig, ax = plt.subplots()
            ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

            plot_title = custom_plotting.DataPlotter._construct_plot_title(
                None,
                leading_substring="",
                metadata_dictionary=metadata_dictionary,
                title_width=100,
                fields_unique_value_dictionary=parameters_value_dictionary,
            )
            ax.set_title(f"{plot_title}", pad=8)

            ax.set(
                xlabel=constants.AXES_LABELS_DICTIONARY["Bare_mass"],
                ylabel=constants.AXES_LABELS_DICTIONARY[
                    "Average_number_of_MV_multiplications_per_spinor_per_configuration"
                ],
            )
            fig.subplots_adjust(left=0.14)

            ax.axvline(0, color="black")  # x = 0

            ax.scatter(x, y, marker="x")
            # s=8,

            # Plot shifted exponential fit
            x_data = np.linspace(np.min(x), np.max(x), 100)
            ax.plot(
                x_data,
                fit_functions.shifted_exponential(
                    x_data, *shifted_exponential_coefficients
                ),
                color="red",
                linestyle="--",
                # linewidth=2,
                # label="Fit",
            )

            # Plot reference levels for constant bare mass
            ax.axvline(REFERENCE_BARE_MASS, color="blue", linestyle="--")
            ax.axhline(
                number_of_MV_multiplications_reference_value_for_constant_bare_mass,
                color="brown",
                linestyle="--",
                label=f"# of MV muls. for const. {x_axis_label}={REFERENCE_BARE_MASS:.2f}",
            )

            # Plot reference levels for constant effective mass
            ax.axvline(bare_mass_reference_value, color="blue", linestyle="--")
            ax.axhline(
                number_of_MV_multiplications_reference_value_for_constant_effective_mass,
                color="brown",
                linestyle="--",
                label=f"# of MV muls. for const. {y_axis_label}={REFERENCE_EFFECTIVE_MASS:.2f}",
            )

            ax.legend(loc="upper right")

            current_plots_base_name = number_of_MV_multiplications_plots_base_name
            plot_path = custom_plotting.DataPlotter._generate_plot_path(
                None,
                number_of_MV_multiplications_plots_subdirectory,
                current_plots_base_name,
                metadata_dictionary,
                single_valued_fields_dictionary=single_valued_fields_dictionary,
            )

            fig.savefig(plot_path)
            plt.close()

            # EXPORT CALCULATED DATA

            parameters_value_dictionary["Critical_bare_mass"] = (
                critical_bare_mass_value.mean,
                critical_bare_mass_value.sdev,
            )

            parameters_value_dictionary[
                "number_of_MV_multiplications_for_constant_bare_mass"
            ] = number_of_MV_multiplications_reference_value_for_constant_bare_mass

            parameters_value_dictionary[
                "number_of_MV_multiplications_for_constant_effective_mass"
            ] = number_of_MV_multiplications_reference_value_for_constant_effective_mass

        critical_bare_mass_values_list.append(parameters_value_dictionary)

    # Create a DataFrame from the extracted data
    critical_bare_mass_dataframe = pd.DataFrame(critical_bare_mass_values_list)

    # Construct output .csv file path
    csv_file_full_path = os.path.join(
        output_files_directory, output_calculation_cost_csv_filename
    )

    # Export the DataFrame to a CSV file
    critical_bare_mass_dataframe.to_csv(csv_file_full_path, index=False)

    # Terminate logging
    logging.info(f"Script '{script_name}' execution terminated successfully.")

    click.echo(
        "   -- Calculation cost estimation from effective mass estimates completed."
    )


if __name__ == "__main__":
    main()
