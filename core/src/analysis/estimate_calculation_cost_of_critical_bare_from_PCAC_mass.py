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
import warnings
from scipy.optimize import OptimizeWarning

from library import (
    filesystem_utilities,
    custom_plotting,
    data_processing,
    fit_functions,
    constants,
    validate_input_directory,
    validate_input_script_log_filename,
)

REFERENCE_BARE_MASS = 0.05
REFERENCE_PCAC_MASS = 0.05

UPPER_BARE_MASS_CUT = 0.06


@click.command()
@click.option(
    "-in_PCAC_csv",
    "--input_PCAC_mass_estimates_csv_file_path",
    "input_PCAC_mass_estimates_csv_file_path",
    required=True,
    callback=filesystem_utilities.validate_input_csv_file,
    help="Path to .csv file containing PCAC mass estimates.",
)
@click.option(
    "-in_jack_hdf5",
    "--input_correlators_jackknife_analysis_hdf5_file_path",
    "input_correlators_jackknife_analysis_hdf5_file_path",
    callback=filesystem_utilities.validate_input_HDF5_file,
    required=True,
    help="Path to input HDF5 file containing extracted correlators values.",
)
@click.option(
    "-out_dir",
    "--output_files_directory",
    "output_files_directory",
    default=None,
    callback=validate_input_directory,
    help="Path to directory where all output files will be stored.",
)
@click.option(
    "-plots_dir",
    "--plots_directory",
    "plots_directory",
    default="../../../output/plots",
    callback=validate_input_directory,
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
    help="Enable plotting cost plots.",
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
    "-out_csv_name",
    "--output_calculation_cost_csv_filename",
    "output_calculation_cost_csv_filename",
    default="calculation_cost_of_critical_bare_mass_from_PCAC_mass.csv",
    help="Specific name for the output .csv files.",
)
@click.option(
    "-log_on",
    "--enable_logging",
    "enable_logging",
    is_flag=True,
    default=False,
    help="Enable logging.",
)
@click.option(
    "-log_file_dir",
    "--log_file_directory",
    "log_file_directory",
    default=None,
    callback=filesystem_utilities.validate_script_log_file_directory,
    help="Directory where the script's log file will be stored.",
)
@click.option(
    "-log_name",
    "--log_filename",
    "log_filename",
    default=None,
    callback=validate_input_script_log_filename,
    help="Specific name for the script's log file.",
)
def main(
    input_PCAC_mass_estimates_csv_file_path,
    input_correlators_jackknife_analysis_hdf5_file_path,
    output_files_directory,
    plots_directory,
    plot_critical_bare_mass,
    plot_calculation_cost,
    annotate_data_points,
    output_calculation_cost_csv_filename,
    enable_logging,
    log_file_directory,
    log_filename,
):
    # HANDLE EMPTY INPUT ARGUMENTS

    # If no output directory is provided, use the directory of the input file
    if output_files_directory is None:
        output_files_directory = os.path.dirname(
            input_PCAC_mass_estimates_csv_file_path
        )

    # INITIATE LOGGING

    # Setup logging
    logger = filesystem_utilities.LoggingWrapper(
        log_file_directory, log_filename, enable_logging
    )

    # Log script start
    logger.initiate_script_logging()

    # CREATE PLOTS SUBDIRECTORIES

    if any([plot_critical_bare_mass, plot_calculation_cost]):
        # Create main plots directory if it does not exist
        plots_main_subdirectory = filesystem_utilities.create_subdirectory(
            plots_directory,
            "Calculation_cost_of_critical_bare_mass",
        )

    # Create deeper-level subdirectories if requested
    if plot_critical_bare_mass:
        critical_bare_mass_plots_base_name = "Critical_bare_mass"
        critical_bare_mass_plots_subdirectory = (
            filesystem_utilities.create_subdirectory(
                plots_main_subdirectory,
                critical_bare_mass_plots_base_name + "_from_PCAC_mass",
                clear_contents=True,
            )
        )
        logger.info("Subdirectory for critical bare mass plots created.")

    if plot_calculation_cost:
        number_of_MV_multiplications_plots_base_name = "Number_MV_multiplications"
        number_of_MV_multiplications_plots_subdirectory = (
            filesystem_utilities.create_subdirectory(
                plots_main_subdirectory,
                number_of_MV_multiplications_plots_base_name + "_per_PCAC_mass",
                clear_contents=True,
            )
        )
        logger.info("Subdirectory for number of MV multiplications plots created.")

        adjusted_core_hours_plots_base_name = "Adjusted_core_hours"
        adjusted_core_hours_plots_subdirectory = (
            filesystem_utilities.create_subdirectory(
                plots_main_subdirectory,
                adjusted_core_hours_plots_base_name + "_per_PCAC_mass",
                clear_contents=True,
            )
        )
        logger.info("Subdirectory for number of MV multiplications plots created.")

        core_hours_Vs_bare_mass_plots_base_name = "Adjusted_core_hours_Vs_bare_mass"
        core_hours_Vs_bare_mass_plots_subdirectory = (
            filesystem_utilities.create_subdirectory(
                plots_main_subdirectory,
                core_hours_Vs_bare_mass_plots_base_name + "_per_PCAC_mass",
                clear_contents=True,
            )
        )
        logger.info("Subdirectory for core-hours Vs bare mass plots plots created.")

    # IMPORT DATASETS AND METADATA

    PCAC_mass_estimates_dataframe = data_processing.load_csv(
        input_PCAC_mass_estimates_csv_file_path
    )

    analyzer = data_processing.DataFrameAnalyzer(PCAC_mass_estimates_dataframe)

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
    for value, group in PCAC_mass_estimates_dataframe.groupby(
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

        PCAC_mass_estimates_array = gv.gvar(group["PCAC_mass_estimate"].to_numpy())

        number_of_gauge_configurations_array = group[
            "Number_of_gauge_configurations"
        ].to_numpy()

        adjusted_average_core_hours_per_spinor_per_configuration_array = group[
            "Adjusted_average_core_hours_per_spinor_per_configuration"
        ].to_numpy()

        average_number_of_MV_multiplications_array = group[
            "Average_number_of_MV_multiplications_per_spinor_per_configuration"
        ].to_numpy()

        # FIT ON PCAC MASS VS BARE MASS DATA

        bare_mass_values_for_fitting_array = bare_mass_values_array
        PCAC_mass_estimates_for_fitting_array = PCAC_mass_estimates_array
        if UPPER_BARE_MASS_CUT is not None:
            condition = bare_mass_values_array < UPPER_BARE_MASS_CUT
            bare_mass_values_for_fitting_array = bare_mass_values_array[condition]
            PCAC_mass_estimates_for_fitting_array = PCAC_mass_estimates_array[condition]

        x = bare_mass_values_for_fitting_array
        y = PCAC_mass_estimates_for_fitting_array

        if len(x) < 3:
            continue

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
        # TODO:
        PCAC_mass_reference_value = fit_functions.linear_function(
            REFERENCE_BARE_MASS, linear_fit.p
        )
        slope, x_intercept = gv.mean(linear_fit.p)
        bare_mass_reference_value = REFERENCE_PCAC_MASS / slope + x_intercept

        # FIT ON # OF MV MULTIPLICATIONS VS BARE MASS DATA

        x = bare_mass_values_array
        y = average_number_of_MV_multiplications_array

        # Check for a minimum amount of data points
        if len(y) < 4:
            # TODO: Log warning
            continue

        # Guess parameters
        a = np.min(y) * 0.9  # Decrease it for safety
        b = np.max(y) - a
        # index = len(y) // 2
        c = (np.log(y[-1] - a) - np.log(y[0] - a)) / (-x[-1] + x[0])
        d = np.min(x)
        shifted_exponential_fit_p0 = [a, b, c, d]

        # Suppress specific warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
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
        number_of_MV_multiplications_reference_value_for_constant_PCAC_mass = (
            fit_functions.shifted_exponential(
                bare_mass_reference_value, *shifted_exponential_coefficients
            )
        )

        # FIT ON # OF MV MULTIPLICATIONS VS BARE MASS DATA

        def power_law(x, a, b):
            return a * x**b

        x = bare_mass_values_array
        y = adjusted_average_core_hours_per_spinor_per_configuration_array

        # Check for a minimum amount of data points
        if len(y) < 4:
            # TODO: Log warning
            continue

        # Suppress specific warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            # Power law fit
            fit_params, _ = curve_fit(power_law, x, y)

        # Calculate corresponding values to the reference levels set by the user
        adjusted_average_core_hours_reference_value_for_constant_bare_mass = (
            power_law(REFERENCE_BARE_MASS, *fit_params)
        )
        adjusted_average_core_hours_reference_value_for_constant_PCAC_mass = (
            power_law(np.abs(bare_mass_reference_value), *fit_params)
        )

        # FIT ON CORE-HOURS VS NUMBER OF MV MULTIPLICATIONS DATA
        x = average_number_of_MV_multiplications_array
        y = adjusted_average_core_hours_per_spinor_per_configuration_array

        # Perform linear fit using numpy's polyfit
        linear_fit_coefficients = np.polyfit(x, y, 1)
        slope, intercept = linear_fit_coefficients
        core_hours_linear_fit = np.poly1d(linear_fit_coefficients)

        # Calculate corresponding values to the reference levels set by the user
        core_hours_reference_value_for_constant_PCAC_mass = (
            slope * number_of_MV_multiplications_reference_value_for_constant_PCAC_mass
            + intercept
        )
        core_hours_reference_value_for_constant_bare_mass = (
            slope * number_of_MV_multiplications_reference_value_for_constant_bare_mass
            + intercept
        )

        # PLOT PCAC MASS VS BARE MASS DATA

        x = bare_mass_values_array
        y = PCAC_mass_estimates_array

        if plot_calculation_cost:

            excluded_title_fields = [
                "Number_of_vectors",
                "Delta_Min",
                "Delta_Max",
                "Lanczos_epsilon",
            ]
            reduced_parameters_value_dictionary = {
                k: v
                for k, v in parameters_value_dictionary.items()
                if k not in excluded_title_fields
            }

            fig, ax = plt.subplots()
            ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

            plot_title = custom_plotting.DataPlotter._construct_plot_title(
                None,
                leading_substring="",
                metadata_dictionary=metadata_dictionary,
                title_width=80,
                fields_unique_value_dictionary=reduced_parameters_value_dictionary,
            )
            ax.set_title(f"{plot_title}", pad=8)

            x_axis_label = constants.AXES_LABELS_DICTIONARY["Bare_mass"]
            y_axis_label = constants.AXES_LABELS_DICTIONARY["PCAC_mass_estimate"]

            ax.set(xlabel=x_axis_label, ylabel=y_axis_label)

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

            # # Plot reference levels for constant bare mass
            # ax.axvline(REFERENCE_BARE_MASS, color="blue", linestyle="--")
            # ax.axhline(
            #     PCAC_mass_reference_value.mean,
            #     color="brown",
            #     linestyle="--",
            #     label=(f"{y_axis_label} for {x_axis_label}={REFERENCE_BARE_MASS:.2f}"),
            # )

            # Plot reference levels for constant PCAC mass
            ax.axhline(REFERENCE_PCAC_MASS, color="brown", linestyle="--")
            ax.axvline(
                bare_mass_reference_value,
                color="blue",
                linestyle="--",
                label=(f"{x_axis_label} for {y_axis_label}={REFERENCE_PCAC_MASS:.2f}"),
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
                title_width=80,
                fields_unique_value_dictionary=reduced_parameters_value_dictionary,
                excluded_title_fields=[
                    "Number_of_vectors",
                    "Delta_Min",
                    "Delta_Max",
                    "Lanczos_epsilon",
                ],
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

            # # Plot reference levels for constant bare mass
            # ax.axvline(REFERENCE_BARE_MASS, color="blue", linestyle="--")
            # ax.axhline(
            #     number_of_MV_multiplications_reference_value_for_constant_bare_mass,
            #     color="brown",
            #     linestyle="--",
            #     label=f"# of MV muls. for const. {x_axis_label}={REFERENCE_BARE_MASS:.2f}",
            # )

            # Plot reference levels for constant PCAC mass
            ax.axvline(bare_mass_reference_value, color="blue", linestyle="--")
            ax.axhline(
                number_of_MV_multiplications_reference_value_for_constant_PCAC_mass,
                color="brown",
                linestyle="--",
                label=f"# of MV muls. for const. {y_axis_label}={REFERENCE_PCAC_MASS:.2f}",
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

            # PLOT CORE-HOURS VS NUMBER OF MV MULTIPLICATIONS

            x = average_number_of_MV_multiplications_array
            y = adjusted_average_core_hours_per_spinor_per_configuration_array

            fig, ax = plt.subplots()
            ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

            plot_title = custom_plotting.DataPlotter._construct_plot_title(
                None,
                leading_substring="",
                metadata_dictionary=metadata_dictionary,
                title_width=80,
                fields_unique_value_dictionary=reduced_parameters_value_dictionary,
            )
            ax.set_title(f"{plot_title}", pad=8)

            ax.set(
                xlabel=constants.AXES_LABELS_DICTIONARY[
                    "Average_number_of_MV_multiplications_per_spinor_per_configuration"
                ],
                ylabel=constants.AXES_LABELS_DICTIONARY[
                    "Adjusted_average_core_hours_per_spinor_per_configuration"
                ],
            )
            # fig.subplots_adjust(left=0.14)

            ax.axhline(0, color="brown")  # y = 0
            ax.axvline(0, color="brown")  # x = 0

            ax.scatter(x, y, marker="x")
            # s=8,

            # Plot shifted exponential fit
            x_data = np.linspace(np.min(x), np.max(x), 100)
            y_data = core_hours_linear_fit(x_data)
            ax.plot(
                x_data,
                y_data,
                color="red",
                linestyle="--",
                # linewidth=2,
                # label="Fit",
            )

            # Plot reference levels for constant PCAC mass
            ax.axvline(
                number_of_MV_multiplications_reference_value_for_constant_PCAC_mass,
                color="blue",
                linestyle="--",
            )
            ax.axhline(
                core_hours_reference_value_for_constant_PCAC_mass,
                color="green",
                linestyle="--",
                label=(
                    "core-hours for const. PCAC mass = "
                    f"{core_hours_reference_value_for_constant_PCAC_mass:.2f}"
                ),
            )
            # Plot reference levels for constant bare mass
            ax.axvline(
                number_of_MV_multiplications_reference_value_for_constant_bare_mass,
                color="blue",
                linestyle="--",
            )
            ax.axhline(
                core_hours_reference_value_for_constant_bare_mass,
                color="green",
                linestyle="--",
                label=(
                    "core-hours for const. bare mass = "
                    f"{core_hours_reference_value_for_constant_bare_mass:.2f}"
                ),
            )

            ax.legend(loc="upper right")

            current_plots_base_name = adjusted_core_hours_plots_base_name
            plot_path = custom_plotting.DataPlotter._generate_plot_path(
                None,
                adjusted_core_hours_plots_subdirectory,
                current_plots_base_name,
                metadata_dictionary,
                single_valued_fields_dictionary=single_valued_fields_dictionary,
            )

            fig.savefig(plot_path)
            plt.close()

##########################################################################################
            # PLOT CORE-HOURS VS BARE MASS

            x = bare_mass_values_array
            y = adjusted_average_core_hours_per_spinor_per_configuration_array

            fig, ax = plt.subplots()
            ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

            plot_title = custom_plotting.DataPlotter._construct_plot_title(
                None,
                leading_substring="",
                metadata_dictionary=metadata_dictionary,
                title_width=80,
                fields_unique_value_dictionary=reduced_parameters_value_dictionary,
            )
            ax.set_title(f"{plot_title}", pad=8)

            ax.set(
                xlabel=constants.AXES_LABELS_DICTIONARY["Bare_mass"],
                ylabel=constants.AXES_LABELS_DICTIONARY[
                    "Adjusted_average_core_hours_per_spinor_per_configuration"
                    ],
            )
            # fig.subplots_adjust(left=0.14)

            ax.axhline(0, color="brown")  # y = 0
            ax.axvline(0, color="brown")  # x = 0

            ax.scatter(x, y, marker="x")
            # s=8,

            # Plot power law fit
            x_data = np.linspace(np.min(x), np.max(x), 100)
            y_data = power_law(x_data, *fit_params)
            ax.plot(
                x_data,
                y_data,
                color="red",
                linestyle="--",
                # linewidth=2,
                # label="Fit",
            )

            # Plot reference levels for constant PCAC mass
            ax.axvline(
                bare_mass_reference_value,
                color="blue",
                linestyle="--",
            )
            ax.axhline(
                adjusted_average_core_hours_reference_value_for_constant_PCAC_mass,
                color="green",
                linestyle="--",
                label=(
                    "core-hours for const. PCAC mass = "
                    f"{adjusted_average_core_hours_reference_value_for_constant_PCAC_mass:.2f}"
                ),
            )
            # Plot reference levels for constant bare mass
            ax.axvline(
                REFERENCE_BARE_MASS,
                color="blue",
                linestyle="--",
            )
            ax.axhline(
                adjusted_average_core_hours_reference_value_for_constant_bare_mass,
                color="green",
                linestyle="--",
                label=(
                    "core-hours for const. bare mass = "
                    f"{adjusted_average_core_hours_reference_value_for_constant_bare_mass:.2f}"
                ),
            )

            ax.legend(loc="upper right")

            current_plots_base_name = core_hours_Vs_bare_mass_plots_base_name
            plot_path = custom_plotting.DataPlotter._generate_plot_path(
                None,
                core_hours_Vs_bare_mass_plots_subdirectory,
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
                "Number_of_MV_multiplications_for_constant_bare_mass"
            ] = number_of_MV_multiplications_reference_value_for_constant_bare_mass

            parameters_value_dictionary[
                "Number_of_MV_multiplications_for_constant_PCAC_mass"
            ] = number_of_MV_multiplications_reference_value_for_constant_PCAC_mass

            parameters_value_dictionary["Core_hours_for_constant_bare_mass"] = (
                adjusted_average_core_hours_reference_value_for_constant_bare_mass
                # core_hours_reference_value_for_constant_bare_mass
            )

            parameters_value_dictionary["Core_hours_for_constant_PCAC_mass"] = (
                adjusted_average_core_hours_reference_value_for_constant_PCAC_mass
                # core_hours_reference_value_for_constant_PCAC_mass
            )

        critical_bare_mass_values_list.append(parameters_value_dictionary)

    # Check if list is empty before exporting
    if not critical_bare_mass_values_list:
        logger.warning(
            "Critical bare mass values calculation produced no results.",
            to_console=True,
        )
        sys.exit(1)

    # Create a DataFrame from the extracted data
    critical_bare_mass_dataframe = pd.DataFrame(critical_bare_mass_values_list)

    # Construct output .csv file path
    csv_file_full_path = os.path.join(
        output_files_directory, output_calculation_cost_csv_filename
    )

    # Export the DataFrame to a CSV file
    critical_bare_mass_dataframe.to_csv(csv_file_full_path, index=False)

    click.echo("   -- Calculation cost estimation from PCAC mass estimates completed.")

    # Terminate logging
    logger.terminate_script_logging()


if __name__ == "__main__":
    main()
