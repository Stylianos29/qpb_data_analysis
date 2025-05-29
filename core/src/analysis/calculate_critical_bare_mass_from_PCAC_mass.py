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
import itertools

import click
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
import lsqfit
import logging
import pandas as pd
import h5py
import copy
from collections import Counter

from library import (
    filesystem_utilities,
    custom_plotting,
    data_processing,
    fit_functions,
    PROCESSED_DATA_FILES_DIRECTORY,
    validate_input_directory,
    validate_input_script_log_filename,
)

UPPER_BARE_MASS_CUT = 0.15


@click.command()
@click.option(
    "-in_PCAC_csv",
    "--input_PCAC_mass_estimates_csv_file_path",
    "input_PCAC_mass_estimates_csv_file_path",
    required=True,
    callback=filesystem_utilities.validate_input_csv_file,
    help="Path to .csv file containing calculated PCAC mass estimates.",
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
    "-apply_fits",
    "--fit_for_critical_bare_mass",
    "fit_for_critical_bare_mass",
    is_flag=True,
    default=False,
    help="Enable performing fits for the calculation of critical bare mass.",
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
    "--output_critical_bare_mass_csv_filename",
    "output_critical_bare_mass_csv_filename",
    default="critical_bare_mass_from_PCAC_mass_estimates.csv",
    callback=filesystem_utilities.validate_output_csv_filename,
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
    fit_for_critical_bare_mass,
    annotate_data_points,
    output_critical_bare_mass_csv_filename,
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

    if plot_critical_bare_mass:
        # Create main plots directory if it does not exist
        plots_main_subdirectory = filesystem_utilities.create_subdirectory(
            plots_directory,
            "Critical_bare_mass_calculation",
        )

        # Create deeper-level subdirectories if requested
        critical_bare_mass_plots_base_name = "Critical_bare_mass"
        critical_bare_mass_plots_subdirectory = (
            filesystem_utilities.create_subdirectory(
                plots_main_subdirectory,
                critical_bare_mass_plots_base_name + "_from_PCAC_mass",
                clear_contents=True,
            )
        )
        logger.info("Subdirectory for critical bare mass plots created.")

    # IMPORT DATASETS AND METADATA

    # Open the input HDF5 file for reading and the output HDF5 file for writing
    with h5py.File(
        input_correlators_jackknife_analysis_hdf5_file_path, "r"
    ) as hdf5_file_read:

        # Construct the path to the processed data files set directory
        processed_data_files_set_directory = os.path.dirname(
            input_correlators_jackknife_analysis_hdf5_file_path
        )
        # The top HDF5 file groups (for both HDF5 files) mirror the directory
        # structure of the data files set directory itself and its parent
        # directories relative to the 'PROCESSED_DATA_FILES_DIRECTORY' directory
        input_data_files_set_group = filesystem_utilities.get_hdf5_target_group(
            hdf5_file_read,
            PROCESSED_DATA_FILES_DIRECTORY,
            processed_data_files_set_directory,
            logger=None,
        )
        logger.info("Top groups of the input HDF5 file identified.")

        # Ensure if input HDF5 file contains any data before initiating analysis
        if not filesystem_utilities.has_subgroups(input_data_files_set_group):
            logger.critical(
                "Input HDF5 file does not contain any data to analyze!", to_console=True
            )
            sys.exit(1)

        # Analyze input PCAC mass estimates .csv file
        PCAC_mass_estimates_dataframe = data_processing.load_csv(
            input_PCAC_mass_estimates_csv_file_path
        )

        analyzer = data_processing.DataFrameAnalyzer(PCAC_mass_estimates_dataframe)

        # Store names and values of tunable parameters with unique values
        single_valued_fields_dictionary = copy.deepcopy(
            analyzer.unique_value_columns_dictionary
        )

        # CONSTRUCT LIST OF RELEVANT MULTIVALUED TUNABLE PARAMETERS FOR GROUPING

        # Groupings will be based on tunable parameters with more than one
        # unique values (multivalued)
        tunable_multivalued_parameter_names_list = copy.deepcopy(
            analyzer.list_of_multivalued_tunable_parameter_names
        )

        # Exclude "MPI_geometry" from tunable multivalued parameters list
        # NOTE: The "MPI_geometry" parameter affects only computation speed,
        # not the results. Excluding it ensures proper grouping based on all
        # relevant tunable multivalued parameters.
        tunable_multivalued_parameter_names_list = [
            parameter_name
            for parameter_name in tunable_multivalued_parameter_names_list
            if parameter_name != "MPI_geometry"
        ]

        # Exclude "Bare_mass" from tunable multivalued parameters list since the
        # final grouping must be based on "Bare_mass" values.
        tunable_multivalued_parameter_names_list = [
            parameter_name
            for parameter_name in tunable_multivalued_parameter_names_list
            if parameter_name != "Bare_mass"
        ]

        # LOOP OVER ALL RELEVANT TUNABLE PARAMETERS GROUPINGS

        # Initialize list with parameter values for the output dataframe
        critical_bare_mass_values_list = []

        # Include counting the iterations for later use
        for critical_bare_mass_calculation_index, (
            combination_of_values,
            dataframe_group,
        ) in enumerate(
            PCAC_mass_estimates_dataframe.groupby(
                tunable_multivalued_parameter_names_list,
                observed=True,
            )
        ):
            # Define a unique name for each grouping as a separate calculation
            critical_bare_mass_calculation_group_name = (
                f"CRITICAL_BARE_MASS_"
                f"CALCULATION_{critical_bare_mass_calculation_index}"
            )

            # STORE PARAMETER VALUES AND DATASETS AND VALIDATE DATA

            # Store specific tunable multivalued parameter names and values in a
            # dedicated metadata dictionary for later use
            if not isinstance(combination_of_values, tuple):
                combination_of_values = [combination_of_values]
            metadata_dictionary = dict(
                zip(tunable_multivalued_parameter_names_list, combination_of_values)
            )
            logger.info(
                f"'{critical_bare_mass_calculation_group_name}' grouping values: "
                f"{metadata_dictionary}."
            )

            bare_mass_values_array = dataframe_group["Bare_mass"].to_numpy()

            # Check for a minimum amount of data points
            sufficient_number_of_data_points = True

            bare_mass_values_for_fitting_array = bare_mass_values_array
            if UPPER_BARE_MASS_CUT is not None:
                condition = bare_mass_values_array < UPPER_BARE_MASS_CUT
                bare_mass_values_for_fitting_array = bare_mass_values_array[condition]

            if len(bare_mass_values_for_fitting_array) < 3:
                logger.warning(
                    "At least three (bare mass, PCAC mass) data points are "
                    "necessary for the calculating the critical bare mass."
                )
                sufficient_number_of_data_points = False

            # Check for a gauge configuration duplicates
            list_of_jackknife_analysis_identifiers = dataframe_group[
                "Jackknife_analysis_identifier"
            ].to_list()
            # Construct a flattened list of all the labels of all the gauge
            # links configurations used to calculate the PCAC mass estimates and
            # eventually the critical bare mass
            list_of_configuration_labels = list(
                itertools.chain.from_iterable(
                    [
                        input_data_files_set_group[jackknife_analysis_identifier][
                            "List_of_gauge_configuration_labels"
                        ][:]
                        for jackknife_analysis_identifier in list_of_jackknife_analysis_identifiers
                    ]
                )
            )
            # Look for duplicates
            if len(list_of_configuration_labels) != len(
                set(list_of_configuration_labels)
            ):
                counter = Counter(list_of_configuration_labels)
                # Get duplicates
                duplicates_dictionary = {
                    label: count for label, count in counter.items() if count > 1
                }
                logger.warning(
                    f"{critical_bare_mass_calculation_group_name}: The "
                    "following gauge links configurations have been used more "
                    "than once in the calculation of the critical bare mass "
                    "for this specific grouping (configuration label, counts):"
                    f"{duplicates_dictionary}",
                    to_console=True,
                )

            # Initialize the parameters values dictionary
            parameters_value_dictionary = copy.deepcopy(single_valued_fields_dictionary)
            # Append metadata dictionary
            parameters_value_dictionary.update(metadata_dictionary)
            logger.info("The parameter values dictionary was filled properly.")

            number_of_gauge_configurations_array = dataframe_group[
                "Number_of_gauge_configurations"
            ].to_numpy()

            PCAC_mass_estimates_array = gv.gvar(
                dataframe_group["PCAC_mass_estimate"].to_numpy()
            )

            PCAC_mass_estimates_for_fitting_array = PCAC_mass_estimates_array
            if UPPER_BARE_MASS_CUT is not None:
                PCAC_mass_estimates_for_fitting_array = PCAC_mass_estimates_array[
                    condition
                ]

            # LINEAR FIT ON BARE MASS VS PCAC MASS ESTIMATES DATA POINTS

            if fit_for_critical_bare_mass and sufficient_number_of_data_points:

                x = bare_mass_values_for_fitting_array
                y = PCAC_mass_estimates_for_fitting_array

                # Find indices of min(x) and max(x)
                min_index = np.argmin(x)
                max_index = np.argmax(x)

                # Get y values corresponding to min(x) and max(x)
                y_min = y[min_index]
                y_max = y[max_index]

                slope = (y_max - y_min) / (np.max(x) - np.min(x))
                x_intercept = y_min / slope + np.min(x)
                linear_fit_p0 = gv.mean([slope, x_intercept])
                linear_fit = lsqfit.nonlinear_fit(
                    data=(x, y),
                    p0=linear_fit_p0,
                    fcn=fit_functions.linear_function,
                    debug=True,
                )
                critical_bare_mass_value = linear_fit.p[1]

            # PLOT BARE MASS VS PCAC MASS ESTIMATES DATA POINTS

            if plot_critical_bare_mass:

                x = bare_mass_values_array
                y = PCAC_mass_estimates_array

                fig, ax = plt.subplots()
                ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

                plot_title = custom_plotting.DataPlotter._construct_plot_title(
                    None,
                    leading_substring="",
                    metadata_dictionary=metadata_dictionary,
                    title_width=105,
                    fields_unique_value_dictionary=parameters_value_dictionary,
                )
                ax.set_title(f"{plot_title}", pad=8)

                ax.set(xlabel="a$m_{{bare}}$", ylabel="a$m_{PCAC}$")

                ax.axvline(0, color="black")  # x = 0
                ax.axhline(0, color="black")  # y = 0

                ax.errorbar(
                    x, gv.mean(y), yerr=gv.sdev(y), fmt=".", markersize=8, capsize=10
                )

                if fit_for_critical_bare_mass and sufficient_number_of_data_points:

                    # Linear fit
                    if gv.mean(critical_bare_mass_value) > 0:
                        margin = 0.06
                    else:
                        margin = -0.06
                    x_data = np.linspace(
                        gv.mean(critical_bare_mass_value) * (1 - margin),
                        np.max(bare_mass_values_for_fitting_array)
                        * (1 + np.abs(margin)),
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

                    ax.legend(loc="upper left")

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

            # EXPORT CALCULATED DATA

            if fit_for_critical_bare_mass and sufficient_number_of_data_points:
                parameters_value_dictionary["Critical_bare_mass"] = (
                    critical_bare_mass_value.mean,
                    critical_bare_mass_value.sdev,
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
            output_files_directory, output_critical_bare_mass_csv_filename
        )

        # Export the DataFrame to a CSV file
        critical_bare_mass_dataframe.to_csv(csv_file_full_path, index=False)

    click.echo(
        "   -- Critical bare mass values calculation from PCAC mass "
        "estimates completed."
    )

    # Terminate logging
    logger.terminate_script_logging()


if __name__ == "__main__":
    main()
