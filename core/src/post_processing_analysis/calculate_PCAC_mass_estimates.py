import os
import sys

import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import gvar as gv
import lsqfit
import logging
import pandas as pd
import h5py
import copy

from library import (
    filesystem_utilities,
    effective_mass,
    plotting,
    jackknife_analysis,
)


@click.command()
@click.option(
    "--input_PCAC_mass_correlator_hdf5_file_path",
    "input_PCAC_mass_correlator_hdf5_file_path",
    "-PCAC_hdf5",
    # default="/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/invert/KL_several_config_and_mu_varying_m/PCAC_mass_correlator_values.h5",
    required=True,
    help="Path to input HDF5 file containing extracted correlators values.",
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
    default="../../output/plots",
    # default="../../../output/plots/invert/KL_several_config_and_mu_varying_m",
    help="Path to the output directory for storing plots.",
)
@click.option(
    "-plot_PCAC_mass",
    "--plot_PCAC_mass_correlators",
    "plot_PCAC_mass_correlators",
    is_flag=True,
    default=False,
    help="Enable plotting PCAC mass correlator values.",
)
@click.option(
    "-zoom_in",
    "--zoom_in_PCAC_mass_correlators_plots",
    "zoom_in_PCAC_mass_correlators_plots",
    is_flag=True,
    default=False,
    help="Enable zooming in on PCAC mass correlators plots.",
)
@click.option(
    "--output_PCAC_mass_estimates_csv_filename",
    "output_PCAC_mass_estimates_csv_filename",
    "-hdf5",
    default="PCAC_mass_estimates.csv",
    help="Specific name for the output HDF5 file.",
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
    input_PCAC_mass_correlator_hdf5_file_path,
    output_files_directory,
    plots_directory,
    plot_PCAC_mass_correlators,
    zoom_in_PCAC_mass_correlators_plots,
    output_PCAC_mass_estimates_csv_filename,
    log_file_directory,
    log_filename,
):
    # VALIDATE INPUT ARGUMENTS

    if not filesystem_utilities.is_valid_file(
        input_PCAC_mass_correlator_hdf5_file_path
    ):
        error_message = "Passed correlator values HDF5 file path is invalid."
        print("ERROR:", error_message)
        sys.exit(1)

    # If no output directory is provided, use the directory of the input file
    if output_files_directory is None:
        output_files_directory = os.path.dirname(
            input_PCAC_mass_correlator_hdf5_file_path
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
    if not output_PCAC_mass_estimates_csv_filename.endswith(".csv"):
        output_PCAC_mass_estimates_csv_filename = (
            output_PCAC_mass_estimates_csv_filename + ".csv"
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

    if plot_PCAC_mass_correlators:
        # Create plots main subdirectory if it does not exist
        plots_main_subdirectory = filesystem_utilities.create_subdirectory(
            plots_directory, "PCAC_mass_estimates_calculation"
        )

        # Create deeper-level subdirectories if they do not exist
        PCAC_mass_plots_base_name = "PCAC_mass_correlator"
        PCAC_mass_plots_subdirectory = filesystem_utilities.create_subdirectory(
            plots_main_subdirectory,
            PCAC_mass_plots_base_name + "s",
            clear_contents=True,
        )

    # IMPORT DATASETS AND METADATA

    # Open the input HDF5 file for reading and the output HDF5 file for writing
    with h5py.File(input_PCAC_mass_correlator_hdf5_file_path, "r") as hdf5_file_read:

        # Initialize group structure of the output HDF5 file
        # NOTE: The assumption here is that the name of the raw data files
        # directory represents the data files set (or experiment) and its parent
        # directory the qpb main program that generated the data files
        parent_directory_name, last_directory_name = (
            filesystem_utilities.extract_directory_names(output_files_directory)
        )

        # Select input HDF5 file's group to read
        input_qpb_main_program_group = hdf5_file_read[parent_directory_name]
        input_data_files_set_group = input_qpb_main_program_group[last_directory_name]

        # Extract attributes of top-level groups into a dictionary
        fields_with_unique_values_dictionary = {
            parameter: attribute
            for parameter, attribute in input_data_files_set_group.attrs.items()
        }

        # List to pass values to dataframe
        PCAC_mass_estimates_list = list()

        # Loop over all PCAC mass correlator Jackknife analysis groups
        for (
            PCAC_mass_correlator_analysis_group_name,
            PCAC_mass_correlator_analysis_group,
        ) in input_data_files_set_group.items():

            # Cautionary check if the item is a HDF5 group
            if not isinstance(PCAC_mass_correlator_analysis_group, h5py.Group):
                # TODO: Log warning
                continue

            # Initialize the parameters values dictionary
            parameters_value_dictionary = copy.deepcopy(
                fields_with_unique_values_dictionary
            )

            # Create a separate metadata dictionary for later use
            metadata_dictionary = dict()
            for (
                parameter,
                attribute,
            ) in PCAC_mass_correlator_analysis_group.attrs.items():
                metadata_dictionary[parameter] = attribute

            # Append metadata dictionary
            parameters_value_dictionary.update(metadata_dictionary)

            # Import Jackknife replicas datasets of PCAC mass correlators
            jackknife_replicas_of_PCAC_mass_correlator_2D_array = (
                PCAC_mass_correlator_analysis_group[
                    "jackknife_samples_of_PCAC_mass_correlator_values_2D_array"
                ][:]
            )

            # VALIDATE VALUES OF IMPORTANT PARAMETERS

            # Ensuring the important parameter values of temporal lattice size
            # and number of gauge configurations are stored and available
            temporal_lattice_size = np.shape(
                jackknife_replicas_of_PCAC_mass_correlator_2D_array
            )[1]
            if "Temporal_lattice_size" not in parameters_value_dictionary:
                parameters_value_dictionary["Temporal_lattice_size"] = (
                    temporal_lattice_size
                )
            elif (
                parameters_value_dictionary["Temporal_lattice_size"]
                != temporal_lattice_size
            ):
                pass
                # TODO: Log warning

            number_of_gauge_configurations = np.shape(
                jackknife_replicas_of_PCAC_mass_correlator_2D_array
            )[0]
            if "Number_of_gauge_configurations" not in parameters_value_dictionary:
                parameters_value_dictionary["Number_of_gauge_configurations"] = (
                    number_of_gauge_configurations
                )
            elif (
                parameters_value_dictionary["Number_of_gauge_configurations"]
                != number_of_gauge_configurations
            ):
                pass
                # TODO: Log warning

            # CALCULATE FURTHER USEFUL QUANTITIES

            if (
                "Total_calculation_time_values_array"
                in PCAC_mass_correlator_analysis_group
                and isinstance(
                    PCAC_mass_correlator_analysis_group[
                        "Total_calculation_time_values_array"
                    ],
                    h5py.Dataset,
                )
            ):
                parameters_value_dictionary[
                    "Average_calculation_time_per_spinor_per_configuration"
                ] = np.average(
                    PCAC_mass_correlator_analysis_group[
                        "Total_calculation_time_values_array"
                    ][:]
                )

            if (
                "Average_number_of_MV_multiplications_per_spinor_values_array"
                in PCAC_mass_correlator_analysis_group
                and isinstance(
                    PCAC_mass_correlator_analysis_group[
                        "Average_number_of_MV_multiplications_per_spinor_values_array"
                    ],
                    h5py.Dataset,
                )
            ):
                parameters_value_dictionary[
                    "Average_number_of_MV_multiplications_per_spinor_per_configuration"
                ] = np.average(
                    PCAC_mass_correlator_analysis_group[
                        "Average_number_of_MV_multiplications_per_spinor_values_array"
                    ][:]
                )

            # TRUNCATE PCAC MASS CORRELATORS

            # Ignore the second half of the PCAC mass correlators arrays since
            # they are by construction symmetrized
            jackknife_replicas_of_PCAC_mass_correlator_2D_array = np.array(
                [
                    PCAC_mass_correlator_replica[: temporal_lattice_size // 2]
                    for (
                        PCAC_mass_correlator_replica
                    ) in jackknife_replicas_of_PCAC_mass_correlator_2D_array
                ]
            )

            # CALCULATE JACKKNIFE AVERAGE MASS CORRELATORS

            jackknife_average_PCAC_mass_correlator_array = np.mean(
                jackknife_replicas_of_PCAC_mass_correlator_2D_array, axis=0
            )

            # Restrict the calculation range to a possible plateau range
            calculation_range = np.arange(
                temporal_lattice_size // 4 - temporal_lattice_size // 8,
                temporal_lattice_size // 4 + temporal_lattice_size // 8,
            )
            integrated_autocorrelation_time = (
                jackknife_analysis.calculate_integrated_autocorrelation_time(
                    jackknife_average_PCAC_mass_correlator_array[calculation_range]
                )
            )

            if integrated_autocorrelation_time < 1:
                integrated_autocorrelation_time = 1

            jackknife_average_PCAC_mass_correlator_array = gv.gvar(
                jackknife_average_PCAC_mass_correlator_array,
                jackknife_analysis.jackknife_correlated_error(
                    jackknife_replicas_of_PCAC_mass_correlator_2D_array,
                    integrated_autocorrelation_time,
                ),
            )

            # PLATEAU RANGE FOR PCAC MASS CORRELATOR

            sigma_criterion_factor = 1.5
            plateau_indices_list = []
            # TODO: Why this number?
            minimum_number_of_data_points = temporal_lattice_size // 8
            while len(plateau_indices_list) < minimum_number_of_data_points:
                plateau_indices_list = effective_mass.plateau_indices_range(
                    jackknife_average_PCAC_mass_correlator_array,
                    sigma_criterion_factor,
                    3,
                )
                sigma_criterion_factor += 0.5

            # PLATEAU FIT ON EVERY REPLICA DATASET

            PCAC_mass_plateau_fit_guess = [
                np.mean(
                    gv.mean(
                        jackknife_average_PCAC_mass_correlator_array[
                            plateau_indices_list
                        ]
                    )
                )
            ]

            plateau_fit_PCAC_mass_estimates_list = list()
            for (
                PCAC_mass_correlator_replica
            ) in jackknife_replicas_of_PCAC_mass_correlator_2D_array:
                y = gv.gvar(
                    PCAC_mass_correlator_replica,
                    gv.sdev(jackknife_average_PCAC_mass_correlator_array),
                )
                x = np.arange(len(y))
                PCAC_mass_plateau_state_fit = lsqfit.nonlinear_fit(
                    data=(x[plateau_indices_list], y[plateau_indices_list]),
                    p0=PCAC_mass_plateau_fit_guess,
                    fcn=effective_mass.plateau_fit_function,
                    debug=True,
                )
                plateau_fit_PCAC_mass_estimates_list.append(
                    PCAC_mass_plateau_state_fit.p
                )

            PCAC_mass_estimate = gv.gvar(
                jackknife_analysis.weighted_mean(
                    gv.mean(plateau_fit_PCAC_mass_estimates_list),
                    gv.sdev(plateau_fit_PCAC_mass_estimates_list),
                    np.sqrt(number_of_gauge_configurations)
                    * np.sqrt(2 * integrated_autocorrelation_time),
                    # TODO: Still need a justification for including this:
                    # * np.sqrt(len(plateau_indices_list)),
                )
            )

            # PLOT PCAC MASS CORRELATORS

            if plot_PCAC_mass_correlators:
                starting_time = 5
                y = jackknife_average_PCAC_mass_correlator_array[starting_time:]
                x = np.arange(starting_time, len(y) + starting_time)

                fig, ax = plt.subplots()
                ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

                plot_title = plotting.DataPlotter._construct_plot_title(
                    None,
                    leading_substring="",
                    metadata_dictionary=dict(),
                    title_width=100,
                    fields_unique_value_dictionary=parameters_value_dictionary,
                )
                ax.set_title(f"{plot_title}", pad=8)

                ax.set(
                    xlabel="$t/a$",
                    ylabel="a$m_{PCAC}(t)$",
                )
                fig.subplots_adjust(left=0.15)  # Adjust left margin

                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                ax.errorbar(
                    x, gv.mean(y), yerr=gv.sdev(y), fmt=".", markersize=8, capsize=10
                )

                if zoom_in_PCAC_mass_correlators_plots:
                    y_limits = [0.9 * y[0].mean, 1.1 * PCAC_mass_estimate.mean]
                    if y[0] > y[temporal_lattice_size // 4]:
                        y_limits = [y_limits[1], y_limits[0]]
                    ax.set_ylim(y_limits)

                plateau_range_minimum = x[plateau_indices_list[0] - starting_time]
                plateau_range_maximum = x[plateau_indices_list[-1] - starting_time]

                ax.axvline(plateau_range_minimum, color="black")
                ax.axvline(plateau_range_maximum, color="black")

                label_string = (
                    f"Plateau fit:\n"
                    f"- fit range: t/a$\\in[${plateau_range_minimum}, "
                    f"{plateau_range_maximum}$]$,\n- $m^{{best\!-\!fit}}_{{PCAC}}$"
                    f"={PCAC_mass_estimate:.3f}\n"
                )

                ax.hlines(
                    y=PCAC_mass_estimate.mean,
                    xmin=plateau_range_minimum,
                    xmax=plateau_range_maximum,
                    color="r",
                    linestyle="--",
                    label=label_string,
                )

                ax.fill_between(
                    np.arange(plateau_range_minimum, plateau_range_maximum + 1),
                    PCAC_mass_estimate.mean - PCAC_mass_estimate.sdev,
                    PCAC_mass_estimate.mean + PCAC_mass_estimate.sdev,
                    color="r",
                    alpha=0.2,
                )

                if y[0] > y[temporal_lattice_size // 4]:
                    ax.legend(loc="upper center")
                else:
                    ax.legend(loc="lower center")

                current_plots_base_name = PCAC_mass_plots_base_name
                plot_path = plotting.DataPlotter._generate_plot_path(
                    None,
                    PCAC_mass_plots_subdirectory,
                    current_plots_base_name,
                    metadata_dictionary,
                    single_valued_fields_dictionary=fields_with_unique_values_dictionary,
                )

                fig.savefig(plot_path)
                plt.close()

            # EXPORT CALCULATED DATA

            parameters_value_dictionary["PCAC_mass_estimate"] = (
                PCAC_mass_estimate.mean,
                PCAC_mass_estimate.sdev,
            )

            PCAC_mass_estimates_list.append(parameters_value_dictionary)

        # Create a DataFrame from the extracted data
        PCAC_mass_estimates_dataframe = pd.DataFrame(PCAC_mass_estimates_list)

        # Construct output .csv file path
        csv_file_full_path = os.path.join(
            output_files_directory, output_PCAC_mass_estimates_csv_filename
        )
        # Export the DataFrame to a CSV file
        PCAC_mass_estimates_dataframe.to_csv(csv_file_full_path, index=False)

    # Terminate logging
    logging.info(f"Script '{script_name}' execution terminated successfully.")

    print("   -- PCAC mass estimates calculation completed.")


if __name__ == "__main__":
    main()
