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
    momentum_correlator,
    jackknife_analysis,
)


@click.command()
@click.option(
    "--input_PCAC_mass_correlator_hdf5_file_path",
    "input_PCAC_mass_correlator_hdf5_file_path",
    "-PCAC_hdf5",
    required=True,
    # default="/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/invert/Chebyshev_several_config_varying_N/PCAC_mass_correlator_values.h5",
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
    # default="/nvme/h/cy22sg1/qpb_data_analysis/output/plots/invert/Chebyshev_several_config_varying_N",
    help="Path to the output directory for storing plots.",
)
@click.option(
    "-plot_g5g5",
    "--plot_g5g5_correlators",
    "plot_g5g5_correlators",
    is_flag=True,
    default=False,
    help="Enable plotting g5-g5 correlator values.",
)
@click.option(
    "-plot_eff_mass",
    "--plot_effective_mass_correlators",
    "plot_effective_mass_correlators",
    is_flag=True,
    default=False,
    help="Enable plotting effective mass correlator values.",
)
@click.option(
    "-zoom_in",
    "--zoom_in_effective_mass_correlators_plots",
    "zoom_in_effective_mass_correlators_plots",
    is_flag=True,
    default=False,
    help="Enable zooming in on effective mass correlators plots.",
)
@click.option(
    "--output_pion_effective_mass_estimates_csv_filename",
    "output_pion_effective_mass_estimates_csv_filename",
    "-out_csv",
    default="pion_effective_mass_estimates.csv",
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
    plot_g5g5_correlators,
    plot_effective_mass_correlators,
    zoom_in_effective_mass_correlators_plots,
    output_pion_effective_mass_estimates_csv_filename,
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
    if not output_pion_effective_mass_estimates_csv_filename.endswith(".csv"):
        output_pion_effective_mass_estimates_csv_filename = (
            output_pion_effective_mass_estimates_csv_filename + ".csv"
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

    if plot_g5g5_correlators or plot_effective_mass_correlators:
        # Create main plots directory if it does not exist
        plots_main_subdirectory = filesystem_utilities.create_subdirectory(
            plots_directory, "pion_effective_mass_calculation"
        )

    # Create deeper-level subdirectories if requested
    if plot_g5g5_correlators:
        g5g5_plots_subdirectory = filesystem_utilities.create_subdirectory(
            plots_main_subdirectory, "g5g5_correlator_values", clear_contents=True
        )

    if plot_effective_mass_correlators:
        effective_mass_plots_subdirectory = filesystem_utilities.create_subdirectory(
            plots_main_subdirectory, "effective_mass_correlator", clear_contents=True
        )

    # IMPORT DATASETS AND METADATA

    # Open the input HDF5 file for reading
    with h5py.File(input_PCAC_mass_correlator_hdf5_file_path, "r") as hdf5_file:

        # Initialize group structure of the output HDF5 file
        # NOTE: The assumption here is that the name of the raw data files
        # directory represents the data files set and its parent
        # directory the qpb main program that generated the data files
        parent_directory_name, data_files_set_name = (
            filesystem_utilities.extract_directory_names(output_files_directory)
        )

        # Select input HDF5 file's group to read
        input_qpb_main_program_group = hdf5_file[parent_directory_name]
        input_data_files_set_group = input_qpb_main_program_group[data_files_set_name]

        # Extract attributes of top-level groups into a dictionary
        fields_with_unique_values_dictionary = {
            parameter: attribute
            for parameter, attribute in input_data_files_set_group.attrs.items()
        }

        # List to pass values to dataframe
        pion_effective_mass_estimates_list = []

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

            # Import g5-g5 correlator Jackknife average dataset
            jackknife_average_g5g5_correlator_array = gv.gvar(
                PCAC_mass_correlator_analysis_group[
                    "jackknife_average_of_g5_g5_correlator_mean_values"
                ][:],
                PCAC_mass_correlator_analysis_group[
                    "jackknife_average_of_g5_g5_correlator_error_values"
                ][:],
            )

            # Import Jackknife replicas datasets of g5-g5 correlators
            jackknife_replicas_of_g5g5_correlator_2D_array = (
                PCAC_mass_correlator_analysis_group[
                    "jackknife_samples_of_g5_g5_correlator_2D_array"
                ][:]
            )

            # SYMMETRIZE G5-G5 CORRELATORS

            jackknife_average_g5g5_correlator_array = (
                momentum_correlator.symmetrization(
                    jackknife_average_g5g5_correlator_array
                )
            )

            jackknife_replicas_of_g5g5_correlator_2D_array = np.array(
                [
                    momentum_correlator.symmetrization(g5_g5_correlator_replica)
                    for g5_g5_correlator_replica in jackknife_replicas_of_g5g5_correlator_2D_array
                ]
            )

            # CALCULATE EFFECTIVE MASS CORRELATORS

            # Halved effective mass replica correlator values from Jackknife
            # replicas of g5g5 correlator values
            effective_mass_correlator_replicas_2D_array = np.array(
                [
                    effective_mass.calculate_two_state_periodic_effective_mass_correlator(
                        g5_g5_correlator_replica
                    )
                    for g5_g5_correlator_replica in jackknife_replicas_of_g5g5_correlator_2D_array
                ]
            )

            jackknife_average_effective_mass_correlator = np.mean(
                effective_mass_correlator_replicas_2D_array, axis=0
            )

            integrated_autocorrelation_time = (
                jackknife_analysis.calculate_integrated_autocorrelation_time(
                    jackknife_average_effective_mass_correlator
                )
            )

            jackknife_average_effective_mass_correlator = gv.gvar(
                jackknife_average_effective_mass_correlator,
                jackknife_analysis.jackknife_correlated_error(
                    effective_mass_correlator_replicas_2D_array,
                    integrated_autocorrelation_time,
                ),
            )

            # VALIDATE VALUES OF IMPORTANT PARAMETERS

            # Ensuring the important parameter values of temporal lattice size
            # and number of gauge configurations are stored and available
            temporal_lattice_size = len(jackknife_average_g5g5_correlator_array)
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
                jackknife_replicas_of_g5g5_correlator_2D_array
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

            ### CALCULATE PLATEAU AND TWO-STATE RANGE FOR EFFECTIVE MASS FIT ###

            # CALCULATE GUESS PARAMETERS FOR G5G5 CORRELATORS SINGLE-STATE FITS

            """Fitting begins with a naive guess of the amplitude of the
            g5-g5 time-dependent correlator C(t) and the effective mass
            estimate. Assuming that either the left or the right symmetric part
            of C(t) has a A e^(-m t) form """
            """Usually the time-dependent pion effective mass values at
            about t=T/4 are a good estimate of the plateau fit effective mass
            estimate, and this the reason it can be used as an effective mass
            guess for the single-state g5-g5 correlator, and correspondingly for
            its amplitude."""

            guess_index = temporal_lattice_size // 4

            single_state_non_periodic_effective_mass_correlator = effective_mass.calculate_single_state_non_periodic_effective_mass_correlator(
                jackknife_average_g5g5_correlator_array
            )
            effective_mass_guess = gv.mean(
                single_state_non_periodic_effective_mass_correlator[guess_index]
            )

            amplitude_factor_guess = (
                momentum_correlator.amplitude_of_single_state_non_periodic_correlator(
                    jackknife_average_g5g5_correlator_array,
                    effective_mass_guess,
                    guess_index,
                )
            )
            amplitude_factor_guess = gv.mean(amplitude_factor_guess)

            # EXTRACT BETTER GUESS PARAMETERS FROM SINGLE STATE NON PERIODIC FIT

            y = jackknife_average_g5g5_correlator_array
            x = np.arange(len(y))

            # A limited fit range
            single_state_non_periodic_fit_range = np.arange(
                guess_index - guess_index // 2, guess_index + guess_index // 2
            )
            g5_g5_correlator_single_state_non_periodic_fit_p0 = [
                amplitude_factor_guess,
                effective_mass_guess,
            ]
            g5_g5_correlator_single_state_fit = lsqfit.nonlinear_fit(
                data=(
                    x[single_state_non_periodic_fit_range],
                    y[single_state_non_periodic_fit_range],
                ),
                p0=g5_g5_correlator_single_state_non_periodic_fit_p0,
                fcn=momentum_correlator.single_state_non_periodic_correlator,
                debug=True,
            )

            # PLOT G5-G5 CORRELATOR FOR TESTING GUESS PARAMETERS IF REQUESTED

            if plot_g5g5_correlators:
                # Exclude first point for a more symmetrical shape
                x = x[1:]
                y = y[1:]

                fig, ax = plt.subplots()
                ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

                ax.set(xlabel="$t/a$", ylabel="$C(t)$")
                ax.set_yscale("log")

                plot_title = plotting.DataPlotter._construct_plot_title(
                    None,
                    leading_substring="",
                    metadata_dictionary=metadata_dictionary,
                    title_width=100,
                    fields_unique_value_dictionary=parameters_value_dictionary,
                )
                ax.set_title(f"{plot_title}", pad=8)

                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                ax.errorbar(
                    x, gv.mean(y), yerr=gv.sdev(y), fmt=".", markersize=8, capsize=10
                )

                # Plot single-state periodic curve using fit non-periodic parameters
                x_data = np.linspace(min(x), max(x), 100)
                label_string = (
                    f"Single-state non-periodic fit:\n"
                    f"- fit range: t/a$\\in[${x[single_state_non_periodic_fit_range.min()]}, "
                    f"{x[single_state_non_periodic_fit_range.max()]}$]$, \n"
                    f"- $\\chi^2$/dof={g5_g5_correlator_single_state_fit.chi2:.2f}"
                    f"/{g5_g5_correlator_single_state_fit.dof}="
                    f"{g5_g5_correlator_single_state_fit.chi2 / g5_g5_correlator_single_state_fit.dof:.3f},\n"
                    f"- $m^{{best\\!-\!fit}}_{{eff.}}$={g5_g5_correlator_single_state_fit.p[1]:.3f}"
                )
                ax.plot(
                    x_data,
                    momentum_correlator.single_state_periodic_correlator(
                        x_data, gv.mean(g5_g5_correlator_single_state_fit.p)
                    ),
                    "r--",
                    label=label_string,
                )

                ax.legend(loc="upper center")

                current_plots_base_name = "g5g5_average_correlator"
                plot_path = plotting.DataPlotter._generate_plot_path(
                    None,
                    g5g5_plots_subdirectory,
                    current_plots_base_name,
                    metadata_dictionary,
                    single_valued_fields_dictionary=fields_with_unique_values_dictionary,
                )
                fig.savefig(plot_path)
                plt.close()

            # PLATEAU RANGE FOR EFFECTIVE MASS CORRELATOR

            sigma_criterion_factor = 1.0
            plateau_indices_list = []
            # TODO: Why 5?
            while len(plateau_indices_list) < 5:
                plateau_indices_list = effective_mass.plateau_indices_range(
                    jackknife_average_effective_mass_correlator, sigma_criterion_factor
                )
                sigma_criterion_factor += 0.5

            # CALCULATE GUESS PARAMETERS FOR EFFECTIVE MASS TWO-STATE FITS

            y = jackknife_average_effective_mass_correlator[
                : plateau_indices_list[-1] + 1
            ]
            x = np.arange(len(y))

            # Use best fit values from g5g5 correlator single-state fit
            amplitude_factor_guess = gv.mean(g5_g5_correlator_single_state_fit.p[0])
            effective_mass_guess = gv.mean(g5_g5_correlator_single_state_fit.p[1])

            c_guess = np.log(
                (y[1] - amplitude_factor_guess * np.exp(-effective_mass_guess))
                / (y[2] - amplitude_factor_guess * np.exp(-2 * effective_mass_guess))
            )
            c_guess = gv.mean(c_guess)

            r_guess = (np.exp(np.sum(y) - len(y) * effective_mass_guess) - 1) / np.exp(
                -c_guess
            )
            r_guess = gv.mean(r_guess)

            effective_mass_two_state_fit_p0 = [effective_mass_guess, r_guess, c_guess]

            # CALCULATE BETTER GUESS PARAMETERS AND FIT RANGE

            optimum_Q = 0  # Initialize optimum p-value variable
            # Investigate the optimum lower index of the fit range. Maintain a
            # distance of at least two point from the plateau range
            for lower_index_cut in range(plateau_indices_list[0] - 1):
                effective_mass_two_state_fit = lsqfit.nonlinear_fit(
                    data=(x[lower_index_cut:], y[lower_index_cut:]),
                    p0=effective_mass_two_state_fit_p0,
                    fcn=effective_mass.two_state_fit_function,
                    debug=True,
                )

                if (
                    (optimum_Q < effective_mass_two_state_fit.Q)
                    and (effective_mass_two_state_fit.p[2] > 0)
                    and (effective_mass_two_state_fit.p[1] > 0)
                ):
                    optimum_Q = effective_mass_two_state_fit.Q
                    effective_mass_two_state_optimum_fit = effective_mass_two_state_fit
                    optimum_lower_index = lower_index_cut

            effective_mass_two_state_fit_p0 = gv.mean(
                effective_mass_two_state_optimum_fit.p
            )

            # TWO-STATE FIT ON EVERY REPLICA DATASET

            two_state_effective_mass_estimates_list = []
            for (
                effective_mass_correlator_replica
            ) in effective_mass_correlator_replicas_2D_array:
                y = gv.gvar(
                    effective_mass_correlator_replica,
                    gv.sdev(jackknife_average_effective_mass_correlator),
                )
                y = y[optimum_lower_index : plateau_indices_list[-1]]
                x = np.arange(optimum_lower_index, len(y) + optimum_lower_index)
                effective_mass_two_state_fit = lsqfit.nonlinear_fit(
                    data=(x, y),
                    p0=effective_mass_two_state_fit_p0,
                    fcn=effective_mass.two_state_fit_function,
                    debug=True,
                )
                two_state_effective_mass_estimates_list.append(
                    effective_mass_two_state_fit.p[0]
                )

            pion_effective_mass_estimate = gv.gvar(
                jackknife_analysis.weighted_mean(
                    gv.mean(two_state_effective_mass_estimates_list),
                    gv.sdev(two_state_effective_mass_estimates_list),
                    np.sqrt(number_of_gauge_configurations)
                    * np.sqrt(2 * integrated_autocorrelation_time),
                    # TODO: Still need a justification for including this:
                    # * np.sqrt(len(plateau_indices_list)),
                )
            )

            # PLOT EFFECTIVE MASS CORRELATORS
            if plot_effective_mass_correlators:
                y = jackknife_average_effective_mass_correlator
                x = np.arange(len(y))

                fig, ax = plt.subplots()
                ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

                plot_title = plotting.DataPlotter._construct_plot_title(
                    None,
                    leading_substring="",
                    # metadata_dictionary=dict(),
                    metadata_dictionary=metadata_dictionary,
                    title_width=110,
                    fields_unique_value_dictionary=fields_with_unique_values_dictionary,
                    additional_excluded_fields=[
                        "Average_calculation_time_per_spinor_per_configuration",
                        "Average_number_of_MV_multiplications_per_spinor_per_configuration",
                    ],
                )
                ax.set_title(f"{plot_title}", pad=8)

                if zoom_in_effective_mass_correlators_plots:
                    ax.set_ylim(
                        [
                            0.5 * pion_effective_mass_estimate.mean,
                            1.5 * pion_effective_mass_estimate.mean,
                        ]
                    )

                ax.set(
                    xlabel="$t/a$",
                    ylabel=(
                        "$m_{eff.}\!(t)=\\frac{1}{2}\\log\!\\left( "
                        "\\frac{C(t-1)+\\sqrt{C^2(t-1)-C^2(T//2)})}{C(t+1)"
                        "+\\sqrt{C^2(t+1)-C^2(T//2)})} \\right) $"
                    ),
                )
                fig.subplots_adjust(left=0.15)  # Adjust left margin

                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                ax.errorbar(
                    x, gv.mean(y), yerr=gv.sdev(y), fmt=".", markersize=8, capsize=10
                )

                label_string = (
                    f"Plateau fit:\n- fit range: t/a$\\in[${x[plateau_indices_list[0]]}, "
                    f"{x[plateau_indices_list[-1]]}$]$,\n- $m^{{best\!-\!fit}}_{{eff.}}$"
                    f"={pion_effective_mass_estimate:.3f}\n"
                )

                ax.hlines(
                    y=pion_effective_mass_estimate.mean,
                    xmin=x[plateau_indices_list[0]],
                    xmax=x[plateau_indices_list[-1]],
                    color="r",
                    linestyle="--",
                    label=label_string,
                )

                ax.fill_between(
                    x[plateau_indices_list],
                    pion_effective_mass_estimate.mean
                    - pion_effective_mass_estimate.sdev,
                    pion_effective_mass_estimate.mean
                    + pion_effective_mass_estimate.sdev,
                    color="r",
                    alpha=0.2,
                )

                ax.legend(loc="upper right")

                current_plots_base_name = "effective_mass_correlator"
                plot_path = plotting.DataPlotter._generate_plot_path(
                    None,
                    effective_mass_plots_subdirectory,
                    current_plots_base_name,
                    metadata_dictionary,
                    single_valued_fields_dictionary=fields_with_unique_values_dictionary,
                )

                fig.savefig(plot_path)
                plt.close()

            # EXPORT CALCULATED DATA

            parameters_value_dictionary["Pion_effective_mass_estimate"] = (
                pion_effective_mass_estimate.mean,
                pion_effective_mass_estimate.sdev,
            )

            pion_effective_mass_estimates_list.append(parameters_value_dictionary)

        # Create a DataFrame from the extracted data
        pion_effective_mass_estimates_dataframe = pd.DataFrame(
            pion_effective_mass_estimates_list
        )

        # Construct output .csv file path
        csv_file_full_path = os.path.join(
            output_files_directory, output_pion_effective_mass_estimates_csv_filename
        )

        # Export the DataFrame to a CSV file
        pion_effective_mass_estimates_dataframe.to_csv(csv_file_full_path, index=False)

    # Terminate logging
    logging.info(f"Script '{script_name}' execution terminated successfully.")

    print("   -- Pion effective mass estimates calculation completed.")


if __name__ == "__main__":
    main()
