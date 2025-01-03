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
import textwrap

from library import filesystem_utilities, effective_mass, plotting, constants


@click.command()
@click.option("--input_PCAC_mass_correlator_values_hdf5_file_path",
              "input_PCAC_mass_correlator_values_hdf5_file_path", "-PCAC_hdf5",
              default=None,
        help="Path to input HDF5 file containing extracted correlators values.")
@click.option("--output_files_directory", "output_files_directory", "-out_dir",
              default=None,
              help="Path to directory where all output files will be stored.")
@click.option("--plots_directory", "plots_directory", "-plots_dir",
              default="../../output/plots",
              help="Path to the output directory for storing plots.")
@click.option("--output_PCAC_mass_csv_filename", "output_PCAC_mass_csv_filename",
              "-hdf5", default="PCAC_mass_estimates.csv",
              help="Specific name for the output HDF5 file.")
@click.option("--log_file_directory", "log_file_directory", "-log_file_dir", 
              default=None, 
              help="Directory where the script's log file will be stored.")
@click.option("--log_filename", "log_filename", "-log", default=None, 
              help="Specific name for the script's log file.")

def main(input_PCAC_mass_correlator_values_hdf5_file_path,
         output_files_directory, plots_directory,
         output_PCAC_mass_csv_filename, log_file_directory, log_filename):

    # VALIDATE INPUT ARGUMENTS

    if not filesystem_utilities.is_valid_file(
                            input_PCAC_mass_correlator_values_hdf5_file_path):
        error_message = "Passed correlator values HDF5 file path is invalid."
        print("ERROR:", error_message)
        sys.exit(1)

    # If no output directory is provided, use the directory of the input file
    if output_files_directory is None:
        output_files_directory = os.path.dirname(
                            input_PCAC_mass_correlator_values_hdf5_file_path)
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
        log_filename = script_name.replace(".py", ".log")

    # Check for proper extensions in provided output filenames
    if not output_PCAC_mass_csv_filename.endswith(".csv"):
        output_PCAC_mass_csv_filename = output_PCAC_mass_csv_filename + ".csv"
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

    # PCAC MASS ESTIMATES CALCULATION

    # Create plots main subdirectory
    plots_main_subdirectory = os.path.join(plots_directory, "PCAC_mass_estimates_calculation")
    # Create "plots_main_subdirectory" if it does not exist
    os.makedirs(plots_main_subdirectory, exist_ok=True)

    PCAC_mass_plots_base_name = "PCAC_mass_correlator_values"
    PCAC_mass_plots_subdirectory = os.path.join(plots_main_subdirectory, PCAC_mass_plots_base_name)
    # Create "PCAC_mass_plots_subdirectory" if it does not exist
    os.makedirs(PCAC_mass_plots_subdirectory, exist_ok=True)

    g5_g5_plots_base_name = "g5g5_correlator_values"
    g5_g5_plots_subdirectory = os.path.join(plots_main_subdirectory, g5_g5_plots_base_name)
    # Create "g5_g5_plots_subdirectory" if it does not exist
    os.makedirs(g5_g5_plots_subdirectory, exist_ok=True)

    g4g5_g5_plots_base_name = "g4g5g5_correlator_values"
    g4g5_g5_plots_subdirectory = os.path.join(plots_main_subdirectory, g4g5_g5_plots_base_name)
    # Create "g5_g5_plots_subdirectory" if it does not exist
    os.makedirs(g4g5_g5_plots_subdirectory, exist_ok=True)

    g4g5_g5_derivative_plots_base_name = "g4g5g5_derivative_correlator_values"
    g4g5_g5_derivative_plots_subdirectory = os.path.join(plots_main_subdirectory, g4g5_g5_derivative_plots_base_name)
    # Create "g5_g5_plots_subdirectory" if it does not exist
    os.makedirs(g4g5_g5_derivative_plots_subdirectory, exist_ok=True)

    # Open the input HDF5 file for reading and the output HDF5 file for writing
    with h5py.File(input_PCAC_mass_correlator_values_hdf5_file_path, "r") \
                                                            as hdf5_file_read:

        # Initialize group structure of the output HDF5 file
        # NOTE: The assumption here is that the name of the raw data files
        # directory represents the data files set (or experiment) and its parent
        # directory the qpb main program that generated the data files
        parent_directory_name, last_directory_name = (
                                filesystem_utilities.extract_directory_names(
                                    output_files_directory)
                                    )

        # Select input HDF5 file's group to read
        input_qpb_main_program_group = hdf5_file_read[parent_directory_name]
        input_data_files_set_group = input_qpb_main_program_group[
                                                        last_directory_name]

        # Extract attributes of top-level groups into a dictionary
        fields_with_unique_values_dictionary = {parameter: attribute 
                            for parameter, attribute \
                                in input_data_files_set_group.attrs.items()}

        # List to pass values to dataframe
        PCAC_mass_estimates_list = list()

        # Loop over all PCAC mass correlator Jackknife analysis groups
        for PCAC_mass_correlator_analysis_group_name, \
                            PCAC_mass_correlator_analysis_group \
                                        in input_data_files_set_group.items():
            
            # Cautionary check if the item is a PCAC_mass_correlator_analysis_group
            if not isinstance(PCAC_mass_correlator_analysis_group, h5py.Group):
                # TODO: Log warning
                continue

            # Initialize the parameters values dictionary
            parameters_value_dictionary = copy.deepcopy(
                                        fields_with_unique_values_dictionary)

            # Extract attributes from current analysis group into the dictionary
            for parameter, attribute \
                        in PCAC_mass_correlator_analysis_group.attrs.items():
                parameters_value_dictionary[parameter] = attribute
            #     print(attribute)
            # print()

            jackknife_average_PCAC_mass_correlator_array = (
                gv.gvar(PCAC_mass_correlator_analysis_group[
    'jackknife_average_of_PCAC_mass_correlator_values_array_mean_values'][:],
                        PCAC_mass_correlator_analysis_group[
    'jackknife_average_of_PCAC_mass_correlator_values_array_error_values'][:]
                    )
                )
            
            temporal_direction_lattice_size = len(
                jackknife_average_PCAC_mass_correlator_array
            )

            # OPTIMUM FIT RANGE INVESTIGATION

            # Choose a lowest index cut for the PCAC mass correlator values array such
            # that the initial erratic points are filtered out
            LOWEST_INDEX_CUT = 4

            # Ignore the second half of the PCAC mass correlator values array since its
            # been by construction symmetrized
            y = jackknife_average_PCAC_mass_correlator_array[
                            LOWEST_INDEX_CUT:temporal_direction_lattice_size//2]

            # The corresponding time values must be shifted by the lowest index cut
            x = range(LOWEST_INDEX_CUT, len(y)+LOWEST_INDEX_CUT)

            # Usually at t=T/4 the time-dependent PCAC mass starts to plateau
            PCAC_mass_plateau_fit_guess = [
                gv.mean(y[temporal_direction_lattice_size//4])]

            # NOTE: 
            PCAC_mass_plateau_fit_minimum_chi2 = 1000 # Set to an arbitrary large number
            minimum_number_of_points = 5
            PCAC_mass_plateau_fit_optimum_range = (3, -3)

            for index_cut in range(len(x)-minimum_number_of_points):
                PCAC_mass_plateau_state_fit = lsqfit.nonlinear_fit(data=(x[index_cut:-1], y[index_cut:-1]), p0=PCAC_mass_plateau_fit_guess, fcn=effective_mass.plateau_fit_function, debug=True)
                if PCAC_mass_plateau_state_fit.chi2/len(x[index_cut:-1]) < PCAC_mass_plateau_fit_minimum_chi2:
                    PCAC_mass_plateau_fit_minimum_chi2 = PCAC_mass_plateau_state_fit.chi2/len(x[index_cut:-1])
                    PCAC_mass_plateau_fit_optimum_range = (index_cut, -1)
                    PCAC_plateau_fit_optimum_mass_estimate = PCAC_mass_plateau_state_fit.p[0]
                else:
                    break

            # PLOT TIME DEPENDENCE OF PCAC MASS VALUES

            fig, ax = plt.subplots()
            # ax.grid()
            plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

            # plot_title = ""
            # # # "Sign squared violation against scaling factor"
            # parameter_values_subtitle=plotting.construct_plot_subtitle(
            #                                     parameters_value_dictionary)
            # # # f"[KL {operator_type} $n$={KL_iterations}, $\\beta$={unique_QCD_beta_value}, $c_{{SW}}$={unique_clover_coefficient}, $\\rho$={unique_rho_value}, APE iters={unique_APE_iterations}, $\\epsilon_{{MSCG}}$={unique_MSCG_epsilon}, config: {configuration_label}, $\\kappa$={condition_number:.2f}]"
            # wrapper = textwrap.TextWrapper(width=100, initial_indent="   ")
            # wrapped_parameter_values_subtitle = wrapper.fill(parameter_values_subtitle)

            # ax.set_title(f'{plot_title}\n{wrapped_parameter_values_subtitle}', pad=7)

            # ax.set_title(f'Jackknife average of $m_{{PCAC}}(t)$ values ({Operator_method} {operator_type}, $\\beta$={QCD_beta_value:.2f}, $c_{{SW}}$={int(Clover_coefficient)},\n$\\rho$={Rho_value}, APE iters={APE_iterations}, $\\mu$={KL_scaling_factor}, $\\epsilon_{{CG}}$={CG_epsilon}, $\\epsilon_{{MSCG}}$={MSCG_epsilon}, $n$={KL_iterations}, $m_b=${bare_mass})', pad=7)
            # ax.set_title(f'Jackknife average of $m_{{PCAC}}(t)$ values ({Operator_method} {operator_type}, $\\beta$={QCD_beta_value:.2f}, $c_{{SW}}$={int(Clover_coefficient)},\n$\\rho$={Rho_value}, APE iters={APE_iterations}, $\\mu$={KL_scaling_factor}, $\\epsilon_{{CG}}$={CG_epsilon}, $n$={KL_iterations}, $m_b=${bare_mass})', pad=7)


            plot_title = ""

            parameter_values_subtitle = plotting.construct_plot_subtitle(
                parameters_value_dictionary
            )
            wrapper = textwrap.TextWrapper(width=100, initial_indent="   ")
            wrapped_parameter_values_subtitle = wrapper.fill(parameter_values_subtitle)

            ax.set_title(f"{plot_title}\n{wrapped_parameter_values_subtitle}", pad=6)

            ax.set(xlabel='$t/a$', ylabel='a$m_{PCAC}(t)$')
            
            # Set x-axis ticks to integer values only
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            ax.axvline(x[PCAC_mass_plateau_fit_optimum_range[0]], color='black') # x = 0
            ax.axvline(x[PCAC_mass_plateau_fit_optimum_range[1]-1], color='black') # x = 0

            ax.errorbar(x, gv.mean(y), yerr=gv.sdev(y), fmt='.', markersize=8, capsize=10
                        # , label=f'# of configs: {number_of_gauge_configurations}'
                        )

            PCAC_mass_estimate = gv.gvar(gv.mean(PCAC_plateau_fit_optimum_mass_estimate), np.sqrt(len(y[PCAC_mass_plateau_fit_optimum_range[0]:PCAC_mass_plateau_fit_optimum_range[1]])-1)*gv.sdev(PCAC_plateau_fit_optimum_mass_estimate))

            x_data = np.linspace(x[PCAC_mass_plateau_fit_optimum_range[0]], x[PCAC_mass_plateau_fit_optimum_range[1]-1], 100)
            plt.plot(x_data, effective_mass.plateau_fit_function(x_data, gv.mean(PCAC_mass_estimate)), 'r--', label=f'Plateau fit:\n- Fitting range: t/a$\in$[{x[PCAC_mass_plateau_fit_optimum_range[0]]}, {x[PCAC_mass_plateau_fit_optimum_range[1]-1]}]\n- $m^{{best\;fit}}_{{PCAC}}$={PCAC_mass_estimate:.5f}, ')
            ax.fill_between(x_data, gv.mean(PCAC_mass_estimate) - gv.sdev(PCAC_mass_estimate), gv.mean(PCAC_mass_estimate) + gv.sdev(PCAC_mass_estimate), color='r', alpha=0.2)

            if y[0] > y[temporal_direction_lattice_size//4]:
                ax.legend(loc="upper center")
            else:
                ax.legend(loc="lower center")
            
            # plot_filename = f'Jackknife_average_PCAC_mass_correlator_values_{Operator_method}_{operator_type}_m{bare_mass}_EpsCG{CG_epsilon}_EpsMSCG{MSCG_epsilon}.png'
            # plot_filename = plots_base_name + input_data_files_set_group.attrs[]
            # PCAC_mass_correlator_analysis_group_name
            # f'Jackknife_average_PCAC_mass_correlator_values_{Operator_method}_{operator_type}_m{bare_mass}_EpsCG{CG_epsilon}.png'
            # plot_path = os.path.join(plots_subdirectory, plot_filename)

            # Initialize characteristic substring
            if "Kernel_operator_type" in fields_with_unique_values_dictionary:
                plots_characteristic_fields_values_string = (
                    fields_with_unique_values_dictionary["Kernel_operator_type"]
                )
            elif "Kernel_operator_type" in input_data_files_set_group.attrs:
                plots_characteristic_fields_values_string = (
                    input_data_files_set_group.attrs["Kernel_operator_type"]
                )
            elif "Kernel_operator_type" in PCAC_mass_correlator_analysis_group.attrs:
                plots_characteristic_fields_values_string = (
                    PCAC_mass_correlator_analysis_group.attrs["Kernel_operator_type"]
                )
            else:
                plots_characteristic_fields_values_string = ""

            for key, value in PCAC_mass_correlator_analysis_group.attrs.items():
                if key in constants.PARAMETERS_PRINTED_LABELS_DICTIONARY:
                    plots_characteristic_fields_values_string += (
                        "_" + constants.PARAMETERS_PRINTED_LABELS_DICTIONARY[key]
                    )
                    plots_characteristic_fields_values_string += str(value)

            plots_characteristic_fields_values_string = plots_characteristic_fields_values_string.replace(".", "p")

            plot_path = os.path.join(
                PCAC_mass_plots_subdirectory,
                f"{PCAC_mass_plots_base_name}_{plots_characteristic_fields_values_string}" + ".png",
            )

            fig.savefig(plot_path)
            plt.close()

            parameters_value_dictionary["PCAC_mass_estimate"] = (
                    gv.mean(PCAC_mass_estimate), gv.sdev(PCAC_mass_estimate), 
                )

            PCAC_mass_estimates_list.append(parameters_value_dictionary)


            # PLOT G5-G5 CORRELATOR VALUES

            jackknife_average_of_g5_g5_correlator_values_array = (
            gv.gvar(PCAC_mass_correlator_analysis_group[
            'jackknife_average_of_g5_g5_correlator_mean_values'][:],
                    PCAC_mass_correlator_analysis_group[
            'jackknife_average_of_g5_g5_correlator_error_values'][:]
                )
            )

            fig, ax = plt.subplots()
            # ax.grid()
            plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

            plot_title = ""
            ax.set_title(f"{plot_title}\n{wrapped_parameter_values_subtitle}", pad=6)

            ax.set(xlabel='$t/a$', ylabel='g5g5(t)')

            ax.set_yscale("log")

            # Set x-axis ticks to integer values only
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Ignore the second half of the PCAC mass correlator values array since its
            # been by construction symmetrized
            y = jackknife_average_of_g5_g5_correlator_values_array[
                            LOWEST_INDEX_CUT:temporal_direction_lattice_size-LOWEST_INDEX_CUT+1]

            # The corresponding time values must be shifted by the lowest index cut
            x = range(LOWEST_INDEX_CUT, len(y)+LOWEST_INDEX_CUT)

            ax.errorbar(x, gv.mean(y), yerr=gv.sdev(y), fmt='.', markersize=8, capsize=10)

            plot_path = os.path.join(
            g5_g5_plots_subdirectory,
            f"{g5_g5_plots_base_name}_{plots_characteristic_fields_values_string}" + ".png",
            )

            fig.savefig(plot_path)
            plt.close()


            # PLOT G4G5-G5 CORRELATOR VALUES

            jackknife_average_of_g4g5_g5_correlator_values_array = (
            gv.gvar(PCAC_mass_correlator_analysis_group[
            'jackknife_average_of_g4g5_g5_correlator_mean_values'][:],
                    PCAC_mass_correlator_analysis_group[
            'jackknife_average_of_g4g5_g5_correlator_error_values'][:]
                )
            )

            fig, ax = plt.subplots()
            # ax.grid()
            plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

            plot_title = ""
            ax.set_title(f"{plot_title}\n{wrapped_parameter_values_subtitle}", pad=6)

            ax.set(xlabel='$t/a$', ylabel='g4g5g5(t)')

            # ax.set_yscale("log")

            # Set x-axis ticks to integer values only
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Ignore the second half of the PCAC mass correlator values array since its
            # been by construction symmetrized
            y = jackknife_average_of_g4g5_g5_correlator_values_array[
                            LOWEST_INDEX_CUT:temporal_direction_lattice_size-LOWEST_INDEX_CUT+1]

            # The corresponding time values must be shifted by the lowest index cut
            x = range(LOWEST_INDEX_CUT, len(y)+LOWEST_INDEX_CUT)

            ax.errorbar(x, gv.mean(y), yerr=gv.sdev(y), fmt='.', markersize=8, capsize=10)

            plot_path = os.path.join(
            g4g5_g5_plots_subdirectory,
            f"{g4g5_g5_plots_base_name}_{plots_characteristic_fields_values_string}" + ".png",
            )

            fig.savefig(plot_path)
            plt.close()

            # PLOT G4G5-G5 DERIVATIVE CORRELATOR VALUES

            jackknife_average_of_g4g5_g5_derivative_correlator_values_array = (
            gv.gvar(PCAC_mass_correlator_analysis_group[
            'g4g5_g5_derivative_correlator_from_jackknife_average_of_g4g5_g5_correlator_mean_values'][:],
                    PCAC_mass_correlator_analysis_group[
            'g4g5_g5_derivative_correlator_from_jackknife_average_of_g4g5_g5_correlator_error_values'][:]
                )
            )

            fig, ax = plt.subplots()
            # ax.grid()
            plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

            plot_title = ""
            ax.set_title(f"{plot_title}\n{wrapped_parameter_values_subtitle}", pad=6)

            ax.set(xlabel='$t/a$', ylabel='$\\partial$g4g5g5(t)')

            ax.set_yscale("log")

            # Set x-axis ticks to integer values only
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Ignore the second half of the PCAC mass correlator values array since its
            # been by construction symmetrized
            y = jackknife_average_of_g4g5_g5_derivative_correlator_values_array[
                            LOWEST_INDEX_CUT:temporal_direction_lattice_size-LOWEST_INDEX_CUT+1]

            # The corresponding time values must be shifted by the lowest index cut
            x = range(LOWEST_INDEX_CUT, len(y)+LOWEST_INDEX_CUT)

            ax.errorbar(x, gv.mean(y), yerr=gv.sdev(y), fmt='.', markersize=8, capsize=10)

            plot_path = os.path.join(
            g4g5_g5_derivative_plots_subdirectory,
            f"{g4g5_g5_derivative_plots_base_name}_{plots_characteristic_fields_values_string}" + ".png",
            )

            fig.savefig(plot_path)
            plt.close()

    # Create a DataFrame from the extracted data
    PCAC_mass_estimates_dataframe = pd.DataFrame(PCAC_mass_estimates_list)

    # Construct output .csv file path
    csv_file_full_path = os.path.join(output_files_directory,
                                                output_PCAC_mass_csv_filename)
    # Export the DataFrame to a CSV file
    PCAC_mass_estimates_dataframe.to_csv(csv_file_full_path, index=False)

    print("   -- PCAC mass estimates calculation completed.")

    # Terminate logging
    logging.info(f"Script '{script_name}' execution terminated successfully.")


if __name__ == "__main__":
    main()

# TODO: Include proper logging!