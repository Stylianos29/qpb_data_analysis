import os
import sys
import itertools
import ast

import click # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from matplotlib.ticker import MaxNLocator # type: ignore
import gvar as gv # type: ignore
import lsqfit # type: ignore
import logging
import pandas as pd
import h5py
import copy

from library import filesystem_utilities, data_processing, plotting, constants


# Function to calculate average of a list of numbers
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

# Function to convert string representation of list to actual list
def convert_to_list(s):
    return ast.literal_eval(s)

def KL_number_of_MV_multiplications(KL_iterations_array):
    return 2*KL_iterations_array + 1



@click.command()
@click.option("--input_csv_file_path",
              "input_csv_file_path", "-csv",
              default="/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/sign_squared_violation/KL_several_rho_varying_n_and_config/qpb_log_files_single_valued_parameters.csv",
        help="Path to input .csv file containing extracted single-valued parameters.")
@click.option("--input_HDF5_file_path",
              "input_HDF5_file_path", "-h5",
              default="/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/sign_squared_violation/KL_several_rho_varying_n_and_config/qpb_log_files_multivalued_parameters.h5",
        help="Path to input HDF5 file containing extracted multivalued parameters.")
@click.option("--output_files_directory", "output_files_directory", "-out_dir",
              default=None,
              help="Path to directory where all output files will be stored.")
@click.option("--plots_directory", "plots_directory", "-plots_dir",
              default="/nvme/h/cy22sg1/qpb_data_analysis/output/plots/sign_squared_violation/KL_several_rho_varying_n_and_config",
              help="Path to the output directory for storing plots.")
@click.option("--log_file_directory", "log_file_directory", "-log_file_dir", 
              default=None, 
              help="Directory where the script's log file will be stored.")
@click.option("--log_filename", "log_filename", "-log", 
              default="calculate_PCAC_mass_correlator_script.log", 
              help="Specific name for the script's log file.")

def main(input_csv_file_path, input_HDF5_file_path, output_files_directory, plots_directory,
         log_file_directory, log_filename):

    # PERFORM VALIDITY CHECKS ON INPUT ARGUMENTS

    if not filesystem_utilities.is_valid_file(
                            input_csv_file_path):
        error_message = "The specified input .csv file path is invalid."
        print("ERROR:", error_message)
        sys.exit(1)

    if not filesystem_utilities.is_valid_file(
                            input_HDF5_file_path):
        error_message = "The specified input HDF5 file path is invalid."
        print("ERROR:", error_message)
        sys.exit(1)

    # If no output directory is provided, use the directory of the input file
    if output_files_directory is None:
        output_files_directory = os.path.dirname(
                            input_csv_file_path)
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

    # Check for proper extensions in provided output filenames
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

    # CRITICAL BARE MASS CALCULATION
    
    # ANALYZE .CSV FILE

    # Load the CSV file into a DataFrame
    qpb_data_files_set_dataframe = pd.read_csv(input_csv_file_path)

    # Read the header of the .csv file to determine available fields
    with open(input_csv_file_path, 'r') as f:
        csv_header = f.readline().strip().split(',')

    # Filter dtype and converters dictionaries to include only columns present in the .csv file
    filtered_dtype_mapping = {field: dtype for field, dtype in constants.dtype_mapping.items() if field in csv_header}
    filtered_converters_mapping = {field: converter for field, converter in constants.converters_mapping.items() if field in csv_header}

    # Import the .csv file using the filtered dictionaries
    qpb_data_files_set_dataframe = pd.read_csv(
        input_csv_file_path,
        dtype=filtered_dtype_mapping,
        converters=filtered_converters_mapping
    )

    # Extract fields with a single unique value
    fields_with_unique_values_dictionary = (
        data_processing.get_fields_with_unique_values(
            qpb_data_files_set_dataframe)
        )

    # Extract list of fields with multiple unique values excluding specified
    selected_field_name = 'KL_diagonal_order'
    excluded_fields = {"Filename", "Plaquette", selected_field_name}
    list_of_fields_with_multiple_values = (
        data_processing.get_fields_with_multiple_values(
            qpb_data_files_set_dataframe, excluded_fields)
        )

    # Get a list of all unique field values
    unique_combinations = [ qpb_data_files_set_dataframe[field].unique()
                        for field in list_of_fields_with_multiple_values ]

    with h5py.File(input_HDF5_file_path, "r") as hdf5_file_read:
        
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
        
        plots_subdirectory = os.path.join(plots_directory, 'TEST')
        # Create "plots_subdirectory" if it does not exist
        os.makedirs(plots_subdirectory, exist_ok=True)
        
        # Use itertools.product to create all combinations of the unique values
        analysis_index = 0 # Initialize counter
        for combination in itertools.product(*unique_combinations):

            # Create a filter for the current combination
            filters = {field: value for field, value in zip(
                            list_of_fields_with_multiple_values, combination)}

            # Get the subset of the dataframe based on the current combination
            current_combination_group = qpb_data_files_set_dataframe
            for field, value in filters.items():
                current_combination_group = current_combination_group[
                                            current_combination_group[field] == value]

            # Skip empty current_combination_groups (no data for this combination)
            if current_combination_group.empty:
                continue

            analysis_index += 1

            # Now 'group' contains the subset for this combination of values
            list_of_qpb_log_filenames = current_combination_group[
                                                        "Filename"].tolist()

            number_of_gauge_configurations = len(list_of_qpb_log_filenames)

            calculation_results_array = np.array([
                input_data_files_set_group[qpb_log_filename]['Calculation_result'][:][0]
                for qpb_log_filename in list_of_qpb_log_filenames
            ])

            print(calculation_results_array)

            selected_field_values_array = current_combination_group[selected_field_name].to_numpy()

            # Plot against n for each configuration
            fig, ax = plt.subplots()
            ax.grid()

            # ax.set_title(f'Sign squared violation against KL iterations ({operator_type}, $\\beta$={unique_QCD_beta_value},\n$c_{{SW}}$={unique_clover_coefficient}, $\\rho$={rho_value}, APE iters={unique_APE_iterations}, $\\mu$={unique_mu_value}, $\\epsilon_{{CG}}$={CG_epsilon}, # of configs: {number_of_configurations})', pad=7)
            ax.set(xlabel='n', ylabel='|| Sign$^2$(X) - 1 ||$^2$')
            plt.subplots_adjust(left=0.13) # Adjust left margin
            ax.set_yscale('log')

            # plt.axhline(y=1e-5, color='green', linestyle='--')
            # plt.axhline(y=1e-10, color='black', linestyle='--')

            # Set x-axis ticks to integer values only
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            ax.scatter(selected_field_values_array, calculation_results_array)

#             # sorted_rho_value_group = rho_value_group.sort_values(by='Condition_number', ascending=False)
#             for configuration_label, configuration_label_group in sorted_rho_value_group.groupby('Configuration_label', sort=False):

#                 KL_iterations_array = np.array(configuration_label_group['KL_iterations'])
#                 sign_squared_violation_array = np.array(configuration_label_group['Sign_squared_violation'])

#                 condition_number = configuration_label_group['Condition_number'].unique()[0]

#                 ax.scatter(KL_iterations_array, sign_squared_violation_array
# #                         #    , label=f'$\\alpha^2$={minimum_eigenvalue**2:.5f}, $\\beta^2$={maximum_eigenvalue**2:.4f}, $\\kappa$={kappa:.2f}, config: {Configuration_label}')
#                             # , label=f'config: {Configuration_label}')
#                         , label=f'config: {configuration_label}, $\\kappa$={condition_number:.2f}')
#                         # , label=f'$\\kappa$={condition_number:.2f}')

#             # Extend the axes range
#             # ax.set_ylim([1e-15, 1e1])  # Adjust the range as needed
            # ax.set_xlim([0, 30])  # Adjust the range as needed

            # ax.legend(loc="upper right", title="Condition number:")

            plot_path = os.path.join(plots_subdirectory, f'test_{analysis_index}'+'.png')
            fig.savefig(plot_path)
            plt.close()

    print("   -- analysis completed.")

    # Terminate logging
    logging.info(f"Script '{script_name}' execution terminated successfully.")


if __name__ == "__main__":
    main()

# TODO: Include proper logging!