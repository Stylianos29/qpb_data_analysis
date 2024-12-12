import os
import sys
import itertools
import ast

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
              default="/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/invert/Chebyshev_several_m_varying_EpsLanczos/PCAC_mass_estimates.csv",
        help="Path to input .csv file containing extracted single-valued parameters.")
@click.option("--input_HDF5_file_path",
              "input_HDF5_file_path", "-h5",
              default="/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/sign_squared_values/Chebyshev_several_configs_varying_EpsLanczos/qpb_log_files_multivalued_parameters.h5",
        help="Path to input HDF5 file containing extracted multivalued parameters.")
@click.option("--output_files_directory", "output_files_directory", "-out_dir",
              default=None,
              help="Path to directory where all output files will be stored.")
@click.option("--plots_directory", "plots_directory", "-plots_dir",
              default="/nvme/h/cy22sg1/qpb_data_analysis/output/plots/invert/Chebyshev_several_m_varying_EpsLanczos",
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
        output_files_directory = os.path.dirname(input_csv_file_path)
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

    # ? CALCULATION
    
    # ANALYZE .CSV FILE

    # Load the CSV file into a DataFrame
    qpb_data_files_set_dataframe = pd.read_csv(input_csv_file_path)

    # Read the header of the .csv file to determine available fields
    with open(input_csv_file_path, 'r') as f:
        csv_header = f.readline().strip().split(',')

    # Filter dtype and converters dictionaries to include only columns present in the .csv file
    filtered_dtype_mapping = {field: dtype for field, dtype in constants.DTYPE_MAPPING.items() if field in csv_header}
    filtered_converters_mapping = {field: converter for field, converter in constants.CONVERTERS_MAPPING.items() if field in csv_header}

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

    output_values_list=[
        'Plaquette',
        'Minimum_eigenvalue_squared',
        'Maximum_eigenvalue_squared',
        'Total_overhead_time',
        'Total_number_of_Lanczos_iterations',
        'PCAC_mass_estimate'
        ]

    # Extract list of fields with multiple unique values excluding specified
    selected_field_name = ['Lanczos_epsilon']
    excluded_fields = {"Filename", 'Lanczos_epsilon', *output_values_list, *selected_field_name}

    list_of_fields_with_multiple_values = (
        data_processing.get_fields_with_multiple_values(
            qpb_data_files_set_dataframe, excluded_fields)
        )
    
    # Get a list of all unique field values
    unique_combinations = [ qpb_data_files_set_dataframe[field].unique()
                        for field in list_of_fields_with_multiple_values ]
    
    plots_subdirectory = os.path.join(plots_directory, 'PCAC_mass_estimate_Vs_Lanczos_epsilon')
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
        Lanczos_epsilon_values_array = current_combination_group['Lanczos_epsilon'].to_numpy()
        maximum_eigenvalue_squared_values_array = gv.gvar(current_combination_group['PCAC_mass_estimate'].to_numpy())

        # Plot against n for each configuration
        fig, ax = plt.subplots()
        ax.grid()

        # ax.set_title(f'Sign squared violation against KL iterations ({operator_type}, $\\beta$={unique_QCD_beta_value},\n$c_{{SW}}$={unique_clover_coefficient}, $\\rho$={rho_value}, APE iters={unique_APE_iterations}, $\\mu$={unique_mu_value}, $\\epsilon_{{CG}}$={CG_epsilon}, # of configs: {number_of_configurations})', pad=7)
        # ax.set(xlabel='n', ylabel='|| Sign$^2$(X) - 1 ||$^2$')
        plt.subplots_adjust(left=0.13) # Adjust left margin
        ax.set_xscale('log')
        # ax.set_yscale('log')

        ax.axhline(0, color='black') # y = 0

        # ax.scatter(Lanczos_epsilon_values_array, maximum_eigenvalue_squared_values_array)

        ax.errorbar(Lanczos_epsilon_values_array, gv.mean(maximum_eigenvalue_squared_values_array), yerr=gv.sdev(maximum_eigenvalue_squared_values_array), fmt='.', markersize=8, capsize=10
                        # , label=f'# of configs: {number_of_gauge_configurations}'
                        )

        plot_path = os.path.join(plots_subdirectory, f'test_{analysis_index}'+'.png')
        fig.savefig(plot_path)
        plt.close()

        # # Now 'group' contains the subset for this combination of values
        # Lanczos_epsilon_values_array = current_combination_group['Lanczos_epsilon'].to_numpy()
        # minimum_eigenvalue_squared_values_array = current_combination_group['Minimum_eigenvalue_squared'].to_numpy()

        # # Plot against n for each configuration
        # fig, ax = plt.subplots()
        # ax.grid()

        # # ax.set_title(f'Sign squared violation against KL iterations ({operator_type}, $\\beta$={unique_QCD_beta_value},\n$c_{{SW}}$={unique_clover_coefficient}, $\\rho$={rho_value}, APE iters={unique_APE_iterations}, $\\mu$={unique_mu_value}, $\\epsilon_{{CG}}$={CG_epsilon}, # of configs: {number_of_configurations})', pad=7)
        # # ax.set(xlabel='n', ylabel='|| Sign$^2$(X) - 1 ||$^2$')
        # plt.subplots_adjust(left=0.13) # Adjust left margin
        # ax.set_xscale('log')
        # # ax.set_yscale('log')

        # ax.scatter(Lanczos_epsilon_values_array, minimum_eigenvalue_squared_values_array)

        # plot_path = os.path.join(plots_subdirectory, f'minimum_{analysis_index}'+'.png')
        # fig.savefig(plot_path)
        # plt.close()

    print("   -- analysis completed.")

    # Terminate logging
    logging.info(f"Script '{script_name}' execution terminated successfully.")


if __name__ == "__main__":
    main()

# TODO: Include proper logging!