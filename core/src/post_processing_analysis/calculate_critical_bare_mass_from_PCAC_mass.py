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

from library import filesystem_utilities, data_processing, plotting


@click.command()
@click.option("--input_PCAC_mass_estimates_csv_file_path",
              "input_PCAC_mass_estimates_csv_file_path", "-PCAC_csv",
              default=None,
        help="Path to input HDF5 file containing extracted correlators values.")
@click.option("--output_files_directory", "output_files_directory", "-out_dir",
              default=None,
              help="Path to directory where all output files will be stored.")
@click.option("--plots_directory", "plots_directory", "-plots_dir",
              default="../../output/plots",
              help="Path to the output directory for storing plots.")
@click.option("--output_critical_bare_mass_csv_filename",
              "output_critical_bare_mass_csv_filename",
              "-hdf5", default="critical_bare_mass_from_PCAC_mass.csv",
              help="Specific name for the output HDF5 file.")
@click.option("--log_file_directory", "log_file_directory", "-log_file_dir", 
              default=None, 
              help="Directory where the script's log file will be stored.")
@click.option("--log_filename", "log_filename", "-log", 
              default="calculate_PCAC_mass_correlator_script.log", 
              help="Specific name for the script's log file.")

def main(input_PCAC_mass_estimates_csv_file_path,
         output_files_directory, plots_directory,
         output_critical_bare_mass_csv_filename, log_file_directory, log_filename):

    # PERFORM VALIDITY CHECKS ON INPUT ARGUMENTS

    if not filesystem_utilities.is_valid_file(
                            input_PCAC_mass_estimates_csv_file_path):
        error_message = "Passed correlator values HDF5 file path is invalid."
        print("ERROR:", error_message)
        sys.exit(1)

    # If no output directory is provided, use the directory of the input file
    if output_files_directory is None:
        output_files_directory = os.path.dirname(
                            input_PCAC_mass_estimates_csv_file_path)
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
    if not output_critical_bare_mass_csv_filename.endswith(".csv"):
        output_critical_bare_mass_csv_filename = output_critical_bare_mass_csv_filename + ".csv"
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
    PCAC_mass_estimates_dataframe = pd.read_csv(
                                        input_PCAC_mass_estimates_csv_file_path)

    PCAC_mass_estimates_dataframe['PCAC_mass_estimate'] = (
        PCAC_mass_estimates_dataframe['PCAC_mass_estimate'].apply(
                                                        ast.literal_eval))

    # Extract fields with a single unique value
    fields_with_unique_values_dictionary = (
        data_processing.get_fields_with_unique_values(
            PCAC_mass_estimates_dataframe)
        )

    # Extract list of fields with multiple unique values excluding specified
    excluded_fields = {"Bare_mass", "Kappa_value", "PCAC_mass_estimate"}
    list_of_fields_with_multiple_values = (
        data_processing.get_fields_with_multiple_values(
            PCAC_mass_estimates_dataframe, excluded_fields)
        )
    
    # Get a list of all unique field values
    unique_combinations = [ PCAC_mass_estimates_dataframe[field].unique()
                        for field in list_of_fields_with_multiple_values ]

    plots_subdirectory = os.path.join(plots_directory,
                                "PCAC_mass_estimated_Vs_bare_mass_values_plots")
    # Create "plots_nested_subdirectory" if it does not exist
    os.makedirs(plots_subdirectory, exist_ok=True)

    critical_bare_mass_values_list = list()

    # Use itertools.product to create all combinations of the unique values
    analysis_index = 0 # Initialize counter
    for combination in itertools.product(*unique_combinations):

        # Initialize the parameters values dictionary
        parameters_value_dictionary = copy.deepcopy(
                                    fields_with_unique_values_dictionary)

        # Create a filter for the current combination
        filters = {field: value for field, value in zip(
                        list_of_fields_with_multiple_values, combination)}

        # Extract attributes from current analysis group into the dictionary
        for parameter, value in filters.items():
            parameters_value_dictionary[parameter] = value

        # Get the subset of the dataframe based on the current combination
        current_combination_group = PCAC_mass_estimates_dataframe
        for field, value in filters.items():
            current_combination_group = current_combination_group[
                                    current_combination_group[field] == value]

        # Skip empty current_combination_groups (no data for this combination)
        if current_combination_group.empty:
            continue

        analysis_index += 1

        bare_mass_values_array = current_combination_group[
            'Bare_mass'].to_numpy()
        PCAC_mass_estimates_array = gv.gvar(current_combination_group[
            'PCAC_mass_estimate'].to_numpy())
        
        # PCAC MASS VS BARE MASS PLOTS

        fig, ax = plt.subplots()
        ax.grid()
        # plot_title = plotting.construct_plot_title('Jackknife-averaged PCAC mass against bare mass values',
        #                                            )
        # ax.set_title()
        # ax.set_title(f'Jackknife-averaged PCAC mass against bare mass values \n({modified_operator_category_plus_enumeration_label}, # of configs={sample_size}, $\\beta=${BETA_VALUE}, $\\rho$={RHO}, $c_{{SW}}=${C_SW})') # T={TEMPORAL_DIRECTION_LATTICE_SIZE}
        ax.set(xlabel='a$m_{{bare}}$', ylabel='a$m_{PCAC}$')
        plt.subplots_adjust(left=0.14) # Adjust left margin

        ax.axhline(0, color='black') # y = 0
        ax.axvline(0, color='black') # x = 0

        # bare_mass_values_array = np.array(list(jackknife_averaged_PCAC_mass_estimates_per_bare_mass_values_dictionary.keys()))
        # PCAC_mass_estimates_array = np.array(list(jackknife_averaged_PCAC_mass_estimates_per_bare_mass_values_dictionary.values()))

        # # Sort
        # sorted_indices = np.argsort(bare_mass_values_array)
        # bare_mass_values_array = bare_mass_values_array[sorted_indices]
        # PCAC_mass_estimates_array = PCAC_mass_estimates_array[sorted_indices]

        # # # Filter: mPCAC > 0
        # # filtered_indices = np.where(gv.mean(PCAC_mass_estimates_array) > 0)
        # # bare_mass_values_array = bare_mass_values_array[filtered_indices]
        # # PCAC_mass_estimates_array = PCAC_mass_estimates_array[filtered_indices]

        # # # Filter: mPCAC.mean - 3*mPCAC.sdev > 0
        # # filtered_indices = np.where(gv.mean(PCAC_mass_estimates_array) - 3*gv.sdev(PCAC_mass_estimates_array) > 0)
        # # bare_mass_values_array = bare_mass_values_array[filtered_indices]
        # # PCAC_mass_estimates_array = PCAC_mass_estimates_array[filtered_indices]

        # if (len(bare_mass_values_array) < 3):
        #     continue

        # jackknife_samples_of_PCAC_mass_estimates_per_bare_mass_array = np.array(jackknife_samples_of_PCAC_mass_estimates_per_bare_mass_list)
        
        ax.errorbar(bare_mass_values_array, gv.mean(PCAC_mass_estimates_array),
                        yerr=gv.sdev(PCAC_mass_estimates_array),
                            fmt='.', markersize=8, capsize=10)

        # # Linear fit
        # min_index = 0
        # x = bare_mass_values_array[min_index:]
        # y = PCAC_mass_estimates_array[min_index:]
        # # # The initial estimate for the effective mass equals the value
        # slope = (max(gv.mean(y)) - min(gv.mean(y)))/(max(gv.mean(x)) - min(gv.mean(x)))
        # linear_fit_p0 = [slope, min(gv.mean(y))/slope+min(x)]
        # linear_fit = lsqfit.nonlinear_fit(data=(x, y), p0=linear_fit_p0, fcn=fit_functions.linear_function, debug=True)

        # A = np.array([[np.sum(x/np.square(gv.sdev(y))), np.sum(1/np.square(gv.sdev(y)))],
        #             [np.sum(np.square(x)/np.square(gv.sdev(y))), np.sum(x/np.square(gv.sdev(y)))]])

        # # Constants vector
        # B = np.array([np.sum(gv.mean(y)/np.square(gv.sdev(y))), np.sum((gv.mean(y)*x)/np.square(gv.sdev(y)))])
        # C = np.array([np.sum(np.square(gv.sdev(y))/np.square(gv.sdev(y))), np.sum((np.square(gv.sdev(y))*x)/np.square(gv.sdev(y)))])


        # test_slop, test_intercept = np.linalg.solve(A, B)
        # test_slop_error, test_intercept_error = np.linalg.solve(A, C)


        # # print(linear_fit.p)
        # # print(test_slop, -test_intercept/test_slop)
        # # print(gv.gvar(-test_intercept/test_slop, np.sqrt(np.abs(-test_intercept_error/test_slop))))

        # # critical_mass_value = linear_fit.p[1]
        # # critical_mass_value = gv.gvar(-test_intercept/test_slop, np.sqrt(np.abs(-test_intercept_error/test_slop)))
        
        # error = np.sqrt(np.sum(((gv.mean(y) - fit_functions.linear_function(x, gv.mean(linear_fit.p)))/1)**2)/len(y)) #gv.sdev(y)
        # critical_mass_value = gv.gvar(gv.mean(linear_fit.p[1]), error)

        # # print(critical_mass_value)

        # margin = 0.03
        # # min(gv.mean(x))
        # if ('Operator' in operator_category):
        #     x_data = np.linspace(gv.mean(critical_mass_value)*(1+margin), max(gv.mean(x))*(1+margin), 100)
        # else:
        #     x_data = np.linspace(min(gv.mean(x))*(1-margin), max(gv.mean(x))*(1+margin), 100)
        # y_data = fit_functions.linear_function(x_data, linear_fit.p)
        # kappa_critical_linear_fit = 0.5/(critical_mass_value+4)
        # plt.plot(x_data, gv.mean(y_data), 'r--', label=f'linear fit ($\\chi^2$/dof={linear_fit.chi2:.2f}/{linear_fit.dof}={linear_fit.chi2/linear_fit.dof:.4f}):\n$a m_c$={critical_mass_value:.5f}, $\\kappa_c$={kappa_critical_linear_fit:.6f}')
        # ax.fill_between(x_data, gv.mean(y_data) - gv.sdev(critical_mass_value), gv.mean(y_data) + gv.sdev(critical_mass_value), color='r', alpha=0.2)

        # match = re.search(r'[Nn][=_]?(\d+)', operator_enumeration)
        # if match:
        #     operator_enumeration_value = int(match.group(1))

        # if (not 'Operator' in operator_category_label):
        #     critical_bare_mass_values_per_operator_enumeration_dictionary[operator_enumeration_value] = (critical_mass_value, sample_size, len(bare_mass_values_array))
        # # 0.5/(critical_mass_value + 4)

        # for _ in range(len(bare_mass_values_array)):
        #     PCAC_mass_values_dictionary['Critical bare mass value'].append((gv.mean(critical_mass_value), gv.sdev(critical_mass_value)))

        #     PCAC_mass_values_dictionary['Best_fit_line_slope_mPCAC_V_mb'].append((gv.mean(linear_fit.p[0]), gv.sdev(linear_fit.p[0])))

        # ax.legend(loc="upper left")
        
        # plot_filename = f'PCAC_mass_Vs_bare_mass_values_{operator_category_label}.png'
        plot_filename = f'PCAC_mass_Vs_bare_mass_values_{analysis_index}'
        plot_path = os.path.join(plots_subdirectory, plot_filename)

        fig.savefig(plot_path)
        plt.close()

        # parameters_value_dictionary["Critical_bare_mass"] = (
        #         gv.mean(PCAC_mass_estimate), gv.sdev(PCAC_mass_estimate), 
        #     )

        # TODO: Append to parameters value dictionary: critical bare mass, and
        # the arrays of values of the excluded fields

        critical_bare_mass_values_list.append(parameters_value_dictionary)

        # print("   - PCAC Vs bare mass values plot created.")

    # Create a DataFrame from the extracted data
    critical_bare_mass_values_dataframe = pd.DataFrame(
        critical_bare_mass_values_list)

    # Construct output .csv file path
    csv_file_full_path = os.path.join(output_files_directory,
                                        output_critical_bare_mass_csv_filename)
    # Export the DataFrame to a CSV file
    critical_bare_mass_values_dataframe.to_csv(csv_file_full_path, index=False)

    print(
     "   -- Critical bare mass calculation from PCAC mass estimates completed.")

    # Terminate logging
    logging.info(f"Script '{script_name}' execution terminated successfully.")


if __name__ == "__main__":
    main()

# TODO: Include proper logging!