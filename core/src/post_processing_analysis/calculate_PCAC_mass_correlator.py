import os
import sys
import itertools

import click  # type: ignore
import numpy as np
import gvar as gv  # type: ignore
import pandas as pd
import ast
import logging
import h5py

from library import momentum_correlator
from library import jackknife_analysis
from library import filesystem_utilities
from library import data_processing


@click.command()
@click.option("--input_qpb_log_files_csv_file_path",
              "input_qpb_log_files_csv_file_path", "-log_csv", default=None,
              help="Path to input .csv file containing extracted info from "\
                                                        "qpb log files sets.")
@click.option("--input_correlator_values_hdf5_file_path",
              "input_correlator_values_hdf5_file_path", "-cor_hdf5",
              default=None,
        help="Path to input HDF5 file containing extracted correlators values.")
@click.option("--output_files_directory", "output_files_directory", "-out_dir",
              default=None,
              help="Path to directory where all output files will be stored.")
@click.option("--output_hdf5_filename", "output_hdf5_filename",
              "-hdf5", default="PCAC_mass_correlator_values.h5",
              help="Specific name for the output HDF5 file.")
@click.option("--log_file_directory", "log_file_directory", "-log_file_dir", 
              default=None, 
              help="Directory where the script's log file will be stored.")
@click.option("--log_filename", "log_filename", "-log", 
              default="calculate_PCAC_mass_correlator_script.log", 
              help="Specific name for the script's log file.")

def main(input_qpb_log_files_csv_file_path, 
        input_correlator_values_hdf5_file_path, output_files_directory,
                    output_hdf5_filename, log_file_directory, log_filename):

    # PERFORM VALIDITY CHECKS ON INPUT ARGUMENTS

    if not filesystem_utilities.is_valid_file(input_qpb_log_files_csv_file_path):
        error_message = "Passed qpb log files .csv file path is invalid!."
        print("ERROR:", error_message)
        sys.exit(1)

    if not filesystem_utilities.is_valid_file(input_correlator_values_hdf5_file_path):
        error_message = "Passed correlator values HDF5 file path is invalid!."
        print("ERROR:", error_message)
        sys.exit(1)

    # If no output directory is provided, use the directory of the input file
    if output_files_directory is None:
        output_files_directory = os.path.dirname(input_correlator_values_hdf5_file_path)
    # Check validity if the provided
    elif not filesystem_utilities.is_valid_file(output_files_directory):
        error_message = (
            "Passed output files directory path is invalid " "or not a directory."
        )
        print("ERROR:", error_message)
        print("Exiting...")
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
    if not output_hdf5_filename.endswith(".h5"):
        output_hdf5_filename = output_hdf5_filename + ".h5"
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

    # PCAC MASS CORRELATOR VALUES CALCULATION

    # Construct output HDF5 file path
    output_PCAC_mass_correlator_hdf5_file_path = os.path.join(
                                output_files_directory, output_hdf5_filename)

    # Open the input HDF5 file for reading and the output HDF5 file for writing
    with h5py.File(input_correlator_values_hdf5_file_path, "r") \
        as hdf5_file_read, h5py.File(
            output_PCAC_mass_correlator_hdf5_file_path, "w") as hdf5_file_write:

        # Initialize group structure of the output HDF5 file
        # NOTE: The assumption here is that the name of the raw data files
        # directory represents the data files set (or experiment) and its parent
        # directory the qpb main program that generated the data files
        parent_directory_name, last_directory_name = (
                                filesystem_utilities.extract_directory_names(
                                    output_files_directory)
                                    )
        qpb_main_program_group = hdf5_file_write.create_group(
                                                        parent_directory_name)
        data_files_set_group = qpb_main_program_group.create_group(
                                                        last_directory_name)

        # Select input HDF5 file's group to read
        input_qpb_main_program_group = hdf5_file_read[parent_directory_name]
        input_data_files_set_group = input_qpb_main_program_group[
                                                        last_directory_name]
        
        # ANALYZE .CSV FILE

        # Load the CSV file into a DataFrame
        qpb_log_files_dataframe = pd.read_csv(input_qpb_log_files_csv_file_path)

        # Extract fields with a single unique value
        fields_with_unique_values_dictionary = (
            data_processing.get_fields_with_unique_values(
                qpb_log_files_dataframe)
            )

        # Extract a list of fields with a multiple unique values
        excluded_fields = {"Filename", "Plaquette", "Configuration_label"}
        list_of_fields_with_multiple_values = (
            data_processing.get_fields_with_multiple_values(
                qpb_log_files_dataframe, excluded_fields)
            )

        # Get a list of all unique field values
        unique_combinations = [ qpb_log_files_dataframe[field].unique()
                            for field in list_of_fields_with_multiple_values ]

        # Use itertools.product to create all combinations of these unique
        # values
        for analysis_index, combination in enumerate(
                                    itertools.product(*unique_combinations)):
            # Create a filter for the current combination
            filters = {field: value for field, value in zip(
                            list_of_fields_with_multiple_values, combination)}

            # Get the subset of the dataframe based on the current combination
            dataframe_group = qpb_log_files_dataframe
            for field, value in filters.items():
                dataframe_group = dataframe_group[
                                            dataframe_group[field] == value]

            # Skip empty dataframe_groups (no data for this combination)
            if dataframe_group.empty:
                continue

            # Now 'group' contains the subset for this combination of values
            list_of_qpb_log_filenames = dataframe_group["Filename"].tolist()

            number_of_gauge_configurations = len(list_of_qpb_log_filenames)

            # Define a unique name for each top-level HDF5 group
            PCAC_mass_correlator_analysis_group_name = (
                        f"PCAC_mass_correlator_analysis_{analysis_index+1}")
            PCAC_mass_correlator_hdf5_group = data_files_set_group.create_group(
                                    PCAC_mass_correlator_analysis_group_name)

            # Add attributes to top-level HDF5 groups
            for key, value in fields_with_unique_values_dictionary.items():
                PCAC_mass_correlator_hdf5_group.attrs[key] = value
            for key, value in filters.items():
                PCAC_mass_correlator_hdf5_group.attrs[key] = value

            # Pass "g5-g5" and "g4g5-g5" datasets, corresponding to different
            # gauge links configuration files, to respective lists common for
            # the current grouping of parameters
            g5_g5_correlator_values_per_configuration_list = []
            g4g5_g5_correlator_values_per_configuration_list = []

            for qpb_log_filename in list_of_qpb_log_filenames:

                correlators_file_name = qpb_log_filename.replace(".txt", ".dat")

                filename_group = input_data_files_set_group[
                                                        correlators_file_name]

                if "g5-g5" in filename_group.keys():
                    g5_g5_dataset = filename_group["g5-g5"][:]
                    g5_g5_correlator_values_per_configuration_list.append(
                                                                g5_g5_dataset)

                if "g4g5-g5" in filename_group.keys():
                    g4g5_g5_dataset = filename_group["g4g5-g5"][:]
                    g4g5_g5_correlator_values_per_configuration_list.append(
                                                                g4g5_g5_dataset)

            # Convert the list of 1D arrays into a 2D NumPy array
            g5_g5_correlator_values_per_configuration_2D_array = np.vstack(
                g5_g5_correlator_values_per_configuration_list)
            g4g5_g5_correlator_values_per_configuration_2D_array = np.vstack(
                g4g5_g5_correlator_values_per_configuration_list)

            # JACKKNIFE ANALYSIS OF THE g5-g5

            # Jackknife analysis of the g5-g5 correlator values
            jackknife_analyzed_g5_g5_correlator_values_per_configuration_object = (
                jackknife_analysis.JackknifeAnalysis(
                    g5_g5_correlator_values_per_configuration_2D_array
                )
            )

            # Jackknife samples of the g5-g5 correlators
            jackknife_samples_of_g5_g5_correlator_2D_array = (
                jackknife_analyzed_g5_g5_correlator_values_per_configuration_object.jackknife_replicas_of_original_2D_array
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_samples_of_g5_g5_correlator_2D_array",
                    data=jackknife_samples_of_g5_g5_correlator_2D_array).attrs[
                        'Description'] = (
                            "Jackknife samples.")

            # Jackknife average of the g5-g5 correlators
            jackknife_average_of_g5_g5_correlator = (
                jackknife_analyzed_g5_g5_correlator_values_per_configuration_object.jackknife_average
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_average_of_g5_g5_correlator_mean_values",
                        data=gv.mean(jackknife_average_of_g5_g5_correlator)).attrs[
                        'Description'] = (
                "Average from Jackknife samples. Mean values.")
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_average_of_g5_g5_correlator_error_values",
                        data=gv.sdev(jackknife_average_of_g5_g5_correlator)).attrs[
                        'Description'] = (
                "Average from Jackknife samples. Error values.")

            # JACKKNIFE ANALYSIS OF THE g4g5-g5 PION CORRELATORS

            # Jackknife analysis of the g4g5-g5 correlator values
            jackknife_analyzed_g4g5_g5_correlator_values_per_configuration_object = (
                jackknife_analysis.JackknifeAnalysis(
                    g4g5_g5_correlator_values_per_configuration_2D_array
                )
            )

            # Jackknife samples of the g4g5-g5 correlators
            jackknife_samples_of_g4g5_g5_correlator_2D_array = (
                jackknife_analyzed_g4g5_g5_correlator_values_per_configuration_object.jackknife_replicas_of_original_2D_array
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_samples_of_g4g5_g5_correlator_2D_array",
                        data=jackknife_samples_of_g4g5_g5_correlator_2D_array).attrs[
                        'Description'] = (
                "Jackknife samples.")

            # Jackknife average of the g4g5-g5 correlators
            jackknife_average_of_g4g5_g5_correlator = (
                jackknife_analyzed_g4g5_g5_correlator_values_per_configuration_object.jackknife_average
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_average_of_g4g5_g5_correlator_mean_values",
                        data=gv.mean(jackknife_average_of_g4g5_g5_correlator)).attrs[
                        'Description'] = (
                "Average from Jackknife samples. Mean values.")
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_average_of_g4g5_g5_correlator_error_values",
                        data=gv.sdev(jackknife_average_of_g4g5_g5_correlator)).attrs[
                        'Description'] = (
                "Average from Jackknife samples. Error values.")

            # PCAC MASS CORRELATOR CALCULATION

            """ 
            NOTE: The PCAC mass is defined as the ratio of the g4g5-g5
            derivative correlator values over the g5-g5 correlator values
            """

            # Jackknife samples of the g4g5-g5 derivate correlators
            jackknife_samples_of_g4g5_g5_derivative_correlator_2D_list = list()
            for index in range(len(jackknife_samples_of_g4g5_g5_correlator_2D_array)):
                jackknife_samples_of_g4g5_g5_derivative_correlator_2D_list.append(
                    momentum_correlator.centered_difference_correlator_derivative(
                        jackknife_samples_of_g4g5_g5_correlator_2D_array[index]
                    )
                )
            jackknife_samples_of_g4g5_g5_derivative_correlator_2D_array = np.array(
                jackknife_samples_of_g4g5_g5_derivative_correlator_2D_list
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_samples_of_g4g5_g5_derivative_correlator_2D_array",
            data=jackknife_samples_of_g4g5_g5_derivative_correlator_2D_array).attrs[
                        'Description'] = (
                "Centered difference derivative.")

            # g4g5-g5 derivative correlator from the jackknife average of g4g5-g5 correlator
            g4g5_g5_derivative_correlator_from_jackknife_average_of_g4g5_g5_correlator = momentum_correlator.centered_difference_correlator_derivative(
                jackknife_average_of_g4g5_g5_correlator
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "g4g5_g5_derivative_correlator_from_jackknife_average_of_g4g5_g5_correlator_mean_values",
            data=gv.mean(g4g5_g5_derivative_correlator_from_jackknife_average_of_g4g5_g5_correlator)).attrs[
                        'Description'] = (
                "Centered difference derivative from average of g4g5-g5 correlator. Mean values.")
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "g4g5_g5_derivative_correlator_from_jackknife_average_of_g4g5_g5_correlator_error_values",
            data=gv.sdev(g4g5_g5_derivative_correlator_from_jackknife_average_of_g4g5_g5_correlator)).attrs[
                        'Description'] = (
                "Centered difference derivative from average of g4g5-g5 correlator. Error values.")

            # Jackknife samples of the time-dependent PCAC mass values
            jackknife_samples_of_time_dependent_PCAC_mass_values_list = list()
            for index in range(len(jackknife_samples_of_g5_g5_correlator_2D_array)):

                jackknife_sample_of_time_dependent_PCAC_mass_values = (
                    0.5
                    * jackknife_samples_of_g4g5_g5_derivative_correlator_2D_array[index]
                    / jackknife_samples_of_g5_g5_correlator_2D_array[index]
                )

                jackknife_sample_of_time_dependent_PCAC_mass_values = (
                    momentum_correlator.symmetrization(
                        jackknife_sample_of_time_dependent_PCAC_mass_values
                    )
                )

                jackknife_samples_of_time_dependent_PCAC_mass_values_list.append(
                    jackknife_sample_of_time_dependent_PCAC_mass_values
                )
            jackknife_samples_of_time_dependent_PCAC_mass_values_2D_array = np.array(
                jackknife_samples_of_time_dependent_PCAC_mass_values_list
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_samples_of_time_dependent_PCAC_mass_values_2D_array",
            data=jackknife_samples_of_time_dependent_PCAC_mass_values_2D_array).attrs[
                        'Description'] = (
                "Jackknife samples.")

            # Jackknife average of the time-dependent PCAC mass values
            jackknife_average_of_time_dependent_PCAC_mass_values_array = gv.gvar(
                np.average(
                    jackknife_samples_of_time_dependent_PCAC_mass_values_2D_array,
                    axis=0,
                ),
                np.sqrt(number_of_gauge_configurations - 1)
                * np.std(
                    jackknife_samples_of_time_dependent_PCAC_mass_values_2D_array,
                    axis=0,
                    ddof=0,
                ),
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_average_of_time_dependent_PCAC_mass_values_array_mean_values",
            data=gv.mean(jackknife_average_of_time_dependent_PCAC_mass_values_array)).attrs[
                        'Description'] = (
                "Average from Jackknife samples. Mean values.")
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_average_of_time_dependent_PCAC_mass_values_array_error_values",
            data=gv.sdev(jackknife_average_of_time_dependent_PCAC_mass_values_array)).attrs[
                        'Description'] = (
                "Average from Jackknife samples. Error values.")
            
            # Time-dependent PCAC mass values from the jackknife averages of the correlators
            time_dependent_PCAC_mass_values_from_jackknife_averages_of_correlators_array = (
                0.5
                * g4g5_g5_derivative_correlator_from_jackknife_average_of_g4g5_g5_correlator
                / jackknife_average_of_g5_g5_correlator
            )
            time_dependent_PCAC_mass_values_from_jackknife_averages_of_correlators_array = momentum_correlator.symmetrization(
                time_dependent_PCAC_mass_values_from_jackknife_averages_of_correlators_array
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "time_dependent_PCAC_mass_values_from_jackknife_averages_of_correlators_array_mean_values",
            data=gv.mean(time_dependent_PCAC_mass_values_from_jackknife_averages_of_correlators_array)).attrs[
                        'Description'] = (
                "Average from average correlators. Mean values.")
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "time_dependent_PCAC_mass_values_from_jackknife_averages_of_correlators_array_error_values",
            data=gv.sdev(time_dependent_PCAC_mass_values_from_jackknife_averages_of_correlators_array)).attrs[
                        'Description'] = (
                "Average from average correlators. Error values.")

    print("   -- PCAC mass correlator values jackknife analysis completed.")

    # Terminate logging
    logging.info(f"Script '{script_name}' execution terminated successfully.")


if __name__ == "__main__":
    main()
