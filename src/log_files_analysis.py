"""
Log Files Analysis Script

This script processes log files stored in a specified directory passed by the
user and extracts relevant parameter values from the files contents and
filenames. It then compiles these extracted values into a Pandas DataFrame and
exports the data to a CSV file.

Input: 1. `qpb_log_files_directory`: Directory containing the log files to be
analyzed. 2. `output_files_directory`: Directory where the generated output CSV
file will  be stored. 3. `filename_analysis`: Flag to perform analysis on
filenames of the log files. 4. `append_to_dataframe`: Flag to append new data to
an existing CSV file instead
  of creating a new one.

Output: 1. A CSV file containing the extracted parameter values from the log
files. 2. A log file

Functionality: 1. The script uses regex patterns to extract specific parameter
values from both the contents and filenames of the log files. 2. It performs
preliminary checks to ensure the log files contain necessary parameter values.
3. Extracted values are stored in a Pandas DataFrame. 4. The DataFrame is then
exported to a CSV file. If the append flag is set, new data is appended to the
existing CSV file.

Usage: - Run the script with the appropriate options to specify the log files
directory, output directory, and flags for filename analysis and appending data.
"""

import glob
import os
import sys

import click # type: ignore
import numpy as np
import pandas as pd
import logging
import h5py

sys.path.append('../')
from library import extraction, filesystem_utilities


@click.command()
@click.option("--qpb_log_files_directory", "qpb_log_files_directory",
                "-qpb_log_dir", default=None,
                help="Directory where the log files to be analyzed are stored.")
@click.option("--output_files_directory", "output_files_directory", "-out_dir",
              default=None,
        help="Directory where all the generated output files will be stored.")
# TODO: Add additional options for source script's log file directory and name

def main(qpb_log_files_directory, output_files_directory):
    
    # PERFORM VALIDITY CHECKS ON INPUT ARGUMENTS

    if not filesystem_utilities.is_valid_directory(qpb_log_files_directory):
        error_message = "Passed log files directory path is invalid or not "\
                                                                  "a directory."
        print("ERROR:", error_message)
        sys.exit(1)

    if not filesystem_utilities.is_valid_directory(output_files_directory):
        error_message = "Passed output files directory path is invalid or "\
                                                              "not a directory."
        print("ERROR:", error_message)
        sys.exit(1)

    script_name = os.path.basename(__file__)  # Get the script's filename
    output_files_name = os.path.basename(output_files_directory)

    # SET CURRENT SCRIPT'S LOG FILE DIRECTORY AND NAME

    log_file_directory = output_files_directory
    log_filename = output_files_name+".log"
    filesystem_utilities.setup_logging(log_file_directory, log_filename)

    logging.info(f"Script '{script_name}' execution initiated.")
    logger = logging.getLogger(__name__)

    # EXTRACT SINGLE-VALUED PARAMETERS
    
    # List to pass values to dataframe
    parameters_values_list = list()
    # Loop over all .txt files in the QPB log files directory
    for qpb_log_file_full_path in glob.glob(
                            os.path.join(qpb_log_files_directory, "*.txt")):
        # Extract the filename from the full path
        qpb_log_filename = os.path.basename(qpb_log_file_full_path)

        # Initialize a dictionary to store extracted parameter values
        extracted_values_dictionary = dict()

        # Add the filename to the dictionary
        extracted_values_dictionary["Filename"] = qpb_log_filename

        # Extract parameter values from the filename
        extracted_values_from_filename_dictionary = \
            extraction.extract_parameters_values_from_filename(qpb_log_filename,
                                                                logger)

        # Read the contents of the log file into a list of lines
        with open(qpb_log_file_full_path, "r") as file:
            qpb_log_file_contents_list = file.readlines()
        
        # Extract parameter values from the contents of the file
        extracted_single_valued_parameters_from_file_contents_dictionary = \
            extraction.extract_parameters_values_from_file_contents(
                                            qpb_log_file_contents_list, logger)

        # Merge extracted values from file contents into the main dictionary
        # File contents are considered the primary source of truth
        extracted_values_dictionary.update(
            extracted_single_valued_parameters_from_file_contents_dictionary)

        # Compare and update the dictionary with values from the filename
        for key, value in extracted_values_from_filename_dictionary.items():
            if key in extracted_values_dictionary:
                # If the key exists in both dictionaries, compare their values
                file_contents_value = extracted_values_dictionary[key]
                if file_contents_value != value:
                    # Log a warning if values differ, favoring file contents
                    if logger:
                        logger.warning(
                            f"Discrepancy for parameter '{key}': "
                            f"filename value='{value}' "
                            f"vs file contents value='{file_contents_value}'. "
                            "Using the value from file contents."
                        )
            else:
                # Add parameters that exist only in filename to the dictionary
                # TODO: This must be accompanied by appropriate checks
                extracted_values_dictionary[key] = value

        # Append the fully constructed dictionary to the list of parameters
        parameters_values_list.append(extracted_values_dictionary)

    # Convert the list of parameter dictionaries into a Pandas DataFrame
    parameter_values_dataframe = pd.DataFrame(parameters_values_list)

    # Save the DataFrame to a CSV file in the output directory
    csv_file_full_path = os.path.join(output_files_directory, 
                                      output_files_name + ".csv")

    # Export dataframe to .cvs file
    parameter_values_dataframe.to_csv(csv_file_full_path, index=False)
    logging.info(f"Extracted single-valued parameters passed to "\
                                        f"{output_files_name}.csv file.")

    # EXTRACT MULTIVALUED PARAMETERS

    hdf5_file_full_path = os.path.join(output_files_directory, 
                                      output_files_name + ".h5")
    with h5py.File(hdf5_file_full_path, 'w') as hdf5_file:

        for qpb_log_file_full_path in glob.glob(
                                os.path.join(qpb_log_files_directory, "*.txt")):
            # Extract the filename from the full path
            qpb_log_filename = os.path.basename(qpb_log_file_full_path)

            # Create a group for the current file (based on the filename)
            file_group = hdf5_file.create_group(qpb_log_filename)

            # Extract multivalued parameters from the contents of the file
            extracted_multivalued_parameters_from_file_contents_dictionary = \
                extraction.extract_multivalued_parameters_from_file_contents(
                                            qpb_log_file_contents_list, logger)
            
            # Loop through each multivalued parameter and store it as a dataset
            for parameter, values in \
        extracted_multivalued_parameters_from_file_contents_dictionary.items():
                # Create a dataset for each parameter in the file
                file_group.create_dataset(parameter, data=values)

    logging.info(f"Extracted multivalued parameters passed to "\
                                        f"{output_files_name}.h5 file.")

    # Log termination of the script
    logging.info(f"Script '{script_name}' execution terminated successfully.")


if __name__ == "__main__":
    main()
