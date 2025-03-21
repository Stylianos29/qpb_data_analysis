# TODO: Revisit this detailed commentary
"""
Log Files Analysis Script

This script processes log files stored in a specified directory passed by the
user and extracts relevant parameter values from the files contents and
filenames. It then compiles these extracted values into a Pandas DataFrame and
exports the data to a CSV file.

Input: 1. `qpb_log_files_csv_file`: Directory containing the log files to be
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

import click
import pandas as pd
import logging
import h5py

from library import extraction, filesystem_utilities, data_processing, constants


@click.command()
@click.option(
    "--qpb_log_files_csv_file",
    "qpb_log_files_csv_file",
    "-qpb_log_dir",
    default="/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/invert/KL_several_config_and_mu_varying_m/single_valued_parameters_values.csv",
    help="Directory where the log files to be analyzed are stored.",
)
@click.option(
    "--output_files_directory",
    "output_files_directory",
    "-out_dir",
    default="/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/invert/KL_several_config_and_mu_varying_m/auxiliary_files",
    help="Directory where all the generated output files will be stored.",
)
@click.option(
    "--output_csv_filename",
    "output_csv_filename",
    "-out_csv_name",
    default="qpb_log_files_single_valued_parameters.csv",
    help="Specific name for the qpb log files .csv output file.",
)
@click.option(
    "--output_hdf5_filename",
    "output_hdf5_filename",
    "-out_hdf5_name",
    default="qpb_log_files_multivalued_parameters.h5",
    help="Specific name for the qpb log files HDF5 output file.",
)
@click.option(
    "--log_file_directory",
    "log_file_directory",
    "-log_file_dir",
    default="/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/invert/KL_several_config_and_mu_varying_m/auxiliary_files",
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
    qpb_log_files_csv_file,
    output_files_directory,
    output_csv_filename,
    output_hdf5_filename,
    log_file_directory,
    log_filename,
):

    # VALIDATE INPUT ARGUMENTS

    if not filesystem_utilities.is_valid_file(qpb_log_files_csv_file):
        error_message = (
            "Passed log files directory path is invalid or not " "a directory."
        )
        print("ERROR:", error_message)
        print("Exiting...")
        sys.exit(1)

    if not filesystem_utilities.is_valid_directory(output_files_directory):
        error_message = (
            "Passed output files directory path is invalid or " "not a directory."
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

    # Get the script's filename
    script_name = os.path.basename(__file__)

    if log_filename is None:
        log_filename = script_name.replace(".py", "_python_script.log")

    # Check for proper extensions in provided filenames
    if not output_csv_filename.endswith(".csv"):
        output_csv_filename = output_csv_filename + ".csv"
    if not output_hdf5_filename.endswith(".h5"):
        output_hdf5_filename = output_hdf5_filename + ".h5"
    if not log_filename.endswith(".log"):
        log_filename = log_filename + ".log"

    # INITIATE LOGGING

    filesystem_utilities.setup_logging(log_file_directory, log_filename)

    # Create a logger instance for the current script using the script's name
    logger = logging.getLogger(__name__)

    # Initiate logging
    logging.info(f"Script '{script_name}' execution initiated.")

    # REMOVE DUPLICATES

    dataframe = data_processing.load_csv(qpb_log_files_csv_file)

    # # Create a temporary DataFrame without the "Filename" column
    # temp_df = dataframe.drop(columns=["Filename"])

    # # Add a new column to identify duplicate groups by their hashable content
    # dataframe["Duplicate_Group_Key"] = temp_df.apply(tuple, axis=1)

    # # Group by the "Duplicate_Group_Key" column to find duplicates
    # duplicate_groups = dataframe.groupby("Duplicate_Group_Key").filter(
    #     lambda x: len(x) > 1
    # )

    # # Create a list of lists of filenames for each duplicate group
    # duplicate_filenames = (
    #     duplicate_groups.groupby("Duplicate_Group_Key")["Filename"].apply(list).tolist()
    # )

    # # Directory containing the raw data files
    # raw_data_files_set_directory = "path/to/raw_data_files_set_directory"

    # # Ask the user if they want to delete the redundant duplicate files
    # user_input = (
    #     input("Do you want to delete the redundant duplicate files? (yes/no): ")
    #     .strip()
    #     .lower()
    # )

    # # Treat Enter (empty input) as "yes"
    # if user_input in {"yes", "y", ""}:  # Empty string "" is treated as "yes"
    #     for duplicates in duplicate_filenames:
    #         # Retain the first file and delete the rest
    #         redundant_files = duplicates[1:]  # Skip the first file

    #         for file in redundant_files:
    #             file_path = os.path.join(raw_data_files_set_directory, file)
    #             try:
    #                 if os.path.exists(file_path):
    #                     os.remove(file_path)
    #                     print(f"Deleted: {file_path}")
    #                 else:
    #                     print(f"File not found, skipping: {file_path}")
    #             except Exception as e:
    #                 print(f"Error deleting {file_path}: {e}")
    # else:
    #     print("No files were deleted.")

    # CHECK FILES NAMING

    analyzer = data_processing.DataFrameAnalyzer(dataframe)

    # Filter the dictionary to retain only keys present in the list
    filtered_dictionary = {
        key: value
        for key, value in analyzer.multivalued_fields_dictionary.items()
        if key in analyzer.list_of_tunable_parameter_names_from_dataframe
    }

    # Sort the dictionary by its values in ascending order
    sorted_items = sorted(filtered_dictionary.items(), key=lambda item: item[1])

    sorted_dictionary = dict(sorted_items)

    # Result as a list of tuples
    # print(sorted_dictionary)

    sorted_fields_list = list(sorted_dictionary.keys())

    if ("Kernel_operator_type" not in sorted_fields_list) and (
        "Kernel_operator_type" in analyzer.list_of_dataframe_fields
    ):
        sorted_fields_list.insert(0, "Kernel_operator_type")

    if "Overlap_operator_method" in analyzer.list_of_dataframe_fields:
        sorted_fields_list.insert(0, "Overlap_operator_method")

    # # Create the "test_string" field
    # dataframe["test_string"] = dataframe[sorted_fields_list].apply(
    #     lambda row: "_".join(map(str, row)), axis=1
    # )

    def construct_test_string(row):
        parts = []
        for field in sorted_fields_list:
            value = row[field]
            label = constants.PARAMETERS_PRINTED_LABELS_DICTIONARY.get(field, "")
            if label:
                parts.append(f"{label}{value}")
            else:
                parts.append(f"{value}")
        return "_".join(parts).replace(" ", "")

    # Apply the function to create the "test_string" column
    dataframe["test_string"] = dataframe.apply(construct_test_string, axis=1)

    # Print a few values to verify
    print(dataframe["test_string"].head())

    # Overlap_operator_method

    # Terminate logging
    logging.info(f"Script '{script_name}' execution terminated successfully.")

    click.echo("   -- Inspecting data files set naming completed.")


if __name__ == "__main__":
    main()
