# TODO: Revisit this detailed commentary
"""
Log Files Analysis Script

This script processes log files stored in a specified directory passed by the
user and extracts relevant parameter values from the files contents and
filenames. It then compiles these extracted values into a Pandas DataFrame and
exports the data to a CSV file.

Input: 1. `raw_data_files_set_directory`: Directory containing the log files to be
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

from library import extraction, filesystem_utilities


@click.command()
@click.option(
    "--raw_data_files_set_directory",
    "raw_data_files_set_directory",
    "-qpb_log_dir",
    default=None,
    help="Directory where the log files to be analyzed are stored.",
)
@click.option(
    "--output_files_directory",
    "output_files_directory",
    "-out_dir",
    default=None,
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
    raw_data_files_set_directory,
    output_files_directory,
    output_csv_filename,
    output_hdf5_filename,
    log_file_directory,
    log_filename,
):

    # VALIDATE INPUT ARGUMENTS

    if not filesystem_utilities.is_valid_directory(raw_data_files_set_directory):
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
        log_filename = script_name.replace(".py", ".log")

    # Check for proper extensions in provided filenames
    if not output_csv_filename.endswith(".csv"):
        output_csv_filename = output_csv_filename + ".csv"
    if not output_hdf5_filename.endswith(".h5"):
        output_hdf5_filename = output_hdf5_filename + ".h5"
    if not log_filename.endswith(".log"):
        log_filename = log_filename + ".log"

    # INITIATE LOGGING

    filesystem_utilities.setup_logging(log_file_directory, log_filename)

    # Create a logger instance for the current script using the script's name.
    logger = logging.getLogger(__name__)

    # Initiate logging
    logging.info(f"Script '{script_name}' execution initiated.")

    # MAIN???

    

    # Terminate logging
    logging.info(f"Script '{script_name}' execution terminated successfully.")

    print("   -- Parsing raw qpb log files completed.")

if __name__ == "__main__":
    main()

# TODO: Included a terminating message
