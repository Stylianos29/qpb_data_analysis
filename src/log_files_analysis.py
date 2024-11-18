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
import re
import sys

import click # type: ignore
import numpy as np
import pandas as pd
import logging


sys.path.append('../')
from library import filesystem_utilities
from library import constants


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

    # SET CURRENT SCRIPT'S LOG FILE DIRECTORY AND NAME

    log_file_directory = output_files_directory
    log_filename = os.path.basename(output_files_directory)+".log"
    filesystem_utilities.setup_logging(log_file_directory, log_filename)


if __name__ == "__main__":
    main()
