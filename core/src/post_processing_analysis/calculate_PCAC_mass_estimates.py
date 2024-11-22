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

from library import filesystem_utilities
import library as effective_mass


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

    print("   -- PCAC mass estimates calculation completed.")

    # Terminate logging
    logging.info(f"Script '{script_name}' execution terminated successfully.")


if __name__ == "__main__":
    main()
