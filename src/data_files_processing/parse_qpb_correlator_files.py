"""
Script Name: parse_qpb_correlator_files.py

Summary:
    This script processes correlator data stored in .dat files within a
    specified directory and converts it into HDF5 format for efficient storage
    and retrieval. It extracts specific correlator values based on predefined
    identifiers and organizes the data into groups within the HDF5 file.

Inputs:
    - raw_data_files_directory (str): Path to the directory containing the input
      .dat files with qpb correlator data.
    - output_files_directory (str): Path to the directory where the output HDF5
      file will be saved.
    - output_hdf5_filename (str): Name of the output HDF5 file (should include
      '.h5' extension).

Outputs:
    - An HDF5 file containing groups for each input .dat file, with datasets
      corresponding to the following correlator identifiers:
        - "1-1"
        - "g5-g5"
        - "g5-g4g5"
        - "g4g5-g5"
        - "g4g5-g4g5"
        - "g1-g1"
        - "g2-g2"
        - "g3-g3"
      Each dataset contains the corresponding correlator values as NumPy arrays.

Usage:
    Run the script from the command line as follows:
        python parse_pion_correlator_files_to_HDF5.py --raw_data_files_directory
        <path_to_data> --output_files_directory <path_to_hdf5>
        --output_hdf5_filename <output_file_name.h5>

Example:
    Given .dat files in the specified data directory, the script will generate
    an HDF5 file in the output directory, structured to allow efficient access
    to the correlator values.
"""

import glob
import os
import sys

import click # type: ignore
import h5py
import numpy as np
import logging

sys.path.append('../')
from library import constants, filesystem_utilities


@click.command()
@click.option("--raw_data_files_directory", "raw_data_files_directory",
              "-raw_dir", default=None,
        help="Directory where the correlator files to be analyzed are stored.")
@click.option("--output_files_directory", "output_files_directory",
              "-out_dir", default=None,
              help="Directory where the HDF5 file will be stored.")
@click.option("--output_hdf5_filename", "output_hdf5_filename",
              "-out_hdf5_file",
              default="qpb_correlators_values.h5",
              help="Name of the output HDF5 file.")
@click.option("--log_file_directory", "log_file_directory", "-log_file_dir", 
              default=None, 
              help="Directory where the script's log file will be stored.")
@click.option("--log_filename", "log_filename", "-log", 
              default="parse_qpb_correlator_files_script.log", 
              help="Specific name for the script's log file.")

def main(raw_data_files_directory, output_files_directory, 
         output_hdf5_filename, log_file_directory, log_filename):
    
    # PERFORM VALIDITY CHECKS ON INPUT ARGUMENTS

    # Check if the provided raw data files directory is valid
    if not filesystem_utilities.is_valid_directory(raw_data_files_directory):
        error_message = (
        "The specified raw data files directory is invalid or does not exist.")
        print("ERROR:", error_message)
        print("Exiting...")
        sys.exit(1)

    # Ensure the directory contains at least one ".dat" file
    if not bool(glob.glob(os.path.join(raw_data_files_directory, "*.dat"))):
        error_message = (
            f"No '.dat' files were found in the specified directory: "
                                            f"'{raw_data_files_directory}'.")
        print("ERROR:", error_message)
        print("Exiting...")
        sys.exit(1)

    # Check if the provided output data files directory is valid
    if not filesystem_utilities.is_valid_directory(output_files_directory):
        # logging.error("The HDF5 file directory path is invalid")
        error_message = ("The specified output data files directory is "
                                                "invalid or does not exist.")
        print("ERROR:", error_message)
        print("Exiting...")
        sys.exit(1)

    # Specify current script's log file directory
    if log_file_directory is None:
        log_file_directory = output_files_directory
    elif not filesystem_utilities.is_valid_directory(log_file_directory):
        error_message = ("The specified directory path to store script's log "
                                        "file is invalid or not a directory.")
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

    # PARSE RAW CORRELATOR CORRELATOR DATA FILES IN SPECIFIED DIRECTORY

    output_hdf5_file_path = os.path.join(output_files_directory, 
                                                        output_hdf5_filename)
    # Open the HDF5 file in 'w' mode (write, replace existing file)
    with h5py.File(output_hdf5_file_path, 'w') as hdf5_file:

        # Initialize group structure of the HDF5 file
        # NOTE: The assumption here is that the name of the raw data files
        # directory represents the data files set (or experiment) and its parent
        # directory the qpb main program that generated the data files
        parent_directory_name, last_directory_name = (
                                filesystem_utilities.extract_directory_names(
                                    raw_data_files_directory)
                                    )
        qpb_main_program_group = hdf5_file.create_group(parent_directory_name)
        data_files_set_group = qpb_main_program_group.create_group(
                                                        last_directory_name)

        # Loop over all .dat files in log files directory
        for data_file_full_path in glob.glob(
                                os.path.join(raw_data_files_directory, "*.dat")):
            data_file = os.path.basename(data_file_full_path)

            # Create a dictionary with correlator identifiers as keys, each
            # initialized to an empty list for storing correlator values.
            correlator_values_dictionary = {}
            for correlator_identifier in constants.CORRELATOR_IDENTIFIERS_LIST:
                correlator_values_dictionary[correlator_identifier] = []

            # Read each correlators values file and fill in the empty lists
            with open(data_file_full_path, 'r') as file:
                lines = file.readlines()

                for line in lines:
                    columns = line.split()

                    # Check and append values for each correlator identifier
                    # NOTE: The identifier is always placed in the last column
                    for correlator_identifier in \
                                        constants.CORRELATOR_IDENTIFIERS_LIST:
                        if columns[-1] == correlator_identifier:
                            correlator_values_dictionary[
                                correlator_identifier].append(float(columns[4]))

            # Convert lists to NumPy arrays for each correlator
            for correlator_identifier in constants.CORRELATOR_IDENTIFIERS_LIST:
                correlator_values_dictionary[correlator_identifier] = np.array(
                            correlator_values_dictionary[correlator_identifier])

            # Create a subgroup in the HDF5 file for this data file
            data_files_group = data_files_set_group.create_group(data_file)

            # Store each correlator array in the correlator group
            for correlator_identifier in constants.CORRELATOR_IDENTIFIERS_LIST:
                data_files_group.create_dataset(correlator_identifier,
                    data=correlator_values_dictionary[correlator_identifier])

    print("     >> Parsing raw correlator files completed.")

    # Terminate logging
    logging.info(f"Script '{script_name}' execution terminated successfully.")


if __name__ == "__main__":
    main()

# TODO: Include logging properly
# TODO: Perform checks on the size of the extracted correlator datasets