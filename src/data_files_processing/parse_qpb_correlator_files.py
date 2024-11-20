"""
Script Name: parse_qpb_correlator_files.py

Summary:
    This script processes correlator data stored in .dat files within a
    specified directory and converts it into HDF5 format for efficient storage
    and retrieval. It extracts specific correlator values based on predefined
    identifiers and organizes the data into groups within the HDF5 file.

Inputs:
    - data_files_directory (str): Path to the directory containing the input
      .dat files with qpb correlator data.
    - hdf5_files_directory (str): Path to the directory where the output HDF5
      file will be saved.
    - output_hdf5_file_name (str): Name of the output HDF5 file (should include
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
        python parse_pion_correlator_files_to_HDF5.py --data_files_directory
        <path_to_data> --hdf5_files_directory <path_to_hdf5>
        --output_hdf5_file_name <output_file_name.h5>

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
@click.option("--data_files_directory", "data_files_directory", "-data_dir",
              default=None,
    help="Directory where the pion correlator files to be analyzed are stored.")
@click.option("--hdf5_files_directory", "hdf5_files_directory", "-hdf5_dir",
              default=None,
              help="Directory where the HDF5 file will be stored.")
@click.option("--output_hdf5_file_name", "output_hdf5_file_name",
              "-out_hdf5_file",
              default="qpb_correlators_values.h5",
              help="Name of the output HDF5 file.")

def main(data_files_directory, hdf5_files_directory, output_hdf5_file_name):
    
    # PERFORM VALIDITY CHECKS ON INPUT ARGUMENTS

    if not filesystem_utilities.is_valid_directory(data_files_directory):
        # logging.error("The log files directory path is invalid")
        error_message = "Passed data files directory path is invalid or not "\
                                                                  "a directory."
        print("ERROR:", error_message)
        sys.exit(1)

    if not bool(glob.glob(os.path.join(data_files_directory, "*.dat"))):
        error_message = f"No '.dat' files found in '{data_files_directory}' "\
                                                        "data files directory."
        print("ERROR:", error_message)
        sys.exit(1)

    if not filesystem_utilities.is_valid_directory(hdf5_files_directory):
        # logging.error("The HDF5 file directory path is invalid")
        error_message = "Passed log files directory path is invalid or not "\
                                                                  "a directory."
        print("ERROR:", error_message)
        sys.exit(1)

    if not output_hdf5_file_name.endswith('.h5'):
        output_hdf5_file_name=output_hdf5_file_name+'.h5'

    hdf5_file_full_path = os.path.join(hdf5_files_directory, 
                                                        output_hdf5_file_name)
    # Open the HDF5 file in 'w' mode (write, replace existing file)
    with h5py.File(hdf5_file_full_path, 'w') as hdf5_file:
        
        # Loop over all .dat files in log files directory
        for data_file_full_path in glob.glob(
                                os.path.join(data_files_directory, "*.dat")):
            data_file = os.path.basename(data_file_full_path)

            # Initialize a dictionary to store lists for each correlator
            # identifier
            correlator_values_dictionary = {}
            for correlator_identifier in constants.CORRELATOR_IDENTIFIERS_LIST:
                correlator_values_dictionary[correlator_identifier] = []

            # Read log file and fill the correlator values
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

            # Create a group in the HDF5 file for this data file
            group = hdf5_file.create_group(data_file)

            # Store each correlator array in the group
            for correlator_identifier in constants.CORRELATOR_IDENTIFIERS_LIST:
                group.create_dataset(correlator_identifier,
                    data=correlator_values_dictionary[correlator_identifier])

    # TODO: Print something about the resulting output
    print("* Correlator files analysis completed.")


if __name__ == "__main__":
    main()

# TODO: Include logging properly