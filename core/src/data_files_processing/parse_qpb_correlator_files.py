"""
data_files_processing/parse_qpb_correlator_files.py

Summary:
    This script processes correlator data stored in .dat files within a
    specified directory and converts it into HDF5 format for efficient storage
    and retrieval. It extracts specific correlator values based on predefined
    identifiers and organizes the data into structured groups within the HDF5
    file.

Inputs:
    - qpb_correlators_files_directory (str): Path to the directory containing
      the input .dat files with qpb correlator data.
    - output_files_directory (str, optional): Path to the directory where the
      output HDF5 file will be saved. If not provided, defaults to the input
      directory.
    - output_hdf5_filename (str, optional): Name of the output HDF5 file 
      (should include '.h5' extension).
    - enable_logging (bool, optional): Flag to enable logging. Default is False.
    - log_file_directory (str, optional): Directory where the script's log file
      will be stored. Required if logging is enabled.
    - log_filename (str, optional): Specific name for the script's log file. 
      Required if logging is enabled.

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
      Each dataset stores extracted correlator values as NumPy arrays.

Processing Steps:
    1. Validate input arguments and directories.
    2. Set up logging if enabled.
    3. Create the HDF5 file and define its group structure.
    4. Iterate over all .dat files in the input directory:
       - Read each file and extract correlator values based on identifiers.
       - Store extracted values as NumPy arrays.
       - Save the data into the HDF5 file under a subgroup named after the 
         original .dat filename.
    5. Log the number of processed files and finalize execution.

Usage:
    Run the script from the command line as follows:
        python parse_qpb_correlator_files.py
        --qpb_correlators_files_directory <path_to_data>
        --output_files_directory <path_to_hdf5>
        --output_hdf5_filename <output_filename.h5>
        --enable_logging
        --log_file_directory <log_directory>
        --log_filename <log_filename.log>

Example:
    Given .dat files in the specified data directory, the script will generate
    an HDF5 file in the output directory, structured to allow efficient access
    to the correlator values.
"""

import glob
import os

import numpy as np
import pandas as pd
import h5py
import click

from library import (
    constants,
    extraction,
    filesystem_utilities,
    RAW_DATA_FILES_DIRECTORY,
    validate_input_directory,
    validate_input_script_log_filename,
)


@click.command()
@click.option(
    "-cors_dir",
    "--qpb_correlators_files_directory",
    "qpb_correlators_files_directory",
    required=True,
    callback=validate_input_directory,
    help="Directory where the correlator files to be analyzed are stored.",
)
@click.option(
    "-out_dir",
    "--output_files_directory",
    "output_files_directory",
    default=None,
    callback=validate_input_directory,
    help="Path to directory where all output files will be stored.",
)
@click.option(
    "-out_hdf5_name",
    "--output_hdf5_filename",
    "output_hdf5_filename",
    default="qpb_log_files_multivalued_parameters.h5",
    callback=filesystem_utilities.validate_output_HDF5_filename,
    help=(
        "Specific name for the output HDF5 file containing extracted values of "
        "multivalued parameters from qpb log files."
    ),
)
@click.option(
    "-log_on",
    "--enable_logging",
    "enable_logging",
    is_flag=True,
    default=False,
    help="Enable logging.",
)
@click.option(
    "-log_file_dir",
    "--log_file_directory",
    "log_file_directory",
    default=None,
    callback=filesystem_utilities.validate_script_log_file_directory,
    help="Directory where the script's log file will be stored.",
)
@click.option(
    "-log_name",
    "--log_filename",
    "log_filename",
    default=None,
    callback=validate_input_script_log_filename,
    help="Specific name for the script's log file.",
)
def main(
    qpb_correlators_files_directory,
    output_files_directory,
    output_hdf5_filename,
    enable_logging,
    log_file_directory,
    log_filename,
):
    # HANDLE EMPTY INPUT ARGUMENTS

    # If no output directory is provided, use the directory of the input file
    if output_files_directory is None:
        output_files_directory = os.path.dirname(qpb_correlators_files_directory)

    # INITIATE LOGGING

    # Setup logging
    logger = filesystem_utilities.LoggingWrapper(
        log_file_directory, log_filename, enable_logging
    )

    # Log script start
    logger.initiate_script_logging()

    #

    scalar_parameter_values_list = []
    # Loop over all .dat files in log files directory
    for count, correlators_file_full_path in enumerate(
        glob.glob(os.path.join(qpb_correlators_files_directory, "*.dat")), start=1
    ):
        correlators_filename = os.path.basename(correlators_file_full_path)

        # Initialize a dictionary to store extracted parameter values
        extracted_values_dictionary = {}

        # Add the filename to the dictionary
        extracted_values_dictionary["Filename"] = correlators_filename

        # Extract parameter values from the filename
        extracted_values_from_filename_dictionary = (
            extraction.extract_parameters_values_from_filename(
                correlators_filename, logger)
        )

        # Update the dictionary with all extracted values from filename
        extracted_values_dictionary.update(extracted_values_from_filename_dictionary)

        # Convert any list values to tuples
        extracted_values_dictionary = {
            k: tuple(v) if isinstance(v, list) else v 
            for k, v in extracted_values_dictionary.items()
        }

        # Append extracted values dictionary to the list of parameters
        scalar_parameter_values_list.append(extracted_values_dictionary)

    # Convert the list of parameter dictionaries into a Pandas DataFrame
    parameter_values_dataframe = pd.DataFrame(scalar_parameter_values_list)

    # Get the counts of unique values for each column
    unique_values_counts = parameter_values_dataframe.nunique()

    # Create lists of parameters based on their unique value counts
    single_valued_parameters_list = unique_values_counts[unique_values_counts == 1].index.tolist()
    multivalued_parameters_list = unique_values_counts[unique_values_counts > 1].index.tolist()

    # Create a dictionary of single-valued parameters and their unique values
    single_valued_parameters_dict = {
        col: parameter_values_dataframe[col].iloc[0] 
        for col in single_valued_parameters_list
    }

    # PARSE RAW CORRELATORS DATA FILES

    output_hdf5_file_path = os.path.join(output_files_directory, output_hdf5_filename)
    # Open the HDF5 file in 'w' mode (write, replace existing file)
    with h5py.File(output_hdf5_file_path, "w") as hdf5_file:

        # The top HDF5 file groups mirror the directory structure of the data
        # files set directory itself and its parent directories relative to the
        # 'data_files/raw/' directory
        data_files_set_group = filesystem_utilities.create_hdf5_group_structure(
            hdf5_file, RAW_DATA_FILES_DIRECTORY, qpb_correlators_files_directory, logger
        )

        # Add single-valued parameters as attributes to the data files set group
        for param_name, param_value in single_valued_parameters_dict.items():
            data_files_set_group.attrs[param_name] = param_value

        # Loop over all .dat files in log files directory
        for count, correlators_file_full_path in enumerate(
            glob.glob(os.path.join(qpb_correlators_files_directory, "*.dat")), start=1
        ):
            correlators_filename = os.path.basename(correlators_file_full_path)

            # Create a dictionary with correlator identifiers as keys, each
            # initialized to an empty list for storing correlator values.
            correlator_values_dictionary = {}
            for correlator_identifier in constants.CORRELATOR_IDENTIFIERS_LIST:
                correlator_values_dictionary[correlator_identifier] = []

            # Read each correlators values file and fill in the empty lists
            with open(correlators_file_full_path, "r") as file:
                lines = file.readlines()

                for line in lines:
                    columns = line.split()

                    # Check and append values for each correlator identifier
                    # NOTE: The identifier is always placed in the last column
                    # and value in the 4 one
                    for correlator_identifier in constants.CORRELATOR_IDENTIFIERS_LIST:
                        if columns[-1] == correlator_identifier:
                            correlator_values_dictionary[correlator_identifier].append(
                                float(columns[4])
                            )

            # Convert lists to NumPy arrays for each correlator
            for correlator_identifier in constants.CORRELATOR_IDENTIFIERS_LIST:
                correlator_values_dictionary[correlator_identifier] = np.array(
                    correlator_values_dictionary[correlator_identifier]
                )

            # Create a subgroup in the HDF5 file for this correlators file
            correlators_file_group = data_files_set_group.create_group(
                correlators_filename
            )

            # Extract parameter values from the filename
            extracted_values_from_filename_dictionary = (
                extraction.extract_parameters_values_from_filename(
                    correlators_filename, logger)
            )

            # Add single-valued parameters as attributes to the data files set group
            for param_name, param_value in extracted_values_from_filename_dictionary.items():
                correlators_file_group.attrs[param_name] = param_value

            # EXPORT CORRELATORS VALUES

            # Store each correlator array in the correlator group
            for correlator_identifier in constants.CORRELATOR_IDENTIFIERS_LIST:
                correlators_file_group.create_dataset(
                    correlator_identifier,
                    data=correlator_values_dictionary[correlator_identifier],
                )

    logger.info(
        f"A total of {count} qpb correlators files "
        f"were parsed for correlator values extraction from the "
        f"'{os.path.basename(qpb_correlators_files_directory)}' raw data "
        "files set directory."
    )

    # Terminate logging
    logger.terminate_script_logging()

    click.echo("   -- Parsing raw correlators files completed.")


if __name__ == "__main__":
    main()
