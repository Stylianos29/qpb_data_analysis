"""
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
import sys
import os
from functools import partial

import numpy as np
import click

from library.constants import RAW_DATA_FILES_DIRECTORY, CORRELATOR_IDENTIFIERS_LIST
from library.extraction import extract_parameters_values_from_filename
from library import (
    LoggingWrapper,
    validate_input_directory,
    validate_output_directory,
    validate_output_file,
)

# Import shared private functions
from src.data_files_processing._shared_processing import (
    _classify_parameters_by_uniqueness,
    _export_arrays_to_hdf5_with_proper_structure,
)


def _process_correlator_files_and_extract_data(correlators_files_directory, logger):
    """
    Process all correlator files and extract both scalar parameters and
    correlator arrays.

    Args:
        correlators_files_directory (str): Directory containing correlator files
        logger: Logger instance

    Returns:
        tuple: (scalar_params_list, correlator_arrays_dict)
            - scalar_params_list: List of dicts with scalar parameters per file
            - correlator_arrays_dict: Dict with filename as key, correlator
              arrays as value
    """
    scalar_params_list = []
    correlator_arrays_dict = {}

    file_count = 0
    for correlators_file_path in glob.glob(
        os.path.join(correlators_files_directory, "*.dat")
    ):
        file_count += 1
        correlators_filename = os.path.basename(correlators_file_path)

        # Initialize parameter dictionary for this file
        scalar_params = {"Filename": correlators_filename}

        # Extract parameters from filename
        filename_params = extract_parameters_values_from_filename(
            correlators_filename, logger
        )
        scalar_params.update(filename_params)

        # Convert list values to tuples for consistency
        scalar_params = {
            k: tuple(v) if isinstance(v, list) else v for k, v in scalar_params.items()
        }

        # Parse correlator arrays from file
        correlator_arrays = _parse_correlator_file(correlators_file_path, logger)

        # Store results
        scalar_params_list.append(scalar_params)
        correlator_arrays_dict[correlators_filename] = correlator_arrays

    logger.info(
        f"A total of {file_count} qpb correlator files were parsed for "
        "parameter values and correlator data extraction from the "
        f"'{os.path.basename(correlators_files_directory)}' raw data files "
        "set directory."
    )

    return scalar_params_list, correlator_arrays_dict


def _parse_correlator_file(file_path, logger):
    """
    Parse a single correlator file and extract correlator values.

    Args:
        file_path (str): Full path to the correlator file
        logger: Logger instance

    Returns:
        dict: Dictionary with correlator identifiers as keys and numpy arrays as values
    """
    # Initialize correlator values dictionary
    correlator_values = {}
    for identifier in CORRELATOR_IDENTIFIERS_LIST:
        correlator_values[identifier] = []

    # Read and parse the correlator file
    with open(file_path, "r") as file:
        for line in file:
            columns = line.split()
            if len(columns) >= 5:  # Ensure line has enough columns
                # Check each correlator identifier (identifier is in last column, value in column 4)
                for identifier in CORRELATOR_IDENTIFIERS_LIST:
                    if columns[-1] == identifier:
                        try:
                            correlator_values[identifier].append(float(columns[4]))
                        except (ValueError, IndexError) as e:
                            if logger:
                                logger.warning(
                                    f"Error parsing line in {file_path}: {e}"
                                )

    # Convert lists to NumPy arrays
    for identifier in CORRELATOR_IDENTIFIERS_LIST:
        correlator_values[identifier] = np.array(correlator_values[identifier])

    return correlator_values


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
    callback=validate_output_directory,
    help="Path to directory where all output files will be stored.",
)
@click.option(
    "-out_hdf5_name",
    "--output_hdf5_filename",
    "output_hdf5_filename",
    default="qpb_correlator_files_values.h5",
    callback=partial(validate_output_file, extensions=[".h5"]),
    help=(
        "Specific name for the output HDF5 file containing extracted correlator "
        "values from qpb correlator files."
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
    callback=partial(validate_output_directory, check_parent_exists=True),
    help="Directory where the script's log file will be stored.",
)
@click.option(
    "-log_name",
    "--log_filename",
    "log_filename",
    default=None,
    callback=partial(validate_output_file, extensions=[".log"]),
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
    if output_files_directory is None:
        output_files_directory = os.path.dirname(qpb_correlators_files_directory)

    if log_file_directory is None and enable_logging:
        log_file_directory = output_files_directory

    if log_filename is None:
        script_name = os.path.basename(sys.argv[0])
        log_filename = script_name.replace(".py", "_python_script.log")

    # INITIATE LOGGING
    logger = LoggingWrapper(log_file_directory, log_filename, enable_logging)
    logger.initiate_script_logging()

    # PROCESS FILES AND EXTRACT DATA
    scalar_params_list, correlator_arrays_dict = (
        _process_correlator_files_and_extract_data(
            qpb_correlators_files_directory, logger
        )
    )

    # CLASSIFY PARAMETERS
    _, constant_params_dict, multivalued_params_list = (
        _classify_parameters_by_uniqueness(scalar_params_list)
    )

    # EXPORT CORRELATOR DATA TO HDF5
    output_hdf5_file_path = os.path.join(output_files_directory, output_hdf5_filename)
    _export_arrays_to_hdf5_with_proper_structure(
        constant_params_dict,
        multivalued_params_list,
        correlator_arrays_dict,
        scalar_params_list,
        output_hdf5_file_path,
        RAW_DATA_FILES_DIRECTORY,
        qpb_correlators_files_directory,
        logger,
        "correlator arrays",
    )

    logger.info(
        f"A total of {len(correlator_arrays_dict)} qpb correlators files "
        f"were parsed for correlator values extraction from the "
        f"'{os.path.basename(qpb_correlators_files_directory)}' raw data "
        "files set directory."
    )

    # TERMINATE LOGGING
    logger.terminate_script_logging()
    click.echo("   -- Parsing raw correlators files completed.")


if __name__ == "__main__":
    main()
