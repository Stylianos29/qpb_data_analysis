"""
Summary:
    This script processes qpb log files stored in a specified directory provided
    by the user, extracting relevant parameter values from both the file
    contents and filenames. It then compiles these extracted values into
    structured data formats, exporting them to both CSV and HDF5 files for
    further analysis.

Input:
1. `qpb_log_files_directory` (Required): Directory containing the qpb
    log files to be analyzed.
2. `output_files_directory` (Optional): Directory where the generated output CSV
   and HDF5 files will be stored. - If not provided, the directory of the input
   log files will be used.
3. `output_csv_filename` (Optional): Name of the CSV file storing extracted
   scalar parameter values.
4. `output_hdf5_filename` (Optional): Name of the HDF5 file storing extracted
   array parameter values.
5. `enable_logging` (Optional, Flag): Enables logging if set.
6. `log_file_directory` (Optional): Directory where the script's log file will
   be stored.
7. `log_filename` (Optional): Name of the script's log file.

Output:
    1. A CSV file containing extracted scalar parameter values
    from the log files. 2. An HDF5 file containing extracted array parameter
    values. 3. A log file (if logging is enabled) documenting the script's
    execution and potential issues.

Functionality:
    1. Uses regex patterns to extract specific parameter values
    from both log file contents and filenames. 2. Ensures extracted parameter
    values meet predefined validity conditions before storing them. 3.
    Constructs a Pandas DataFrame to organize scalar parameter values and
    exports it to a CSV file. 4. Organizes array parameter values in an HDF5
    hierarchical structure. 5. Logs execution details, warnings, and errors for
    traceability and debugging.

Usage:
    - Execute the script with appropriate options to specify the log files
    directory, output directory, and filenames for generated CSV and HDF5 files.
    - Enable logging if required for tracking script execution.
"""

import glob
import sys
import os

import click
from functools import partial

from library import (
    parsing,
    LoggingWrapper,
    RAW_DATA_FILES_DIRECTORY,
    validate_input_directory,
    validate_output_directory,
    validate_output_file,
)

# Import shared private functions
from src.parsing._shared_parsing import (
    _classify_parameters_by_uniqueness,
    _export_dataframe_to_csv,
    _export_arrays_to_hdf5_with_proper_structure,
    _check_parameter_mismatches,
)


def _process_log_files_and_extract_parameters(qpb_log_files_directory, logger):
    """
    Process all log files once and extract both scalar and array parameters.

    Args:
        - qpb_log_files_directory (str): Directory containing log files
        - logger: Logger instance

    Returns:
        tuple: (scalar_params_list, array_params_dict)
            - scalar_params_list: List of dicts with scalar parameters per file
            - array_params_dict: Dict with filename as key, array parameters as
              value
    """
    scalar_params_list = []
    array_params_dict = {}

    # Process all .txt files in the directory
    file_count = 0
    for qpb_log_file_full_path in glob.glob(
        os.path.join(qpb_log_files_directory, "*.txt")
    ):
        file_count += 1
        qpb_log_filename = os.path.basename(qpb_log_file_full_path)

        # Initialize dictionary for this file's scalar parameters
        scalar_params = {"Filename": qpb_log_filename}

        # Extract parameters from filename
        filename_params = parsing.extract_scalar_parameters_from_filename(
            qpb_log_filename, logger
        )

        # Read file contents
        with open(qpb_log_file_full_path, "r") as file:
            file_contents = file.readlines()

        # Extract scalar parameters from file contents
        file_scalar_params = (
            parsing.extract_scalar_parameters_from_file_contents(
                file_contents, logger
            )
        )

        # Extract array parameters from file contents
        file_array_params = (
            parsing.extract_array_parameters_from_file_contents(
                file_contents, logger
            )
        )

        # Merge scalar parameters (file contents take precedence over filename)
        scalar_params.update(file_scalar_params)

        # Add filename parameters only if not already present
        for param_name, filename_value in filename_params.items():
            scalar_params.setdefault(param_name, filename_value)

        # Check for mismatches and log warnings
        _check_parameter_mismatches(
            filename_params,
            scalar_params,
            qpb_log_filename,
            logger,
            "filename",
            "file contents",
        )

        # Convert list values to tuples for pandas compatibility
        scalar_params = {
            k: tuple(v) if isinstance(v, list) else v for k, v in scalar_params.items()
        }

        # Store results
        scalar_params_list.append(scalar_params)
        array_params_dict[qpb_log_filename] = file_array_params

    logger.info(
        f"A total of {file_count} qpb log files were parsed for parameter "
        "values parsing from the "
        f"'{os.path.basename(qpb_log_files_directory)}' "
        "raw data files set directory."
    )

    return scalar_params_list, array_params_dict


@click.command()
@click.option(
    "-qpb_log_dir",
    "--qpb_log_files_directory",
    "qpb_log_files_directory",
    required=True,
    callback=validate_input_directory,
    help="Directory where the qpb log files to be analyzed are stored.",
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
    "-out_csv_name",
    "--output_csv_filename",
    "output_csv_filename",
    default="qpb_log_files_scalar_parameters.csv",
    callback=partial(validate_output_file, extensions=[".csv"]),
    help=(
        "Specific name for the output .csv file containing extracted values of "
        "scalar parameters from qpb log files."
    ),
)
@click.option(
    "-out_hdf5_name",
    "--output_hdf5_filename",
    "output_hdf5_filename",
    default="qpb_log_files_array_parameters.h5",
    callback=partial(validate_output_file, extensions=[".h5"]),
    help=(
        "Specific name for the output HDF5 file containing extracted values of "
        "array parameters from qpb log files."
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
    qpb_log_files_directory,
    output_files_directory,
    output_csv_filename,
    output_hdf5_filename,
    enable_logging,
    log_file_directory,
    log_filename,
):
    # HANDLE EMPTY INPUT ARGUMENTS
    if output_files_directory is None:
        output_files_directory = os.path.dirname(qpb_log_files_directory)

    if log_file_directory is None and enable_logging:
        log_file_directory = output_files_directory

    if log_filename is None:
        script_name = os.path.basename(sys.argv[0])
        log_filename = script_name.replace(".py", "_python_script.log")

    # INITIATE LOGGING
    logger = LoggingWrapper(log_file_directory, log_filename, enable_logging)
    logger.initiate_script_logging()

    # PROCESS FILES AND EXTRACT PARAMETERS
    scalar_params_list, array_params_dict = _process_log_files_and_extract_parameters(
        qpb_log_files_directory, logger
    )

    # CLASSIFY PARAMETERS
    dataframe, constant_params_dict, multivalued_params_list = (
        _classify_parameters_by_uniqueness(scalar_params_list)
    )

    # EXPORT SCALAR PARAMETERS TO CSV
    csv_file_path = os.path.join(output_files_directory, output_csv_filename)
    _export_dataframe_to_csv(dataframe, csv_file_path, logger, "scalar parameters")

    # EXPORT ARRAY PARAMETERS TO HDF5
    hdf5_file_path = os.path.join(output_files_directory, output_hdf5_filename)
    _export_arrays_to_hdf5_with_proper_structure(
        constant_params_dict,
        multivalued_params_list,
        array_params_dict,
        scalar_params_list,
        hdf5_file_path,
        RAW_DATA_FILES_DIRECTORY,
        qpb_log_files_directory,
        logger,
        "array parameters",
    )

    # TERMINATE LOGGING
    logger.terminate_script_logging()
    click.echo("   -- Parsing raw qpb log files completed.")


if __name__ == "__main__":
    main()
