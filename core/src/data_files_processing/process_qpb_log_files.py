"""
data_files_processing/process_qpb_log_files.py

Summary:
    This script processes log files stored in a specified directory provided by
    the user, extracting relevant parameter values from both the file contents
    and filenames. It then compiles these extracted values into structured data
    formats, exporting them to both CSV and HDF5 files for further analysis.

Input:
1. `qpb_log_files_directory` (Required): Directory containing the qpb
    log files to be analyzed.
2. `output_files_directory` (Optional): Directory where the generated output CSV
   and HDF5 files will be stored.
   - If not provided, the directory of the input log files will be used.
3. `output_csv_filename` (Optional): Name of the CSV file storing extracted
   single-valued parameter values.
4. `output_hdf5_filename` (Optional): Name of the HDF5 file storing extracted
   multi-valued parameter values.
5. `enable_logging` (Optional, Flag): Enables logging if set.
6. `log_file_directory` (Optional): Directory where the script's log file will
   be stored.
7. `log_filename` (Optional): Name of the script's log file.

Output:
    1. A CSV file containing extracted single-valued parameter values
    from the log files. 2. An HDF5 file containing extracted multi-valued
    parameter values. 3. A log file (if logging is enabled) documenting the
    script's execution and potential issues.

Functionality:
    1. Uses regex patterns to extract specific parameter values
    from both log file contents and filenames. 2. Ensures extracted parameter
    values meet predefined validity conditions before storing them. 3.
    Constructs a Pandas DataFrame to organize single-valued parameter values and
    exports it to a CSV file. 4. Organizes multi-valued parameter values in an
    HDF5 hierarchical structure. 5. Logs execution details, warnings, and errors
    for traceability and debugging.

Usage:
    - Execute the script with appropriate options to specify the log files
    directory, output directory, and filenames for generated CSV and HDF5 files.
    - Enable logging if required for tracking script execution.
"""

import glob
import os

import pandas as pd
import h5py
import click

from library import (
    extraction,
    filesystem_utilities,
    RAW_DATA_FILES_DIRECTORY,
    validate_input_directory,
    validate_input_script_log_filename,
)


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
    callback=validate_input_directory,
    help="Path to directory where all output files will be stored.",
)
@click.option(
    "-out_csv_name",
    "--output_csv_filename",
    "output_csv_filename",
    default="qpb_log_files_single_valued_parameters.csv",
    callback=filesystem_utilities.validate_output_csv_filename,
    help=(
        "Specific name for the output .csv file containing extracted values of "
        "single-valued parameters from qpb log files."
    ),
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
    qpb_log_files_directory,
    output_files_directory,
    output_csv_filename,
    output_hdf5_filename,
    enable_logging,
    log_file_directory,
    log_filename,
):
    # HANDLE EMPTY INPUT ARGUMENTS

    # If no output directory is provided, use the directory of the input file
    if output_files_directory is None:
        output_files_directory = os.path.dirname(qpb_log_files_directory)

    # INITIATE LOGGING

    # Setup logging
    logger = filesystem_utilities.LoggingWrapper(
        log_file_directory, log_filename, enable_logging
    )

    # Log script start
    logger.initiate_script_logging()

    # EXTRACT SINGLE-VALUED PARAMETERS

    # Create list to pass values to dataframe
    single_valued_parameter_values_list = []
    # Loop over all .txt files in the qpb log files directory
    for count, qpb_log_file_full_path in enumerate(
        glob.glob(os.path.join(qpb_log_files_directory, "*.txt")), start=1
    ):
        # Extract the filename from the full path
        qpb_log_filename = os.path.basename(qpb_log_file_full_path)

        # Initialize a dictionary to store extracted parameter values
        extracted_values_dictionary = {}

        # Add the filename to the dictionary
        extracted_values_dictionary["Filename"] = qpb_log_filename

        # Extract parameter values from the filename
        extracted_values_from_filename_dictionary = (
            extraction.extract_parameters_values_from_filename(qpb_log_filename, logger)
        )

        # Read the contents of the log file into a list of lines
        with open(qpb_log_file_full_path, "r") as file:
            qpb_log_file_contents_list = file.readlines()

        # Extract parameter values from the contents of the file
        extracted_single_valued_parameters_from_file_contents_dictionary = (
            extraction.extract_single_valued_parameter_values_from_file_contents(
                qpb_log_file_contents_list, logger
            )
        )

        # Add values from file contents into extracted values dictionary
        extracted_values_dictionary.update(
            extracted_single_valued_parameters_from_file_contents_dictionary
        )

        # Add values from 'extracted_values_from_filename_dictionary' into
        # 'extracted_values_dictionary' only if their parameter names are not
        # already present, ensuring existing values remain unchanged.
        for (
            parameter_name,
            filename_value,
        ) in extracted_values_from_filename_dictionary.items():
            extracted_values_dictionary.setdefault(parameter_name, filename_value)

        # Check for common keys between the
        # 'extracted_values_from_filename_dictionary' and
        # 'extracted_values_dictionary' dictionaries. If the values are
        # different, log a warning to indicate a potential mismatch.
        # NOTE: Values from file contents supersede those from filename
        for parameter_name in (
            extracted_values_from_filename_dictionary.keys()
            & extracted_values_dictionary.keys()
        ):
            if (
                extracted_values_from_filename_dictionary[parameter_name]
                != extracted_values_dictionary[parameter_name]
            ):
                logger.warning(
                    f"Mismatch for '{parameter_name}' parameter in file {qpb_log_filename}. "
                    f"Filename value: {extracted_values_from_filename_dictionary[parameter_name]}, "
                    f"file contents value: {extracted_values_dictionary[parameter_name]}."
                )

        # Append extracted values dictionary to the list of parameters
        single_valued_parameter_values_list.append(extracted_values_dictionary)

    logger.info(
        f"A total of {count} qpb log files were parsed for parameter values "
        f"extraction from the '{os.path.basename(qpb_log_files_directory)}' "
        "raw data files set directory."
    )

    # EXPORT EXTRACTED SINGLE-VALUED PARAMETER VALUES

    # Convert the list of parameter dictionaries into a Pandas DataFrame
    parameter_values_dataframe = pd.DataFrame(single_valued_parameter_values_list)

    # Save the DataFrame to a CSV file in the output directory
    csv_file_full_path = os.path.join(output_files_directory, output_csv_filename)

    # Export dataframe to .cvs file
    parameter_values_dataframe.to_csv(csv_file_full_path, index=False)
    logger.info(
        f"Extracted single-valued parameters are stored in the "
        f"'{output_csv_filename}' file."
    )

    # EXTRACT MULTIVALUED PARAMETERS

    hdf5_file_full_path = os.path.join(output_files_directory, output_hdf5_filename)
    with h5py.File(hdf5_file_full_path, "w") as hdf5_file:

        # The top HDF5 file groups mirror the directory structure of the data
        # files set directory itself and its parent directories relative to the
        # 'data_files/raw/' directory
        data_files_set_group = filesystem_utilities.create_hdf5_group_structure(
            hdf5_file,
            RAW_DATA_FILES_DIRECTORY,
            qpb_log_files_directory,
            logger,
        )

        for qpb_log_file_full_path in glob.glob(
            os.path.join(qpb_log_files_directory, "*.txt")
        ):
            # Read the contents of the log file into a list of lines
            with open(qpb_log_file_full_path, "r") as file:
                qpb_log_file_contents_list = file.readlines()

            # Extract the filename from the full path
            qpb_log_filename = os.path.basename(qpb_log_file_full_path)

            # Create a group for the current file (based on the filename)
            qpb_log_file_group = data_files_set_group.create_group(qpb_log_filename)

            # Extract multivalued parameters from the contents of the file
            extracted_multivalued_parameters_from_file_contents_dictionary = (
                extraction.extract_multivalued_parameters_from_file_contents(
                    qpb_log_file_contents_list, logger
                )
            )

            # EXPORT EXTRACTED MULTIVALUED PARAMETER VALUES

            # Loop through each multivalued parameter and store it as a dataset
            for (
                parameter,
                values,
            ) in extracted_multivalued_parameters_from_file_contents_dictionary.items():
                # Create a dataset for each parameter in the file
                qpb_log_file_group.create_dataset(parameter, data=values)

    logger.info(
        f"Extracted multivalued parameters are stored in the "
        f"'{output_hdf5_filename}' file."
    )

    # Terminate logging
    logger.terminate_script_logging()

    click.echo("   -- Parsing raw qpb log files completed.")


if __name__ == "__main__":
    main()
