import os
import sys
import glob

import click
import datetime

from library import filesystem_utilities


def get_yes_no_response(prompt_text, logger=None):
    """Helper function to handle yes/no prompts with validation"""
    response = click.prompt(prompt_text, type=str)
    while response.lower() not in ["y", "yes", "", "n", "no"]:
        response = click.prompt(
            "Invalid input. Please enter 'y' for yes or 'n' for no (y[Y]/n[N])",
            type=str,
        )
    return response.lower() in ["y", "yes", ""]


@click.command()
@click.option(
    "-raw_dir",
    "--raw_data_files_set_directory_path",
    "raw_data_files_set_directory_path",
    required=True,
    callback=filesystem_utilities.validate_directory,
    help="Path to the raw data files set directory",
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
    "-ax_files_dir",
    "--auxiliary_files_directory",
    "auxiliary_files_directory",
    default=None,
    callback=filesystem_utilities.validate_script_log_file_directory,
    help="Directory where the script's log file will be stored.",
)
@click.option(
    "-log_name",
    "--log_filename",
    "log_filename",
    default=None,
    callback=filesystem_utilities.validate_script_log_filename,
    help="Specific name for the script's log file.",
)
def main(
    raw_data_files_set_directory_path,
    enable_logging,
    auxiliary_files_directory,
    log_filename,
):

    ### INITIATE SCRIPT AND LOGGING ###

    click.echo("   -- Validating raw qpb data files set initiated.")

    # Setup logging
    logger = filesystem_utilities.LoggingWrapper(
        auxiliary_files_directory, log_filename, enable_logging
    )

    # Log script start
    logger.initiate_script_logging()

    ### MAIN PART OF THE SCRIPT ###

    # REMOVE ANY UNSUPPORTED FILES FROM THE DIRECTORY

    # Find files with extensions other than .txt, .err, or .dat (unsupported)
    all_files = glob.glob(os.path.join(raw_data_files_set_directory_path, "*"))
    invalid_extension_files = [
        file
        for file in all_files
        if not any(file.endswith(ext) for ext in [".txt", ".err", ".dat"])
    ]

    # Unsupported files must be deleted before validation can proceed
    if invalid_extension_files:
        # Log a list of unsupported files
        logger.warning(
            f"Found {len(invalid_extension_files)} files with invalid extensions:",
            to_console=True,
        )
        for file in invalid_extension_files:
            logger.info(f"Invalid file: {os.path.basename(file)}")

        # Ask user about deleting unsupported files
        response = get_yes_no_response(
            "Found unsupported file types. Delete them to continue? "
            "Selecting 'n' will exit the program (y[Y]/n[N])"
        )
        if response:
            for file_path in invalid_extension_files:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted invalid file: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(
                        f"Error deleting file {file_path}: {str(e)}",
                        to_console=True,
                    )
        else:
            logger.error(
                "Validation cannot proceed with unsupported file types "
                "present inside the qpb data files set directory. "
                "Exiting the program.",
                to_console=True,
            )
            sys.exit(1)

    # REMOVE ANY EMPTY FILES FROM THE DIRECTORY

    # Find empty .txt and .dat files, ignore .err files for later
    empty_files = [
        file
        for file in all_files
        if (file.endswith(".txt") or file.endswith(".dat"))
        and os.path.getsize(file) == 0
    ]

    # Empty files better be deleted at the start of the validation process
    if empty_files:
        # Log a list of empty files
        logger.warning(
            f"Found {len(empty_files)} empty files with .txt or .dat extensions:",
            to_console=True,
        )
        for file in empty_files:
            logger.info(f"Empty file: {os.path.basename(file)}")

        # Ask used about deleting empty files
        response = get_yes_no_response(
            "Do you want to delete all empty .txt and .dat files? (y[Y]/n[N])"
        )
        if response:
            for file_path in empty_files:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted empty file: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(
                        f"Error deleting file {file_path}: {str(e)}",
                        to_console=True,
                    )

    # CHECK PRESENCE OF QPB LOG FILES

    # Check if there are any .txt files present
    if not any(glob.iglob(os.path.join(raw_data_files_set_directory_path, "*.txt"))):
        logger.error(
            f"No qpb log files (*.txt) found in '{raw_data_files_set_directory_path}'. "
            "Validation cannot proceed without log files.",
            to_console=True,
        )
        sys.exit(1)

    # RETRIEVE STORED FILE PATHS

    # Initialize empty lists
    list_of_stored_qpb_log_file_paths = []
    list_of_stored_qpb_error_file_paths = []
    list_of_stored_qpb_correlators_file_paths = []

    # Define file mappings
    file_mappings = {
        "list_of_stored_qpb_log_files.txt": list_of_stored_qpb_log_file_paths,
        "list_of_stored_qpb_error_files.txt": list_of_stored_qpb_error_file_paths,
        "list_of_stored_qpb_correlators_files.txt": list_of_stored_qpb_correlators_file_paths,
    }

    # Process each input list file
    for filename, file_list in file_mappings.items():
        file_path = os.path.join(auxiliary_files_directory, filename)
        try:
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    # Read non-empty lines and strip whitespace
                    file_list.extend(line.strip() for line in file if line.strip())
            else:
                # Create empty file if it doesn't exist
                with open(file_path, "w") as file:
                    pass
                logger.info(
                    f"Created empty file: {filename}",
                    to_console=True,
                )
        except Exception as e:
            logger.error(
                f"Error processing {filename}: {str(e)}",
                to_console=True,
            )

    # SELECT DATA FILES TO BE VALIDATED

    # Create lists of all current qpb data file paths
    list_of_qpb_log_file_paths = glob.glob(
        os.path.join(raw_data_files_set_directory_path, "*.txt")
    )
    list_of_qpb_error_file_paths = glob.glob(
        os.path.join(raw_data_files_set_directory_path, "*.err")
    )
    list_of_qpb_correlators_file_paths = glob.glob(
        os.path.join(raw_data_files_set_directory_path, "*.dat")
    )

    # Create lists for files to validate by comparing with stored files
    list_of_qpb_log_files_to_validate = [
        path
        for path in list_of_qpb_log_file_paths
        if path not in list_of_stored_qpb_log_file_paths
    ]
    list_of_qpb_error_files_to_validate = [
        path
        for path in list_of_qpb_error_file_paths
        if path not in list_of_stored_qpb_error_file_paths
    ]
    list_of_qpb_correlators_files_to_validate = [
        path
        for path in list_of_qpb_correlators_file_paths
        if path not in list_of_stored_qpb_correlators_file_paths
    ]

    # Log information about files to validate
    if list_of_qpb_log_files_to_validate:
        logger.info(
            f"Found {len(list_of_qpb_log_files_to_validate)} log files to validate.",
            to_console=True,
        )
    if list_of_qpb_error_files_to_validate:
        logger.info(
            f"Found {len(list_of_qpb_error_files_to_validate)} error "
            "files to validate.",
            to_console=True,
        )
    if list_of_qpb_correlators_files_to_validate:
        logger.info(
            f"Found {len(list_of_qpb_correlators_files_to_validate)} "
            "correlator files to validate.",
            to_console=True,
        )

    # Check if there are no files to validate
    if not (
        list_of_qpb_log_files_to_validate
        or list_of_qpb_error_files_to_validate
        or list_of_qpb_correlators_files_to_validate
    ):
        logger.warning(f"No new files found to validate.")

        # Ask user if they want to repeat the validation of already existing files
        response = get_yes_no_response(
            "No new files found to validate. Would you like to repeat "
            "validation of already existing qpb data files? "
            "Selecting 'n' will exit the program (y[Y]/n[N])"
        )
        if not response:
            logger.info(
                "Exiting validation as no new files were found.", to_console=True
            )
            sys.exit(0)
        else:
            logger.info(
                "Repeating validation of already existing data files.", to_console=True
            )

            list_of_qpb_log_files_to_validate = list_of_qpb_log_file_paths
            list_of_qpb_error_files_to_validate = list_of_qpb_error_file_paths
            list_of_qpb_correlators_files_to_validate = (
                list_of_qpb_correlators_file_paths
            )

    # KEEP OR REMOVE QPB ERROR FILES

    # Check for and handle .err files
    if list_of_qpb_error_files_to_validate:
        # Ask user to delete .err files
        response = get_yes_no_response(
            "Do you want to delete all .err files set for validation? (y[Y]/n[N])"
        )
        if response:
            number_of_qpb_error_files = len(list_of_qpb_error_files_to_validate)
            for file_path in list_of_qpb_error_files_to_validate:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted .err file: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(
                        f"Error deleting file {file_path}: {str(e)}",
                        to_console=True,
                    )
            logger.info(f"A total of {number_of_qpb_error_files} were deleted.")

    # REMOVE CORRUPTED QPB DATA FILES

    # Lists to store categorized files based on containing specific flag phrases
    list_of_invert_qpb_log_file_paths = []  # Containing "CG done"
    list_of_non_invert_qpb_log_file_paths = []  # Containing "per stochastic source"
    # Files lacking "CG done" or "per stochastic source" are marked as corrupted
    list_of_corrupted_qpb_log_file_paths = []

    # Check each log file for required phrases
    for log_file_path in list_of_qpb_log_files_to_validate:
        try:
            with open(log_file_path, "r") as file:
                content = file.read()

                # Check for phrases in file content
                if "CG done" in content:
                    list_of_invert_qpb_log_file_paths.append(log_file_path)
                elif "per stochastic source" in content:
                    list_of_non_invert_qpb_log_file_paths.append(log_file_path)
                else:
                    list_of_corrupted_qpb_log_file_paths.append(log_file_path)
                    logger.info(
                        f"Corrupted log file found: {os.path.basename(log_file_path)}"
                    )
        except Exception as e:
            # Handle file reading errors
            logger.error(
                f"Error reading file {log_file_path}: {str(e)}", to_console=True
            )
            list_of_corrupted_qpb_log_file_paths.append(log_file_path)

    # Check if any corrupted files were found and delete them if requested.
    # Associated .err and .dat files will be removed as well
    if list_of_corrupted_qpb_log_file_paths:
        # Log the number of corrupted qpb files found
        logger.warning(
            f"A total of {len(list_of_corrupted_qpb_log_file_paths)} corrupted "
            "qpb log files found.",
            to_console=True,
        )

        # Ask user about deleting corrupted files. Otherwise, exit the program
        response = get_yes_no_response(
            "Do you want to delete all corrupted qpb log files? "
            "Selecting 'n' will exit the program (y[Y]/n[N])"
        )
        if response:
            # Delete corrupted files
            for file_path in list_of_corrupted_qpb_log_file_paths:
                try:
                    # Remove the log file
                    os.remove(file_path)
                    logger.info(
                        f"Deleted corrupted file: {os.path.basename(file_path)}"
                    )
                    # Remove corresponding .err and .dat files if they exist
                    basename = os.path.splitext(file_path)[0]
                    for ext in [".err", ".dat"]:
                        associated_file = basename + ext
                        if os.path.exists(associated_file):
                            try:
                                os.remove(associated_file)
                                logger.info(
                                    "Deleted associated file: "
                                    f"{os.path.basename(associated_file)}"
                                )
                            except Exception as e:
                                logger.error(
                                    "Error deleting file "
                                    f"{associated_file}: {str(e)}.",
                                    to_console=True,
                                )
                except Exception as e:
                    logger.error(
                        f"Error deleting file {file_path}: {str(e)}.",
                        to_console=True,
                    )
        else:
            logger.error(
                "Validation cannot proceed with corrupted files present "
                "inside the qpb data files set directory. "
                "Exiting the program.",
                to_console=True,
            )
            sys.exit(1)

    # IDENTIFY MAIN PROGRAM TYPE AND REMOVE INCOMPATIBLE FILES

    # Check if metadata file exists and get main program type if present
    metadata_file = os.path.join(auxiliary_files_directory, "metadata.md")
    main_program_type = None
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as file:
            for line in file:
                if line.startswith("Main program type: "):
                    program_type = line.strip().split("Main program type: ")[1]
                    if program_type in ["invert", "non-invert"]:
                        main_program_type = program_type
                    break
            if main_program_type:
                logger.info(
                    f"Main program type found in metadata file: {main_program_type}.",
                    to_console=True,
                )

    incompatible_pairs_dictionary = {
        "invert": list_of_non_invert_qpb_log_file_paths,
        "non-invert": list_of_invert_qpb_log_file_paths,
    }

    # If there is an established main program type
    if main_program_type is not None:
        # Check for incompatible files based on main program type
        incompatible_files = incompatible_pairs_dictionary.get(main_program_type, [])
        if incompatible_files:
            logger.error(
                f"Found {len(incompatible_files)} {main_program_type} incompatible"
                "files. This conflicts with the established main program type."
            )
            response = get_yes_no_response(
                f"Do you want to delete all incompatible {main_program_type} qpb log files? "
                "Selecting 'n' will exit the program. (y[Y]/n[N])?"
            )
            if response:
                # Delete incompatible files
                for file_path in incompatible_files:
                    try:
                        # Remove the log file
                        os.remove(file_path)
                        logger.info(
                            f"Deleted incompatible file: {os.path.basename(file_path)}"
                        )
                        # Remove corresponding .err and .dat files if they exist
                        basename = os.path.splitext(file_path)[0]
                        for ext in [".err", ".dat"]:
                            associated_file = basename + ext
                            if os.path.exists(associated_file):
                                try:
                                    os.remove(associated_file)
                                    logger.info(
                                        "Deleted associated file: "
                                        f"{os.path.basename(associated_file)}"
                                    )
                                except Exception as e:
                                    logger.error(
                                        "Error deleting file "
                                        f"{associated_file}: {str(e)}.",
                                        to_console=True,
                                    )
                    except Exception as e:
                        logger.error(
                            f"Error deleting file {file_path}: {str(e)}.",
                            to_console=True,
                        )
                else:
                    logger.error(
                        "Validation cannot proceed with incompatible "
                        f"{main_program_type} file types present inside the "
                        "qpb data files set directory. Exiting the program.",
                        to_console=True,
                    )
                    sys.exit(1)
        # Filter log files to match the main program type
        list_of_qpb_log_files_to_validate = (
            list_of_invert_qpb_log_file_paths
            if main_program_type == "non-invert"
            else list_of_non_invert_qpb_log_file_paths
        )

    else:
        if (
            list_of_invert_qpb_log_file_paths
            and not list_of_non_invert_qpb_log_file_paths
        ):
            # Case: Only invert files present
            with open(metadata_file, "w") as file:
                file.write("Main program type: invert.")
            list_of_qpb_log_files_to_validate = list_of_invert_qpb_log_file_paths

        elif (
            list_of_non_invert_qpb_log_file_paths
            and not list_of_invert_qpb_log_file_paths
        ):
            # Case: Only non-invert files present
            with open(metadata_file, "w") as file:
                file.write("Main program type: non-invert.")
            list_of_qpb_log_files_to_validate = list_of_non_invert_qpb_log_file_paths

        else:
            # Both types present - this is incompatible
            logger.error(
                "Both invert and non-invert files detected in directory. "
                "These file types cannot coexist. Please manually review "
                "and remove incompatible files.",
                to_console=True,
            )
            sys.exit(1)

    # REMOVE UNMATCHED FILES

    # Check for matching .txt and .dat files
    log_file_basenames = {
        os.path.splitext(path)[0] for path in list_of_qpb_log_files_to_validate
    }
    correlators_file_basenames = {
        os.path.splitext(path)[0] for path in list_of_qpb_correlators_files_to_validate
    }

    # Find unmatched files
    unmatched_log_files_only = log_file_basenames - correlators_file_basenames
    unmatched_correlators_files_only = correlators_file_basenames - log_file_basenames

    # Handle unmatched .txt files
    if unmatched_log_files_only:
        logger.warning(
            f"Found {len(unmatched_log_files_only)} qpb log files without "
            "matching correlator files."
        )
        unmatched_txt_files = [
            path
            for path in list_of_qpb_log_file_paths
            if os.path.splitext(path)[0] in unmatched_log_files_only
        ]

        response = get_yes_no_response(
            "Do you want to delete qpb log files without matching "
            "correlators files? (y[Y]/n[N])"
        )
        if response:
            for file_path in unmatched_txt_files:
                try:
                    os.remove(file_path)
                    logger.info(
                        f"Deleted unmatched log file: {os.path.basename(file_path)}"
                    )
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")

    # Handle unmatched .dat files
    if unmatched_correlators_files_only:
        logger.warning(
            f"Found {len(unmatched_correlators_files_only)} qpb correlators "
            "files without matching log files."
        )
        unmatched_dat_files = [
            path
            for path in list_of_qpb_correlators_file_paths
            if os.path.splitext(path)[0] in unmatched_correlators_files_only
        ]

        response = get_yes_no_response(
            "Do you want to delete qpb correlators files without matching "
            "log files? (y[Y]/n[N])"
        )
        if response:
            for file_path in unmatched_dat_files:
                try:
                    os.remove(file_path)
                    logger.info(
                        "Deleted unmatched qpb correlators "
                        f"file: {os.path.basename(file_path)}"
                    )
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")

    # STORE REMAINING FILE PATHS IN SEPARATE TEXT FILES

    # Update lists of qpb data file paths
    list_of_qpb_log_file_paths = glob.glob(
        os.path.join(raw_data_files_set_directory_path, "*.txt")
    )
    list_of_qpb_error_file_paths = glob.glob(
        os.path.join(raw_data_files_set_directory_path, "*.err")
    )
    list_of_qpb_correlators_file_paths = glob.glob(
        os.path.join(raw_data_files_set_directory_path, "*.dat")
    )

    output_files_dictionary = {
        "list_of_stored_qpb_log_files.txt": list_of_qpb_log_file_paths,
        "list_of_stored_qpb_error_files.txt": list_of_qpb_error_file_paths,
        "list_of_stored_qpb_correlators_files.txt": list_of_qpb_correlators_file_paths,
    }
    for filename, file_list in output_files_dictionary.items():
        output_path = os.path.join(raw_data_files_set_directory_path, filename)
        try:
            file_exists = os.path.exists(output_path)
            with open(output_path, "w") as file:
                for file_path in file_list:
                    file.write(f"{file_path}\n")
            initial_substring = "Updated" if file_exists else "Created"
            logger.info(f"{initial_substring} file list: {filename}", to_console=True)
        except Exception as e:
            logger.error(f"Error writing to {filename}: {str(e)}", to_console=True)

    # INCLUDE ADDITIONAL INFORMATION IN THE METADATA FILE

    with open(metadata_file, "a") as file:
        file.write(f"\nNumber of qpb log files: {len(list_of_qpb_log_file_paths)}")
        file.write(f"\nNumber of qpb error files: {len(list_of_qpb_error_file_paths)}")
        file.write(
            f"\nNumber of qpb correlators files: {len(list_of_qpb_correlators_file_paths)}"
        )
        file.write("\n")
        file.write(
            f"\nValidation completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    ### FINALIZE SCRIPT AND LOGGING ###

    # Terminate logging
    logger.terminate_script_logging()

    click.echo("   -- Validating raw qpb data files set completed.")


if __name__ == "__main__":
    main()
