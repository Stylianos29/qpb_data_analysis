import os
import sys
import glob

import click

from library import filesystem_utilities


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
    callback=filesystem_utilities.validate_script_log_filename,
    help="Specific name for the script's log file.",
)
def main(
    raw_data_files_set_directory_path,
    enable_logging,
    log_file_directory,
    log_filename
):

    # INITIATE LOGGING

    # Setup logging
    logger = filesystem_utilities.LoggingWrapper(
        log_file_directory, log_filename, enable_logging
    )

    # Log script start
    logger.initiate_script_logging()

    # MAIN PART OF THE SCRIPT

    list_of_qpb_log_file_paths = glob.glob(
        os.path.join(raw_data_files_set_directory_path, "*.txt")
    )
    list_of_qpb_error_file_paths = glob.glob(
        os.path.join(raw_data_files_set_directory_path, "*.err")
    )
    list_of_qpb_correlators_file_paths = glob.glob(
        os.path.join(raw_data_files_set_directory_path, "*.dat")
    )

    # Check if there are no .txt files. Exit with error if so.
    if len(list_of_qpb_log_file_paths) == 0:
        click.echo(
            f"ERROR: No qpb log files found in "
            "'{raw_data_files_set_directory_path}'. No validation "
            "process of the data files set can be performed without qpb "
            "log files present.",
            err=True,
        )
        sys.exit(1)

    # Lists to store categorized files
    list_of_invert_qpb_log_file_paths = []  # Files containing "CG done"
    list_of_non_invert_qpb_log_file_paths = (
        []
    )  # Files containing "per stochastic source"
    list_of_corrupted_qpb_log_file_paths = []  # Files containing neither phrase

    # Check each log file for required phrases
    for log_file_path in list_of_qpb_log_file_paths:
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
            logger.error(f"Error reading file {log_file_path}: {str(e)}")
            list_of_corrupted_qpb_log_file_paths.append(log_file_path)

    # Print summary
    click.echo(f"Total qpb log files: {len(list_of_qpb_log_file_paths)}")
    click.echo(
        f"Invert files (with 'CG done'): {len(list_of_invert_qpb_log_file_paths)}"
    )
    click.echo(
        f"Non-invert files (with 'per stochastic source'): {len(list_of_non_invert_qpb_log_file_paths)}"
    )

    # Check if any corrupted files were found
    if list_of_corrupted_qpb_log_file_paths:
        click.echo(
            f"WARNING: {len(list_of_corrupted_qpb_log_file_paths)} corrupted qpb log files found:",
            err=True,
        )
        for file_path in list_of_corrupted_qpb_log_file_paths:
            click.echo(f"  - {os.path.basename(file_path)}", err=True)

    # Terminate logging
    logger.terminate_script_logging()

    click.echo("   -- Validating raw qpb data files set completed.")


if __name__ == "__main__":
    main()
