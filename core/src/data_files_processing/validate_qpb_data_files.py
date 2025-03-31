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
    raw_data_files_set_directory_path, enable_logging, log_file_directory, log_filename
):

    # INITIATE LOGGING

    # Setup logging
    logger = filesystem_utilities.LoggingWrapper(
        log_file_directory, log_filename, enable_logging
    )

    # Log script start
    logger.initiate_script_logging()


    # Terminate logging
    logger.terminate_script_logging()

    click.echo("   -- Validating raw qpb data files set completed.")

if __name__ == "__main__":
    main()
