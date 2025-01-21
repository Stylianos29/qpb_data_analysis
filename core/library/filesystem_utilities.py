import os

import logging
import textwrap


import os
import shutil


def create_subdirectory(base_path, subdirectory_name, clear_contents=False):
    """
    Joins a base path with a subdirectory name to form a full path,
    creates the directory if it does not exist, and optionally clears
    all existing contents if the directory already exists.

    Parameters:
    - base_path (str): The parent directory path.
    - subdirectory_name (str): The name of the subdirectory to create.
    - clear_contents (bool): If True, clears all contents of the directory
                             if it already exists.

    Returns:
    - str: The full path of the created directory.
    """
    full_path = os.path.join(base_path, subdirectory_name)

    if os.path.exists(full_path):
        if clear_contents:
            # Remove all contents of the directory
            for item in os.listdir(full_path):
                item_path = os.path.join(full_path, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)  # Remove file or symlink
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Remove directory tree
    else:
        # Create the directory if it does not exist
        os.makedirs(full_path, exist_ok=True)

    return full_path


def is_valid_directory(directory_path):
    """Check if a given path is a valid directory."""

    return os.path.exists(directory_path) and os.path.isdir(directory_path)


def is_valid_file(file_path):
    """Check if a given path is a valid file."""

    return os.path.isfile(file_path)


def create_directory(directory_path):
    """Create a directory if it does not exist."""

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


class WrappedFormatter(logging.Formatter):
    def format(self, record):
        # Use textwrap to wrap the log message to 80 characters
        wrapped_message = textwrap.fill(super().format(record), width=80)
        return wrapped_message


def setup_logging(log_directory, log_filename):
    """Setup logging configuration."""

    log_path = os.path.join(log_directory, log_filename)

    logging.basicConfig(
        # Set the logging level to DEBUG
        level=logging.DEBUG,  # Set the log level (DEBUG, INFO, WARNING, etc.)
        # Set the log message format
        format="%(asctime)s - %(levelname)s - %(message)s",
        # Overwrite an existing log file
        handlers=[logging.FileHandler(log_path, mode="w")],
    )

    # Get the default handler
    file_handler = logging.getLogger().handlers[0]

    # Set the custom WrappedFormatter
    file_handler.setFormatter(
        WrappedFormatter("%(asctime)s - %(levelname)s - %(message)s")
    )


def extract_directory_names(directory_path):
    """
    Extracts the parent directory name and the last directory name from a given path.

    Parameters:
        directory_path (str): The full path to the directory.

    Returns:
        tuple: A tuple containing the parent directory name and the last directory name.
    """
    parent_directory_name = os.path.basename(os.path.dirname(directory_path))
    last_directory_name = os.path.basename(os.path.normpath(directory_path))

    return parent_directory_name, last_directory_name
