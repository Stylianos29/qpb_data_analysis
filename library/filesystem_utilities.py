import os

import logging
import textwrap


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
        handlers=[logging.FileHandler(log_path, mode='w')]
    )

    # Get the default handler
    file_handler = logging.getLogger().handlers[0]

    # Set the custom WrappedFormatter
    file_handler.setFormatter(WrappedFormatter("%(asctime)s - %(levelname)s - %(message)s"))
