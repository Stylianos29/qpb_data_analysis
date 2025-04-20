import os
import sys
import shutil

import click
import logging
import textwrap
import h5py
import pandas as pd
import inspect


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


def setup_logging(log_directory, log_filename, enable_logging=True):
    """Setup logging configuration.

    Args:
        log_directory (str): Directory where the log file will be stored.
        log_filename (str): Name of the log file.
        enable_logging (bool): If False, disable logging output.
    """
    log_path = os.path.join(log_directory, log_filename)

    if enable_logging:
        logging.basicConfig(
            level=logging.DEBUG,  # Set log level
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_path, mode="w")],
        )

        # Get the default handler
        file_handler = logging.getLogger().handlers[0]

        # Set custom formatter
        file_handler.setFormatter(
            WrappedFormatter("%(asctime)s - %(levelname)s - %(message)s")
        )
    else:
        # Disable logging by setting level to CRITICAL+1 (or removing handlers)
        logging.getLogger().setLevel(logging.CRITICAL + 1)


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


def create_hdf5_group_structure(
    hdf5_file, base_directory, target_directory, logger=None
):
    """
    Creates an HDF5 group structure that mirrors the directory structure of
    target_directory beyond base_directory.

    This function iteratively creates HDF5 groups corresponding to each
    directory in target_directory after removing base_directory. If
    base_directory and target_directory are the same, no groups are created, and
    the root group is returned.

    Parameters:
        hdf5_file (h5py.File): Open HDF5 file handle. base_directory (str): The
        base directory to remove from target_directory. target_directory (str):
        The directory whose structure should be mirrored.
        logger (logging.Logger, optional): Logger instance for warnings
        (default: None).

    Returns:
        h5py.Group: The deepest group corresponding to target_directory.
    """

    # Convert both paths to their real full paths
    base_directory = os.path.realpath(base_directory)
    target_directory = os.path.realpath(target_directory)

    # Get the relative path beyond the base directory
    relative_path = os.path.relpath(target_directory, base_directory)

    # Start from the root group
    current_group = hdf5_file

    if relative_path:
        # Split the relative path into its hierarchical components
        group_hierarchy = relative_path.split(os.sep)

        # Iteratively create or retrieve groups
        for group_name in group_hierarchy:
            current_group = current_group.require_group(group_name)
    else:
        if logger:
            logger.warning(
                f"Target directory is the same as base directory: "
                "{base_directory}. No groups created."
            )

    return current_group  # Return the deepest group created (or the root group)


def get_hdf5_target_group(hdf5_file, base_directory, target_directory, logger=None):
    """
    Retrieves the HDF5 group corresponding to target_directory, assuming
    the group structure mirrors the directory structure beyond base_directory.

    Parameters:
        hdf5_file (h5py.File): Open HDF5 file handle.
        base_directory (str): The base directory to remove from target_directory.
        target_directory (str): The directory whose structure should be mirrored.

    Returns:
        h5py.Group: The deepest existing group corresponding to target_directory.
                    Returns the root group if base_directory == target_directory.
    """

    # Convert both paths to their real full paths
    base_directory = os.path.realpath(base_directory)
    target_directory = os.path.realpath(target_directory)

    # Get the relative path beyond the base directory
    relative_path = os.path.relpath(target_directory, base_directory)

    # Traverse the HDF5 group hierarchy
    current_group = hdf5_file
    if relative_path != ".":
        for group_name in relative_path.split(os.sep):
            if group_name in current_group:
                current_group = current_group[group_name]
            else:
                raise KeyError(f"Group '{group_name}' not found in HDF5 file.")

    return current_group


# def get_hdf5_target_group(hdf5_file, base_directory, target_directory, logger=None):
#     """
#     Retrieves the deepest HDF5 group corresponding to target_directory,
#     mirroring the directory structure beyond base_directory.

#     Parameters:
#         hdf5_file (h5py.File): Open HDF5 file handle in read mode.
#         base_directory (str): The base directory to remove from target_directory.
#         target_directory (str): The directory whose structure should be mirrored.

#     Returns:
#         h5py.Group or None: The deepest group if found, otherwise None.
#     """
#     # Compute the relative path beyond base_directory
#     relative_path = os.path.relpath(target_directory, base_directory)

#     # print(relative_path)

#     # If target and base directories are the same, return root group
#     if relative_path == ".":
#         if logger:
#             logger.warning(
#                 "Target directory is the same as base directory: %s.", base_directory
#             )
#         return hdf5_file  # Root group

#     # Split into directories
#     group_hierarchy = relative_path.split(os.sep)

#     # Navigate through the HDF5 file structure
#     current_group = hdf5_file
#     for group_name in group_hierarchy:
#         if group_name in current_group:
#             current_group = current_group[group_name]
#         else:
#             if logger:
#                 logger.error("Group '%s' not found in HDF5 file.", group_name)
#             return None  # Group does not exist

#     return current_group  # Return the deepest group found


def validate_file(ctx, param, value):
    if value is None:
        return None  # Skip validation for None
    # Validate the file path using click.Path
    path_type = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
    return path_type.convert(value, param, ctx)


def validate_input_directory(ctx, param, value):
    if value is None:
        return None  # Skip validation for None
    # Validate the directory path using click.Path
    path_type = click.Path(exists=True, file_okay=True, dir_okay=True, readable=True)
    return path_type.convert(value, param, ctx)


def validate_output_HDF5_filename(ctx, param, value):
    if not value.endswith(".h5"):
        raise click.BadParameter(
            f"The file name '{value}' is invalid. Output HDF5 file names must "
            "end with '.h5'."
        )
    return value


def validate_output_csv_filename(ctx, param, value):
    if not value.endswith(".csv"):
        raise click.BadParameter(
            f"The file name '{value}' is invalid. Output csv file names must "
            "end with '.csv'."
        )
    return value


def validate_input_script_log_filename(ctx, param, value):
    # Get the name of the script being executed (entry point)
    script_name = os.path.basename(sys.argv[0])

    # If no log filename is provided, generate a default name
    if value is None:
        default_log_filename = script_name.replace(".py", "_python_script.log")
        return default_log_filename

    # Validate the provided log filename (e.g., ensure it's a valid string)
    if not value.strip():  # Ensure it's not empty or just spaces
        raise click.BadParameter("Log filename cannot be an empty string.")

    if not value.endswith(".log"):
        raise click.BadParameter(
            f"The file name '{value}' is invalid. Current script's log file "
            "names must end with '.log'."
        )

    return value


def validate_script_log_file_directory(ctx, param, value):
    # TODO: I need to rethink this case, there isn't always a
    # "output_files_directory" directory
    if value is None:
        # Get the value of 'input_correlators_hdf5_file_path' from ctx.params
        output_files_directory = ctx.params.get("output_files_directory")
        if output_files_directory:
            # Default to the directory of the input file
            return output_files_directory

    # Validate the directory path
    path_type = click.Path(exists=True, file_okay=False, dir_okay=True, writable=True)
    return path_type.convert(value, param, ctx)


def validate_input_HDF5_file(ctx, param, value):
    if value is None:
        return None  # Skip validation for None

    # Validate the file path using click.Path
    path_type = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
    file_path = path_type.convert(value, param, ctx)

    # Ensure the file has a valid HDF5 extension
    if not file_path.lower().endswith((".hdf5", ".h5")):
        raise click.BadParameter(
            f"Invalid file extension: '{file_path}'. "
            "Expected a '.hdf5' or '.h5' file."
        )

    # Verify the file is a readable HDF5 file
    try:
        with h5py.File(file_path, "r") as _:
            pass  # Successfully opened in read mode
    except Exception as e:
        raise click.BadParameter(
            f"File '{file_path}' is not a valid HDF5 file or cannot be opened: {e}"
        )

    return file_path  # Return the validated file path


def validate_input_csv_file(ctx, param, value):
    if value is None:
        return None  # Skip validation for None

    # Validate the file path using click.Path
    path_type = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
    file_path = path_type.convert(value, param, ctx)

    # Ensure the file has a valid CSV extension
    if not file_path.lower().endswith(".csv"):
        raise click.BadParameter(
            f"Invalid file extension: '{file_path}'. Expected a '.csv' file."
        )

    # Verify the file is a valid CSV by attempting to read it
    try:
        df = pd.read_csv(
            file_path, nrows=1
        )  # Read only the first row to check validity
    except Exception as e:
        raise click.BadParameter(
            f"File '{file_path}' is not a valid CSV file or cannot be opened: {e}"
        )

    return file_path  # Return the validated file path


class LoggingWrapper:
    """Encapsulates logging setup and provides convenient logging methods."""

    def __init__(self, log_directory, log_filename, enable_logging=True):
        """Initialize the logging system.

        Args:
            log_directory (str): Directory where the log file will be stored.
            log_filename (str): Name of the log file.
            enable_logging (bool): If False, disable logging output.
        """
        # Get the calling script's filename
        caller_frame = inspect.stack()[1]
        self.script_name = os.path.basename(caller_frame.filename)

        self.logger = None  # Default to None if logging is disabled
        if enable_logging:
            self.logger = self._setup_logging(log_directory, log_filename)

    def _setup_logging(self, log_directory, log_filename):
        """Internal method to configure logging."""
        log_path = os.path.join(log_directory, log_filename)

        # Create log directory if it doesn't exist
        os.makedirs(log_directory, exist_ok=True)

        logger = logging.getLogger(self.script_name)  # Use script name as logger name
        logger.setLevel(logging.DEBUG)

        # File handler (writes to log file)
        file_handler = logging.FileHandler(log_path, mode="w")

        # TODO: Make sure that log message is wrapped indeed at 80 characters width
        # Apply text wrapping inline instead of using a separate class
        class WrappedFormatter(logging.Formatter):
            def format(self, record):
                record.msg = textwrap.fill(record.getMessage(), width=80)
                return super().format(record)

        # Define formatter
        formatter = WrappedFormatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Clear previous handlers to avoid duplicates
        if logger.hasHandlers():
            logger.handlers.clear()

        # Add file handler
        logger.addHandler(file_handler)

        return logger

    def initiate_script_logging(self):
        """Log script execution start."""
        if self.logger:
            self.logger.info(f"Script '{self.script_name}' execution initiated.")

    def terminate_script_logging(self):
        """Log script execution termination."""
        if self.logger:
            self.logger.info(
                f"Script '{self.script_name}' execution terminated successfully."
            )

    # Forward logging methods with optional console printing
    def info(self, message, to_console=False):
        if self.logger:
            self.logger.info(message)
            if to_console:
                print(f"INFO: {message}")
# TODO: Create a list option
    def warning(self, message, to_console=False):
        if self.logger:
            self.logger.warning(message)
            if to_console:
                print(f"WARNING: {message}")

    def error(self, message, to_console=False):
        if self.logger:
            self.logger.error(message)
            if to_console:
                print(f"ERROR: {message}")

    def debug(self, message, to_console=False):
        if self.logger:
            self.logger.debug(message)
            if to_console:
                print(f"DEBUG: {message}")

    def critical(self, message, to_console=False):
        if self.logger:
            self.logger.critical(message)
            if to_console:
                print(f"CRITICAL: {message}")


def has_subgroups(hdf5_group):
    """Check if an HDF5 group contains any subgroups."""
    for key in hdf5_group:
        if isinstance(hdf5_group[key], h5py.Group):
            return True  # Found at least one subgroup, exit early
    return False  # No subgroups found
