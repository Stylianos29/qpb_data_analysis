import os

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
