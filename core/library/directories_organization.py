import os


def extract_relative_path(file_full_path, root_directory):
    '''Returns the relative path from the given full path of a file or directory with respect to the given root directory. NOTE: A leading "/" needs to be removed as well from the resulting relative path.'''

    return file_full_path.replace(root_directory, "")


def replicate_directory_structure(source_directory, target_directory):
    
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        print(f"Warning: Target directory '{target_directory}' does not exist.")
        print("Exiting.")
        
        return

    new_subdirectories_lists = list()
    # Walk through the source directory
    for root, dirs, _ in os.walk(source_directory):
        # Calculate the relative path from the source directory
        relative_path = os.path.relpath(root, source_directory)

        # Combine the relative path with the target directory to get the new directory path
        new_subdirectory = os.path.join(target_directory, relative_path)

        # Create the new directory if it doesn't exist
        new_subdirectories_lists.append(new_subdirectory)
        if not os.path.exists(new_subdirectory):
            os.makedirs(new_subdirectory)

        # Limit the depth to 2 by removing subdirectories beyond depth 2
        if new_subdirectory.count(os.path.sep) - target_directory.count(os.path.sep) >= 2:
            del dirs[:]
    
    return new_subdirectories_lists


def list_leaf_subdirectories(directory):
    '''Return a list of all the subdirectories excluding those with other further subdirectories and files.'''

    leaf_subdirectories = []
    for root, dirs, files in os.walk(directory):
        if not dirs and files:
            leaf_subdirectories.append(root)

    return leaf_subdirectories


def change_root(old_path, old_root, new_root):
    # Ensure old_path starts with old_root
    if not old_path.startswith(old_root):
        raise ValueError("old_path does not start with old_root")

    # Remove the old_root from the beginning of old_path
    relative_path = old_path[len(old_root)+1:]

    # Create the new path by joining new_root and the relative path
    new_path = os.path.join(new_root, relative_path)
    
    return new_path
