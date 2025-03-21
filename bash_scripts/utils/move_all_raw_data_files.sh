#!/bin/bash

################################################################################
# move_all_raw_data_files.sh - Automates the movement of raw data files between
# hierarchical directory structures.
#
# DESCRIPTION:
# This script moves raw data files organized in a hierarchical directory 
# structure from a source directory to a destination directory. The source 
# directory contains subdirectories representing the outputs of main programs, 
# each of which has subdirectories corresponding to data files sets.
#
# The script:
# 1. Traverses the source directory recursively to locate raw data files sets.
# 2. Checks for empty directories and skips them with a warning.
# 3. Moves raw data files from each identified data files set directory to the 
#    corresponding location in the destination directory, maintaining the 
#    original hierarchy.
#
# DEPENDENCIES:
# - Library scripts in the "library" directory containing utility functions 
#   such as:
#   - check_if_directory_exists
#   - replace_parent_directory
#
# ENVIRONMENT VARIABLES:
# - SOURCE_DIRECTORY: Path to the root directory containing raw data files.
# - DESTINATION_DIRECTORY: Path to the target directory where files will be 
#   moved.
#
# USAGE:
#  ./move_all_raw_data_files.sh
#    - Moves raw data files from the source to the destination directory while 
#      preserving the directory structure.
#
# EXAMPLES:
#  ./move_all_raw_data_files.sh
#    - Moves data files under "../../data_files/raw" 
#                                       to "${HOME}/scratch/raw_qpb_data_files".
#
# NOTES:
# - The script assumes that library scripts are located in "../library" relative 
#   to the script's directory.
# - Empty data files set directories are not processed and will be skipped.
#
################################################################################

# SOURCE DEPENDENCIES

export LIBRARY_SCRIPTS_DIRECTORY_PATH=$(realpath "../library")
if [[ ! -d "$LIBRARY_SCRIPTS_DIRECTORY_PATH" ]]; then
    echo "Invalid library scripts path."
    exit 1
fi
# Source all library scripts to load utility functions.
for library_script in "${LIBRARY_SCRIPTS_DIRECTORY_PATH}"/*.sh; do
    [ -f "$library_script" ] && source "$library_script"
done

# ENVIRONMENT VARIABLES

SOURCE_DIRECTORY="../../data_files/raw"
check_if_directory_exists "$SOURCE_DIRECTORY" || exit 1

DESTINATION_DIRECTORY="${HOME}/scratch/raw_qpb_data_files"
check_if_directory_exists "$DESTINATION_DIRECTORY" || exit 1

# TODO: Set up logging functionality
# LOG_FILE="${HOME}/Data_analysis/rsync.log"
# check_if_directory_exists "$DESTINATION_DIRECTORY" || exit 1

# BACK UP ALL RAW DATA FILES

# Traverse the source directory to process main program subdirectories
for main_program_directory in "$SOURCE_DIRECTORY"/*; do
    # Skip non-directory entries
    [ -d "$main_program_directory" ] || continue

    echo
    echo "Working within '${main_program_directory/"../"}/':"

    # Process each data files set within the current main program directory
    for data_files_set_directory in "$main_program_directory"/*; do
        # Skip non-directory entries.
        [ -d "$data_files_set_directory" ] || continue

        # Calculate the destination subdirectory path by replacing the parent
        # directory of the source data files set with the destination directory.
        # This ensures that the directory structure of the raw data files is
        # maintained in the destination.
        destination_subdirectory=$(replace_parent_directory \
            "$data_files_set_directory" "$SOURCE_DIRECTORY" \
            "$DESTINATION_DIRECTORY")

        # Ensure the destination subdirectory exists (create if necessary)
        check_if_directory_exists "$destination_subdirectory" -c -s

        # Extract the data files set name
        data_files_set_name=$(basename "$data_files_set_directory")

        # Check if source data files set directory is empty
        if [ -z "$(ls -A $data_files_set_directory)" ]; then
            echo "'$data_files_set_name' data files set directory is empty."
            echo "Skipping..."
            continue
        fi

        echo "- Moving raw data files from '${data_files_set_name}/' directory."

        # Count the number of files to be moved
        file_count=$(ls -1 "${data_files_set_directory}" | wc -l)

        # Move the files
        mv "${data_files_set_directory}"/* "${destination_subdirectory}/"

        # Log the number of files moved
        echo "Moved ${file_count} files to '${destination_subdirectory}/'."

    done
done

echo
echo "Moving all raw data files sets completed!"

# Unset the library scripts path variable to avoid conflicts.
unset LIBRARY_SCRIPTS_DIRECTORY_PATH
