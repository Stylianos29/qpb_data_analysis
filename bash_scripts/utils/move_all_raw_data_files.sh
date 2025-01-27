#!/bin/bash

# TODO: Rewrite comment
################################################################################
# validate_all_raw_data_files_sets.sh - Automates the validation of raw data
# files sets stored in a hierarchical directory structure.
#
# DESCRIPTION:
# This script validates raw data files sets organized in a specific directory
# structure. Each subdirectory of "raw" represents a main program's output,
# containing subdirectories corresponding to data files sets. The script:
#
# 1. Recursively traverses this structure to locate data files sets.
# 2. Uses timestamp checks to avoid re-validating unmodified sets unless the
#    "--all" flag is provided.
# 3. For each identified data files set:
#     a. Validates the set using "validate_raw_data_files_set.sh."
#     b. Updates a timestamp file to track validation.
#
# FLAGS:
# --all       Validate all data files sets, bypassing timestamp checks.
# -u, --usage Display usage instructions and exit.
#
# USAGE EXAMPLES:
#   ./validate_all_raw_data_files_sets.sh         # Validate only modified sets.
#   ./validate_all_raw_data_files_sets.sh --all   # Validate all sets.
#   ./validate_all_raw_data_files_sets.sh -u      # Show usage information.
#
# DEPENDENCIES:
# - Library scripts located in the "library" directory.
# - "validate_raw_data_files_set.sh" script for individual set validation.
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

DESTINATION_DIRECTORY="/nvme/h/cy22sg1/scratch/raw_qpb_data_files"
check_if_directory_exists "$DESTINATION_DIRECTORY" || exit 1

LOG_FILE="/nvme/h/cy22sg1/Data_analysis/rsync.log"
check_if_directory_exists "$DESTINATION_DIRECTORY" || exit 1

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
            continue
        fi

        echo "- Moving raw data files from '${data_files_set_name}/' directory."

        mv "${data_files_set_directory}"/* "${destination_subdirectory}/"

    done
done

echo
echo "Moving all raw data files sets completed!"

# Unset the library scripts path variable to avoid conflicts.
unset LIBRARY_SCRIPTS_DIRECTORY_PATH

# # Step 1: Compress subdirectories (depth-2) into .tar.gz files inside their
# # respective depth-1 subdirectory
# find "$SOURCE_DIR" -mindepth 2 -maxdepth 2 -type d | while read subdirDepth2; do
#     # Get the parent directory (depth-1 subdirectory)
#     parent_dir=$(dirname "$subdirDepth2")
    
#     # Ensure the subdirectory is correctly compressed into a tar.gz file
#     # inside the parent directory
#     tar -czf "$parent_dir/$(basename "$subdirDepth2").tar.gz" -C "$parent_dir" \
#                                                 "$(basename "$subdirDepth2")"
# done

# # Step 2: Rsync to backup files (only .tar.gz and markdown files, maintaining
# # depth-1 structure, and excluding depth-2 directories entirely)
# rsync -av --partial --delete --log-file="$LOG_FILE" \
#     --include="/*/" --include="*.tar.gz" --include="*.md" \
#     --exclude="/*/*/" --exclude="*" \
#     "$SOURCE_DIR/" "$DEST_DIR/"

# # Step 3: Clean up by deleting the .tar.gz files from the source directory
# # (keeping the subdirectories intact)
# find "$SOURCE_DIR" -type f -name "*.tar.gz" -exec rm -f {} \;
