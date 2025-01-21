#!/bin/bash

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

export LIBRARY_SCRIPTS_DIRECTORY_PATH=$(realpath "./library")
if [[ ! -d "$LIBRARY_SCRIPTS_DIRECTORY_PATH" ]]; then
    echo "Invalid library scripts path."
    exit 1
fi
# Source all library scripts to load utility functions.
for library_script in "${LIBRARY_SCRIPTS_DIRECTORY_PATH}"/*.sh; do
    [ -f "$library_script" ] && source "$library_script"
done

# ENVIRONMENT VARIABLES

RAW_DATA_FILES_DIRECTORY="../data_files/raw"
check_if_directory_exists "$RAW_DATA_FILES_DIRECTORY" || exit 1

PROCESSED_DATA_FILES_DIRECTORY="../data_files/processed"
check_if_directory_exists "$PROCESSED_DATA_FILES_DIRECTORY" || exit 1

WORKING_SCRIPT_NAME="validate_raw_data_files_set"
working_script_path="./checks/${WORKING_SCRIPT_NAME}.sh"
check_if_file_exists $working_script_path || exit 1

VALIDATE_ALL_FLAG=false #Initialize flag

# PARSE FLAGS

function usage() {
    echo "Usage: $0 [--all] [-u | --usage]"
    echo
    echo "Flags:"
    echo " --all       Validate all data files sets, bypassing timestamp checks."
    echo " -u, --usage Display this usage message and exit."
    echo
    exit 0
}

# Process command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --all)
            VALIDATE_ALL_FLAG=true
            shift
            ;;
        -u|--usage)
            usage
            ;;
        *)
            echo "Error: Unknown flag '$1'."
            usage
            ;;
    esac
done

# VALIDATE ALL RAW DATA FILES SETS

# Traverse the "raw" directory to process main program subdirectories
for main_program_directory in "$RAW_DATA_FILES_DIRECTORY"/*; do
    # Skip non-directory entries
    [ -d "$main_program_directory" ] || continue

    echo
    echo "Working within '${main_program_directory/"../"}/':"

    # Process each data files set within the current main program directory
    for data_files_set_directory in "$main_program_directory"/*; do
        # Skip non-directory entries.
        [ -d "$data_files_set_directory" ] || continue

        # CONSTRUCT AUXILIARY FILES DIRECTORY PATH

        # Convert raw directory path to corresponding processed directory path
        processed_data_files_set_directory=$(replace_parent_directory \
            "$data_files_set_directory" "$RAW_DATA_FILES_DIRECTORY" \
            "$PROCESSED_DATA_FILES_DIRECTORY")

        # Define the auxiliary files directory path.
        auxiliary_files_directory="${processed_data_files_set_directory}"
        auxiliary_files_directory+="/auxiliary_files"

        # Ensure the auxiliary files directory exists (create if necessary)
        check_if_directory_exists "$auxiliary_files_directory" -c -s

        # CHECK IF DATA FILES SET DIRECTORY WAS MODIFIED

        # Extract the data files set name
        data_files_set_name=$(basename "$data_files_set_directory")

        # Define the path to the timestamp file
        timestamp_file_path=$(get_timestamp_file_path \
            "$data_files_set_directory" "$auxiliary_files_directory" \
            "$WORKING_SCRIPT_NAME")

        # Ensure the timestamp file exists (create if necessary)
        check_if_file_exists "$timestamp_file_path" -c -s

        # Determine if the data files set directory has been modified
        modified_raw_data_files_set_flag=false
        if check_directory_for_changes "$data_files_set_directory" \
                                                "$timestamp_file_path"; then
            modified_raw_data_files_set_flag=true
        fi

        # Skip validation if not modified and "--all" is not specified
        if ! ($VALIDATE_ALL_FLAG || $modified_raw_data_files_set_flag); then
            warning_message="- Skipping '${data_files_set_name}' raw data "
            warning_message+="files set, already validated."
            echo $warning_message
            continue
        fi

        # VALIDATE CURRENT RAW DATA FILES SET

        echo "- Inspecting '${data_files_set_name}/' directory:"

        $working_script_path \
            --data_files_set_directory "$data_files_set_directory" \
            --scripts_log_files_directory "$auxiliary_files_directory" \
            || { echo; continue; }

        # Update the timestamp file after successful validation.
        update_timestamp "$data_files_set_directory" "$timestamp_file_path"
    done

done

echo
echo "Validating all raw data files sets completed!"

# Unset the library scripts path variable to avoid conflicts.
unset LIBRARY_SCRIPTS_DIRECTORY_PATH
