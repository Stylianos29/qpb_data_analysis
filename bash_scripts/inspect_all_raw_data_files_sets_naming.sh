#!/bin/bash

################################################################################
# inspect_all_raw_data_files_sets_naming.sh - Automates the processing of raw
# data files sets stored in a hierarchical directory structure.
#
# DESCRIPTION:
# This script processes raw data files sets organized in a specific directory
# structure. Each subdirectory of "raw" represents a main program's output,
# containing subdirectories corresponding to data files sets. The script:
#
# 1. Recursively traverses this structure to locate data files sets.
# 2. Uses timestamp checks to avoid re-validating unmodified sets unless the
#    "--all" flag is provided.
# 3. For each identified data files set:
#     a. Processes the set using "process_all_raw_data_files_sets.sh."
#     b. Updates a timestamp file to track validation.
#
# FLAGS:
# --all       Process all raw data files sets, bypassing timestamp checks.
# -u, --usage Display usage instructions and exit.
#
# USAGE EXAMPLES:
#   ./process_all_raw_data_files_sets.sh         # Validate only modified sets.
#   ./process_all_raw_data_files_sets.sh --all   # Validate all sets.
#   ./process_all_raw_data_files_sets.sh -u      # Show usage information.
#
# DEPENDENCIES:
# - Library scripts located in the "library" directory.
# - "process_all_raw_data_files_sets.sh" script for individual set validation.
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

# RAW_DATA_FILES_DIRECTORY="../data_files/raw"
RAW_DATA_FILES_DIRECTORY="/nvme/h/cy22sg1/scratch/raw_qpb_data_files"
check_if_directory_exists "$RAW_DATA_FILES_DIRECTORY" || exit 1

PROCESSED_DATA_FILES_DIRECTORY="../data_files/processed"
check_if_directory_exists "$PROCESSED_DATA_FILES_DIRECTORY" || exit 1

export PYTHON_SCRIPTS_DIRECTORY="../core/src"
check_if_directory_exists "$PYTHON_SCRIPTS_DIRECTORY" || exit 1

WORKING_SCRIPT_NAME="inspect_raw_data_files_set_naming"
working_script_path="./checks/${WORKING_SCRIPT_NAME}.sh"
check_if_file_exists $working_script_path || exit 1

DEPENDED_SCRIPT_NAME="validate_raw_data_files_set"
depended_script_path="./checks/${DEPENDED_SCRIPT_NAME}.sh"
check_if_file_exists $depended_script_path || exit 1

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

# PROCESS ALL RAW DATA FILES SETS

# Traverse the "raw" directory to process main program subdirectories
for main_program_directory in "$RAW_DATA_FILES_DIRECTORY"/*; do
    # Skip non-directory entries
    [ -d "$main_program_directory" ] || continue

    echo
    echo "Working within '${main_program_directory/"../"}/':"

    # Process each data files set within the current main program directory
    for raw_data_files_set_directory in "$main_program_directory"/*; do
        # Skip non-directory entries.
        [ -d "$raw_data_files_set_directory" ] || continue

        # CONSTRUCT AUXILIARY FILES DIRECTORY PATH

        # Convert raw directory path to corresponding processed directory path
        processed_data_files_set_directory=$(replace_parent_directory \
            "$raw_data_files_set_directory" "$RAW_DATA_FILES_DIRECTORY" \
            "$PROCESSED_DATA_FILES_DIRECTORY")

        # Define the auxiliary files directory path.
        auxiliary_files_directory="${processed_data_files_set_directory}"
        auxiliary_files_directory+="/auxiliary_files"

        # Ensure the auxiliary files directory exists (create if necessary)
        check_if_directory_exists "$auxiliary_files_directory" -c -s

        # CHECK IF DATA FILES SET DIRECTORY WAS MODIFIED

        # Extract the data files set name
        data_files_set_name=$(basename "$raw_data_files_set_directory")

        # Define the path to the working script timestamp file
        working_script_timestamp_file_path=$(get_timestamp_file_path \
            "$raw_data_files_set_directory" "$auxiliary_files_directory" \
            "$WORKING_SCRIPT_NAME")

        # Ensure the working script timestamp file exists (create if necessary)
        check_if_file_exists "$working_script_timestamp_file_path" -c -s

        # Determine if the raw data files set directory has been modified
        modified_raw_data_files_set_flag=false
        if check_directory_for_changes "$raw_data_files_set_directory" \
                                    "$working_script_timestamp_file_path"; then
            modified_raw_data_files_set_flag=true
        fi

        # Define the path to the depended script timestamp file
        depended_script_timestamp_file_path=$(get_timestamp_file_path \
            "$raw_data_files_set_directory" "$auxiliary_files_directory" \
            "$DEPENDED_SCRIPT_NAME")

        # Ensure the depended script timestamp file exists (create if necessary)
        check_if_file_exists "$depended_script_timestamp_file_path" -c -s

        # Determine if the raw data files set has been validated properly
        validated_raw_data_files_set_flag=true
        if check_directory_for_changes "$raw_data_files_set_directory" \
                                    "$depended_script_timestamp_file_path"; then
            validated_raw_data_files_set_flag=false
        fi

        # Skip processing if raw data files set modified but not validated yet
        if $modified_raw_data_files_set_flag && \
                                    ! $validated_raw_data_files_set_flag; then
            error_message="- ERROR: '${data_files_set_name}' raw data "
            error_message+="files set has not been validated yet, skipping..."
            echo $error_message
            echo
            continue
        fi

        # Skip processing if raw data files set has not been modified and
        # "--all" is not specified
        if ! ($VALIDATE_ALL_FLAG || $modified_raw_data_files_set_flag); then
            warning_message="- Skipping '${data_files_set_name}' raw data "
            warning_message+="files set, already inspected."
            echo $warning_message
            continue
        fi

        # PROCESS CURRENT RAW DATA FILES SET

        echo "- Inspecting '${data_files_set_name}' raw data files set naming:"

        $working_script_path \
            --data_files_set_directory "$raw_data_files_set_directory" \
            --output_files_directory "$processed_data_files_set_directory" \
            --scripts_log_files_directory "$auxiliary_files_directory" \
            || { echo; continue; }

        # Update the timestamp file after successful processing.
        update_timestamp "$raw_data_files_set_directory" \
                                        "$working_script_timestamp_file_path"

    done

done

echo
echo "Inspecting all raw data files sets naming completed!"

# Unset the library scripts path variable to avoid conflicts.
unset PYTHON_SCRIPTS_DIRECTORY
unset LIBRARY_SCRIPTS_DIRECTORY_PATH