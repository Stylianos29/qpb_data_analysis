#!/bin/bash

################################################################################
# analyze_all_invert_processed_data_files_sets.sh - Automates the processing of
# raw data files sets stored in a hierarchical directory structure.
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

export PROCESSED_DATA_FILES_DIRECTORY="../data_files/processed"
check_if_directory_exists "$PROCESSED_DATA_FILES_DIRECTORY" || exit 1

export PYTHON_SCRIPTS_DIRECTORY="../core/src"
check_if_directory_exists "$PYTHON_SCRIPTS_DIRECTORY" || exit 1

export PLOTS_DIRECTORY="../output/plots"
check_if_directory_exists "$PLOTS_DIRECTORY" || exit 1

WORKING_SCRIPT_NAME="analyze_invert_processed_data_files_set"
working_script_path="./workflows/${WORKING_SCRIPT_NAME}.sh"
check_if_file_exists $working_script_path || exit 1

DEPENDED_SCRIPT_NAME="process_raw_data_files_set"
depended_script_path="./workflows/${DEPENDED_SCRIPT_NAME}.sh"
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

        # if [[ ! $data_files_set_name == 'Chebyshev_several_config_varying_N' ]]; then
        if [[ ! $data_files_set_name == 'KL_several_config_varying_n' ]]; then
            continue
        fi

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

        # Determine if the raw data files set has been processed properly
        processed_raw_data_files_set_flag=true
        if check_directory_for_changes "$raw_data_files_set_directory" \
                                "$depended_script_timestamp_file_path"; then
            processed_raw_data_files_set_flag=false
        fi

        is_invert_flag=false
        if $processed_raw_data_files_set_flag; then
            # Check if the current data files set directory contains any .dat files
            if find "$raw_data_files_set_directory" -maxdepth 1 -type f \
                                    -name "*.dat" -print -quit | grep -q .; then
                is_invert_flag=true
            fi
        fi

        if ! $is_invert_flag; then
            warning_message="- Skipping '${data_files_set_name}' processed "
            warning_message+="data files set, not generated from an 'invert' "
            warning_message+="main program."
            echo $warning_message
            continue
        fi

        # Skip processing if raw data files set modified but not processed yet
        if $modified_raw_data_files_set_flag && \
                                    ! $processed_raw_data_files_set_flag; then
            error_message="- ERROR: '${data_files_set_name}' processed data "
            error_message+="files set has not been processed yet, skipping..."
            echo $error_message
            echo
            continue
        fi

        # Skip processing if raw data files set has not been modified and
        # "--all" is not specified
        if ! ($VALIDATE_ALL_FLAG || $modified_raw_data_files_set_flag); then
            warning_message="- Skipping '${data_files_set_name}' processed "
            warning_message+="data files set, already analyzed."
            echo $warning_message
            continue
        fi

        # PROCESS CURRENT RAW DATA FILES SET

        message="- Analyzing '${data_files_set_name}' processed invert "
        message+="data files set:"
        echo $message
        
        $working_script_path \
            --data_files_set_directory "$processed_data_files_set_directory" \
            --scripts_log_files_directory "$auxiliary_files_directory" \
            || { echo; continue; }

        # Update the timestamp file after successful processing.
        update_timestamp "$raw_data_files_set_directory" \
                                        "$working_script_timestamp_file_path"

    done

done

echo
echo "Analyzing all invert data files sets completed!"

# Unset the library scripts path variable to avoid conflicts.
unset PLOTS_DIRECTORY
unset PYTHON_SCRIPTS_DIRECTORY
unset PROCESSED_DATA_FILES_DIRECTORY
unset LIBRARY_SCRIPTS_DIRECTORY_PATH
