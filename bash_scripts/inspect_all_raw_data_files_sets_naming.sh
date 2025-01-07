#!/bin/bash

################################################################################
# validate_all_raw_data_files_sets.sh - Script to automate the 
#
# FLAGS:
# --all        Validates all data files sets, bypassing timestamp checks.
# -u, --usage  Displays this usage message and exits.
#
# USAGE EXAMPLES:
#
################################################################################

# SOURCE DEPENDENCIES

export LIBRARY_SCRIPTS_DIRECTORY_PATH=$(realpath "./library")
[[ ! -d "$LIBRARY_SCRIPTS_DIRECTORY_PATH" ]] \
                            && echo "Invalid library scripts path." && exit 1
# Source all library scripts from "bash_scripts/library" using a loop avoiding
# this way name-specific sourcing and thus potential typos
for library_script in "${LIBRARY_SCRIPTS_DIRECTORY_PATH}"/*.sh; do
    # Check if the current file in the loop is a regular file
    if [ -f "$library_script" ]; then
        source "$library_script"
    fi
done

# ENVIRONMENT VARIABLES

RAW_DATA_FILES_DIRECTORY="../data_files/raw"
check_directory_exists $RAW_DATA_FILES_DIRECTORY
PROCESSED_DATA_FILES_DIRECTORY="../data_files/processed"
check_directory_exists $PROCESSED_DATA_FILES_DIRECTORY

ALL_FLAG=false

# PARSE FLAGS

function usage() {
    echo "Usage: $0 [--all] [-u | --usage]"
    echo
    echo "Flags:"
    echo " --all      Validate all data files sets, bypassing timestamp checks."
    echo " -u, --usage  Display this usage message and exit."
    echo
    exit 0
}

# Process arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --all)
            ALL_FLAG=true
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

# CHECK ALL DATA FILES SETS

# Loop over all subdirectories in the raw data files directory. These
# subdirectories are expected to represent the qpb main programs that generated
# the respective data files.
for main_program_directory in "$RAW_DATA_FILES_DIRECTORY"/*; do

    # Ensure the current entry is a directory; skip otherwise.
    if [ ! -d "$main_program_directory" ]; then
        continue
    fi

    echo
    echo "Working within '${main_program_directory/"../"}/':"

    # Loop over all subdirectories of the current main program directory. These
    # are expected to represent specific data files sets.
    for data_files_set_directory in "$main_program_directory"/*; do

        # Ensure the current entry is a directory; skip otherwise.
        if [ ! -d "$data_files_set_directory" ]; then
            continue
        fi

        # Extract name of data files set directory
        data_files_set_name=$(basename "$data_files_set_directory")

        # Construct the corresponding data files set processed data directory
        processed_data_files_set_directory=$(replace_parent_directory\
                "$data_files_set_directory" "$RAW_DATA_FILES_DIRECTORY" \
                                            "$PROCESSED_DATA_FILES_DIRECTORY")
        if [ ! -d "$processed_data_files_set_directory" ]; then
            mkdir -p "$processed_data_files_set_directory"
        fi

        # Construct auxiliary files directory
        auxiliary_files_directory="${processed_data_files_set_directory}"
        auxiliary_files_directory+="/auxiliary_files"
        if [ ! -d "$auxiliary_files_directory" ]; then
            mkdir -p "$auxiliary_files_directory"
        fi

        # Check if data files set directory is new or has been modified in any
        # way to proceed with validation, otherwise skip
        if ! $ALL_FLAG && ! check_directory_for_changes \
                                "$data_files_set_directory"\
                                    "$auxiliary_files_directory"; then
            warning_message="- Skipping '${data_files_set_name}' data files "
            warning_message+="set, already validated."
            echo "$warning_message"
            continue
        fi

        echo "- Inspecting '${data_files_set_name}/' directory:"

        # Validate specific data files set
        ./checks/check_data_files_set_naming.sh \
            -p "$data_files_set_directory" \
            -l "$auxiliary_files_directory"

    done
done

echo
echo "Checking all data files sets naming completed!"

unset LIBRARY_SCRIPTS_DIRECTORY_PATH
