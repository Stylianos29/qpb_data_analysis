#!/bin/bash

################################################################################
# check_raw_data_files_set_naming.sh
#
# Description: 
#
# Purpose:
#
# Usage:
#
# Flags:
#
# Note:
################################################################################

# CUSTOM FUNCTIONS DEFINITIONS

usage() {
    # Function to display usage information

    echo "Usage: $0 -p <data_files_set_directory>"
    echo "  -p, --path   Specify the directory containing raw files"
    exit 1
}

# ENVIRONMENT VARIABLES

CURRENT_SCRIPT_FULL_PATH=$(realpath "$0")
# Extract the current script's name from its full path
CURRENT_SCRIPT_NAME="$(basename "$CURRENT_SCRIPT_FULL_PATH")"
# Extract the current script's parent directory from its full path
CURRENT_SCRIPT_DIRECTORY="$(dirname "$CURRENT_SCRIPT_FULL_PATH")"
# Replace ".sh" with "_script.log" to create the log file name
SCRIPT_LOG_FILE_NAME=$(echo "$CURRENT_SCRIPT_NAME" | sed 's/\.sh$/_script.log/')
# Construct full path of library scripts directory if not set yet
if [ -z "$LIBRARY_SCRIPTS_DIRECTORY_PATH" ]; then
    LIBRARY_SCRIPTS_DIRECTORY_PATH=$(\
                            realpath "${CURRENT_SCRIPT_DIRECTORY}/../library")
    [[ ! -d "$LIBRARY_SCRIPTS_DIRECTORY_PATH" ]] \
                            && echo "Invalid library scripts path." && exit 1
fi

NON_INVERT_LOG_FILES_SUCCESS_FLAG="per stochastic source"
INVERT_LOG_FILES_SUCCESS_FLAG="CG done"
ERROR_FILES_FAILURE_FLAG="terminated"
NON_NUMERICAL_FAILURE_FLAG="= nan|= inf"

# Export script termination message to be used for finalizing logging
export SCRIPT_TERMINATION_MESSAGE="\n\t\t"$(echo "$CURRENT_SCRIPT_NAME" \
                    | tr '[:lower:]' '[:upper:]')" SCRIPT EXECUTION TERMINATED"

# SOURCE DEPENDENCIES

# Source all library scripts from "bash_scripts/library" using a loop avoiding
# this way name-specific sourcing and thus potential typos
for library_script in "${LIBRARY_SCRIPTS_DIRECTORY_PATH}"/*.sh;
do
    # Check if the current file in the loop is a regular file
    if [ -f "$library_script" ]; then
        source "$library_script"
    fi
done

# PARSE INPUT ARGUMENTS

data_files_set_directory=""
script_log_file_directory=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--path)
            data_files_set_directory="$2"
            shift 2
            ;;
        -l|--log)
            script_log_file_directory="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            usage
            ;;
    esac
done

# VALIDATE INPUT

# Ensure a data files set directory path is provided
if [ -z "$data_files_set_directory" ]; then
    echo "ERROR: No data files set directory path specified."
    usage
fi
# Verify the data files set directory exists
check_directory_exists "$data_files_set_directory"
data_files_set_directory_name=$(basename $data_files_set_directory)

# Check if a log directory is provided
if [ -z "$script_log_file_directory" ]; then
    # if not, then set it to the parent of the data files set directory
    script_log_file_directory=$(dirname $data_files_set_directory)
else
    # if it was provided, then check if it exists
    check_directory_exists "$script_log_file_directory"
fi

# INITIATE LOGGING

# Export log file path as a global variable to be used by custom functions
SCRIPT_LOG_FILE_PATH="${script_log_file_directory}/${SCRIPT_LOG_FILE_NAME}"
export SCRIPT_LOG_FILE_PATH

# Create or override a log file. Initiate logging
echo -e "\t\t"$(echo "$CURRENT_SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') \
                "SCRIPT EXECUTION INITIATED\n" > "$SCRIPT_LOG_FILE_PATH"

log_message="Initiate inspecting '${data_files_set_directory_name}' "
log_message+="data files set directory."
log "INFO" "$log_message"

# MAIN ???????????????????

# SUCCESSFUL COMPLETION OUTPUT

# Construct the final message
final_message="'${data_files_set_directory_name}' data files set "
final_message+="validation completed!"
# Print the final message
echo "!! $final_message"

log "INFO" "${final_message}"
echo # Empty line

echo -e $SCRIPT_TERMINATION_MESSAGE >> "$SCRIPT_LOG_FILE_PATH"

unset SCRIPT_TERMINATION_MESSAGE
unset SCRIPT_LOG_FILE_PATH
