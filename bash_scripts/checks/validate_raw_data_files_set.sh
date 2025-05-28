#!/bin/bash

################################################################################
# validate_raw_data_files_set.sh - Validates the integrity and completeness of 
# QPB simulation data files within a given data files set directory.
#
# DESCRIPTION:
# This script performs validation checks on QPB (Quantum Parallel Bridge) 
# simulation data files in a specified directory. It verifies:
# 1. The presence and structure of error, log, and correlator files
# 2. The successful completion of simulations by checking for specific flags
# 3. The absence of numerical errors (NaN/Inf values) in output files
#
# The script is designed to be called either directly or by 
# validate_all_raw_data_files_sets.sh for batch processing.
#
# USAGE:
# ./validate_raw_data_files_set.sh -set_dir <data_dir> [-log_dir <aux_dir>] \
#                                 [-log_name <log_name>] [--disable-cache]
#
# FLAGS:
#   -set_dir, --data_files_set_directory   Directory containing raw data files
#   -log_dir, --auxiliary_files_directory  Directory for auxiliary files
#   -log_name, --script_log_filename       Custom log file name
#   --disable-cache                        Disable caching of validated paths
#   -u, --usage                           Display usage information
#
# EXAMPLES:
#   ./validate_raw_data_files_set.sh -set_dir /path/to/data/set
#   ./validate_raw_data_files_set.sh -set_dir /path/to/data/set \
#                                   -log_dir /path/to/aux \
#                                   -log_name custom_log
#
# DEPENDENCIES:
# - Python script: validate_qpb_data_files.py
# - Library scripts in bash_scripts/library/
#
# SUCCESS CRITERIA:
# - Non-invert log files: Contains "per stochastic source"
# - Invert log files: Contains "CG done"
# - Error files: No "terminated" messages
# - All files: No "nan" or "inf" values
#
# OUTPUT:
# - Generates a detailed log file with validation results
# - Returns 0 on successful validation, non-zero on failure
#
# NOTES:
# - If no auxiliary directory is specified, uses parent of data directory
# - Log files are created with .log extension if not specified
# - Cache is enabled by default to improve performance on large datasets
################################################################################

# CUSTOM FUNCTIONS DEFINITIONS

usage() {
    # Function to display usage information

    echo "Usage: $0 -set_dir <data_files_set_directory> [-log_dir <auxiliary_files_directory>] [-log_name <script_log_filename>] [--disable-cache]"
    echo "  -set_dir, --data_files_set_directory   Specify the directory containing raw files"
    echo "  -log_dir, --auxiliary_files_directory  Specify the directory for auxiliary files (default: parent of data files set directory)"
    echo "  -log_name, --script_log_filename       Specify the log file name (default: script name with .log extension)"
    echo "  --disable-cache                        Disable caching of validated file paths"
    exit 1
}

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

# ENVIRONMENT VARIABLES

CURRENT_SCRIPT_FULL_PATH=$(realpath "$0")
# Extract the current script's name from its full path
CURRENT_SCRIPT_NAME="$(basename "$CURRENT_SCRIPT_FULL_PATH")"
# Extract the current script's parent directory from its full path
CURRENT_SCRIPT_DIRECTORY="$(dirname "$CURRENT_SCRIPT_FULL_PATH")"
# Replace ".sh" with "_script.log" to create the default log file name
DEFAULT_SCRIPT_LOG_FILE_NAME=$(echo "$CURRENT_SCRIPT_NAME" | sed 's/\.sh$/_script.log/')
# Construct full path to library scripts directory if not set yet
if [ -z "$LIBRARY_SCRIPTS_DIRECTORY_PATH" ]; then
    LIBRARY_SCRIPTS_DIRECTORY_PATH=$(\
                            realpath "${CURRENT_SCRIPT_DIRECTORY}/../library")
    [[ ! -d "$LIBRARY_SCRIPTS_DIRECTORY_PATH" ]] \
                            && echo "Invalid library scripts path." && exit 1
fi
# Construct full path to python scripts directory if not set yet
if [ -z "$PYTHON_SCRIPTS_DIRECTORY" ]; then
    PYTHON_SCRIPTS_DIRECTORY=$(\
                        realpath "${CURRENT_SCRIPT_DIRECTORY}/../../core/src")
    check_if_directory_exists "$PYTHON_SCRIPTS_DIRECTORY" || exit 1
fi

# Export script termination message to be used for finalizing logging
export SCRIPT_TERMINATION_MESSAGE="\n\t\t"$(echo "$CURRENT_SCRIPT_NAME" \
                    | tr '[:lower:]' '[:upper:]')" SCRIPT EXECUTION TERMINATED"

# PARSE INPUT ARGUMENTS

data_files_set_directory=""
auxiliary_files_directory=""
script_log_file_name=""
enable_cache=true
while [[ $# -gt 0 ]]; do
    case $1 in
        -set_dir|--data_files_set_directory)
            data_files_set_directory="$2"
            shift 2
            ;;
        -log_dir|--auxiliary_files_directory)
            auxiliary_files_directory="$2"
            shift 2
            ;;
        -log_name|--script_log_filename)
            script_log_file_name="$2"
            shift 2
            ;;
        --disable-cache)
            enable_cache=false
            shift
            ;;
        -u|--usage)
            usage
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            usage
            ;;
    esac
done

# VALIDATE INPUT ARGUMENTS

# Ensure a data files set directory path is provided
if [ -z "$data_files_set_directory" ]; then
    echo "ERROR: No data files set directory path specified."
    usage
fi
# Verify the data files set directory exists
check_directory_exists "$data_files_set_directory"
data_files_set_directory_name=$(basename $data_files_set_directory)

# Check if an auxiliary files directory is provided
if [ -z "$auxiliary_files_directory" ]; then
    # if not, then set it to the parent of the data files set directory
    auxiliary_files_directory=$(dirname $data_files_set_directory)
else
    # if it was provided, then check if it exists
    check_directory_exists "$auxiliary_files_directory"
fi

# Set log filename to default if not provided
if [ -z "$script_log_file_name" ]; then
    script_log_file_name=$DEFAULT_SCRIPT_LOG_FILE_NAME
else
    # Check if the provided script log filename ends with a ".log". If it does
    # not, append the ".log" extension to it.
    if [[ "$script_log_file_name" != *.log ]]; then
        script_log_file_name="${script_log_file_name}.log"
    fi
fi

# INITIATE LOGGING

# Export log file path as a global variable to be used by custom functions
SCRIPT_LOG_FILE_PATH="${auxiliary_files_directory}/${script_log_file_name}"
export SCRIPT_LOG_FILE_PATH

# Create or override a log file. Initiate logging
echo -e "\t\t"$(echo "$CURRENT_SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') \
                "SCRIPT EXECUTION INITIATED\n" > "$SCRIPT_LOG_FILE_PATH"

log_message="Initiate inspecting '${data_files_set_directory_name}' "
log_message+="data files set directory."
log "INFO" "$log_message"

# VALIDATE THE RAW SATA FILES OF THE DATA FILES SET

python_script_path="${PYTHON_SCRIPTS_DIRECTORY}"
python_script_path+="/utils/validate_qpb_data_files.py"
check_if_file_exists "$python_script_path" || exit 1

python $python_script_path \
    --raw_data_files_set_directory_path "$data_files_set_directory" \
    --enable_logging \
    --auxiliary_files_directory "$auxiliary_files_directory" \
    || failed_python_script $python_script_path

log_message="All qpb data files of the ${data_files_set_directory} data "
log_message+="files set were successfully validated."
log "INFO" "$log_message"

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
