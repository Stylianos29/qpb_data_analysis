#!/bin/bash

################################################################################
# process_raw_data_files_set.sh - Script for automating the processing of qpb
# data files stored in nested subdirectories. The script utilizes two Python
# scripts to handle the processing of qpb log files and qpb correlator data
# files.
#
# Functionalities:
# - Iterates through a specified raw data directory structure.
# - Identifies subdirectories corresponding to specific qpb main programs and
#   data files sets.
# - Processes qpb log files and correlator data files (if present) within each
#   data files sets directory using the appropriate Python scripts.
# - Automatically creates corresponding output directories in the processed data
#   directory if they do not exist.
#
# Processing Logic:
# - **QPB Log Files:**
#   * The Python script `process_qpb_log_files.py` is called to process any
#     `.txt` files found in the data files sets directory.
#   * The script automatically filters and processes qpb log files, generating
#     output files with appropriate names in the processed data directory.
#
# - **QPB Correlator Data Files:**
#   * The Python script `parse_qpb_correlator_files.py` is called to process any
#     `.dat` files found in the data files sets directory.
#   * The script parses the correlator data and generates output in HDF5 format
#     in the processed data directory.
#
# Input:
# - Paths to the raw data and processed data directories, defined as script
#   variables.
# - The raw data directory is expected to have a structure where:
#   * First-level subdirectories represent qpb main programs.
#   * Second-level subdirectories represent specific experiments or analyses
#     (data file sets).
#
# Output:
# - Processed files are saved in a parallel directory structure under the
#   processed data directory. These include:
#   * Output files from processing `.txt` files (qpb log files).
#   * Output files from parsing `.dat` files (correlator data files).
#
# Directory Structure Assumptions:
# - The raw data directory structure is as follows:
#   * `<RAW_DATA_FILES_DIRECTORY>/<qpb_main_program>/<experiment_directory>/`
# - The processed data directory mirrors this structure:
#   * `<PROCESSED_DATA_FILES_DIRECTORY>/<qpb_main_program>/<experiment_directory>/`
#
# Additional Notes:
# - The script ensures that only directories are processed, skipping invalid
#   entries.
# - If a required output subdirectory does not exist, it is created
#   automatically with a warning message.
# - Both Python scripts handle their own filtering logic for file types within
#   the given directory.
################################################################################

# CUSTOM FUNCTIONS DEFINITIONS

usage() {
    # Function to display usage information

    echo "Usage: $0 -p <raw_data_files_set_directory>"
    echo "  -p, --set_dir   Specify the directory containing raw files"
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
# Construct full path to library scripts directory if not set yet
if [ -z "$LIBRARY_SCRIPTS_DIRECTORY_PATH" ]; then
    LIBRARY_SCRIPTS_DIRECTORY_PATH=$(\
                            realpath "${CURRENT_SCRIPT_DIRECTORY}/../library")
    check_if_directory_exists "$LIBRARY_SCRIPTS_DIRECTORY_PATH" || exit 1
fi
# Construct full path to python scripts directory if not set yet
if [ -z "$PYTHON_SCRIPTS_DIRECTORY" ]; then
    PYTHON_SCRIPTS_DIRECTORY=$(\
                        realpath "${CURRENT_SCRIPT_DIRECTORY}/../../core/src")
    check_if_directory_exists "$PYTHON_SCRIPTS_DIRECTORY" || exit 1
fi

# Set common output filenames independent of the data files set name
PREPROCESSED_CSV_FILENAME="single_valued_parameters_values.csv"
PREPROCESSED_HDF5_FILENAME="multivalued_parameters_values.h5"
PROCESSED_VALUES_CSV_FILENAME="processed_parameter_values.csv"
CORRELATORS_VALUES_HDF5_FILENAME="pion_correlators_values.h5"

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

raw_data_files_set_directory=""
output_directory_path=""
auxiliary_files_directory=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -set_dir|--data_files_set_directory)
            raw_data_files_set_directory="$2"
            shift 2
            ;;
        -out_dir|--output_files_directory)
            output_directory_path="$2"
            shift 2
            ;;
        -log_dir|--scripts_log_files_directory)
            auxiliary_files_directory="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            usage
            ;;
    esac
done

# VALIDATE INPUT

# Ensure a data files set directory path was provided
if [ -z "$raw_data_files_set_directory" ]; then
    echo "ERROR: No raw data files set directory path specified."
    usage
fi
# Ensure the raw data files set directory exists
check_if_directory_exists "$raw_data_files_set_directory" || exit 1

# Extract name of raw data files set directory 
data_files_set_directory_name=$(basename $raw_data_files_set_directory)

# Check if an output directory was provided
if [ -z "$output_directory_path" ]; then
    # if not, then set it to the parent of the data files set directory
    output_directory_path=$(dirname $raw_data_files_set_directory)
else
    # if it was provided, then check if it exists
    check_if_directory_exists "$auxiliary_files_directory" || exit 1
fi

# Check if a log directory was provided
if [ -z "$auxiliary_files_directory" ]; then
    # if not, then set it to be the same with the output directory
    auxiliary_files_directory=$output_directory_path
else
    # if it was provided, then check if it exists
    check_if_directory_exists "$auxiliary_files_directory" || exit 1
fi

# INITIATE LOGGING

# Export log file path as a global variable to be used by custom functions
SCRIPT_LOG_FILE_PATH="${auxiliary_files_directory}/${SCRIPT_LOG_FILE_NAME}"
export SCRIPT_LOG_FILE_PATH

# Create or override a log file. Initiate logging
echo -e "\t\t"$(echo "$CURRENT_SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') \
                "SCRIPT EXECUTION INITIATED\n" > "$SCRIPT_LOG_FILE_PATH"

log_message="Initiate inspecting '${data_files_set_directory_name}' "
log_message+="data files set directory."
log "INFO" "$log_message"

# QPB LOG FILES DATASETS PROCESSING

# Check if provided raw data files set directory contains any .txt files
# NOTE: This is a rudimentary check on the raw data files set but not a proper
# validation. The assumption is that the set has been already properly validated
if ! find "$raw_data_files_set_directory" \
                -maxdepth 1 -type f -name "*.txt" -print -quit | grep -q .; then
    error_message="No qpb log files found in '$raw_data_files_set_directory'."
    termination_output "$error_message"
    exit 1
fi

# EXTRACT VALUES FROM QPB LOG FILES

# Process all qpb log files of the data files set by extracting all useful
# pieces of information they might contain without any further processing
python_script_path="${PYTHON_SCRIPTS_DIRECTORY}"
python_script_path+="/data_files_processing/process_qpb_log_files.py"
check_if_file_exists "$python_script_path" || exit 1

python $python_script_path \
    --qpb_log_files_directory "$raw_data_files_set_directory" \
    --output_files_directory "$output_directory_path" \
    --output_csv_filename "$PREPROCESSED_CSV_FILENAME" \
    --output_hdf5_filename "$PREPROCESSED_HDF5_FILENAME" \
    --enable_logging \
    --log_file_directory "$auxiliary_files_directory" \
    || failed_python_script $python_script_path

log_message="All the qpb log files of the ${raw_data_files_set_directory} data "
log_message+="files set were successfully processed."
log "INFO" "$log_message"

# Generate a summary for the output .csv file and save it to a text file
preprocessed_csv_file_path="${output_directory_path}/"
preprocessed_csv_file_path+="$PREPROCESSED_CSV_FILENAME"
check_if_file_exists "$preprocessed_csv_file_path" || exit 1

python_script_path="${PYTHON_SCRIPTS_DIRECTORY}"
python_script_path+="/utils/inspect_csv_file.py"
check_if_file_exists "$python_script_path" || exit 1

python $python_script_path \
    --csv_file_path "$preprocessed_csv_file_path" \
    || failed_python_script $python_script_path

log_message="A summary of the ${PREPROCESSED_CSV_FILENAME} .csv file "
log_message+="was generated."
log "INFO" "$log_message"

# Generate and save the tree structure of the output HDF5 file
preprocessed_hdf5_file_path="${output_directory_path}/"
preprocessed_hdf5_file_path+="$PREPROCESSED_HDF5_FILENAME"
check_if_file_exists "$preprocessed_hdf5_file_path" || exit 1

h5glance "$preprocessed_hdf5_file_path" \
                                > "${preprocessed_hdf5_file_path%.h5}_tree.txt"

log_message="A hierarchy tree of the ${PREPROCESSED_HDF5_FILENAME} HDF5 file "
log_message+="was generated."
log "INFO" "$log_message"

# PROCESS EXTRACTED VALUES

# Process the extracted values from the qpb log files from the 
python_script_path="${PYTHON_SCRIPTS_DIRECTORY}/post_processing_analysis"
python_script_path+="/process_qpb_log_files_extracted_values.py"
check_if_file_exists "$python_script_path" || exit 1

python $python_script_path \
    --input_single_valued_csv_file_path $preprocessed_csv_file_path \
    --input_multivalued_hdf5_file_path $preprocessed_hdf5_file_path \
    --output_files_directory "$output_directory_path" \
    --output_csv_filename $PROCESSED_VALUES_CSV_FILENAME \
    --log_file_directory "$auxiliary_files_directory" \
    || failed_python_script $python_script_path

log_message="All the qpb log files of the ${raw_data_files_set_directory} data "
log_message+="files set were successfully processed."
log "INFO" "$log_message"

# Generate a summary for output .csv file and save it to a text file
processed_csv_file_path="${output_directory_path}/"
processed_csv_file_path+="$PROCESSED_VALUES_CSV_FILENAME"
check_if_file_exists "$processed_csv_file_path" || exit 1

python_script_path="${PYTHON_SCRIPTS_DIRECTORY}"
python_script_path+="/utils/inspect_csv_file.py"
check_if_file_exists "$python_script_path" || exit 1

python $python_script_path \
    --csv_file_path "$processed_csv_file_path" \
    || failed_python_script $python_script_path

log_message="A summary of the ${PROCESSED_VALUES_CSV_FILENAME} .csv file "
log_message+="was generated."
log "INFO" "$log_message"

# QPB CORRELATOR FILES PARSING

# Check if the current data files set directory contains any .dat files
if find "$raw_data_files_set_directory" \
            -maxdepth 1 -type f -name "*.dat" -print -quit | grep -q .; then

    # Parse 
    python_script_path="${PYTHON_SCRIPTS_DIRECTORY}"
    python_script_path+="/data_files_processing/parse_qpb_correlator_files.py"
    check_if_file_exists "$python_script_path" || exit 1

    python $python_script_path \
        --qpb_correlators_files_directory "$raw_data_files_set_directory" \
        --output_files_directory "$output_directory_path" \
        --output_hdf5_filename "$CORRELATORS_VALUES_HDF5_FILENAME" \
        --enable_logging \
        --log_file_directory "$auxiliary_files_directory" \
    || failed_python_script $python_script_path

    log_message="All the qpb log files of the ${raw_data_files_set_directory} "
    log_message+="data files set were successfully processed."
    log "INFO" "$log_message"

    correlators_values_hdf5_file_path="${output_directory_path}/"
    correlators_values_hdf5_file_path+="$CORRELATORS_VALUES_HDF5_FILENAME"
    check_if_file_exists "$correlators_values_hdf5_file_path" || exit 1

    # Generate and save the tree structure of the output HDF5 file
    h5glance "$correlators_values_hdf5_file_path" \
                        > "${correlators_values_hdf5_file_path%.h5}_tree.txt"
    
    log_message="A hierarchy tree of the ${CORRELATORS_VALUES_HDF5_FILENAME} "
    log_message+="HDF5 file was generated."
    log "INFO" "$log_message"
fi

# SUCCESSFUL COMPLETION OUTPUT

# Construct the final message
final_message="'${data_files_set_directory_name}' raw data files set "
final_message+="processing completed!"
# Print the final message
echo "!! $final_message"

log "INFO" "${final_message}"
echo # Empty line

echo -e $SCRIPT_TERMINATION_MESSAGE >> "$SCRIPT_LOG_FILE_PATH"

unset SCRIPT_TERMINATION_MESSAGE
unset SCRIPT_LOG_FILE_PATH
