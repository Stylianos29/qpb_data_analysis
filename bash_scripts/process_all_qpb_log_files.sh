#!/bin/bash


################################################################################
# process_all_qpb_log_files.sh - Script for automating the processing of qpb
# log files stored in nested subdirectories, using a Python script to handle
# the processing logic.
#
# Functionalities:
# - Iterates through a specified raw data directory structure.
# - Identifies subdirectories corresponding to specific qpb main programs and
#   experiments.
# - Passes each experiment directory to a Python script for processing.
# - The Python script filters and processes only qpb log files within each
#   directory and saves the output in the corresponding processed data
#   directory.
#
# Input:
# - Paths to the raw data and processed data directories, defined in the
#   script variables.
# - The raw data directory is expected to have a structure where:
#   * First-level subdirectories refer to qpb main programs.
#   * Second-level subdirectories refer to specific experiments or analyses.
#
# Output:
# - Processed files generated by the Python script are saved in a parallel
#   directory structure under the processed data directory.
################################################################################


# Define paths for the source scripts, raw data files, and processed data files.
SOURCE_SCRIPTS_DIRECTORY="../src"
RAW_DATA_FILES_DIRECTORY="../data_files/raw"
PROCESSED_DATA_FILES_DIRECTORY="../data_files/processed"

# Loop over all subdirectories in the raw data files directory.
# These subdirectories are expected to represent the qpb main programs that 
# generated the respective data files.
for data_files_main_program_directory in "$RAW_DATA_FILES_DIRECTORY"/*; do

    # Ensure the current entry is a directory; skip otherwise.
    if [ ! -d "$data_files_main_program_directory" ]; then
        continue
    fi

    # Loop over all subdirectories of the current main program directory.
    # These are expected to represent specific experiments or analyses.
    for data_files_experiment_directory in \
                                    "$data_files_main_program_directory"/*; do

        # Ensure the current entry is a directory; skip otherwise.
        if [ ! -d "$data_files_experiment_directory" ]; then
            continue
        fi

        # Construct the corresponding output directory path by replacing the base 
        # raw data directory path with the processed data directory path.
output_directory_path="${data_files_experiment_directory/$RAW_DATA_FILES_DIRECTORY\
/$PROCESSED_DATA_FILES_DIRECTORY}"

        # Call the Python script to process qpb log files in the current 
        # experiment directory.
        # The Python script:
        # - Distinguishes qpb log files from other file types automatically.
        # - Generates output files with appropriate names.
        python "${SOURCE_SCRIPTS_DIRECTORY}/process_qpb_log_files.py" \
            -qpb_log_dir "$data_files_experiment_directory" \
            -out_dir "$output_directory_path"

    done
done
