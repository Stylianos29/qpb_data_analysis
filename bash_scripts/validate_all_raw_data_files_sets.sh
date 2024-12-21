#!/bin/bash

################################################################################
# process_all_qpb_data_files.sh - Script for automating the processing of qpb
#
################################################################################

# Define paths for the source scripts, raw data files, and processed data files.
RAW_DATA_FILES_DIRECTORY="../data_files/raw"

# Loop over all subdirectories in the raw data files directory.
# These subdirectories are expected to represent the qpb main programs that 
# generated the respective data files.
for main_program_directory in "$RAW_DATA_FILES_DIRECTORY"/*; do

    # Ensure the current entry is a directory; skip otherwise.
    if [ ! -d "$main_program_directory" ]; then
        continue
    fi

    echo
    echo "Working within '${main_program_directory}':"

    # Loop over all subdirectories of the current main program directory.
    # These are expected to represent specific data files sets.
    for data_files_set_directory in "$main_program_directory"/*; do

        # Ensure the current entry is a directory; skip otherwise.
        if [ ! -d "$data_files_set_directory" ]; then
            continue
        fi

        # Extract name of data files set directory
        data_files_set_name=$(basename $data_files_set_directory)

        echo "- '${data_files_set_name}' data files set:"

    done
done

echo
echo "Processing all qpb data_files completed!"