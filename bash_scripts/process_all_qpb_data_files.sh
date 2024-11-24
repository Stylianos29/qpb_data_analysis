#!/bin/bash


################################################################################
# process_all_qpb_data_files.sh - Script for automating the processing of qpb
# data files stored in nested subdirectories. The script utilizes two Python
# scripts to handle the processing of qpb log files and qpb correlator data
# files.
#
# Functionalities:
# - Iterates through a specified raw data directory structure.
# - Identifies subdirectories corresponding to specific qpb main programs and
#   experiments.
# - Processes qpb log files and correlator data files (if present) within each
#   experiment directory using separate Python scripts.
# - Automatically creates corresponding output directories in the processed data
#   directory if they do not exist.
#
# Processing Logic:
# - **QPB Log Files:**
#   * The Python script `process_qpb_log_files.py` is called to process any
#     `.txt` files found in the experiment directory.
#   * The script automatically filters and processes qpb log files, generating
#     output files with appropriate names in the processed data directory.
#
# - **QPB Correlator Data Files:**
#   * The Python script `parse_qpb_correlator_files.py` is called to process any
#     `.dat` files found in the experiment directory.
#   * The script parses the correlator data and generates output in HDF5 format
#     in the processed data directory.
#
# Input:
# - Paths to the raw data and processed data directories, defined as script
#   variables.
# - The raw data directory is expected to have a structure where:
#   * First-level subdirectories represent qpb main programs.
#   * Second-level subdirectories represent specific experiments or analyses.
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


# Define paths for the source scripts, raw data files, and processed data files.
SOURCE_SCRIPTS_DIRECTORY="../core/src/data_files_processing"
RAW_DATA_FILES_DIRECTORY="../data_files/raw"
PROCESSED_DATA_FILES_DIRECTORY="../data_files/processed"

# Set a common output HDF5 filename for all qpb main programs and datasets
OUTPUT_HDF5_FILE_NAME="pion_correlators_values.h5"

# Loop over all subdirectories in the raw data files directory.
# These subdirectories are expected to represent the qpb main programs that 
# generated the respective data files.
for data_files_main_program_directory in "$RAW_DATA_FILES_DIRECTORY"/*; do

    # Ensure the current entry is a directory; skip otherwise.
    if [ ! -d "$data_files_main_program_directory" ]; then
        continue
    fi

    echo
    echo "Working within '${data_files_main_program_directory}':"

    # Loop over all subdirectories of the current main program directory.
    # These are expected to represent specific experiments or analyses.
    for data_files_experiment_directory in \
                                    "$data_files_main_program_directory"/*; do

        # Ensure the current entry is a directory; skip otherwise.
        if [ ! -d "$data_files_experiment_directory" ]; then
            continue
        fi

        # Extract name of data files set directory
        data_files_set_directory=$(basename $data_files_experiment_directory)

        echo "- '${data_files_set_directory}' data files set:"

        # Construct path to corresponding processed data files subdirectory
        output_directory_path=$PROCESSED_DATA_FILES_DIRECTORY
        output_directory_path+="/$(basename $data_files_main_program_directory)"
        output_directory_path+="/${data_files_set_directory}"

        # Create the processed data files subdirectory if it does not exit
        if [ ! -d "$output_directory_path" ]; then
            mkdir -p "$output_directory_path"
            warning_message="   ++ WARNING: Subdirectory "
            warning_message+="'${output_directory_path}' does not exist, so "
            warning_message+="it was created."
            echo "${warning_message}"
        fi

        # Initialize a boolean variable for warning purposes
        NO_DATA_FILES="True"

        # QPB LOG FILES DATASETS PROCESSING

        # Check if the current experiment subdirectory contains any .txt files
        if find "$data_files_experiment_directory" \
                            -maxdepth 1 -type f -name "*.txt" | grep -q .; then
            
            echo "   ++ It contains qpb log files."
            NO_DATA_FILES="False"

            # Call the Python script to process qpb log files in the current
            # experiment directory. The Python script:
            # - Distinguishes qpb log files from other file types automatically.
            # - Generates output files with appropriate names.
            python "${SOURCE_SCRIPTS_DIRECTORY}/process_qpb_log_files.py" \
                -qpb_log_dir "$data_files_experiment_directory" \
                -out_dir "$output_directory_path"

            # TODO: Include a tree graph in the output files
            # # Accompany the HDF5 file with a detailed tree graph of its structure
            # h5glance "$output_hdf5_file_path" >> "${output_hdf5_file_path%.h5}_tree.txt"
        fi

        # QPB CORRELATOR FILES PARSING

        # Check if the current experiment subdirectory contains any .dat files
        if find "$data_files_experiment_directory" \
                            -maxdepth 1 -type f -name "*.dat" | grep -q .; then

            echo "   ++ It contains qpb correlator files."
            NO_DATA_FILES="False"

            # Call the Python script to process correlator data file in the
            # current experiment directory. The Python script:
            # - Distinguishes correlator data files from other file types
            #   automatically.
            # - Generates output files with appropriate names.
            python "${SOURCE_SCRIPTS_DIRECTORY}/parse_qpb_correlator_files.py" \
                -raw_dir "$data_files_experiment_directory" \
                -out_dir "$output_directory_path" \
                -out_hdf5_file "$OUTPUT_HDF5_FILE_NAME"
                
            # TODO: Include a tree graph in the output files
            # # Accompany the HDF5 file with a detailed tree graph of its structure
            # h5glance "$output_hdf5_file_path" >> "${output_hdf5_file_path%.h5}_tree.txt"
        fi

        # If no data files were found print a warning
        if [ NO_DATA_FILES == "True" ]; then
            echo "   ++ WARNING: No data files were found in this directory!"
        fi

    done
done

echo
echo "Processing all qpb data_files completed!"

# TODO: Generate a summary files for each .csv file