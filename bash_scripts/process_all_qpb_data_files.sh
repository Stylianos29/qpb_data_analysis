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


# Define paths for the source scripts, raw data files, and processed data files.
SOURCE_SCRIPTS_DIRECTORY="../core/src"
RAW_DATA_FILES_DIRECTORY="../data_files/raw"
PROCESSED_DATA_FILES_DIRECTORY="../data_files/processed"

# Set a common output HDF5 filename for all qpb main programs and datasets
OUTPUT_HDF5_FILE_NAME="pion_correlators_values.h5"

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
    # These are expected to represent specific experiments or analyses.
    for data_files_set_directory in "$main_program_directory"/*; do

        # Ensure the current entry is a directory; skip otherwise.
        if [ ! -d "$data_files_set_directory" ]; then
            continue
        fi

        # Extract name of data files set directory
        data_files_set_name=$(basename $data_files_set_directory)

        echo "- '${data_files_set_name}' data files set:"

        # Construct path to corresponding processed data files subdirectory
        output_directory_path=$PROCESSED_DATA_FILES_DIRECTORY
        output_directory_path+="/$(basename $main_program_directory)"
        output_directory_path+="/${data_files_set_name}"

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

        # Create paths to the two types of output files for storing extracted
        # values form the qpb files.
        # NOTE: Filenames are independent of the data files sets names
        output_csv_filename="qpb_log_files_single_valued_parameters.csv"
        output_csv_file_path="${output_directory_path}/${output_csv_filename}"
        output_hdf5_filename="qpb_log_files_multivalued_parameters.h5"
        output_hdf5_file_path="${output_directory_path}/${output_hdf5_filename}"

        # Check if current data files set subdirectory contains any .txt files
        if find "$data_files_set_directory" \
                            -maxdepth 1 -type f -name "*.txt" | grep -q .; then
            
            echo "   ++ It contains qpb log files."
            NO_DATA_FILES="False"

            # Process the current set of qpb log files
            script_path="${SOURCE_SCRIPTS_DIRECTORY}"
            script_path+="/data_files_processing/process_qpb_log_files.py"
            python $script_path \
                -qpb_log_dir "$data_files_set_directory" \
                -out_dir "$output_directory_path" \
                -csv_name "${output_csv_filename%.csv}_preprocessed.csv" \
                -hdf5_name "$output_hdf5_filename"
            
            # Generate a summary for output .csv file and save it to a text file
            script_path="${SOURCE_SCRIPTS_DIRECTORY}"
            script_path+="/utils/inspect_csv_file.py"
            python $script_path \
                -in_csv_path "${output_csv_file_path%.csv}_preprocessed.csv" \
                --output_file "${output_csv_file_path%.csv}_preprocessed_summary.txt" \
                --sample_rows 0
            
            # Process the extracted values from the qpb log files
            script_path="${SOURCE_SCRIPTS_DIRECTORY}/post_processing_analysis"
            script_path+="/process_qpb_log_files_extracted_values.py"
            python $script_path \
                -in_csv_path "${output_csv_file_path%.csv}_preprocessed.csv" \
                -in_hdf5_path "$output_hdf5_file_path" \
                -out_csv $output_csv_filename

            # Generate a summary for output .csv file and save it to a text file
            script_path="${SOURCE_SCRIPTS_DIRECTORY}"
            script_path+="/utils/inspect_csv_file.py"
            python $script_path \
                -in_csv_path "$output_csv_file_path" \
                --output_file "${output_csv_file_path%.csv}_summary.txt" \
                --sample_rows 0

            # Generate and save the tree structure of the output HDF5 file
            h5glance "$output_hdf5_file_path" \
                                    > "${output_hdf5_file_path%.h5}_tree.txt"
        fi

        # QPB CORRELATOR FILES PARSING

        # Check if the current experiment subdirectory contains any .dat files
        if find "$data_files_set_directory" \
                            -maxdepth 1 -type f -name "*.dat" | grep -q .; then

            echo "   ++ It contains qpb correlator files."
            NO_DATA_FILES="False"

            # Call the Python script to process correlator data file in the
            # current experiment directory. The Python script:
            # - Distinguishes correlator data files from other file types
            #   automatically.
            # - Generates output files with appropriate names.
            python "${SOURCE_SCRIPTS_DIRECTORY}/data_files_processing/parse_qpb_correlator_files.py" \
                -raw_dir "$data_files_set_directory" \
                -out_dir "$output_directory_path" \
                -out_hdf5_file "$OUTPUT_HDF5_FILE_NAME"
                
            # TODO: Include a tree graph in the output files
            # # Accompany the HDF5 file with a detailed tree graph of its structure
            output_hdf5_file_path="${output_directory_path}"
            output_hdf5_file_path+="/pion_correlators_values.h5"
            h5glance "$output_hdf5_file_path" >> "${output_hdf5_file_path%.h5}_tree.txt"
        fi

        # If no data files were found print a warning
        if [ NO_DATA_FILES == "True" ]; then
            echo "   ++ WARNING: No data files were found in this directory!"
        fi

    done
done

echo
echo "Processing all qpb data_files completed!"

# TODO: Remove old output files when rerunning