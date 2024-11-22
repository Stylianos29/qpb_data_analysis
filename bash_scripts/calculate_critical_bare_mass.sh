#!/bin/bash

# Define paths for the source scripts, raw data files, and processed data files.
SOURCE_SCRIPTS_DIRECTORY="../src/post_processing_analysis"
PROCESSED_DATA_FILES_DIRECTORY="../data_files/processed"

# Set a common output HDF5 filename for all qpb main programs and datasets
INPUT_LOG_FILES_CSV_FILE_NAME="qpb_log_files_single_valued_parameters.csv"
INPUT_HDF5_FILE_NAME="pion_correlators_values.h5"

# Loop over all subdirectories in the raw data files directory.
# These subdirectories are expected to represent the qpb main programs that 
# generated the respective data files.
for data_files_main_program_directory in "$PROCESSED_DATA_FILES_DIRECTORY"/*; do

    # Ensure the current entry is a directory; skip otherwise.
    if [ ! -d "$data_files_main_program_directory" ]; then
        continue
    fi

    echo
    echo "Working within '${data_files_main_program_directory}':"

    # Loop over all subdirectories of the current main program directory.
    # These are expected to represent specific experiments or analyses.
    for data_files_sets_directory in "$data_files_main_program_directory"/*; do

        # Ensure the current entry is a directory; skip otherwise.
        if [ ! -d "$data_files_sets_directory" ]; then
            continue
        fi
        
        echo "- '$(basename ${data_files_sets_directory})' data files set:"

        # Construct input HDF5 and .csv files paths
        input_log_files_csv_file_path="${data_files_sets_directory}"
        input_log_files_csv_file_path+="/${INPUT_LOG_FILES_CSV_FILE_NAME}"

        input_hdf5_file_path="${data_files_sets_directory}"
        input_hdf5_file_path+="/${INPUT_HDF5_FILE_NAME}"

        # Check if both the correlator values HDF5 file and qpb log files .csv
        # file exist; skip otherwise.
        if [ ! -f "${input_hdf5_file_path}" ] \
                        || [ ! -f "${input_log_files_csv_file_path}" ]; then
            echo "   ++ No input HDF5 and .csv files to analyze."
            continue
        fi

        # 
        python "${SOURCE_SCRIPTS_DIRECTORY}/calculate_PCAC_mass_correlator.py" \
            -log_csv "$input_log_files_csv_file_path" \
            -cor_hdf5 "$input_hdf5_file_path"

    done
done

echo
echo "Script termination."
