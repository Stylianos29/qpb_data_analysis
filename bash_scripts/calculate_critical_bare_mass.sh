#!/bin/bash

# Define paths for the source scripts, raw data files, and processed data files.
SOURCE_SCRIPTS_DIRECTORY="../core/src/post_processing_analysis"
PROCESSED_DATA_FILES_DIRECTORY="../data_files/processed"
PLOTS_DIRECTORY="../output/plots"

# Set a common output HDF5 filename for all qpb main programs and datasets
INPUT_LOG_FILES_CSV_FILE_NAME="qpb_log_files_single_valued_parameters.csv"
INPUT_HDF5_FILE_NAME="pion_correlators_values.h5"

# Loop over all subdirectories in the raw data files directory.
# These subdirectories are expected to represent the qpb main programs that 
# generated the respective data files.
for qpb_main_program_directory in "$PROCESSED_DATA_FILES_DIRECTORY"/*; do

    # Ensure the current entry is a directory; skip otherwise.
    if [ ! -d "$qpb_main_program_directory" ]; then
        continue
    fi

    echo
    echo "Working within '${qpb_main_program_directory}':"

    # Loop over all subdirectories of the current main program directory.
    # These are expected to represent specific experiments or analyses.
    for data_files_set_directory in "$qpb_main_program_directory"/*; do

        # Ensure the current entry is a directory; skip otherwise.
        if [ ! -d "$data_files_set_directory" ]; then
            continue
        fi

        echo "- '$(basename ${data_files_set_directory})' data files set:"

        # Construct input HDF5 and .csv files paths
        input_log_files_csv_file_path="${data_files_set_directory}"
        input_log_files_csv_file_path+="/${INPUT_LOG_FILES_CSV_FILE_NAME}"

        input_hdf5_file_path="${data_files_set_directory}"
        input_hdf5_file_path+="/${INPUT_HDF5_FILE_NAME}"

        # Check if both the correlator values HDF5 file and qpb log files .csv
        # file exist; skip otherwise.
        if [ ! -f "${input_hdf5_file_path}" ] \
                        || [ ! -f "${input_log_files_csv_file_path}" ]; then
            echo "   ++ No input HDF5 and .csv files to analyze."
            continue
        fi

        # PCAC mass correlator analysis
        python "${SOURCE_SCRIPTS_DIRECTORY}/calculate_PCAC_mass_correlator.py" \
            -log_csv "$input_log_files_csv_file_path" \
            -cor_hdf5 "$input_hdf5_file_path"

        # input_hdf5_file_path="${data_files_set_directory}"
        # input_hdf5_file_path+="/PCAC_mass_correlator_values.h5"

        # # Accompany the HDF5 file with a detailed tree graph of its structure
        # h5glance "$input_hdf5_file_path" >> "${input_hdf5_file_path%.h5}_tree.txt"

        # # Construct path to corresponding plots subdirectory
        # plots_subdirectory_path=$PLOTS_DIRECTORY
        # plots_subdirectory_path+="/$(basename $qpb_main_program_directory)"
        # plots_subdirectory_path+="/$(basename $data_files_set_directory)"

        # # Create the plots subdirectory if it does not exit
        # if [ ! -d "$plots_subdirectory_path" ]; then
        #     mkdir -p "$plots_subdirectory_path"
        #     warning_message="   ++ WARNING: Subdirectory "
        #     warning_message+="'${plots_subdirectory_path}' does not exist, so "
        #     warning_message+="it was created."
        #     echo "${warning_message}"
        # fi

        # # PCAC mass estimates analysis
        # python "${SOURCE_SCRIPTS_DIRECTORY}/calculate_PCAC_mass_estimates.py" \
        #     -PCAC_hdf5 "$input_hdf5_file_path" \
        #     -plots_dir "$plots_subdirectory_path"

        # # Summary of .csv file
        # output_csv_file_path="${data_files_set_directory}"
        # output_csv_file_path+="/PCAC_mass_estimates.csv"
        # python "${SOURCE_SCRIPTS_DIRECTORY}/../utils/inspect_csv_file.py" \
        #     -in_csv_path "$output_csv_file_path" \
        #     --output_file "${output_csv_file_path%.csv}_summary.txt" \
        #     --sample_rows 0

        # input_csv_file_path="${data_files_set_directory}"
        # input_csv_file_path+="/PCAC_mass_estimates.csv"
        
        # # Critical bare mass from PCAC mass estimates analysis
        # python "${SOURCE_SCRIPTS_DIRECTORY}/calculate_critical_bare_mass_from_PCAC_mass.py" \
        #     -PCAC_csv "$input_csv_file_path" \
        #     -plots_dir "$plots_subdirectory_path"

        # # Summary of .csv file
        # output_csv_file_path="${data_files_set_directory}"
        # output_csv_file_path+="/critical_bare_mass_from_PCAC_mass.csv"
        # python "${SOURCE_SCRIPTS_DIRECTORY}/../utils/inspect_csv_file.py" \
        #     -in_csv_path "$output_csv_file_path" \
        #     --output_file "${output_csv_file_path%.csv}_summary.txt" \
        #     --sample_rows 0
        
    done
done

echo
echo "Script termination."

# TODO: Provide better names for inputs and outputs
# TODO: Generate a summary files for each .csv file
