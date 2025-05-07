#!/bin/bash

################################################################################
# analyze_invert_processed_data_files_set.sh
#
# Description: 
#   Script for analyzing processed inversion data files sets. Performs various
#   analyses including correlator jackknife analysis, PCAC mass estimation, pion
#   effective mass calculations, and critical bare mass determinations.
#
# Purpose:
#   - Execute jackknife analysis on correlator data
#   - Calculate PCAC mass estimates and generate related plots
#   - Determine pion effective mass estimates
#   - Calculate critical bare mass values using both PCAC and effective mass
#   - Estimate calculation costs for different methods
#   - Generate comprehensive plots and data summaries
#
# Usage: ./analyze_invert_processed_data_files_set.sh [options]
#
# Flags:
#   - set_dir, --data_files_set_directory  Directory containing processed data
#   files
#   - out_dir, --output_files_directory    Directory for output files (optional)
#   - log_dir, --scripts_log_files_directory  Directory for log files (optional)
#
# Note:
#   - Requires associated Python scripts in the core/src directory
#   - Generates multiple CSV and HDF5 output files
#   - Creates plots in the specified output directory
#   - Maintains detailed logging of all operations
################################################################################

# CUSTOM FUNCTIONS DEFINITIONS

usage() {
    # Function to display usage information

    echo "Usage: $0 -p <data_files_set_directory>"
    echo "  -p, --path   Specify the directory containing raw files"
    exit 1
}

# TODO: Print usage when standalone execution

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
if [ -z "$PROCESSED_DATA_FILES_DIRECTORY" ]; then
    PROCESSED_DATA_FILES_DIRECTORY=$(\
            realpath "${CURRENT_SCRIPT_DIRECTORY}/../../data_files/processed")
    check_if_directory_exists "$PROCESSED_DATA_FILES_DIRECTORY" || exit 1
fi
# Construct full path to python scripts directory if not set yet
if [ -z "$PYTHON_SCRIPTS_DIRECTORY" ]; then
    PYTHON_SCRIPTS_DIRECTORY=$(\
                        realpath "${CURRENT_SCRIPT_DIRECTORY}/../../core/src")
    check_if_directory_exists "$PYTHON_SCRIPTS_DIRECTORY" || exit 1
fi
# Construct full path to plots directory if not set yet
if [ -z "$PLOTS_DIRECTORY" ]; then
    PLOTS_DIRECTORY=$(\
                    realpath "${CURRENT_SCRIPT_DIRECTORY}/../../output/plots")
    check_if_directory_exists "$PLOTS_DIRECTORY" || exit 1
fi

# Set common output filenames independent of the data files set name
PROCESSED_VALUES_CSV_FILENAME="processed_parameter_values.csv"
PION_CORRELATORS_HDF5_FILENAME="pion_correlators_values.h5"
CORRELATORS_JACKKNIFE_ANALYSIS_HDF5_FILENAME="correlators_jackknife_analysis.h5"
PCAC_MASS_ESTIMATE_CSV_FILENAME="PCAC_mass_estimates.csv"
PION_EFFECTIVE_MASS_ESTIMATE_CSV_FILENAME="Pion_effective_mass_estimates.csv"
CRITICAL_BARE_FROM_PCAC_MASS_CSV_FILENAME="critical_bare_mass_from_PCAC_mass_estimates.csv"
CRITICAL_BARE_FROM_EFFECTIVE_MASS_CSV_FILENAME="critical_bare_mass_from_pion_effective_mass.csv"
CALCULATION_COST_FROM_PCAC_MASS_CSV_FILENAME="calculation_cost_of_critical_bare_mass_from_PCAC_mass.csv"
CALCULATION_COST_FROM_EFFECTIVE_MASS_CSV_FILENAME="calculation_cost_of_critical_bare_mass_from_effective_mass.csv"

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

processed_data_files_set_directory=""
output_directory_path=""
auxiliary_files_directory=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -set_dir|--data_files_set_directory)
            processed_data_files_set_directory="$2"
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
if [ -z "$processed_data_files_set_directory" ]; then
    echo "ERROR: No raw data files set directory path specified."
    usage
fi
# Ensure the raw data files set directory exists
check_if_directory_exists "$processed_data_files_set_directory" || exit 1

# Extract name of raw data files set directory 
data_files_set_directory_name=$(basename $processed_data_files_set_directory)

# Check if an output directory was provided
if [ -z "$output_directory_path" ]; then
    # if not, then set it to the parent of the data files set directory
    output_directory_path=$(dirname $processed_data_files_set_directory)
else
    # if it was provided, then check if it exists
    # output_directory_path=$(realpath "$output_directory_path")
    check_if_directory_exists "$output_directory_path" || exit 1
fi

# Check if a log directory was provided
if [ -z "$auxiliary_files_directory" ]; then
    # if not, then set it to be the same with the output directory
    auxiliary_files_directory=$output_directory_path
else
    # if it was provided, then check if it exists
    # auxiliary_files_directory=$(realpath "$auxiliary_files_directory")
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

# CORRELATORS JACKKNIFE ANALYSIS 

# Validate input files
parameter_values_csv_file_path="${processed_data_files_set_directory}"
parameter_values_csv_file_path+="/${PROCESSED_VALUES_CSV_FILENAME}"
check_if_file_exists "$parameter_values_csv_file_path" || exit 1

correlators_hdf5_file_path="${processed_data_files_set_directory}/"
correlators_hdf5_file_path+="$PION_CORRELATORS_HDF5_FILENAME"
check_if_file_exists "$correlators_hdf5_file_path" || exit 1

# Convert processed directory path to corresponding plots directory path
data_files_set_plots_directory=$(replace_parent_directory \
    "$processed_data_files_set_directory" "$PROCESSED_DATA_FILES_DIRECTORY" \
    "$PLOTS_DIRECTORY")
check_if_directory_exists "$data_files_set_plots_directory" -c

# Validate python script
python_script_path="${PYTHON_SCRIPTS_DIRECTORY}/post_processing_analysis"
python_script_path+="/perform_jackknife_analysis_on_correlators.py"
check_if_file_exists "$python_script_path" || exit 1

python $python_script_path \
    --input_parameter_values_csv_file_path "$parameter_values_csv_file_path" \
    --input_correlators_hdf5_file_path "$correlators_hdf5_file_path" \
    --output_hdf5_filename "$CORRELATORS_JACKKNIFE_ANALYSIS_HDF5_FILENAME" \
    --plots_directory "$data_files_set_plots_directory" \
    --symmetrize_correlators \
    --plot_g5g5_correlators \
    --plot_g4g5g5_correlators \
    --plot_g4g5g5_derivative_correlators \
    --enable_logging \
    --log_file_directory "$auxiliary_files_directory" \
    || failed_python_script $python_script_path

# Log successful calculation
log_message="Correlators jackknife analysis for "
log_message+="${processed_data_files_set_directory} data files set completed."
log "INFO" "$log_message"

# Validate output file
correlators_jackknife_analysis_hdf5_file_path="${processed_data_files_set_directory}"
correlators_jackknife_analysis_hdf5_file_path+="/${CORRELATORS_JACKKNIFE_ANALYSIS_HDF5_FILENAME}"
check_if_file_exists "$correlators_jackknife_analysis_hdf5_file_path" || exit 1

# Generate outout HDF5 file structure tree
h5glance "$correlators_jackknife_analysis_hdf5_file_path" \
            > "${correlators_jackknife_analysis_hdf5_file_path%.h5}_tree.txt"

# Log generation of output HDF5 file structure tree
log_message="A hierarchy tree of the ${CORRELATORS_JACKKNIFE_ANALYSIS_HDF5_FILENAME} "
log_message+="HDF5 file was generated."
log "INFO" "$log_message"

# PCAC MASS ESTIMATES ANALYSIS

# Validate python script
python_script_path="${PYTHON_SCRIPTS_DIRECTORY}/post_processing_analysis"
python_script_path+="/calculate_PCAC_mass_estimates.py"
check_if_file_exists "$python_script_path" || exit 1

python $python_script_path \
    --input_correlators_jackknife_analysis_hdf5_file_path \
                                        "$correlators_jackknife_analysis_hdf5_file_path" \
    --plots_directory "$data_files_set_plots_directory" \
    --plot_PCAC_mass_correlators \
    --zoom_in_PCAC_mass_correlators_plots \
    --output_PCAC_mass_estimates_csv_filename \
                                            "$PCAC_MASS_ESTIMATE_CSV_FILENAME" \
    --enable_logging \
    --log_file_directory "$auxiliary_files_directory" \
    || failed_python_script $python_script_path

# Log successful calculation
log_message="PCAC mass estimate analysis for "
log_message+="${processed_data_files_set_directory} data files set completed."
log "INFO" "$log_message"

# Validate output .csv file
PCAC_mass_estimate_csv_file_path="${processed_data_files_set_directory}/"
PCAC_mass_estimate_csv_file_path+="$PCAC_MASS_ESTIMATE_CSV_FILENAME"
check_if_file_exists "$PCAC_mass_estimate_csv_file_path" || exit 1

# Log successful calculation
inspect_csv_file_python_script_path="${PYTHON_SCRIPTS_DIRECTORY}"
inspect_csv_file_python_script_path+="/utils/inspect_csv_file.py"
check_if_file_exists "$inspect_csv_file_python_script_path" || exit 1

# Generate a summary for output .csv file
python $inspect_csv_file_python_script_path \
    --csv_file_path "$PCAC_mass_estimate_csv_file_path" \
    --output_directory "$processed_data_files_set_directory" \
    --uniqueness_report \
    || failed_python_script $inspect_csv_file_python_script_path

# Log generation of output .csv file summary
log_message="A summary of the ${PCAC_MASS_ESTIMATE_CSV_FILENAME} .csv file "
log_message+="was generated."
log "INFO" "$log_message"

# PION EFFECTIVE MASS ESTIMATES ANALYSIS

# Validate python script
python_script_path="${PYTHON_SCRIPTS_DIRECTORY}/post_processing_analysis"
python_script_path+="/calculate_effective_mass_estimates.py"
check_if_file_exists "$python_script_path" || exit 1

python $python_script_path \
    --input_correlators_jackknife_analysis_hdf5_file_path \
                            "$correlators_jackknife_analysis_hdf5_file_path" \
    --plots_directory "$data_files_set_plots_directory" \
    --plot_g5g5_correlators --plot_effective_mass_correlators \
    --output_pion_effective_mass_estimates_csv_filename \
                                "$PION_EFFECTIVE_MASS_ESTIMATE_CSV_FILENAME" \
    --enable_logging \
    --log_file_directory "$auxiliary_files_directory" \
    || failed_python_script $python_script_path
    # --zoom_in_effective_mass_correlators_plots \

# Log successful calculation
log_message="Pion effective mass estimate analysis for "
log_message+="${processed_data_files_set_directory} data files set completed."
log "INFO" "$log_message"

# Validate output .csv file
pion_effective_mass_estimate_csv_file_path="${processed_data_files_set_directory}/"
pion_effective_mass_estimate_csv_file_path+="$PION_EFFECTIVE_MASS_ESTIMATE_CSV_FILENAME"
check_if_file_exists "$pion_effective_mass_estimate_csv_file_path" || exit 1

# Generate a summary for output .csv file
python $inspect_csv_file_python_script_path \
    --csv_file_path "$pion_effective_mass_estimate_csv_file_path" \
    --output_directory "$processed_data_files_set_directory" \
    --uniqueness_report \
    || failed_python_script $python_script_path

# Log generation of output .csv file summary
log_message="A summary of the ${PION_EFFECTIVE_MASS_ESTIMATE_CSV_FILENAME} .csv"
log_message+=" file was generated."
log "INFO" "$log_message"

# CRITICAL BARE VALUES FROM PCAC MASS ESTIMATES CALCULATION

# Validate python script
python_script_path="${PYTHON_SCRIPTS_DIRECTORY}/post_processing_analysis"
python_script_path+="/calculate_critical_bare_mass_from_PCAC_mass.py"
check_if_file_exists "$python_script_path" || exit 1

python $python_script_path \
    --input_PCAC_mass_estimates_csv_file_path \
                                        "$PCAC_mass_estimate_csv_file_path" \
    --input_correlators_jackknife_analysis_hdf5_file_path \
                            "$correlators_jackknife_analysis_hdf5_file_path" \
    --plots_directory "$data_files_set_plots_directory" \
    --plot_critical_bare_mass \
    --fit_for_critical_bare_mass \
    --annotate_data_points \
    --enable_logging \
    --output_critical_bare_mass_csv_filename \
                                "$CRITICAL_BARE_FROM_PCAC_MASS_CSV_FILENAME" \
    --log_file_directory "$auxiliary_files_directory" \
    || failed_python_script $python_script_path

# Log successful calculation
log_message="Critical bare mass values calculation from PCAC mass estimates for"
log_message+=" ${processed_data_files_set_directory} data files set completed."
log "INFO" "$log_message"

# Validate output .csv file
critical_bare_mass_csv_file_path="${processed_data_files_set_directory}/"
critical_bare_mass_csv_file_path+="$CRITICAL_BARE_FROM_PCAC_MASS_CSV_FILENAME"
check_if_file_exists "$critical_bare_mass_csv_file_path" || exit 1

# Generate a summary for output .csv file
python $inspect_csv_file_python_script_path \
    --csv_file_path "$critical_bare_mass_csv_file_path" \
    --output_directory "$processed_data_files_set_directory" \
    --uniqueness_report \
    || failed_python_script $python_script_path

# Log generation of output .csv file summary
log_message="A summary of the ${CRITICAL_BARE_FROM_PCAC_MASS_CSV_FILENAME} .csv"
log_message+=" file was generated."
log "INFO" "$log_message"

# CRITICAL BARE VALUES FROM PION EFFECTIVE MASS ESTIMATES CALCULATION

# Validate python script
python_script_path="${PYTHON_SCRIPTS_DIRECTORY}/post_processing_analysis"
python_script_path+="/calculate_critical_bare_mass_from_effective_mass.py"
check_if_file_exists "$python_script_path" || exit 1

python $python_script_path \
    --input_effective_mass_csv_file_path \
                                "$pion_effective_mass_estimate_csv_file_path" \
    --plots_directory "$data_files_set_plots_directory" \
    --plot_critical_bare_mass --fit_for_critical_bare_mass \
    --output_critical_bare_mass_csv_filename \
                            "$CRITICAL_BARE_FROM_EFFECTIVE_MASS_CSV_FILENAME" \
    --log_file_directory "$auxiliary_files_directory" \
    || failed_python_script $python_script_path

# Log successful calculation
log_message="Critical bare mass values calculation from Pion effective mass for"
log_message+=" ${processed_data_files_set_directory} data files set completed."
log "INFO" "$log_message"

# Validate output .csv file
critical_bare_mass_csv_file_path="${processed_data_files_set_directory}/"
critical_bare_mass_csv_file_path+="$CRITICAL_BARE_FROM_EFFECTIVE_MASS_CSV_FILENAME"
check_if_file_exists "$critical_bare_mass_csv_file_path" || exit 1

# Generate a summary for output .csv file
python $inspect_csv_file_python_script_path \
    --csv_file_path "$critical_bare_mass_csv_file_path" \
    --output_directory "$processed_data_files_set_directory" \
    --uniqueness_report \
    || failed_python_script $python_script_path

# Log generation of output .csv file summary
log_message="A summary of the ${CRITICAL_BARE_FROM_EFFECTIVE_MASS_CSV_FILENAME}"
log_message+=" .csv file was generated."
log "INFO" "$log_message"

# CALCULATION COST OF CRITICAL BARE FROM PCAC MASS ESTIMATES

# Validate python script
python_script_path="${PYTHON_SCRIPTS_DIRECTORY}/post_processing_analysis"
python_script_path+="/estimate_calculation_cost_of_critical_bare_from_PCAC_mass.py"
check_if_file_exists "$python_script_path" || exit 1

python $python_script_path \
    --input_PCAC_mass_estimates_csv_file_path \
                                        "$PCAC_mass_estimate_csv_file_path" \
    --input_correlators_jackknife_analysis_hdf5_file_path \
                            "$correlators_jackknife_analysis_hdf5_file_path" \
    --plots_directory "$data_files_set_plots_directory" \
    --plot_critical_bare_mass --plot_calculation_cost \
    --output_calculation_cost_csv_filename \
                            "$CALCULATION_COST_FROM_PCAC_MASS_CSV_FILENAME" \
    --enable_logging \
    --log_file_directory "$auxiliary_files_directory" \
    || failed_python_script $python_script_path

# Log successful calculation
log_message="Estimation of calculation cost from PCAC mass estimates for"
log_message+=" ${processed_data_files_set_directory} data files set completed."
log "INFO" "$log_message"

# Validate output .csv file
calculation_cost_csv_file_path="${processed_data_files_set_directory}/"
calculation_cost_csv_file_path+="$CALCULATION_COST_FROM_PCAC_MASS_CSV_FILENAME"
check_if_file_exists "$calculation_cost_csv_file_path" || exit 1

# Generate a summary for output .csv file
python $inspect_csv_file_python_script_path \
    --csv_file_path "$calculation_cost_csv_file_path" \
    --output_directory "$processed_data_files_set_directory" \
    --uniqueness_report \
    || failed_python_script $python_script_path

# Log generation of output .csv file summary
log_message="A summary of the ${CALCULATION_COST_FROM_PCAC_MASS_CSV_FILENAME} .csv"
log_message+=" file was generated."
log "INFO" "$log_message"

# CALCULATION COST OF CRITICAL BARE FROM EFFECTIVE MASS ESTIMATES

# Validate python script
python_script_path="${PYTHON_SCRIPTS_DIRECTORY}/post_processing_analysis"
python_script_path+="/estimate_calculation_cost_of_critical_bare_from_effective_mass.py"
check_if_file_exists "$python_script_path" || exit 1

python $python_script_path \
    --input_pion_effective_mass_estimates_csv_file_path \
                                "$pion_effective_mass_estimate_csv_file_path" \
    --plots_directory "$data_files_set_plots_directory" \
    --plot_critical_bare_mass --plot_calculation_cost \
    --output_calculation_cost_csv_filename \
                        "$CALCULATION_COST_FROM_EFFECTIVE_MASS_CSV_FILENAME" \
    --log_file_directory "$auxiliary_files_directory" \
    || failed_python_script $python_script_path

# Log successful calculation
log_message="Estimation of calculation cost from effective mass estimates for"
log_message+=" ${processed_data_files_set_directory} data files set completed."
log "INFO" "$log_message"

# Validate output .csv file
calculation_cost_csv_file_path="${processed_data_files_set_directory}/"
calculation_cost_csv_file_path+="$CALCULATION_COST_FROM_EFFECTIVE_MASS_CSV_FILENAME"
check_if_file_exists "$calculation_cost_csv_file_path" || exit 1

# Generate a summary for output .csv file
python $inspect_csv_file_python_script_path \
    --csv_file_path "$calculation_cost_csv_file_path" \
    --output_directory "$processed_data_files_set_directory" \
    --uniqueness_report \
    || failed_python_script $python_script_path

# Log generation of output .csv file summary
log_message="A summary of the ${CALCULATION_COST_FROM_EFFECTIVE_MASS_CSV_FILENAME} .csv"
log_message+=" file was generated."
log "INFO" "$log_message"

# SUCCESSFUL COMPLETION OUTPUT

# Construct the final message
final_message="'${data_files_set_directory_name}' data files set "
final_message+="analysis completed!"
# Print the final message
echo "!! $final_message"

log "INFO" "${final_message}"
echo # Empty line

echo -e $SCRIPT_TERMINATION_MESSAGE >> "$SCRIPT_LOG_FILE_PATH"

unset SCRIPT_TERMINATION_MESSAGE
unset SCRIPT_LOG_FILE_PATH
