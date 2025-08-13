#!/bin/bash

################################################################################
# run_processing_pipeline.sh - Sequential execution script for QPB data 
# analysis pipeline processing.
#
# DESCRIPTION:
# This script executes the three main data analysis processing scripts in the 
# correct sequence with proper input/output handling and comprehensive error 
# checking. The pipeline processes QPB correlator data through:
#
# 1. Raw parameter processing (process_raw_parameters.py)
# 2. Jackknife analysis (apply_jackknife_analysis.py) 
# 3. Visualization (visualize_jackknife_samples.py) - optional
#
# The script ensures proper data flow between stages, validates intermediate
# outputs, and provides detailed logging throughout the process.
#
# USAGE:
#   ./run_processing_pipeline.sh -csv <input_csv> -hdf5 <input_hdf5> [options]
#
# REQUIRED ARGUMENTS:
#   -csv, --input_csv_file          Path to input CSV file with single-valued parameters
#   -hdf5, --input_hdf5_file        Path to input HDF5 file with correlator data
#
# OPTIONAL ARGUMENTS:
#   -out_dir, --output_directory    Output directory (default: input file directory)
#   -log_dir, --log_directory      Log files directory (default: output directory)
#   -viz, --enable_visualization   Enable visualization step (optional)
#   --skip_checks                  Skip intermediate file validation checks
#   -h, --help                     Show this help message
#
# DEPENDENCIES:
# - Python scripts in qpb_data_analysis/core/src/processing/
# - Library scripts for validation and logging utilities
#
################################################################################

# ENVIRONMENT VARIABLES AND SETUP

CURRENT_SCRIPT_FULL_PATH=$(realpath "$0")
CURRENT_SCRIPT_NAME="$(basename "$CURRENT_SCRIPT_FULL_PATH")"
CURRENT_SCRIPT_DIRECTORY="$(dirname "$CURRENT_SCRIPT_FULL_PATH")"
SCRIPT_LOG_FILE_NAME=$(echo "$CURRENT_SCRIPT_NAME" | sed 's/\.sh$/_pipeline.log/')

# Construct paths to required directories
if [ -z "$LIBRARY_SCRIPTS_DIRECTORY_PATH" ]; then
    LIBRARY_SCRIPTS_DIRECTORY_PATH=$(realpath "${CURRENT_SCRIPT_DIRECTORY}/../library")
fi

if [ -z "$PYTHON_SCRIPTS_DIRECTORY" ]; then
    PYTHON_SCRIPTS_DIRECTORY=$(realpath "${CURRENT_SCRIPT_DIRECTORY}/../../core/src")
fi

# Processing scripts paths
PROCESSING_SCRIPTS_DIR="${PYTHON_SCRIPTS_DIRECTORY}/processing"
PROCESS_RAW_SCRIPT="${PROCESSING_SCRIPTS_DIR}/process_raw_parameters.py"
JACKKNIFE_SCRIPT="${PROCESSING_SCRIPTS_DIR}/apply_jackknife_analysis.py"
VISUALIZATION_SCRIPT="${PROCESSING_SCRIPTS_DIR}/visualize_jackknife_samples.py"

# Output filenames following project conventions
PROCESSED_CSV_FILENAME="processed_parameter_values.csv"
JACKKNIFE_HDF5_FILENAME="correlators_jackknife_analysis.h5"

# Export script termination message for logging
export SCRIPT_TERMINATION_MESSAGE="\n\t\t$(echo "$CURRENT_SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') PIPELINE EXECUTION TERMINATED"

# UTILITY FUNCTIONS

usage() {
    cat << EOF
Usage: $0 -csv <input_csv> -hdf5 <input_hdf5> [options]

REQUIRED ARGUMENTS:
  -csv,  --input_csv_file          Path to input CSV file with single-valued parameters
  -hdf5, --input_hdf5_file         Path to input HDF5 file with correlator data

OPTIONAL ARGUMENTS:
  -out_dir, --output_directory     Output directory (default: input file directory)
  -log_dir, --log_directory        Log files directory (default: output directory)
  -viz,     --enable_visualization Enable visualization step (optional)
  --skip_checks                    Skip intermediate file validation checks
  -h,       --help                 Show this help message

EXAMPLES:
  $0 -csv data.csv -hdf5 correlators.h5
  $0 -csv data.csv -hdf5 correlators.h5 -out_dir ./results -viz
  $0 -csv data.csv -hdf5 correlators.h5 -log_dir ./logs --skip_checks

EOF
    exit 1
}

# Error handling function for failed Python scripts
failed_python_script() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    error_message="PIPELINE FAILURE: ${script_name} execution failed."
    log "ERROR" "$error_message"
    echo "ERROR: $error_message" >&2
    
    termination_output "$error_message"
    exit 1
}

# Function to validate file exists and is readable
validate_input_file() {
    local file_path="$1"
    local file_type="$2"
    
    if [[ ! -f "$file_path" ]]; then
        echo "ERROR: ${file_type} file not found: $file_path" >&2
        exit 1
    fi
    
    if [[ ! -r "$file_path" ]]; then
        echo "ERROR: ${file_type} file not readable: $file_path" >&2
        exit 1
    fi
    
    log "INFO" "${file_type} file validated: $file_path"
}

# Function to validate Python script exists
validate_python_script() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    if [[ ! -f "$script_path" ]]; then
        echo "ERROR: Python script not found: $script_path" >&2
        echo "Please ensure the qpb_data_analysis repository structure is intact." >&2
        exit 1
    fi
    
    if [[ ! -x "$script_path" ]]; then
        echo "WARNING: Making $script_name executable..."
        chmod +x "$script_path"
    fi
    
    log "INFO" "Python script validated: $script_name"
}

# Function to check intermediate outputs
check_intermediate_output() {
    local file_path="$1"
    local stage_name="$2"
    
    if [[ ! -f "$file_path" ]]; then
        error_message="PIPELINE FAILURE: ${stage_name} did not produce expected output: $file_path"
        log "ERROR" "$error_message"
        echo "ERROR: $error_message" >&2
        termination_output "$error_message"
        exit 1
    fi
    
    # Check if file is not empty
    if [[ ! -s "$file_path" ]]; then
        error_message="PIPELINE FAILURE: ${stage_name} produced empty output file: $file_path"
        log "ERROR" "$error_message"
        echo "ERROR: $error_message" >&2
        termination_output "$error_message"
        exit 1
    fi
    
    log "INFO" "${stage_name} output validated: $file_path"
}

# SOURCE DEPENDENCIES

# Check if library directory exists
if [[ ! -d "$LIBRARY_SCRIPTS_DIRECTORY_PATH" ]]; then
    echo "ERROR: Library scripts directory not found: $LIBRARY_SCRIPTS_DIRECTORY_PATH" >&2
    echo "Please ensure you're running this script from the correct location." >&2
    exit 1
fi

# Source all library scripts for utility functions
for library_script in "${LIBRARY_SCRIPTS_DIRECTORY_PATH}"/*.sh; do
    if [[ -f "$library_script" ]]; then
        source "$library_script"
    fi
done

# Verify that essential functions are available
if ! command -v check_if_file_exists &> /dev/null; then
    echo "ERROR: Required library functions not loaded. Check library scripts." >&2
    exit 1
fi

# PARSE COMMAND LINE ARGUMENTS

input_csv_file=""
input_hdf5_file=""
output_directory=""
log_directory=""
enable_visualization=false
skip_checks=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -csv|--input_csv_file)
            input_csv_file="$2"
            shift 2
            ;;
        -hdf5|--input_hdf5_file)
            input_hdf5_file="$2"
            shift 2
            ;;
        -out_dir|--output_directory)
            output_directory="$2"
            shift 2
            ;;
        -log_dir|--log_directory)
            log_directory="$2"
            shift 2
            ;;
        -viz|--enable_visualization)
            enable_visualization=true
            shift
            ;;
        --skip_checks)
            skip_checks=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "ERROR: Unknown argument '$1'" >&2
            echo "Use -h or --help for usage information." >&2
            exit 1
            ;;
    esac
done

# VALIDATE REQUIRED ARGUMENTS

if [[ -z "$input_csv_file" ]]; then
    echo "ERROR: Input CSV file not specified. Use -csv <file_path>" >&2
    usage
fi

if [[ -z "$input_hdf5_file" ]]; then
    echo "ERROR: Input HDF5 file not specified. Use -hdf5 <file_path>" >&2
    usage
fi

# VALIDATE AND SETUP DIRECTORIES

# Set default output directory to input file directory if not specified
if [[ -z "$output_directory" ]]; then
    output_directory=$(dirname "$input_hdf5_file")
    echo "INFO: Using default output directory: $output_directory"
fi

# Ensure output directory exists
check_if_directory_exists "$output_directory" -c || exit 1

# Set default log directory to output directory if not specified  
if [[ -z "$log_directory" ]]; then
    log_directory="$output_directory"
fi

# Ensure log directory exists
check_if_directory_exists "$log_directory" -c || exit 1

# VALIDATE INPUT FILES AND SCRIPTS

echo "=== QPB DATA ANALYSIS PIPELINE VALIDATION ==="
echo "Validating input files and Python scripts..."

validate_input_file "$input_csv_file" "Input CSV"
validate_input_file "$input_hdf5_file" "Input HDF5"

validate_python_script "$PROCESS_RAW_SCRIPT"
validate_python_script "$JACKKNIFE_SCRIPT"

if $enable_visualization; then
    validate_python_script "$VISUALIZATION_SCRIPT"
fi

# INITIALIZE LOGGING

SCRIPT_LOG_FILE_PATH="${log_directory}/${SCRIPT_LOG_FILE_NAME}"
export SCRIPT_LOG_FILE_PATH

echo -e "\t\t$(echo "$CURRENT_SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') PIPELINE EXECUTION INITIATED\n" > "$SCRIPT_LOG_FILE_PATH"

log_message="Starting QPB data analysis pipeline with inputs:"
log_message+="\n  - CSV: $input_csv_file"
log_message+="\n  - HDF5: $input_hdf5_file"
log_message+="\n  - Output: $output_directory"
log_message+="\n  - Visualization: $enable_visualization"
log "INFO" "$log_message"

# DEFINE OUTPUT PATHS

processed_csv_path="${output_directory}/${PROCESSED_CSV_FILENAME}"
jackknife_hdf5_path="${output_directory}/${JACKKNIFE_HDF5_FILENAME}"

# ============================================================================
# STAGE 1: PROCESS RAW PARAMETERS
# ============================================================================

echo ""
echo "=== STAGE 1: PROCESSING RAW PARAMETERS ==="
echo "Processing QPB log file parameters..."

log "INFO" "STAGE 1: Starting raw parameter processing"

python "$PROCESS_RAW_SCRIPT" \
    --input_single_valued_csv_file_path "$input_csv_file" \
    --input_multivalued_hdf5_file_path "$input_hdf5_file" \
    --output_directory "$output_directory" \
    --output_csv_filename "$PROCESSED_CSV_FILENAME" \
    --enable_logging \
    --log_directory "$log_directory" \
    || failed_python_script "$PROCESS_RAW_SCRIPT"

if ! $skip_checks; then
    check_intermediate_output "$processed_csv_path" "Raw parameter processing"
fi

log_message="STAGE 1 COMPLETED: Raw parameters processed successfully"
log_message+="\n  - Output: $processed_csv_path"
log "INFO" "$log_message"
echo "✓ Stage 1 completed: Raw parameters processed"

# ============================================================================
# STAGE 2: APPLY JACKKNIFE ANALYSIS  
# ============================================================================

echo ""
echo "=== STAGE 2: APPLYING JACKKNIFE ANALYSIS ==="
echo "Applying jackknife resampling to correlator data..."

log "INFO" "STAGE 2: Starting jackknife analysis"

python "$JACKKNIFE_SCRIPT" \
    --input_hdf5_file "$input_hdf5_file" \
    --output_hdf5_file "$jackknife_hdf5_path" \
    --output_directory "$output_directory" \
    --enable_logging \
    --log_directory "$log_directory" \
    || failed_python_script "$JACKKNIFE_SCRIPT"

if ! $skip_checks; then
    check_intermediate_output "$jackknife_hdf5_path" "Jackknife analysis"
fi

log_message="STAGE 2 COMPLETED: Jackknife analysis completed successfully"
log_message+="\n  - Output: $jackknife_hdf5_path"
log "INFO" "$log_message"
echo "✓ Stage 2 completed: Jackknife analysis applied"

# ============================================================================
# STAGE 3: VISUALIZATION (OPTIONAL)
# ============================================================================

if $enable_visualization; then
    echo ""
    echo "=== STAGE 3: GENERATING VISUALIZATIONS ==="
    echo "Creating jackknife sample visualizations..."
    
    log "INFO" "STAGE 3: Starting visualization generation"
    
    python "$VISUALIZATION_SCRIPT" \
        --input_hdf5_file "$jackknife_hdf5_path" \
        --output_directory "$output_directory" \
        --enable_logging \
        --log_directory "$log_directory" \
        || failed_python_script "$VISUALIZATION_SCRIPT"
    
    # Check for visualization output directory
    plots_dir="${output_directory}/jackknife_plots"
    if ! $skip_checks && [[ -d "$plots_dir" ]]; then
        log "INFO" "STAGE 3 COMPLETED: Visualizations generated in $plots_dir"
        echo "✓ Stage 3 completed: Visualizations generated"
    elif ! $skip_checks; then
        log "WARNING" "Visualization completed but no plots directory found"
        echo "⚠ Stage 3 completed with warnings"
    else
        echo "✓ Stage 3 completed: Visualization script executed"
    fi
else
    log "INFO" "STAGE 3 SKIPPED: Visualization not requested"
    echo "○ Stage 3 skipped: Visualization not enabled"
fi

# ============================================================================
# PIPELINE COMPLETION
# ============================================================================

echo ""
echo "=== PIPELINE EXECUTION COMPLETED ==="
echo "All processing stages completed successfully!"
echo ""
echo "Output files:"
echo "  - Processed parameters: $processed_csv_path"
echo "  - Jackknife analysis:   $jackknife_hdf5_path"

if $enable_visualization; then
    echo "  - Visualizations:       ${output_directory}/jackknife_plots/"
fi

echo ""
echo "Log file: $SCRIPT_LOG_FILE_PATH"

# Final logging
log_message="QPB DATA ANALYSIS PIPELINE COMPLETED SUCCESSFULLY"
log_message+="\n  - Total stages executed: $($enable_visualization && echo "3" || echo "2")"
log_message+="\n  - All outputs validated: $(!$skip_checks && echo "YES" || echo "SKIPPED")"
log_message+="\n  - Log file: $SCRIPT_LOG_FILE_PATH"
log "INFO" "$log_message"

# Terminate logging properly
if command -v termination_output &> /dev/null; then
    termination_output "Pipeline execution completed successfully"
else
    echo -e "$SCRIPT_TERMINATION_MESSAGE" >> "$SCRIPT_LOG_FILE_PATH"
fi

echo "Pipeline execution completed successfully!"
exit 0
