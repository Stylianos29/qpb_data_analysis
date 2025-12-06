#!/bin/bash

################################################################################
# run_parsing_pipeline.sh - Execute Stage 1 (Parsing) of QPB data analysis
#
# DESCRIPTION:
# This script orchestrates the parsing stage of the QPB data analysis pipeline.
# It processes raw data files from QPB program output (.txt log files and 
# optional .dat correlator files) and converts them into structured formats
# for downstream analysis.
#
# The script executes two Python parsing scripts in sequence:
# 1. parse_log_files.py - Extracts parameters from .txt log files
#    Output: CSV (single-valued params) + HDF5 (multivalued params)
# 
# 2. parse_correlator_files.py - Extracts correlator data from .dat files
#    Output: HDF5 (correlator arrays)
#    Note: Only runs if .dat files are present in the input directory
#
# USAGE:
#   ./run_parsing_pipeline.sh -i <raw_data_dir> [options]
#
# REQUIRED ARGUMENTS:
#   -i,  --input_directory      Directory containing raw QPB data files
#                               (.txt files required, .dat files optional)
#
# OPTIONAL ARGUMENTS:
#   -o,  --output_directory     Output directory for parsed files
#                               (default: input directory)
#   -log_dir, --log_directory   Directory for log files
#                               (default: output directory)
#   --csv_name                  Custom name for output CSV file
#                               (default: single_valued_parameters.csv)
#   --hdf5_log_name            Custom name for log params HDF5 file
#                               (default: multivalued_parameters.h5)
#   --hdf5_corr_name           Custom name for correlators HDF5 file
#                               (default: correlators_raw_data.h5)
#   --skip_checks              Skip intermediate file validation
#   --skip_summaries           Skip generation of summary files
#   -h,  --help                Display this help message
#
# EXAMPLES:
#   # Basic usage - parse data set in current directory
#   ./run_parsing_pipeline.sh -i ../data_files/raw/my_experiment/
#
#   # Specify custom output location
#   ./run_parsing_pipeline.sh -i raw_data/ -o processed_data/
#
#   # Custom output filenames
#   ./run_parsing_pipeline.sh -i raw_data/ \
#       --csv_name params.csv \
#       --hdf5_log_name arrays.h5
#
#   # Skip summary file generation (faster)
#   ./run_parsing_pipeline.sh -i raw_data/ --skip_summaries
#
# DEPENDENCIES:
# - Python scripts: parse_log_files.py, parse_correlator_files.py
# - Library scripts in bash_scripts/library/
# - Python environment with qpb_data_analysis package
#
# OUTPUT FILES:
# - single_valued_parameters.csv     : Scalar parameters from log files
# - multivalued_parameters.h5        : Array parameters from log files
# - correlators_raw_data.h5          : Correlator data (if .dat files exist)
# - run_parsing_pipeline.log         : Script execution log
#
# SUMMARY FILES (unless --skip_summaries):
# - single_valued_parameters_uniqueness_report.txt : CSV column summary
# - multivalued_parameters_tree.txt                : HDF5 structure tree
# - correlators_raw_data_tree.txt                  : Correlators HDF5 tree
#
################################################################################

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Determine script paths
SCRIPT_PATH="$(realpath "$0")"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Setup library path
if [[ -z "$LIBRARY_SCRIPTS_DIRECTORY_PATH" ]]; then
    LIBRARY_SCRIPTS_DIRECTORY_PATH="$(realpath "${SCRIPT_DIR}/../library")"
fi

# Validate library directory exists
if [[ ! -d "$LIBRARY_SCRIPTS_DIRECTORY_PATH" ]]; then
    echo "ERROR: Library scripts directory not found: $LIBRARY_SCRIPTS_DIRECTORY_PATH" >&2
    exit 1
fi

# Source all library functions
for library_script in "${LIBRARY_SCRIPTS_DIRECTORY_PATH}"/*.sh; do
    [[ -f "$library_script" ]] && source "$library_script"
done

# Setup Python scripts path
if [[ -z "$PYTHON_SCRIPTS_DIRECTORY" ]]; then
    PYTHON_SCRIPTS_DIRECTORY="$(realpath "${SCRIPT_DIR}/../../core/src")"
fi

# Validate Python scripts directory
if [[ ! -d "$PYTHON_SCRIPTS_DIRECTORY" ]]; then
    echo "ERROR: Python scripts directory not found: $PYTHON_SCRIPTS_DIRECTORY" >&2
    exit 1
fi

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Python script paths
PARSE_LOG_FILES_SCRIPT="${PYTHON_SCRIPTS_DIRECTORY}/parsing/parse_log_files.py"
PARSE_CORRELATOR_FILES_SCRIPT="${PYTHON_SCRIPTS_DIRECTORY}/parsing/parse_correlator_files.py"

# Default output filenames (using constants from constants.sh)
DEFAULT_CSV_FILENAME="$PARSING_CSV_SINGLE_VALUED"
DEFAULT_HDF5_LOG_FILENAME="$PARSING_HDF5_MULTIVALUED"
DEFAULT_HDF5_CORR_FILENAME="$PARSING_HDF5_CORRELATORS"

# Script log filename
SCRIPT_LOG_FILENAME="${SCRIPT_NAME%.sh}.log"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function usage() {
    cat << EOF
Usage: $SCRIPT_NAME -i <raw_data_directory> [options]

REQUIRED ARGUMENTS:
  -i, --input_directory      Directory containing raw QPB data files

OPTIONAL ARGUMENTS:
  -o, --output_directory     Output directory (default: input directory)
  -log_dir, --log_directory  Log files directory (default: output directory)
  --csv_name                 Custom CSV filename (default: $DEFAULT_CSV_FILENAME)
  --hdf5_log_name           Custom log HDF5 filename (default: $DEFAULT_HDF5_LOG_FILENAME)
  --hdf5_corr_name          Custom correlators HDF5 filename (default: $DEFAULT_HDF5_CORR_FILENAME)
  --skip_checks             Skip intermediate file validation
  --skip_summaries          Skip summary file generation
  -h, --help                Display this help message

EXAMPLES:
  $SCRIPT_NAME -i ../data_files/raw/my_experiment/
  $SCRIPT_NAME -i raw_data/ -o processed_data/
  $SCRIPT_NAME -i raw_data/ --skip_summaries

EOF
    # Clear exit handlers before exiting
    trap - EXIT
    exit 0
}

function validate_prerequisites() {
    # Validate that required files and scripts exist
    #
    # Returns:
    #   0 - All prerequisites valid
    #   1 - Validation failed
    
    local input_dir="$1"
    
    # Check for .txt log files (required)
    if ! find "$input_dir" -maxdepth 1 -type f -name "*.txt" -print -quit | grep -q .; then
        echo "ERROR: No .txt log files found in $input_dir" >&2
        log_error "No .txt log files found in input directory"
        return 1
    fi
    
    # Validate Python scripts exist
    validate_python_script "$PARSE_LOG_FILES_SCRIPT" -s || return 1
    validate_python_script "$PARSE_CORRELATOR_FILES_SCRIPT" -s || return 1
    
    echo "All prerequisites validated successfully"
    log_info "Prerequisites validation completed successfully"
    return 0
}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

function main() {
    # Parse command line arguments
    local input_dir=""
    local output_dir=""
    local log_dir=""
    local csv_filename="$DEFAULT_CSV_FILENAME"
    local hdf5_log_filename="$DEFAULT_HDF5_LOG_FILENAME"
    local hdf5_corr_filename="$DEFAULT_HDF5_CORR_FILENAME"
    local skip_checks=false
    local skip_summaries=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--input_directory)
                input_dir="$2"
                shift 2
                ;;
            -o|--output_directory)
                output_dir="$2"
                shift 2
                ;;
            -log_dir|--log_directory)
                log_dir="$2"
                shift 2
                ;;
            --csv_name)
                csv_filename="$2"
                shift 2
                ;;
            --hdf5_log_name)
                hdf5_log_filename="$2"
                shift 2
                ;;
            --hdf5_corr_name)
                hdf5_corr_filename="$2"
                shift 2
                ;;
            --skip_checks)
                skip_checks=true
                shift
                ;;
            --skip_summaries)
                skip_summaries=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                echo "ERROR: Unknown argument '$1'" >&2
                usage
                ;;
        esac
    done
    
    # Setup and validate directories using library function
    setup_pipeline_directories "$input_dir" "$output_dir" "$log_dir" || return 1
    
    # Initialize logging
    local log_file="${log_directory}/${SCRIPT_LOG_FILENAME}"
    export SCRIPT_LOG_FILE_PATH="$log_file"
    echo -e "\t\t$(echo "$SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') EXECUTION INITIATED\n" > "$SCRIPT_LOG_FILE_PATH"
    export SCRIPT_TERMINATION_MESSAGE="\n\t\t$(echo "$SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') EXECUTION TERMINATED"
    
    log_info "Script: $SCRIPT_NAME"
    log_info "Input directory: $input_directory"
    log_info "Output directory: $output_directory"
    log_info "Log directory: $log_directory"
    
    # Validate prerequisites
    echo ""
    echo "=== VALIDATING PREREQUISITES ==="
    validate_prerequisites "$input_directory" || return 1
    
    # Detect correlator files
    local dat_files_found=false
    if detect_correlator_files "$input_directory"; then
        dat_files_found=true
        log_info "Correlator files (.dat) detected"
    else
        log_info "No correlator files detected - Stage 1B will be skipped"
    fi
    
    # Define output file paths
    local csv_output_path="${output_directory}/${csv_filename}"
    local hdf5_log_output_path="${output_directory}/${hdf5_log_filename}"
    local hdf5_corr_output_path="${output_directory}/${hdf5_corr_filename}"
    
    # =========================================================================
    # STAGE 1A: PARSE LOG FILES
    # =========================================================================
    
    echo ""
    echo "=== STAGE 1A: PARSING LOG FILES ==="
    echo "Processing .txt log files..."
    log_info "STAGE 1A: Starting log file parsing"
    
    execute_python_script "$PARSE_LOG_FILES_SCRIPT" "parse_log_files" \
        --qpb_log_files_directory "$input_directory" \
        --output_files_directory "$output_directory" \
        --output_csv_filename "$csv_filename" \
        --output_hdf5_filename "$hdf5_log_filename" \
        --enable_logging \
        --log_file_directory "$log_directory" \
        || {
            log_error "Stage 1A: Log file parsing failed"
            return 1
        }
    
    # Validate outputs
    if ! $skip_checks; then
        validate_output_file "$csv_output_path" "CSV output" || return 1
        validate_output_file "$hdf5_log_output_path" "HDF5 log output" || return 1
    fi
    
    echo "${PROGRESS_SUCCESS} Stage 1A completed: Log files parsed"
    echo "  - CSV output: $(basename "$csv_output_path")"
    echo "  - HDF5 output: $(basename "$hdf5_log_output_path")"
    
    # Generate summary files for Stage 1A
    if ! $skip_summaries; then
        echo ""
        echo "Generating summary files..."
        generate_csv_summary "$csv_output_path" "$output_directory"
        generate_hdf5_tree "$hdf5_log_output_path" "$output_directory"
    fi
    
    # =========================================================================
    # STAGE 1B: PARSE CORRELATOR FILES (if present)
    # =========================================================================
    
    if $dat_files_found; then
        echo ""
        echo "=== STAGE 1B: PARSING CORRELATOR FILES ==="
        echo "Processing .dat correlator files..."
        log_info "STAGE 1B: Starting correlator file parsing"
        
        execute_python_script "$PARSE_CORRELATOR_FILES_SCRIPT" "parse_correlator_files" \
            --qpb_correlators_files_directory "$input_directory" \
            --output_files_directory "$output_directory" \
            --output_hdf5_filename "$hdf5_corr_filename" \
            --enable_logging \
            --log_file_directory "$log_directory" \
            || {
                log_error "Stage 1B: Correlator file parsing failed"
                return 1
            }
        
        # Validate output
        if ! $skip_checks; then
            validate_output_file "$hdf5_corr_output_path" "HDF5 correlators output" || return 1
        fi
        
        echo "${PROGRESS_SUCCESS} Stage 1B completed: Correlator files parsed"
        echo "  - HDF5 output: $(basename "$hdf5_corr_output_path")"
        
        # Generate summary for correlators
        if ! $skip_summaries; then
            echo ""
            echo "Generating HDF5 tree..."
            generate_hdf5_tree "$hdf5_corr_output_path" "$output_directory"
        fi
    else
        echo ""
        echo "${PROGRESS_SKIPPED} Stage 1B skipped: No correlator files found"
    fi
    
    # =========================================================================
    # COMPLETION
    # =========================================================================
    
    echo ""
    echo "=== PARSING PIPELINE COMPLETED ==="
    echo "All parsing stages completed successfully!"
    echo ""
    echo "Output files:"
    echo "  - Parameters CSV:  $(get_display_path "$csv_output_path")"
    echo "  - Parameters HDF5: $(get_display_path "$hdf5_log_output_path")"
    if $dat_files_found; then
        echo "  - Correlators HDF5: $(get_display_path "$hdf5_corr_output_path")"
    fi
    
    if ! $skip_summaries; then
        echo ""
        echo "Summary files:"
        echo "  - CSV summary:     $(get_display_path "${csv_output_path%.csv}_uniqueness_report.txt")"
        echo "  - HDF5 tree (log): $(get_display_path "${hdf5_log_output_path%.h5}_tree.txt")"
        if $dat_files_found; then
            echo "  - HDF5 tree (corr): $(get_display_path "${hdf5_corr_output_path%.h5}_tree.txt")"
        fi
    fi
    
    echo ""
    echo "Log file: $(get_display_path "$log_file")"
    
    # Final logging
    log_info "PARSING PIPELINE COMPLETED SUCCESSFULLY"
    log_info "  Input directory: $input_directory"
    log_info "  Output files created: $(( dat_files_found ? 3 : 2 ))"
    log_info "  Correlator data: $(( dat_files_found ? "YES" : "NO" ))"
    log_info "  Summary files: $(( skip_summaries ? "SKIPPED" : "GENERATED" ))"
    
    echo -e "$SCRIPT_TERMINATION_MESSAGE" >> "$SCRIPT_LOG_FILE_PATH"
    echo ""
    echo "${PROGRESS_SUCCESS} Parsing pipeline execution completed successfully!"
    
    return 0
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Set up trap handlers for robust error handling
trap 'handle_interrupt' INT TERM
trap 'cleanup_on_exit' EXIT

# Only execute main if script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
    exit_code=$?
    exit $exit_code
fi
