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

# Default output filenames (following project conventions)
DEFAULT_CSV_FILENAME="single_valued_parameters.csv"
DEFAULT_HDF5_LOG_FILENAME="multivalued_parameters.h5"
DEFAULT_HDF5_CORR_FILENAME="correlators_raw_data.h5"

# Script log filename
SCRIPT_LOG_FILENAME="${SCRIPT_NAME%.sh}.log"

# Export termination message for logging utilities
export SCRIPT_TERMINATION_MESSAGE="\n\t\t$(echo "$SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') EXECUTION TERMINATED"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function usage() {
    cat << EOF
Usage: $SCRIPT_NAME -i <input_directory> [options]

REQUIRED ARGUMENTS:
  -i,  --input_directory      Directory containing raw QPB data files

OPTIONAL ARGUMENTS:
  -o,  --output_directory     Output directory for parsed files (default: input dir)
  -log_dir, --log_directory   Log files directory (default: output dir)
  --csv_name                  Output CSV filename (default: $DEFAULT_CSV_FILENAME)
  --hdf5_log_name            Log params HDF5 filename (default: $DEFAULT_HDF5_LOG_FILENAME)
  --hdf5_corr_name           Correlators HDF5 filename (default: $DEFAULT_HDF5_CORR_FILENAME)
  --skip_checks              Skip intermediate file validation
  --skip_summaries           Skip generation of summary files (HDF5 trees and CSV reports)
  -h,  --help                Display this help message

EXAMPLES:
  $SCRIPT_NAME -i ../data_files/raw/my_experiment/
  $SCRIPT_NAME -i raw_data/ -o processed_data/
  $SCRIPT_NAME -i raw_data/ --csv_name params.csv
  $SCRIPT_NAME -i raw_data/ --skip_summaries

EOF
    exit 0
}

function validate_prerequisites() {
    local input_dir="$1"
    
    log "INFO" "Validating prerequisites..."
    
    # Check for .txt log files (required)
    local txt_count=$(find "$input_dir" -maxdepth 1 -type f -name "*.txt" | wc -l)
    if [[ $txt_count -eq 0 ]]; then
        log "ERROR" "No .txt log files found in input directory"
        echo "ERROR: No .txt log files found in: $input_dir" >&2
        return 1
    fi
    log "INFO" "Found $txt_count .txt log file(s)"
    
    # Check for .dat correlator files (optional, informational only)
    local dat_count=$(find "$input_dir" -maxdepth 1 -type f -name "*.dat" | wc -l)
    if [[ $dat_count -gt 0 ]]; then
        log "INFO" "Found $dat_count .dat correlator file(s) - will parse correlator data"
    else
        log "INFO" "No .dat files found - will skip correlator parsing"
    fi
    
    # Validate Python scripts exist
    if [[ ! -f "$PARSE_LOG_FILES_SCRIPT" ]]; then
        log "ERROR" "Python script not found: $PARSE_LOG_FILES_SCRIPT"
        echo "ERROR: parse_log_files.py not found" >&2
        return 1
    fi
    
    if [[ ! -f "$PARSE_CORRELATOR_FILES_SCRIPT" ]]; then
        log "ERROR" "Python script not found: $PARSE_CORRELATOR_FILES_SCRIPT"
        echo "ERROR: parse_correlator_files.py not found" >&2
        return 1
    fi
    
    log "INFO" "All prerequisites validated successfully"
    return 0
}

function check_intermediate_output() {
    local file_path="$1"
    local stage_name="$2"
    
    if [[ ! -f "$file_path" ]]; then
        local error_msg="$stage_name did not produce expected output: $file_path"
        log "ERROR" "$error_msg"
        echo "ERROR: $error_msg" >&2
        return 1
    fi
    
    # Check if file is not empty
    if [[ ! -s "$file_path" ]]; then
        local error_msg="$stage_name produced empty output: $file_path"
        log "ERROR" "$error_msg"
        echo "ERROR: $error_msg" >&2
        return 1
    fi
    
    log "INFO" "Validated output: $(basename "$file_path")"
    return 0
}

function cleanup() {
    # Cleanup function called on exit
    log "INFO" "Cleanup completed"
}

function generate_hdf5_tree() {
    local hdf5_file_path="$1"
    local stage_name="$2"
    
    if ! command -v h5glance &> /dev/null; then
        log "WARNING" "h5glance not found - skipping HDF5 tree generation"
        echo "  ⚠ h5glance not available - skipping tree generation"
        return 0
    fi
    
    local tree_file_path="${hdf5_file_path%.h5}_tree.txt"
    
    h5glance "$hdf5_file_path" > "$tree_file_path" 2>&1 || {
        log "WARNING" "Failed to generate HDF5 tree for $(basename "$hdf5_file_path")"
        echo "  ⚠ Failed to generate HDF5 tree"
        return 1
    }
    
    log "INFO" "Generated HDF5 tree: $(basename "$tree_file_path")"
    echo "  → HDF5 tree: $(basename "$tree_file_path")"
    return 0
}

function generate_csv_summary() {
    local csv_file_path="$1"
    local output_directory="$2"
    
    local inspect_script="${PYTHON_SCRIPTS_DIRECTORY}/utils/inspect_csv_file.py"
    
    if [[ ! -f "$inspect_script" ]]; then
        log "WARNING" "inspect_csv_file.py not found - skipping CSV summary"
        echo "  ⚠ CSV inspection script not available"
        return 0
    fi
    
    python "$inspect_script" \
        --csv_file_path "$csv_file_path" \
        --output_directory "$output_directory" \
        --uniqueness_report \
        || {
            log "WARNING" "Failed to generate CSV summary for $(basename "$csv_file_path")"
            echo "  ⚠ Failed to generate CSV summary"
            return 1
        }
    
    log "INFO" "Generated CSV summary for $(basename "$csv_file_path")"
    echo "  → CSV summary generated"
    return 0
}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

function main() {
    # Parse command line arguments
    local input_directory=""
    local output_directory=""
    local log_directory=""
    local csv_filename="$DEFAULT_CSV_FILENAME"
    local hdf5_log_filename="$DEFAULT_HDF5_LOG_FILENAME"
    local hdf5_corr_filename="$DEFAULT_HDF5_CORR_FILENAME"
    local skip_checks=false
    local skip_summaries=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--input_directory)
                input_directory="$2"
                shift 2
                ;;
            -o|--output_directory)
                output_directory="$2"
                shift 2
                ;;
            -log_dir|--log_directory)
                log_directory="$2"
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
    
    # Validate required arguments
    if [[ -z "$input_directory" ]]; then
        echo "ERROR: Input directory not specified. Use -i <directory>" >&2
        usage
    fi
    
    # Validate input directory exists
    if [[ ! -d "$input_directory" ]]; then
        echo "ERROR: Input directory not found: $input_directory" >&2
        return 1
    fi
    
    # Convert to absolute path
    input_directory="$(realpath "$input_directory")"
    
    # Set default output directory to input directory if not specified
    if [[ -z "$output_directory" ]]; then
        output_directory="$input_directory"
        echo "INFO: Using input directory as output directory"
    fi
    
    # Ensure output directory exists
    check_if_directory_exists "$output_directory" -c || return 1
    output_directory="$(realpath "$output_directory")"
    
    # Set default log directory to output directory if not specified
    if [[ -z "$log_directory" ]]; then
        log_directory="$output_directory"
    fi
    
    # Ensure log directory exists
    check_if_directory_exists "$log_directory" -c || return 1
    log_directory="$(realpath "$log_directory")"
    
    # Initialize logging
    local log_file="${log_directory}/${SCRIPT_LOG_FILENAME}"
    export SCRIPT_LOG_FILE_PATH="$log_file"
    
    # Write log header
    echo "=== QPB DATA PARSING PIPELINE STARTED: $(date) ===" > "$log_file"
    log "INFO" "Script: $SCRIPT_NAME"
    log "INFO" "Input directory: $input_directory"
    log "INFO" "Output directory: $output_directory"
    log "INFO" "Log directory: $log_directory"
    
    # Validate prerequisites
    echo ""
    echo "=== VALIDATING PREREQUISITES ==="
    validate_prerequisites "$input_directory" || return 1
    
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
    log "INFO" "STAGE 1A: Starting log file parsing"
    
    python "$PARSE_LOG_FILES_SCRIPT" \
        --qpb_log_files_directory "$input_directory" \
        --output_files_directory "$output_directory" \
        --output_csv_filename "$csv_filename" \
        --output_hdf5_filename "$hdf5_log_filename" \
        --enable_logging \
        --log_file_directory "$log_directory" \
        || {
            log "ERROR" "parse_log_files.py execution failed"
            echo "ERROR: Log file parsing failed" >&2
            return 1
        }
    
    # Validate outputs
    if ! $skip_checks; then
        check_intermediate_output "$csv_output_path" "Log file parsing (CSV)" || return 1
        check_intermediate_output "$hdf5_log_output_path" "Log file parsing (HDF5)" || return 1
    fi
    
    log "INFO" "STAGE 1A COMPLETED: Log files parsed successfully"
    echo "✓ Stage 1A completed: Log files parsed"
    echo "  - CSV output: $csv_filename"
    echo "  - HDF5 output: $hdf5_log_filename"
    
    # Generate summary files for Stage 1A outputs
    if ! $skip_summaries; then
        echo ""
        echo "Generating summary files..."
        generate_csv_summary "$csv_output_path" "$output_directory"
        generate_hdf5_tree "$hdf5_log_output_path" "Stage 1A"
    fi
    
    # =========================================================================
    # STAGE 1B: PARSE CORRELATOR FILES (CONDITIONAL)
    # =========================================================================
    
    # Check if .dat files exist
    local dat_files_found=false
    if find "$input_directory" -maxdepth 1 -type f -name "*.dat" -print -quit | grep -q .; then
        dat_files_found=true
    fi
    
    if $dat_files_found; then
        echo ""
        echo "=== STAGE 1B: PARSING CORRELATOR FILES ==="
        echo "Processing .dat correlator files..."
        log "INFO" "STAGE 1B: Starting correlator file parsing"
        
        python "$PARSE_CORRELATOR_FILES_SCRIPT" \
            --qpb_correlators_files_directory "$input_directory" \
            --output_files_directory "$output_directory" \
            --output_hdf5_filename "$hdf5_corr_filename" \
            --enable_logging \
            --log_file_directory "$log_directory" \
            || {
                log "ERROR" "parse_correlator_files.py execution failed"
                echo "ERROR: Correlator file parsing failed" >&2
                return 1
            }
        
        # Validate output
        if ! $skip_checks; then
            check_intermediate_output "$hdf5_corr_output_path" "Correlator file parsing" || return 1
        fi
        
        log "INFO" "STAGE 1B COMPLETED: Correlator files parsed successfully"
        echo "✓ Stage 1B completed: Correlator files parsed"
        echo "  - HDF5 output: $hdf5_corr_filename"
        
        # Generate summary file for Stage 1B output
        if ! $skip_summaries; then
            echo ""
            echo "Generating HDF5 tree..."
            generate_hdf5_tree "$hdf5_corr_output_path" "Stage 1B"
        fi
    else
        echo ""
        echo "○ Stage 1B skipped: No correlator files found"
        log "INFO" "STAGE 1B SKIPPED: No .dat files found in input directory"
    fi
    
    # =========================================================================
    # PIPELINE COMPLETION
    # =========================================================================
    
    echo ""
    echo "=== PARSING PIPELINE COMPLETED ==="
    echo "All parsing stages completed successfully!"
    echo ""
    echo "Output files:"
    echo "  - Parameters CSV:  $csv_output_path"
    echo "  - Parameters HDF5: $hdf5_log_output_path"
    if $dat_files_found; then
        echo "  - Correlators HDF5: $hdf5_corr_output_path"
    fi
    if ! $skip_summaries; then
        echo ""
        echo "Summary files:"
        echo "  - CSV summary:     ${csv_output_path%.csv}_uniqueness_report.txt"
        echo "  - HDF5 tree (log): ${hdf5_log_output_path%.h5}_tree.txt"
        if $dat_files_found; then
            echo "  - HDF5 tree (corr): ${hdf5_corr_output_path%.h5}_tree.txt"
        fi
    fi
    echo ""
    echo "Log file: $log_file"
    
    # Final logging
    local log_msg="PARSING PIPELINE COMPLETED SUCCESSFULLY"
    log_msg+="\n  - Input directory: $input_directory"
    log_msg+="\n  - Output files created: $(( dat_files_found ? 3 : 2 ))"
    log_msg+="\n  - Correlator data: $(( dat_files_found ? "YES" : "NO" ))"
    log_msg+="\n  - Summary files: $(( skip_summaries ? "SKIPPED" : "GENERATED" ))"
    log "INFO" "$log_msg"
    
    # Terminate logging properly
    log "INFO" "Parsing pipeline completed successfully"
    echo -e "$SCRIPT_TERMINATION_MESSAGE" >> "$SCRIPT_LOG_FILE_PATH"
    
    echo "Parsing pipeline execution completed successfully!"
    return 0
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Set up trap for cleanup
trap cleanup EXIT

# Only execute main if script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
    exit_code=$?
    
    # Final status logging
    if [[ $exit_code -eq 0 ]]; then
        echo "=== PARSING PIPELINE COMPLETED SUCCESSFULLY ===" >> "${SCRIPT_LOG_FILE_PATH:-/dev/null}"
    else
        echo "=== PARSING PIPELINE FAILED WITH EXIT CODE $exit_code ===" >> "${SCRIPT_LOG_FILE_PATH:-/dev/null}"
    fi
    
    exit $exit_code
fi