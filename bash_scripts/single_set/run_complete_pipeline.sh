#!/bin/bash

################################################################################
# run_complete_pipeline.sh - Execute complete QPB data analysis pipeline
#
# DESCRIPTION:
# Master orchestrator script that runs the complete data analysis pipeline on
# a single data file set. Intelligently detects data file set type and executes
# appropriate stages:
#
# Stage 1 (Parsing) - ALWAYS:
#   - Parses .txt log files (required)
#   - Parses .dat correlator files (if present)
#   - Calls: run_parsing_pipeline.sh
#
# Stage 2 (Processing) - CONDITIONAL (only if correlators present):
#   - Processes raw parameters
#   - Applies jackknife analysis
#   - Optional visualization
#   - Calls: run_processing_pipeline.sh
#
# Stage 3 (Analysis) - FUTURE:
#   - Correlator calculations (PCAC/Pion)
#   - Plateau extraction
#   - Critical mass extrapolation
#   - Cost extrapolation
#   - [Not yet implemented - placeholders included]
#
# USAGE:
#   ./run_complete_pipeline.sh -i <raw_data_set_dir> [options]
#
# REQUIRED ARGUMENTS:
#   -i, --input_directory       Path to raw data file set directory
#                               (must contain .txt log files)
#
# OPTIONAL ARGUMENTS:
#   -o, --output_directory      Output directory for processed data
#                               (default: mirrors raw path in processed/)
#   -log_dir, --log_directory   Directory for log files
#                               (default: output directory)
#   -viz, --enable_visualization Enable visualization in processing stage
#   --skip_checks               Skip intermediate file validation
#   --skip_summaries            Skip generation of summary files
#   -h, --help                  Display this help message
#
# EXAMPLES:
#   # Basic usage - auto-detects output location
#   ./run_complete_pipeline.sh \
#       -i ../data_files/raw/invert/Chebyshev_experiment/
#
#   # Custom output location
#   ./run_complete_pipeline.sh \
#       -i ../data_files/raw/experiment1/ \
#       -o ../data_files/processed/experiment1/
#
#   # With visualization enabled
#   ./run_complete_pipeline.sh \
#       -i raw_data/ \
#       -viz
#
#   # Skip validation for faster execution
#   ./run_complete_pipeline.sh \
#       -i raw_data/ \
#       --skip_checks --skip_summaries
#
# DEPENDENCIES:
# - Workflow scripts: run_parsing_pipeline.sh, run_processing_pipeline.sh
# - Library scripts in bash_scripts/library/
# - Python scripts in core/src/parsing/ and core/src/processing/
#
# OUTPUT STRUCTURE:
# output_directory/
# ├── single_valued_parameters.csv
# ├── multivalued_parameters.h5
# ├── correlators_raw_data.h5          (if correlators present)
# ├── processed_parameter_values.csv   (if correlators present)
# ├── correlators_jackknife_analysis.h5 (if correlators present)
# ├── jackknife_plots/                 (if visualization enabled)
# └── logs/
#     ├── run_parsing_pipeline.log
#     ├── run_processing_pipeline.log
#     └── run_complete_pipeline.log
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

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Workflow scripts directory
WORKFLOWS_DIR="$(realpath "${SCRIPT_DIR}/../workflows")"

# Workflow script paths
PARSING_PIPELINE_SCRIPT="${WORKFLOWS_DIR}/run_parsing_pipeline.sh"
PROCESSING_PIPELINE_SCRIPT="${WORKFLOWS_DIR}/run_processing_pipeline.sh"

# Future workflow scripts (not yet implemented)
# CORRELATOR_ANALYSIS_SCRIPT="${WORKFLOWS_DIR}/run_correlator_analysis_pipeline.sh"
# PLATEAU_ANALYSIS_SCRIPT="${WORKFLOWS_DIR}/run_plateau_analysis_pipeline.sh"
# CRITICAL_MASS_SCRIPT="${WORKFLOWS_DIR}/run_critical_mass_pipeline.sh"
# COST_ANALYSIS_SCRIPT="${WORKFLOWS_DIR}/run_cost_analysis_pipeline.sh"

# Script log filename
SCRIPT_LOG_FILENAME="${SCRIPT_NAME%.sh}.log"

# Export termination message for logging utilities
export SCRIPT_TERMINATION_MESSAGE="\n\t\t$(echo "$SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') EXECUTION TERMINATED"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function usage() {
    cat << EOF
Usage: $SCRIPT_NAME -i <raw_data_set_directory> [options]

REQUIRED ARGUMENTS:
  -i, --input_directory        Path to raw data file set directory

OPTIONAL ARGUMENTS:
  -o, --output_directory       Output directory (default: mirrors raw in processed/)
  -log_dir, --log_directory    Log files directory (default: output directory)
  -viz, --enable_visualization Enable visualization in processing stage
  --skip_checks                Skip intermediate file validation
  --skip_summaries             Skip generation of summary files
  -h, --help                   Display this help message

EXAMPLES:
  $SCRIPT_NAME -i ../data_files/raw/invert/Chebyshev_experiment/
  $SCRIPT_NAME -i raw_data/ -o processed_data/
  $SCRIPT_NAME -i raw_data/ -viz --skip_checks

EOF
    exit 0
}

function detect_correlator_files() {
    # Detect if data file set contains correlator files (.dat)
    #
    # Arguments:
    #   $1 - input_directory : Directory to check for .dat files
    #
    # Returns:
    #   0 - Correlator files found
    #   1 - No correlator files found
    
    local input_directory="$1"
    
    if find "$input_directory" -maxdepth 1 -type f -name "*.dat" -print -quit | grep -q .; then
        return 0  # Found correlators
    else
        return 1  # No correlators
    fi
}

function validate_prerequisites() {
    local input_dir="$1"
    
    log_info "Validating prerequisites..."
    
    # Check for .txt log files (required)
    local txt_count=$(find "$input_dir" -maxdepth 1 -type f -name "*.txt" | wc -l)
    if [[ $txt_count -eq 0 ]]; then
        log_error "No .txt log files found in input directory"
        echo "ERROR: No .txt log files found in: $input_dir" >&2
        return 1
    fi
    log_info "Found $txt_count .txt log file(s)"
    
    # Check for .dat correlator files (optional, informational)
    local dat_count=$(find "$input_dir" -maxdepth 1 -type f -name "*.dat" | wc -l)
    if [[ $dat_count -gt 0 ]]; then
        log_info "Found $dat_count .dat correlator file(s)"
        echo "  → Correlator data detected - full pipeline will execute"
    else
        log_info "No .dat files found"
        echo "  → No correlator data - pipeline will stop after parsing stage"
    fi
    
    # Validate workflow scripts exist
    if [[ ! -f "$PARSING_PIPELINE_SCRIPT" ]]; then
        log_error "Workflow script not found: $PARSING_PIPELINE_SCRIPT"
        echo "ERROR: run_parsing_pipeline.sh not found" >&2
        return 1
    fi
    
    if [[ ! -f "$PROCESSING_PIPELINE_SCRIPT" ]]; then
        log_error "Workflow script not found: $PROCESSING_PIPELINE_SCRIPT"
        echo "ERROR: run_processing_pipeline.sh not found" >&2
        return 1
    fi
    
    log_info "All prerequisites validated successfully"
    echo "All prerequisites validated successfully"
    return 0
}

function check_stage_outputs() {
    # Check if required outputs from a stage exist
    #
    # Arguments:
    #   $1 - stage_name    : Name of stage for error messages
    #   $@ - file_paths... : Paths to files that should exist
    
    local stage_name="$1"
    shift
    local files=("$@")
    
    for file_path in "${files[@]}"; do
        if [[ ! -f "$file_path" ]]; then
            log_error "$stage_name did not produce expected output: $file_path"
            echo "ERROR: $stage_name failed - missing output: $(basename "$file_path")" >&2
            return 1
        fi
    done
    
    return 0
}

function cleanup() {
    # Cleanup function called on exit via trap
    log_info "Cleanup completed"
}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

function main() {
    # Parse command line arguments
    local input_directory=""
    local output_directory=""
    local log_directory=""
    local enable_visualization=false
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
            -viz|--enable_visualization)
                enable_visualization=true
                shift
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
    
    # Validate input directory exists using validation helper
    validate_input_directory "$input_directory" || return 1
    
    # Convert to absolute path
    input_directory="$(realpath "$input_directory")"
    local input_dir_name="$(basename "$input_directory")"
    
    # Set default output directory if not specified
    # Mirror the raw data structure in processed/
    if [[ -z "$output_directory" ]]; then
        # Try to replace 'raw' with 'processed' in path
        if [[ "$input_directory" == *"/data_files/raw/"* ]]; then
            output_directory="${input_directory/\/raw\//\/processed\/}"
            echo "INFO: Auto-detected output directory: $output_directory"
        else
            # Fallback: use input directory
            output_directory="$input_directory"
            echo "INFO: Using input directory as output directory"
        fi
    fi
    
    # Ensure output directory exists using validation helper
    validate_output_directory "$output_directory" -c || return 1
    output_directory="$(realpath "$output_directory")"
    
    # Set default log directory to output directory if not specified
    if [[ -z "$log_directory" ]]; then
        log_directory="$output_directory"
    fi
    
    # Ensure log directory exists using validation helper
    validate_log_directory "$log_directory" || return 1
    log_directory="$(realpath "$log_directory")"
    
    # Initialize logging using init_logging helper
    local log_file="${log_directory}/${SCRIPT_LOG_FILENAME}"
    init_logging "$log_file" -c || return 1
    
    log_info "=== COMPLETE PIPELINE EXECUTION ==="
    log_info "Script: $SCRIPT_NAME"
    log_info "Data file set: $input_dir_name"
    log_info "Input directory: $input_directory"
    log_info "Output directory: $output_directory"
    log_info "Log directory: $log_directory"
    log_info "Visualization: $enable_visualization"
    
    # Display banner
    echo ""
    echo "==================================================================="
    echo "   QPB DATA ANALYSIS - COMPLETE PIPELINE"
    echo "==================================================================="
    echo "Data file set: $input_dir_name"
    echo "Input:  $input_directory"
    echo "Output: $output_directory"
    echo "==================================================================="
    
    # Validate prerequisites
    echo ""
    echo "=== VALIDATING PREREQUISITES ==="
    validate_prerequisites "$input_directory" || return 1
    
    # Detect if data set has correlator files
    local has_correlators=false
    if detect_correlator_files "$input_directory"; then
        has_correlators=true
        log_info "Correlator files detected - full pipeline will execute"
    else
        log_info "No correlator files detected - pipeline will stop after parsing"
    fi
    
    # =========================================================================
    # STAGE 1: PARSING
    # =========================================================================
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 1: PARSING"
    echo "==================================================================="
    log_info "Starting Stage 1: Parsing"
    
    # Build arguments for parsing pipeline
    local parsing_args=(
        "-i" "$input_directory"
        "-o" "$output_directory"
        "-log_dir" "$log_directory"
    )
    
    # Add optional flags
    $skip_checks && parsing_args+=("--skip_checks")
    $skip_summaries && parsing_args+=("--skip_summaries")
    
    # Execute parsing pipeline
    "$PARSING_PIPELINE_SCRIPT" "${parsing_args[@]}" || {
        log_error "Parsing pipeline failed"
        echo ""
        echo "${PROGRESS_FAILED} Stage 1 FAILED - Pipeline terminated"
        return 1
    }
    
    # Verify parsing outputs
    if ! $skip_checks; then
        local csv_output="${output_directory}/${PARSING_CSV_SINGLE_VALUED}"
        local hdf5_output="${output_directory}/${PARSING_HDF5_MULTIVALUED}"
        
        check_stage_outputs "Parsing stage" "$csv_output" "$hdf5_output" || return 1
    fi
    
    log_info "Stage 1 completed successfully"
    echo ""
    echo "${PROGRESS_SUCCESS} Stage 1 COMPLETED"
    
    # =========================================================================
    # STAGE 2: PROCESSING (CONDITIONAL)
    # =========================================================================
    
    if $has_correlators; then
        echo ""
        echo "==================================================================="
        echo "   STAGE 2: PROCESSING"
        echo "==================================================================="
        log_info "Starting Stage 2: Processing"
        
        # Define input files for processing stage
        local csv_input="${output_directory}/${PARSING_CSV_SINGLE_VALUED}"
        local hdf5_input="${output_directory}/${PARSING_HDF5_CORRELATORS}"
        
        # Build arguments for processing pipeline
        local processing_args=(
            "-csv" "$csv_input"
            "-hdf5" "$hdf5_input"
            "-o" "$output_directory"
            "-log_dir" "$log_directory"
        )
        
        # Add optional flags
        $enable_visualization && processing_args+=("-viz")
        $skip_checks && processing_args+=("--skip_checks")
        $skip_summaries && processing_args+=("--skip_summaries")
        
        # Execute processing pipeline
        "$PROCESSING_PIPELINE_SCRIPT" "${processing_args[@]}" || {
            log_error "Processing pipeline failed"
            echo ""
            echo "${PROGRESS_FAILED} Stage 2 FAILED - Pipeline terminated"
            return 1
        }
        
        # Verify processing outputs
        if ! $skip_checks; then
            local processed_csv="${output_directory}/${PROCESSING_CSV_PROCESSED}"
            local jackknife_hdf5="${output_directory}/${PROCESSING_HDF5_JACKKNIFE}"
            
            check_stage_outputs "Processing stage" "$processed_csv" "$jackknife_hdf5" || return 1
        fi
        
        log_info "Stage 2 completed successfully"
        echo ""
        echo "${PROGRESS_SUCCESS} Stage 2 COMPLETED"
    else
        echo ""
        echo "==================================================================="
        echo "${PROGRESS_SKIPPED} Stage 2: PROCESSING - SKIPPED (no correlator data)"
        echo "==================================================================="
        log_info "Stage 2 skipped - no correlator files in data set"
    fi
    
    # =========================================================================
    # STAGE 3: ANALYSIS (FUTURE - NOT YET IMPLEMENTED)
    # =========================================================================
    
    # TODO: Implement Stage 3 analysis workflows
    # This will include:
    # - Correlator calculations (PCAC/Pion)
    # - Plateau extraction
    # - Critical mass extrapolation
    # - Cost extrapolation
    #
    # if $has_correlators; then
    #     echo ""
    #     echo "==================================================================="
    #     echo "   STAGE 3: ANALYSIS"
    #     echo "==================================================================="
    #     
    #     # Stage 3.1: Correlator Calculations
    #     "$CORRELATOR_ANALYSIS_SCRIPT" [args] || return 1
    #     
    #     # Stage 3.2: Plateau Extraction
    #     "$PLATEAU_ANALYSIS_SCRIPT" [args] || return 1
    #     
    #     # Stage 3.3: Critical Mass Extrapolation
    #     "$CRITICAL_MASS_SCRIPT" [args] || return 1
    #     
    #     # Stage 3.4: Cost Extrapolation
    #     "$COST_ANALYSIS_SCRIPT" [args] || return 1
    # fi
    
    # =========================================================================
    # PIPELINE COMPLETION
    # =========================================================================
    
    echo ""
    echo "==================================================================="
    echo "   PIPELINE EXECUTION COMPLETED"
    echo "==================================================================="
    
    # Summary
    echo ""
    echo "Summary:"
    echo "  - Data file set:  $input_dir_name"
    echo "  - Stages completed: $(( has_correlators ? 2 : 1 ))"
    echo "  - Correlator data: $(( has_correlators ? "YES" : "NO" ))"
    echo "  - Output location: $output_directory"
    echo ""
    
    # List key output files
    echo "Output files generated:"
    echo "  ${PROGRESS_SUCCESS} ${PARSING_CSV_SINGLE_VALUED}"
    echo "  ${PROGRESS_SUCCESS} ${PARSING_HDF5_MULTIVALUED}"
    
    if $has_correlators; then
        echo "  ${PROGRESS_SUCCESS} ${PARSING_HDF5_CORRELATORS}"
        echo "  ${PROGRESS_SUCCESS} ${PROCESSING_CSV_PROCESSED}"
        echo "  ${PROGRESS_SUCCESS} ${PROCESSING_HDF5_JACKKNIFE}"
        
        if $enable_visualization; then
            echo "  ${PROGRESS_SUCCESS} jackknife_plots/"
        fi
    fi
    
    echo ""
    echo "Log file: $log_file"
    echo "==================================================================="
    
    # Final logging
    local log_msg="COMPLETE PIPELINE EXECUTION FINISHED"
    log_msg+="\n  - Data file set: $input_dir_name"
    log_msg+="\n  - Stages completed: $(( has_correlators ? 2 : 1 ))"
    log_msg+="\n  - Correlator data: $(( has_correlators ? "YES" : "NO" ))"
    log_msg+="\n  - Output directory: $output_directory"
    log_info "$log_msg"
    
    # Terminate logging properly using close_logging
    log_info "Complete pipeline execution finished successfully"
    close_logging
    
    echo ""
    echo "${PROGRESS_SUCCESS} Complete pipeline execution finished successfully!"
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
