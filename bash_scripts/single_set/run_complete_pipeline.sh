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
# Stage 2 (Processing) - CONDITIONAL:
#   Stage 2A (Process Raw Parameters) - ALWAYS runs:
#     - Processes raw parameters
#     - Validates and transforms data
#     - Calculates derived quantities
#   Stage 2B (Jackknife Analysis) - Only if correlators present:
#     - Applies jackknife resampling
#     - Generates statistical samples
#   Stage 2C (Visualization) - Optional (only if correlators present):
#     - Optional visualization of jackknife samples
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
# For detailed usage information, run with -h or --help
#
################################################################################

# =============================================================================
# SECTION 1: BASIC PATH SETUP (Minimal, no side effects)
# =============================================================================

SCRIPT_PATH="$(realpath "$0")"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# =============================================================================
# SECTION 2: LIBRARY PATH SETUP AND SOURCING
# =============================================================================

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
# SECTION 3: CONSTANTS AND CONFIGURATION
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

# Expected output filenames from parsing stage
PARSED_CSV_FILENAME="single_valued_parameters.csv"
PARSED_HDF5_LOG_FILENAME="multivalued_parameters.h5"
PARSED_HDF5_CORR_FILENAME="correlators_raw_data.h5"

# Auxiliary directory structure
AUXILIARY_DIR_NAME="auxiliary"
AUXILIARY_LOGS_SUBDIR="logs"
AUXILIARY_SUMMARIES_SUBDIR="summaries"

# =============================================================================
# SECTION 4: FUNCTION DEFINITIONS
# =============================================================================

function usage() {
    cat << EOF
Usage: $SCRIPT_NAME -i <raw_data_set_directory> [options]

REQUIRED ARGUMENTS:
  -i, --input_directory        Path to raw data file set directory

OPTIONAL ARGUMENTS:
  -o, --output_directory       Output directory (default: mirrors raw in processed/)
  -plots_dir, --plots_directory Plots directory (optional, default: output_directory)
  -log_dir, --log_directory    Log files directory (default: output_directory)
  -viz, --enable_visualization Enable visualization in processing stage
  --skip_checks                Skip intermediate file validation
  --skip_summaries             Skip generation of summary files
  -h, --help                   Display this help message

EXAMPLES:
  $SCRIPT_NAME -i ../data_files/raw/invert/Chebyshev_experiment/
  $SCRIPT_NAME -i raw_data/ -o processed_data/
  $SCRIPT_NAME -i raw_data/ -plots_dir output/plots/ -viz
  $SCRIPT_NAME -i raw_data/ -viz --skip_checks

EOF
    # Clear exit handlers before exiting
    trap - EXIT
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
        return 0
    else
        return 1
    fi
}

function validate_prerequisites() {
    # Validate that required workflow scripts exist and are executable
    #
    # Arguments:
    #   $1 - input_directory : Directory to validate (for file checks)
    #
    # Returns:
    #   0 - All prerequisites valid
    #   1 - Validation failed
    
    local input_directory="$1"
    
    # Check for .txt log files (required)
    if ! find "$input_directory" -maxdepth 1 -type f -name "*.txt" -print -quit | grep -q .; then
        echo "ERROR: No .txt log files found in $input_directory" >&2
        log_error "No .txt log files found in input directory"
        return 1
    fi
    
    # Validate workflow scripts exist and are executable
    validate_workflow_script "$PARSING_PIPELINE_SCRIPT" "parsing pipeline" || return 1
    validate_workflow_script "$PROCESSING_PIPELINE_SCRIPT" "processing pipeline" || return 1
    
    echo "✓ All prerequisites validated"
    log_info "Prerequisites validation completed successfully"
    return 0
}

function run_parsing_stage() {
    # Execute Stage 1: Parsing pipeline
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 1: PARSING"
    echo "==================================================================="
    echo "Parsing .txt log files and .dat correlator files (if present)..."
    
    log_info "=== STAGE 1: PARSING STAGE ==="
    log_info "Executing parsing pipeline script"
    
    # Build parsing command
    local parsing_cmd="$PARSING_PIPELINE_SCRIPT"
    parsing_cmd+=" -i \"$input_directory\""
    parsing_cmd+=" -o \"$output_directory\""
    parsing_cmd+=" -log_dir \"$log_directory\""
    
    if $skip_checks; then
        parsing_cmd+=" --skip_checks"
    fi
    
    if $skip_summaries; then
        parsing_cmd+=" --skip_summaries"
    fi
    
    log_info "Command: $parsing_cmd"
    
    # Execute parsing pipeline
    if eval "$parsing_cmd"; then
        echo "✓ Parsing stage completed successfully"
        log_info "Parsing stage completed successfully"
        return 0
    else
        echo "ERROR: Parsing stage failed" >&2
        log_error "Parsing stage failed"
        return 1
    fi
}

function run_processing_stage() {
    # Execute Stage 2: Processing pipeline (2A always, 2B/2C conditional)
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 2: PROCESSING"
    echo "==================================================================="
    echo "Processing parameters and applying analysis..."
    
    log_info "=== STAGE 2: PROCESSING STAGE ==="
    log_info "Executing processing pipeline script"
    
    # Define paths to parsed files from Stage 1
    local parsed_csv_path="${output_directory}/${PARSED_CSV_FILENAME}"
    local parsed_hdf5_log_path="${output_directory}/${PARSED_HDF5_LOG_FILENAME}"
    local parsed_hdf5_corr_path="${output_directory}/${PARSED_HDF5_CORR_FILENAME}"
    
    # Build processing command
    local processing_cmd="$PROCESSING_PIPELINE_SCRIPT"
    processing_cmd+=" -csv \"$parsed_csv_path\""
    processing_cmd+=" -hdf5_param \"$parsed_hdf5_log_path\""
    
    # Add correlator file if it exists
    if $has_correlators && [[ -f "$parsed_hdf5_corr_path" ]]; then
        processing_cmd+=" -hdf5_corr \"$parsed_hdf5_corr_path\""
    fi
    
    # Add output directories
    processing_cmd+=" -out_dir \"$output_directory\""
    
    # Add plots directory only if user specified it
    if [[ -n "$plots_directory" ]]; then
        processing_cmd+=" -plots_dir \"$plots_directory\""
    fi
    # Otherwise, processing pipeline will use its default (output_directory)
    
    processing_cmd+=" -log_dir \"$log_directory\""
    
    # Add visualization flag
    if $enable_visualization; then
        processing_cmd+=" -viz"
    fi
    
    # Add skip checks flag
    if $skip_checks; then
        processing_cmd+=" --skip_checks"
    fi
    
    log_info "Command: $processing_cmd"
    
    # Execute processing pipeline
    if eval "$processing_cmd"; then
        echo "✓ Processing stage completed successfully"
        log_info "Processing stage completed successfully"
        return 0
    else
        echo "ERROR: Processing stage failed" >&2
        log_error "Processing stage failed"
        return 1
    fi
}

function organize_auxiliary_files() {
    # Organize auxiliary files (logs and summaries) into subdirectories
    #
    # Moves log files and summary files from output_directory into
    # auxiliary/logs/ and auxiliary/summaries/ subdirectories.
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    
    echo ""
    echo "=== ORGANIZING AUXILIARY FILES ==="
    log_info "Organizing auxiliary files into subdirectories"
    
    # Create auxiliary directory structure
    local aux_base="${output_directory}/${AUXILIARY_DIR_NAME}"
    local aux_logs="${aux_base}/${AUXILIARY_LOGS_SUBDIR}"
    local aux_summaries="${aux_base}/${AUXILIARY_SUMMARIES_SUBDIR}"
    
    mkdir -p "$aux_logs" || {
        echo "WARNING: Failed to create auxiliary logs directory" >&2
        log_warning "Failed to create auxiliary logs directory"
        return 1
    }
    
    mkdir -p "$aux_summaries" || {
        echo "WARNING: Failed to create auxiliary summaries directory" >&2
        log_warning "Failed to create auxiliary summaries directory"
        return 1
    }
    
    # Move log files (excluding the orchestrator's own log)
    local logs_moved=0
    for log_file in "${output_directory}"/*_python_script.log "${output_directory}"/run_parsing_pipeline.log "${output_directory}"/run_processing_pipeline.log; do
        if [[ -f "$log_file" ]]; then
            mv "$log_file" "$aux_logs/" 2>/dev/null && ((logs_moved++))
        fi
    done
    # Note: run_complete_pipeline.log stays in root as the main orchestrator log
    
    # Move summary files
    local summaries_moved=0
    for summary_file in "${output_directory}"/*_summary.txt "${output_directory}"/*_tree.txt "${output_directory}"/*_report.txt; do
        if [[ -f "$summary_file" ]]; then
            mv "$summary_file" "$aux_summaries/" 2>/dev/null && ((summaries_moved++))
        fi
    done
    
    echo "  → Moved $logs_moved log files to auxiliary/logs/"
    echo "  → Moved $summaries_moved summary files to auxiliary/summaries/"
    log_info "Organized auxiliary files: $logs_moved logs, $summaries_moved summaries"
    
    return 0
}

# =============================================================================
# SECTION 5: ARGUMENT PARSING
# =============================================================================

# Initialize variables
input_directory=""
output_directory=""
plots_directory=""
log_directory=""
enable_visualization=false
skip_checks=false
skip_summaries=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input_directory)
            input_directory="$2"
            shift 2
            ;;
        -o|--output_directory)
            output_directory="$2"
            shift 2
            ;;
        -plots_dir|--plots_directory)
            plots_directory="$2"
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
            echo "ERROR: Unknown option: $1" >&2
            usage
            ;;
    esac
done

# =============================================================================
# SECTION 6: HELP CHECK (Before any logging setup!)
# =============================================================================

# Note: Help is already handled in argument parsing above
# This section is here for clarity in the structure

# =============================================================================
# SECTION 7: INPUT VALIDATION AND LOGGING SETUP
# =============================================================================

# Validate required input
if [[ -z "$input_directory" ]]; then
    echo "ERROR: Input directory not specified. Use -i <directory>" >&2
    usage
fi

# Validate input directory exists
if [[ ! -d "$input_directory" ]]; then
    echo "ERROR: Input directory does not exist: $input_directory" >&2
    exit 1
fi

# Convert to absolute path
input_directory="$(realpath "$input_directory")"
input_dir_name="$(basename "$input_directory")"

# Set default output directory if not specified
# Mirror the raw data structure in processed/
if [[ -z "$output_directory" ]]; then
    # Try to replace 'raw' with 'processed' in path
    if [[ "$input_directory" == *"/data_files/raw/"* ]]; then
        output_directory="${input_directory/\/raw\//\/processed\/}"
        echo "INFO: Auto-detected output directory: $(get_display_path "$output_directory")"
    else
        # Fallback: use input directory
        output_directory="$input_directory"
        echo "INFO: Using input directory as output directory"
    fi
fi

# Ensure output directory exists
if [[ ! -d "$output_directory" ]]; then
    mkdir -p "$output_directory" || {
        echo "ERROR: Failed to create output directory: $output_directory" >&2
        exit 1
    }
    echo "INFO: Created output directory: $(get_display_path "$output_directory")"
fi
output_directory="$(realpath "$output_directory")"

# Set default log directory to output directory if not specified
if [[ -z "$log_directory" ]]; then
    log_directory="$output_directory"
fi

# Ensure log directory exists
if [[ ! -d "$log_directory" ]]; then
    mkdir -p "$log_directory" || {
        echo "ERROR: Failed to create log directory: $log_directory" >&2
        exit 1
    }
fi
log_directory="$(realpath "$log_directory")"

# Handle plots directory if specified
if [[ -n "$plots_directory" ]]; then
    # Ensure plots directory exists if user specified it
    if [[ ! -d "$plots_directory" ]]; then
        mkdir -p "$plots_directory" || {
            echo "ERROR: Failed to create plots directory: $plots_directory" >&2
            exit 1
        }
        echo "INFO: Created plots directory: $(get_display_path "$plots_directory")"
    fi
    plots_directory="$(realpath "$plots_directory")"
    echo "INFO: Using plots directory: $(get_display_path "$plots_directory")"
fi
# If not specified, processing pipeline will use its own default (output_directory)

# NOW setup logging infrastructure (after help check and validation)
export SCRIPT_TERMINATION_MESSAGE="\n\t\t$(echo "$SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') EXECUTION TERMINATED"

# Initialize logging
SCRIPT_LOG_FILE_PATH="${log_directory}/${SCRIPT_LOG_FILENAME}"
export SCRIPT_LOG_FILE_PATH

echo -e "\t\t$(echo "$SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') EXECUTION INITIATED\n" > "$SCRIPT_LOG_FILE_PATH"

log_info "=== COMPLETE PIPELINE EXECUTION ==="
log_info "Script: $SCRIPT_NAME"
log_info "Data file set: $input_dir_name"
log_info "Input directory: $input_directory"
log_info "Output directory: $output_directory"
if [[ -n "$plots_directory" ]]; then
    log_info "Plots directory: $plots_directory"
fi
log_info "Log directory: $log_directory"
log_info "Visualization: $enable_visualization"
log_info "Skip checks: $skip_checks"
log_info "Skip summaries: $skip_summaries"

# =============================================================================
# SECTION 8: MAIN EXECUTION
# =============================================================================

# Display banner
echo ""
echo "==================================================================="
echo "   QPB DATA ANALYSIS - COMPLETE PIPELINE"
echo "==================================================================="
echo "Data file set: $input_dir_name"
echo "Input:  $(get_display_path "$input_directory")"
echo "Output: $(get_display_path "$output_directory")"
if [[ -n "$plots_directory" ]]; then
    echo "Plots:  $(get_display_path "$plots_directory")"
fi
echo "==================================================================="

# Validate prerequisites
echo ""
echo "=== VALIDATING PREREQUISITES ==="
if ! validate_prerequisites "$input_directory"; then
    echo "ERROR: Prerequisites validation failed" >&2
    exit 1
fi

# Detect if data set has correlator files
has_correlators=false
if detect_correlator_files "$input_directory"; then
    has_correlators=true
    echo "✓ Correlator files (.dat) detected - Full pipeline will execute (Stages 1, 2A, 2B, 2C)"
    log_info "Correlator files detected - all processing stages will run"
else
    echo "ℹ Only log files (.txt) detected - Stages 2B & 2C will be skipped"
    log_info "No correlator files detected - only Stages 1 and 2A will run"
fi

# Execute Stage 1: Parsing
if ! run_parsing_stage; then
    echo ""
    echo "PIPELINE FAILED: Error in parsing stage" >&2
    log_error "Pipeline terminated: Parsing stage failed"
    exit 1
fi

# Execute Stage 2: Processing (always runs 2A, conditionally runs 2B/2C)
if ! run_processing_stage; then
    echo ""
    echo "PIPELINE FAILED: Error in processing stage" >&2
    log_error "Pipeline terminated: Processing stage failed"
    exit 1
fi

# Organize auxiliary files into subdirectories
organize_auxiliary_files

# Pipeline completion
echo ""
echo "==================================================================="
echo "   PIPELINE COMPLETED SUCCESSFULLY"
echo "==================================================================="
echo "All applicable stages completed"
echo ""
echo "Output structure:"
echo "  Data files:    $(get_display_path "$output_directory")"
if [[ -n "$plots_directory" ]]; then
    echo "  Plots:         $(get_display_path "$plots_directory")"
fi
echo "  Auxiliary:"
echo "    - Logs:      $(get_display_path "${output_directory}/${AUXILIARY_DIR_NAME}/${AUXILIARY_LOGS_SUBDIR}")"
echo "    - Summaries: $(get_display_path "${output_directory}/${AUXILIARY_DIR_NAME}/${AUXILIARY_SUMMARIES_SUBDIR}")"
echo ""
echo "Main log file: $(get_display_path "$SCRIPT_LOG_FILE_PATH")"
echo "==================================================================="

log_info "=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ==="
log_info "All stages completed without errors"
log_info "Auxiliary files organized successfully"

exit 0