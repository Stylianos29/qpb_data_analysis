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
# Note: Processing stages (2A, 2B, 2C) are implemented directly in this script
# PROCESSING_PIPELINE_SCRIPT is no longer used but kept for reference
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
    # Clear any exit handlers before exiting
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
    # Validate that required workflow scripts exist
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
    
    # Validate parsing pipeline script exists
    if [[ ! -f "$PARSING_PIPELINE_SCRIPT" ]]; then
        echo "ERROR: Parsing pipeline script not found: $PARSING_PIPELINE_SCRIPT" >&2
        log_error "Missing parsing pipeline script"
        return 1
    fi
    
    # Validate processing pipeline script exists
    if [[ ! -f "$PROCESSING_PIPELINE_SCRIPT" ]]; then
        echo "ERROR: Processing pipeline script not found: $PROCESSING_PIPELINE_SCRIPT" >&2
        log_error "Missing processing pipeline script"
        return 1
    fi
    
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

function run_processing_stage_2a() {
    # Execute Stage 2A: Process raw parameters (ALWAYS runs)
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 2A: PROCESSING RAW PARAMETERS"
    echo "==================================================================="
    echo "Processing and validating raw parameters..."
    
    log_info "=== STAGE 2A: PROCESS RAW PARAMETERS ==="
    log_info "This stage always runs regardless of correlator presence"
    
    # Define paths to parsed files from Stage 1
    local parsed_csv_path="${output_directory}/${PARSED_CSV_FILENAME}"
    local parsed_hdf5_log_path="${output_directory}/${PARSED_HDF5_LOG_FILENAME}"
    
    # Validate that parsed CSV file exists
    if [[ ! -f "$parsed_csv_path" ]]; then
        echo "ERROR: Parsed CSV file not found: $parsed_csv_path" >&2
        log_error "Missing parsed CSV file from Stage 1"
        return 1
    fi
    
    # Validate that parsed HDF5 log file exists
    if [[ ! -f "$parsed_hdf5_log_path" ]]; then
        echo "ERROR: Parsed HDF5 log file not found: $parsed_hdf5_log_path" >&2
        log_error "Missing parsed HDF5 log file from Stage 1"
        return 1
    fi
    
    # Define Python scripts directory
    local python_scripts_dir="$(realpath "${SCRIPT_DIR}/../../core/src")"
    local process_raw_script="${python_scripts_dir}/processing/process_raw_parameters.py"
    
    # Validate script exists
    if [[ ! -f "$process_raw_script" ]]; then
        echo "ERROR: Python script not found: $process_raw_script" >&2
        log_error "Missing process_raw_parameters.py script"
        return 1
    fi
    
    # Build command
    local cmd="python \"$process_raw_script\""
    cmd+=" --input_single_valued_csv_file_path \"$parsed_csv_path\""
    cmd+=" --input_multivalued_hdf5_file_path \"$parsed_hdf5_log_path\""
    cmd+=" --output_directory \"$output_directory\""
    cmd+=" --output_csv_filename \"processed_parameter_values.csv\""
    cmd+=" --enable_logging"
    cmd+=" --log_directory \"$log_directory\""
    
    log_info "Command: $cmd"
    
    # Execute
    if eval "$cmd"; then
        echo "✓ Stage 2A completed: Raw parameters processed"
        log_info "Stage 2A completed successfully"
        return 0
    else
        echo "ERROR: Stage 2A failed" >&2
        log_error "Stage 2A failed"
        return 1
    fi
}

function run_processing_stage_2b() {
    # Execute Stage 2B: Jackknife analysis (only if correlators exist)
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 2B: JACKKNIFE ANALYSIS"
    echo "==================================================================="
    echo "Applying jackknife resampling for error estimation..."
    
    log_info "=== STAGE 2B: JACKKNIFE ANALYSIS ==="
    
    # Define path to correlator file from Stage 1
    local correlator_hdf5_path="${output_directory}/${PARSED_HDF5_CORR_FILENAME}"
    
    # Validate that correlator file exists
    if [[ ! -f "$correlator_hdf5_path" ]]; then
        echo "ERROR: Correlator HDF5 file not found: $correlator_hdf5_path" >&2
        log_error "Missing correlator HDF5 file from Stage 1"
        return 1
    fi
    
    # Define Python scripts directory
    local python_scripts_dir="$(realpath "${SCRIPT_DIR}/../../core/src")"
    local jackknife_script="${python_scripts_dir}/processing/apply_jackknife_analysis.py"
    
    # Validate script exists
    if [[ ! -f "$jackknife_script" ]]; then
        echo "ERROR: Python script not found: $jackknife_script" >&2
        log_error "Missing apply_jackknife_analysis.py script"
        return 1
    fi
    
    # Build command
    local cmd="python \"$jackknife_script\""
    cmd+=" --input_hdf5_file_path \"$correlator_hdf5_path\""
    cmd+=" --output_directory \"$output_directory\""
    cmd+=" --output_hdf5_filename \"correlators_jackknife_analysis.h5\""
    cmd+=" --enable_logging"
    cmd+=" --log_directory \"$log_directory\""
    
    if $skip_checks; then
        cmd+=" --skip_validation"
    fi
    
    log_info "Command: $cmd"
    
    # Execute
    if eval "$cmd"; then
        echo "✓ Stage 2B completed: Jackknife analysis applied"
        log_info "Stage 2B completed successfully"
        return 0
    else
        echo "ERROR: Stage 2B failed" >&2
        log_error "Stage 2B failed"
        return 1
    fi
}

function run_processing_stage_2c() {
    # Execute Stage 2C: Visualization (optional, only if correlators exist)
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 2C: VISUALIZATION (OPTIONAL)"
    echo "==================================================================="
    echo "Generating jackknife sample diagnostic plots..."
    
    log_info "=== STAGE 2C: VISUALIZATION ==="
    
    # Define path to jackknife results from Stage 2B
    local jackknife_hdf5_path="${output_directory}/correlators_jackknife_analysis.h5"
    
    # Validate that jackknife file exists
    if [[ ! -f "$jackknife_hdf5_path" ]]; then
        echo "ERROR: Jackknife HDF5 file not found: $jackknife_hdf5_path" >&2
        log_error "Missing jackknife HDF5 file from Stage 2B"
        return 1
    fi
    
    # Define Python scripts directory
    local python_scripts_dir="$(realpath "${SCRIPT_DIR}/../../core/src")"
    local viz_script="${python_scripts_dir}/processing/visualize_jackknife_samples.py"
    
    # Validate script exists
    if [[ ! -f "$viz_script" ]]; then
        echo "ERROR: Python script not found: $viz_script" >&2
        log_error "Missing visualize_jackknife_samples.py script"
        return 1
    fi
    
    # Build command
    local cmd="python \"$viz_script\""
    cmd+=" --input_hdf5_file_path \"$jackknife_hdf5_path\""
    cmd+=" --output_directory \"$output_directory\""
    cmd+=" --enable_logging"
    cmd+=" --log_directory \"$log_directory\""
    
    log_info "Command: $cmd"
    
    # Execute
    if eval "$cmd"; then
        echo "✓ Stage 2C completed: Visualizations generated"
        log_info "Stage 2C completed successfully"
        return 0
    else
        echo "WARNING: Stage 2C failed (visualization is optional)" >&2
        log_warning "Stage 2C failed but continuing (visualization is optional)"
        return 0  # Don't fail pipeline for optional visualization
    fi
}

# =============================================================================
# SECTION 5: ARGUMENT PARSING
# =============================================================================

# Initialize variables
input_directory=""
output_directory=""
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
        echo "INFO: Auto-detected output directory: $output_directory"
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
    echo "INFO: Created output directory: $output_directory"
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

# Execute Stage 2A: Process Raw Parameters (ALWAYS runs)
if ! run_processing_stage_2a; then
    echo ""
    echo "PIPELINE FAILED: Error in Stage 2A (process raw parameters)" >&2
    log_error "Pipeline terminated: Stage 2A failed"
    exit 1
fi

# Execute Stage 2B & 2C: Jackknife Analysis (only if correlators present)
if $has_correlators; then
    # Stage 2B: Jackknife Analysis
    if ! run_processing_stage_2b; then
        echo ""
        echo "PIPELINE FAILED: Error in Stage 2B (jackknife analysis)" >&2
        log_error "Pipeline terminated: Stage 2B failed"
        exit 1
    fi
    
    # Stage 2C: Visualization (optional, only if enabled)
    if $enable_visualization; then
        run_processing_stage_2c
        # Note: This stage is optional and won't fail the pipeline
    else
        echo ""
        echo "○ Stage 2C skipped: Visualization not enabled"
        log_info "Stage 2C skipped (visualization not enabled)"
    fi
else
    echo ""
    echo "==================================================================="
    echo "   STAGES 2B & 2C: SKIPPED"
    echo "==================================================================="
    echo "No correlator files detected - jackknife analysis not applicable"
    log_info "Stages 2B and 2C skipped (no correlator files)"
fi

# Pipeline completion
echo ""
echo "==================================================================="
echo "   PIPELINE COMPLETED SUCCESSFULLY"
echo "==================================================================="
echo "All applicable stages completed"
echo "Output location: $(get_display_path "$output_directory")"
echo "Log file: $(get_display_path "$SCRIPT_LOG_FILE_PATH")"
echo "==================================================================="

log_info "=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ==="
log_info "All stages completed without errors"

exit 0
