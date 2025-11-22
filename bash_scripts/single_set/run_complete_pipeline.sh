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
# Stage 3 (Analysis) - Only if correlators present:
#   Stage 3.1 (Correlator Calculations):
#     - PCAC mass time series calculation
#     - Pion effective mass calculation
#     - Optional visualization (if plots directory provided)
#   Stage 3.2 (Plateau Extraction):
#     - PCAC mass plateau extraction
#     - Pion mass plateau extraction
#     - Optional visualization (if plots directory provided)
#   Stage 3.3 (Critical Mass Extrapolation):
#     - Critical mass from PCAC
#     - Critical mass from Pion
#     - Optional visualization (if plots directory provided)
#     - Note: Requires multiple bare mass values; skips gracefully if insufficient
#   Stage 3.4 (Cost Extrapolation):
#     - Cost extrapolation from PCAC
#     - Cost extrapolation from Pion
#     - Optional visualization (if plots directory provided)
#     - Note: Only runs if Stage 3.3 produces results
#   - Calls: run_correlator_calculations.sh, run_plateau_extraction.sh,
#            run_critical_mass.sh, run_cost_extrapolation.sh
#
# VISUALIZATION:
#   Visualization is automatically enabled for all stages when a plots directory
#   is provided via -plots_dir option.
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

# Workflow script paths - Stages 1 and 2
PARSING_PIPELINE_SCRIPT="${WORKFLOWS_DIR}/run_parsing_pipeline.sh"
PROCESSING_PIPELINE_SCRIPT="${WORKFLOWS_DIR}/run_processing_pipeline.sh"

# Workflow script paths - Stage 3 (Analysis)
ANALYSIS_WORKFLOWS_DIR="${WORKFLOWS_DIR}/analysis"
CORRELATOR_CALCULATIONS_SCRIPT="${ANALYSIS_WORKFLOWS_DIR}/run_correlator_calculations.sh"
PLATEAU_EXTRACTION_SCRIPT="${ANALYSIS_WORKFLOWS_DIR}/run_plateau_extraction.sh"
CRITICAL_MASS_SCRIPT="${ANALYSIS_WORKFLOWS_DIR}/run_critical_mass.sh"
COST_EXTRAPOLATION_SCRIPT="${ANALYSIS_WORKFLOWS_DIR}/run_cost_extrapolation.sh"

# Script log filename
SCRIPT_LOG_FILENAME="${SCRIPT_NAME%.sh}.log"

# Expected output filenames from parsing stage (Stage 1)
PARSED_CSV_FILENAME="single_valued_parameters.csv"
PARSED_HDF5_LOG_FILENAME="multivalued_parameters.h5"
PARSED_HDF5_CORR_FILENAME="correlators_raw_data.h5"

# Expected output filenames from processing stage (Stage 2)
PROCESSED_CSV_FILENAME="processed_parameter_values.csv"
JACKKNIFE_HDF5_FILENAME="correlators_jackknife_analysis.h5"

# Expected output filenames from analysis stage (Stage 3)
# Stage 3.1 outputs
PCAC_MASS_HDF5_FILENAME="PCAC_mass_analysis.h5"
PION_MASS_HDF5_FILENAME="pion_effective_mass_analysis.h5"

# Stage 3.2 outputs
PCAC_PLATEAU_CSV_FILENAME="plateau_PCAC_mass_estimates.csv"
PCAC_PLATEAU_HDF5_FILENAME="plateau_PCAC_mass_estimates.h5"
PION_PLATEAU_CSV_FILENAME="plateau_pion_mass_estimates.csv"
PION_PLATEAU_HDF5_FILENAME="plateau_pion_mass_estimates.h5"

# Stage 3.3 outputs
CRITICAL_PCAC_CSV_FILENAME="critical_bare_mass_from_pcac.csv"
CRITICAL_PION_CSV_FILENAME="critical_bare_mass_from_pion.csv"

# Stage 3.4 outputs
COST_PCAC_CSV_FILENAME="cost_extrapolation_from_pcac.csv"
COST_PION_CSV_FILENAME="cost_extrapolation_from_pion.csv"

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
  -plots_dir, --plots_directory Plots directory (enables visualization for all stages)
  -log_dir, --log_directory    Log files directory (default: output_directory)
  --skip_checks                Skip intermediate file validation
  --skip_summaries             Skip generation of summary files
  -h, --help                   Display this help message

PIPELINE BEHAVIOR:
  With correlator files (.dat):
    - Executes Stages 1, 2A, 2B, 2C (if plots), 3.1, 3.2, 3.3, 3.4
    - Complete end-to-end analysis including cost extrapolation
    - Stage 3.3/3.4 may skip if insufficient bare mass variation
  
  Without correlator files:
    - Executes Stages 1A, 2A only
    - Parameters-only processing

VISUALIZATION:
  Visualization is automatically enabled when -plots_dir is provided.
  Applies to: Stage 2C, Stage 3.1, Stage 3.2, Stage 3.3, Stage 3.4

EXAMPLES:
  # Basic usage - no visualization
  $SCRIPT_NAME -i ../data_files/raw/invert/Chebyshev_experiment/
  
  # With visualization (auto-enabled by providing plots directory)
  $SCRIPT_NAME -i raw_data/ -plots_dir output/plots/
  
  # Custom output and plots directories
  $SCRIPT_NAME -i raw_data/ -o processed_data/ -plots_dir plots/
  
  # Fast processing (skip validation and summaries)
  $SCRIPT_NAME -i raw_data/ --skip_checks --skip_summaries

STAGES EXECUTED:
  Stage 1:  Parsing (always)
  Stage 2A: Processing parameters (always)
  Stage 2B: Jackknife analysis (if correlators present)
  Stage 2C: Jackknife visualization (if correlators + plots_dir)
  Stage 3.1: Correlator calculations (if correlators present)
  Stage 3.2: Plateau extraction (if correlators present)
  Stage 3.3: Critical mass extrapolation (if correlators + sufficient data)
  Stage 3.4: Cost extrapolation (if 3.3 produces results)

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
    
    # Validate Stage 3 scripts if correlators are present
    if $has_correlators; then
        validate_workflow_script "$CORRELATOR_CALCULATIONS_SCRIPT" "correlator calculations" || return 1
        validate_workflow_script "$PLATEAU_EXTRACTION_SCRIPT" "plateau extraction" || return 1
        validate_workflow_script "$CRITICAL_MASS_SCRIPT" "critical mass extraction" || return 1
        validate_workflow_script "$COST_EXTRAPOLATION_SCRIPT" "cost extrapolation" || return 1
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
        # Enable visualization in Stage 2 if plots directory provided
        processing_cmd+=" -viz"
    fi
    # Otherwise, processing pipeline will use its default (output_directory)
    
    processing_cmd+=" -log_dir \"$log_directory\""
    
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

function run_stage_3_1() {
    # Execute Stage 3.1: Correlator Calculations
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 3.1: CORRELATOR CALCULATIONS"
    echo "==================================================================="
    echo "Calculating PCAC mass and pion effective mass..."
    
    log_info "=== STAGE 3.1: CORRELATOR CALCULATIONS ==="
    log_info "Executing correlator calculations script"
    
    # Define input file from Stage 2B
    local jackknife_hdf5="${output_directory}/${JACKKNIFE_HDF5_FILENAME}"
    
    # Build command
    local cmd="$CORRELATOR_CALCULATIONS_SCRIPT"
    cmd+=" -hdf5_jack \"$jackknife_hdf5\""
    cmd+=" -o \"$output_directory\""
    
    # Add plots directory if specified (visualization auto-enabled in substage)
    if [[ -n "$plots_directory" ]]; then
        cmd+=" -p \"$plots_directory\" --enable-viz"
    fi
    
    cmd+=" -log_dir \"$log_directory\""
    
    # Note: Visualization is auto-enabled in substage if plots_directory provided
    # Not passing --enable-viz flag to avoid compatibility issues
    
    # Add skip flags
    if $skip_checks; then
        cmd+=" --skip-checks"
    fi
    
    if $skip_summaries; then
        cmd+=" --skip-summaries"
    fi
    
    log_info "Command: $cmd"
    
    # Execute
    if eval "$cmd"; then
        echo "✓ Stage 3.1 completed successfully"
        log_info "Stage 3.1 completed successfully"
        return 0
    else
        echo "ERROR: Stage 3.1 failed" >&2
        log_error "Stage 3.1 failed"
        return 1
    fi
}

function run_stage_3_2() {
    # Execute Stage 3.2: Plateau Extraction
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 3.2: PLATEAU EXTRACTION"
    echo "==================================================================="
    echo "Extracting PCAC and pion mass plateaus..."
    
    log_info "=== STAGE 3.2: PLATEAU EXTRACTION ==="
    log_info "Executing plateau extraction script"
    
    # Define input files from Stage 3.1
    local pcac_mass_hdf5="${output_directory}/${PCAC_MASS_HDF5_FILENAME}"
    local pion_mass_hdf5="${output_directory}/${PION_MASS_HDF5_FILENAME}"
    
    # Build command
    local cmd="$PLATEAU_EXTRACTION_SCRIPT"
    cmd+=" -i_pcac \"$pcac_mass_hdf5\""
    cmd+=" -i_pion \"$pion_mass_hdf5\""
    cmd+=" -o \"$output_directory\""
    
    # Add plots directory if specified (visualization auto-enabled in substage)
    if [[ -n "$plots_directory" ]]; then
        cmd+=" -p \"$plots_directory\" --enable-viz"
    fi
    
    cmd+=" -log_dir \"$log_directory\""
    
    # Note: Visualization is auto-enabled in substage if plots_directory provided
    # Not passing --enable-viz flag to avoid compatibility issues
    
    # Add skip flags
    if $skip_checks; then
        cmd+=" --skip-checks"
    fi
    
    if $skip_summaries; then
        cmd+=" --skip-summaries"
    fi
    
    log_info "Command: $cmd"
    
    # Execute
    if eval "$cmd"; then
        echo "✓ Stage 3.2 completed successfully"
        log_info "Stage 3.2 completed successfully"
        return 0
    else
        echo "ERROR: Stage 3.2 failed" >&2
        log_error "Stage 3.2 failed"
        return 1
    fi
}

function run_stage_3_3() {
    # Execute Stage 3.3: Critical Mass Extrapolation
    #
    # Returns:
    #   0 - Success (at least one branch succeeded OR both gracefully skipped)
    #   1 - Failure (hard error occurred)
    #   2 - Success but no output (both branches skipped gracefully)
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 3.3: CRITICAL MASS EXTRAPOLATION"
    echo "==================================================================="
    echo "Extrapolating critical bare mass from PCAC and pion..."
    
    log_info "=== STAGE 3.3: CRITICAL MASS EXTRAPOLATION ==="
    log_info "Executing critical mass extraction script"
    
    # Define input files from Stage 3.2
    local pcac_plateau_csv="${output_directory}/${PCAC_PLATEAU_CSV_FILENAME}"
    local pion_plateau_csv="${output_directory}/${PION_PLATEAU_CSV_FILENAME}"
    
    # Build command
    local cmd="$CRITICAL_MASS_SCRIPT"
    cmd+=" -i_pcac \"$pcac_plateau_csv\""
    cmd+=" -i_pion \"$pion_plateau_csv\""
    cmd+=" -o \"$output_directory\""
    
    # Add plots directory if specified (visualization auto-enabled in substage)
    if [[ -n "$plots_directory" ]]; then
        cmd+=" -p \"$plots_directory\" --enable-viz"
    fi
    
    cmd+=" -log_dir \"$log_directory\""
    
    # Note: Visualization is auto-enabled in substage if plots_directory provided
    # Not passing --enable-viz flag to avoid compatibility issues
    
    # Add skip flags
    if $skip_checks; then
        cmd+=" --skip-checks"
    fi
    
    if $skip_summaries; then
        cmd+=" --skip-summaries"
    fi
    
    log_info "Command: $cmd"
    
    # Execute
    if eval "$cmd"; then
        echo "✓ Stage 3.3 completed successfully"
        log_info "Stage 3.3 completed successfully"
        
        # Check if any critical mass files were created
        local pcac_critical_csv="${output_directory}/${CRITICAL_PCAC_CSV_FILENAME}"
        local pion_critical_csv="${output_directory}/${CRITICAL_PION_CSV_FILENAME}"
        
        if [[ ! -f "$pcac_critical_csv" && ! -f "$pion_critical_csv" ]]; then
            echo ""
            echo "⚠ WARNING: No critical mass results produced"
            echo "  → Insufficient bare mass variation for extrapolation"
            echo "  → Skipping Stage 3.4 (Cost Extrapolation)"
            log_warning "Stage 3.3 produced no results - insufficient data variation"
            log_info "Stage 3.4 will be skipped"
            return 2  # Special return code: success but no output
        fi
        
        return 0
    else
        echo "ERROR: Stage 3.3 failed" >&2
        log_error "Stage 3.3 failed"
        return 1
    fi
}

function run_stage_3_4() {
    # Execute Stage 3.4: Cost Extrapolation
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 3.4: COST EXTRAPOLATION"
    echo "==================================================================="
    echo "Extrapolating computational costs..."
    
    log_info "=== STAGE 3.4: COST EXTRAPOLATION ==="
    log_info "Executing cost extrapolation script"
    
    # Define input files
    local pcac_plateau_csv="${output_directory}/${PCAC_PLATEAU_CSV_FILENAME}"
    local pion_plateau_csv="${output_directory}/${PION_PLATEAU_CSV_FILENAME}"
    local processed_csv="${output_directory}/${PROCESSED_CSV_FILENAME}"
    
    # Build command
    local cmd="$COST_EXTRAPOLATION_SCRIPT"
    cmd+=" -i_pcac \"$pcac_plateau_csv\""
    cmd+=" -i_pion \"$pion_plateau_csv\""
    cmd+=" -i_cost \"$processed_csv\""
    cmd+=" -o \"$output_directory\""
    
    # Add plots directory if specified (visualization auto-enabled in substage)
    if [[ -n "$plots_directory" ]]; then
        cmd+=" -p \"$plots_directory\" --enable-viz"
    fi
    
    cmd+=" -log_dir \"$log_directory\""
    
    # Note: Visualization is auto-enabled in substage if plots_directory provided
    # Not passing --enable-viz flag to avoid compatibility issues
    
    # Add skip flags
    if $skip_checks; then
        cmd+=" --skip-checks"
    fi
    
    if $skip_summaries; then
        cmd+=" --skip-summaries"
    fi
    
    log_info "Command: $cmd"
    
    # Execute
    if eval "$cmd"; then
        echo "✓ Stage 3.4 completed successfully"
        log_info "Stage 3.4 completed successfully"
        return 0
    else
        echo "ERROR: Stage 3.4 failed" >&2
        log_error "Stage 3.4 failed"
        return 1
    fi
}

function run_analysis_stage() {
    # Execute Stage 3: Complete Analysis Pipeline
    # Runs all four substages sequentially: 3.1 → 3.2 → 3.3 → 3.4
    # Stops gracefully if Stage 3.3 produces no results (insufficient data)
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 3: ANALYSIS"
    echo "==================================================================="
    echo "Running complete analysis pipeline (up to 4 substages)..."
    
    log_info "=== STAGE 3: ANALYSIS STAGE ==="
    log_info "Executing all analysis substages"
    
    # Stage 3.1: Correlator Calculations
    if ! run_stage_3_1; then
        echo ""
        echo "PIPELINE FAILED: Error in Stage 3.1" >&2
        log_error "Pipeline terminated: Stage 3.1 failed"
        return 1
    fi
    
    # Stage 3.2: Plateau Extraction
    if ! run_stage_3_2; then
        echo ""
        echo "PIPELINE FAILED: Error in Stage 3.2" >&2
        log_error "Pipeline terminated: Stage 3.2 failed"
        return 1
    fi
    
    # Stage 3.3: Critical Mass Extrapolation
    run_stage_3_3
    local stage_3_3_result=$?
    
    if [[ $stage_3_3_result -eq 1 ]]; then
        # Hard failure
        echo ""
        echo "PIPELINE FAILED: Error in Stage 3.3" >&2
        log_error "Pipeline terminated: Stage 3.3 failed"
        return 1
    elif [[ $stage_3_3_result -eq 2 ]]; then
        # Graceful skip - no results produced
        echo ""
        echo "○ Stage 3.4 skipped: No critical mass results from Stage 3.3"
        log_info "Stage 3.4 skipped: insufficient data for critical mass extrapolation"
        log_info "Stage 3 completed successfully (Stages 3.1 and 3.2 only)"
        return 0
    fi
    
    # Stage 3.4: Cost Extrapolation (only if 3.3 produced results)
    if ! run_stage_3_4; then
        echo ""
        echo "PIPELINE FAILED: Error in Stage 3.4" >&2
        log_error "Pipeline terminated: Stage 3.4 failed"
        return 1
    fi
    
    echo ""
    echo "✓ Stage 3 (Analysis) completed successfully"
    log_info "Stage 3 completed successfully - all substages executed"
    return 0
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
    for log_file in "${output_directory}"/*_python_script.log "${output_directory}"/run_*.log; do
        # Skip the main orchestrator log
        if [[ -f "$log_file" && "$log_file" != "$SCRIPT_LOG_FILE_PATH" ]]; then
            mv "$log_file" "$aux_logs/" 2>/dev/null && ((logs_moved++))
        fi
    done
    
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
    echo "INFO: Plots directory provided - visualization will be enabled for all stages"
fi

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
    log_info "Plots directory: $plots_directory (visualization enabled)"
else
    log_info "Plots directory: not specified (visualization disabled)"
fi
log_info "Log directory: $log_directory"
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
    echo "Plots:  $(get_display_path "$plots_directory") (visualization enabled)"
fi
echo "==================================================================="

# Detect if data set has correlator files (determines pipeline path)
has_correlators=false
if detect_correlator_files "$input_directory"; then
    has_correlators=true
    echo "✓ Correlator files (.dat) detected"
    echo "  → Full pipeline will execute: Stages 1, 2A, 2B, 2C, 3.1, 3.2, 3.3, 3.4"
    log_info "Correlator files detected - full pipeline execution"
else
    echo "ℹ Only log files (.txt) detected"
    echo "  → Limited pipeline: Stages 1, 2A only"
    log_info "No correlator files detected - parameters-only pipeline"
fi

# Validate prerequisites
echo ""
echo "=== VALIDATING PREREQUISITES ==="
if ! validate_prerequisites "$input_directory"; then
    echo "ERROR: Prerequisites validation failed" >&2
    exit 1
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

# Execute Stage 3: Analysis (only if correlators present)
if $has_correlators; then
    if ! run_analysis_stage; then
        echo ""
        echo "PIPELINE FAILED: Error in analysis stage" >&2
        log_error "Pipeline terminated: Analysis stage failed"
        exit 1
    fi
else
    echo ""
    echo "○ Stage 3 (Analysis) skipped: No correlator data"
    log_info "Stage 3 skipped (no correlator data)"
fi

# Organize auxiliary files into subdirectories
organize_auxiliary_files

# =============================================================================
# PIPELINE COMPLETION
# =============================================================================

echo ""
echo "==================================================================="
echo "   PIPELINE COMPLETED SUCCESSFULLY"
echo "==================================================================="

# Display completion summary based on pipeline path
if $has_correlators; then
    echo "Complete analysis pipeline executed:"
    echo "  Stage 1:   Parsing ✓"
    echo "  Stage 2A:  Processing parameters ✓"
    echo "  Stage 2B:  Jackknife analysis ✓"
    if [[ -n "$plots_directory" ]]; then
        echo "  Stage 2C:  Jackknife visualization ✓"
    fi
    echo "  Stage 3.1: Correlator calculations ✓"
    echo "  Stage 3.2: Plateau extraction ✓"
    
    # Check if Stage 3.3/3.4 ran
    if [[ -f "${output_directory}/${CRITICAL_PCAC_CSV_FILENAME}" || -f "${output_directory}/${CRITICAL_PION_CSV_FILENAME}" ]]; then
        echo "  Stage 3.3: Critical mass extrapolation ✓"
        echo "  Stage 3.4: Cost extrapolation ✓"
    else
        echo "  Stage 3.3: Critical mass extrapolation ○ (skipped - insufficient data)"
        echo "  Stage 3.4: Cost extrapolation ○ (skipped - no Stage 3.3 results)"
    fi
else
    echo "Parameters-only pipeline executed:"
    echo "  Stage 1:  Parsing ✓"
    echo "  Stage 2A: Processing parameters ✓"
fi

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
if $has_correlators; then
    if [[ -f "${output_directory}/${CRITICAL_PCAC_CSV_FILENAME}" || -f "${output_directory}/${CRITICAL_PION_CSV_FILENAME}" ]]; then
        log_info "Full pipeline executed: All stages (1, 2A, 2B, 2C, 3.1-3.4) completed"
    else
        log_info "Partial pipeline executed: Stages 1, 2A, 2B, 2C, 3.1, 3.2 completed (3.3/3.4 skipped)"
    fi
else
    log_info "Parameters-only pipeline executed: Stages 1, 2A completed"
fi
log_info "Auxiliary files organized successfully"

exit 0
