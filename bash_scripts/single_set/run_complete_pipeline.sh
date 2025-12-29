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
#     - Optional filename filtering (via --filter_config)
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
# SELECTIVE STAGE EXECUTION:
#   Use --stages flag to run only specific stages (e.g., --stages 2,3 to skip
#   parsing and only rerun processing and analysis). Requires appropriate input
#   files from previous stages.
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
  -filter, --filter_config  Path to JSON filter configuration for jackknife analysis
                            (optional, applies to Stage 2B only)
  -plots_dir, --plots_directory Plots directory (enables visualization for all stages)
  -log_dir, --log_directory    Log files directory (default: output_directory)
  --stages <1,2,3>            Comma-separated stages to execute (default: all)
                               Examples: --stages 2,3 (rerun processing+analysis)
                                        --stages 3 (rerun analysis only)
                                        --stages 1,2 (parse+process, no analysis)
  --skip_checks                Skip intermediate file validation
  --skip_summaries             Skip generation of summary files
  --clean-plots                Remove existing plots before generating new ones  
  -h, --help                   Display this help message

PIPELINE BEHAVIOR:
  With correlator files (.dat):
    - Executes Stages 1, 2A, 2B, 2C (if plots), 3.1, 3.2, 3.3, 3.4
    - Complete end-to-end analysis including cost extrapolation
    - Stage 3.3/3.4 may skip if insufficient bare mass variation
  
  Without correlator files:
    - Executes Stages 1A, 2A only
    - Parameters-only processing

STAGE EXECUTION:
  Stage 1 (Parsing):
    - Requires: Raw .txt and/or .dat files in input directory
    - Outputs: single_valued_parameters.csv, multivalued_parameters.h5
    - Optional: correlators_raw_data.h5 (if .dat files present)
    
  Stage 2 (Processing):
    - Requires: Stage 1 outputs (single_valued_parameters.csv, multivalued_parameters.h5)
    - Outputs: processed_parameter_values.csv
    - Optional: correlators_jackknife_analysis.h5 (if correlators present)
    
  Stage 3 (Analysis):
    - Requires: Stage 2 outputs (processed_parameter_values.csv, correlators_jackknife_analysis.h5)
    - Only valid for data sets with correlators
    - Outputs: Various analysis CSVs and HDF5 files

  When using --stages:
    - Script validates required input files exist for each requested stage
    - Errors if inputs missing (strict validation)
    - Only creates output directory if running Stage 1

VISUALIZATION:
  Visualization is automatically enabled when -plots_dir is provided.
  Applies to all applicable stages (2C, 3.1, 3.2, 3.3, 3.4).

EXAMPLES:
  # Full pipeline with auto-detected output
  $SCRIPT_NAME -i ../data_files/raw/invert/my_experiment/

  # Full pipeline with custom output and plots
  $SCRIPT_NAME -i ../raw/experiment1/ -o ../processed/experiment1/ -plots_dir ../plots/

  # Rerun processing and analysis only (after parsing already done)
  $SCRIPT_NAME -i ../raw/experiment1/ --stages 2,3

  # Rerun only analysis (after parsing and processing already done)
  $SCRIPT_NAME -i ../raw/experiment1/ --stages 3

  # Parse and process only (skip analysis)
  $SCRIPT_NAME -i ../raw/experiment1/ --stages 1,2

  # Fast processing (skip checks and summaries)
  $SCRIPT_NAME -i ../raw/experiment1/ --skip_checks --skip_summaries

  # Process with filtering
  $SCRIPT_NAME \\
      -i raw_data_directory/ \\
      --filter_config filters/experiment_subset.json

NOTES:
  - Output directory structure mirrors input directory structure
  - Auxiliary files (logs, summaries) organized automatically
  - Selective stage execution useful for iterative development
  - Stage validation ensures data integrity across pipeline stages

EOF
    exit 0
}

function validate_stages_argument() {
    # Validate and normalize the --stages argument
    #
    # Arguments:
    #   $1 - stages : Comma-separated stage numbers (e.g., "2,3" or "1,2,3")
    #
    # Returns:
    #   Prints normalized stages string (sorted, deduplicated)
    #   Exit code 0 on success, 1 on validation failure
    #
    # Example:
    #   normalized=$(validate_stages_argument "2,1,2") → "1,2"
    
    local stages="$1"
    
    # Check for valid format (only 1,2,3 and commas, no spaces)
    if [[ ! "$stages" =~ ^[1-3](,[1-3])*$ ]]; then
        echo "ERROR: Invalid --stages format: '$stages'" >&2
        echo "  Valid format: comma-separated numbers (1, 2, or 3)" >&2
        echo "  Examples: --stages 1, --stages 2,3, --stages 1,2,3" >&2
        return 1
    fi
    
    # Remove duplicates and sort (e.g., "2,1,2" → "1,2")
    local normalized
    normalized=$(echo "$stages" | tr ',' '\n' | sort -u | tr '\n' ',' | sed 's/,$//')
    
    echo "$normalized"
    return 0
}

function should_run_stage() {
    # Check if a stage should be executed based on --stages argument
    #
    # Arguments:
    #   $1 - stage : Stage number to check (1, 2, or 3)
    #
    # Returns:
    #   0 - Stage should run
    #   1 - Stage should be skipped
    #
    # Example:
    #   if should_run_stage 2; then
    #       run_processing_stage
    #   fi
    
    local stage="$1"
    
    # If --stages not specified, run all stages (default behavior)
    if [[ -z "$stages_to_run" ]]; then
        return 0
    fi
    
    # Check if stage is in the comma-separated list
    if [[ ",$stages_to_run," == *",$stage,"* ]]; then
        return 0
    else
        return 1
    fi
}

function validate_stage_inputs() {
    # Validate that required input files exist for a given stage
    #
    # Arguments:
    #   $1 - stage : Stage number to validate (1, 2, or 3)
    #
    # Returns:
    #   0 - All required inputs exist
    #   1 - Missing required inputs (prints error messages)
    #
    # Stage requirements:
    #   Stage 1: Raw .txt files in input directory
    #   Stage 2: single_valued_parameters.csv, multivalued_parameters.h5
    #   Stage 3: processed_parameter_values.csv, correlators_jackknife_analysis.h5
    
    local stage="$1"
    
    case "$stage" in
        1)
            # Stage 1: Need raw data files
            if ! find "$input_directory" -maxdepth 1 -type f -name "*.txt" -print -quit | grep -q .; then
                echo "ERROR: Stage 1 requires .txt log files in input directory" >&2
                echo "  Missing: .txt files in $input_directory" >&2
                return 1
            fi
            return 0
            ;;
        2)
            # Stage 2: Need Stage 1 outputs
            local csv="${output_directory}/${PARSED_CSV_FILENAME}"
            local hdf5="${output_directory}/${PARSED_HDF5_LOG_FILENAME}"
            
            if [[ ! -f "$csv" ]]; then
                echo "ERROR: Stage 2 requires single-valued parameters CSV from Stage 1" >&2
                echo "  Missing: $csv" >&2
                echo "  Run Stage 1 first or use --stages 1,2" >&2
                return 1
            fi
            
            if [[ ! -f "$hdf5" ]]; then
                echo "ERROR: Stage 2 requires multivalued parameters HDF5 from Stage 1" >&2
                echo "  Missing: $hdf5" >&2
                echo "  Run Stage 1 first or use --stages 1,2" >&2
                return 1
            fi
            
            return 0
            ;;
        3)
            # Stage 3: Need Stage 2 outputs
            local csv="${output_directory}/${PROCESSED_CSV_FILENAME}"
            local hdf5="${output_directory}/${JACKKNIFE_HDF5_FILENAME}"
            
            if [[ ! -f "$csv" ]]; then
                echo "ERROR: Stage 3 requires processed CSV from Stage 2" >&2
                echo "  Missing: $csv" >&2
                echo "  Run Stage 2 first or use --stages 2,3" >&2
                return 1
            fi
            
            # Check for correlators - error if Stage 3 requested but no correlators
            if [[ ! -f "$hdf5" ]]; then
                echo "ERROR: Stage 3 requires jackknife HDF5 from Stage 2B" >&2
                echo "  This data set has no correlators - Stage 3 cannot run" >&2
                echo "  Missing: $hdf5" >&2
                return 1
            fi
            
            return 0
            ;;
        *)
            echo "ERROR: Invalid stage number: $stage" >&2
            return 1
            ;;
    esac
}

function validate_workflow_script() {
    # Validate that a workflow script exists and is executable
    #
    # Arguments:
    #   $1 - script_path : Path to workflow script
    #   $2 - script_name : Descriptive name for error messages
    #
    # Returns:
    #   0 - Script is valid
    #   1 - Script missing or not executable
    
    local script_path="$1"
    local script_name="$2"
    
    if [[ ! -f "$script_path" ]]; then
        echo "ERROR: $script_name script not found: $script_path" >&2
        log_error "$script_name script not found: $script_path"
        return 1
    fi
    
    if [[ ! -x "$script_path" ]]; then
        echo "ERROR: $script_name script is not executable: $script_path" >&2
        log_error "$script_name script is not executable: $script_path"
        return 1
    fi
    
    echo "  ✓ $script_name script validated"
    log_info "$script_name script validated: $script_path"
    return 0
}

function validate_prerequisites() {
    # Validate that required workflow scripts exist and are executable
    #
    # Returns:
    #   0 - All prerequisites valid
    #   1 - Validation failed
    
    # Validate workflow scripts based on which stages will run
    if should_run_stage 1; then
        validate_workflow_script "$PARSING_PIPELINE_SCRIPT" "parsing pipeline" || return 1
    fi
    
    if should_run_stage 2; then
        validate_workflow_script "$PROCESSING_PIPELINE_SCRIPT" "processing pipeline" || return 1
    fi
    
    # Validate Stage 3 scripts if Stage 3 will run and correlators are present
    if should_run_stage 3 && $has_correlators; then
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
    # Execute Stage 2: Processing
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 2: PROCESSING"
    echo "==================================================================="
    echo "Processing parsed parameters..."
    
    log_info "=== STAGE 2: PROCESSING ==="
    log_info "Executing processing pipeline script"
    
    # Build processing command
    local processing_cmd="$PROCESSING_PIPELINE_SCRIPT"
    processing_cmd+=" -csv \"${output_directory}/${PARSED_CSV_FILENAME}\""
    processing_cmd+=" -hdf5_param \"${output_directory}/${PARSED_HDF5_LOG_FILENAME}\""
    
    if $has_correlators; then
        processing_cmd+=" -hdf5_corr \"${output_directory}/${PARSED_HDF5_CORR_FILENAME}\""
    fi
    
    processing_cmd+=" -out_dir \"$output_directory\""
    processing_cmd+=" -log_dir \"$log_directory\""
    
    # Add filter config if provided
    if [[ -n "$filter_config" ]]; then
        processing_cmd+=" --filter_config \"$filter_config\""
    fi
    
    if [[ -n "$plots_directory" ]]; then
        processing_cmd+=" -plots_dir \"$plots_directory\" -viz"
    fi
    
    if $skip_checks; then
        processing_cmd+=" --skip_checks"
    fi
    
    if $skip_summaries; then
        processing_cmd+=" --skip_summaries"
    fi

    if [[ "$clean_plots" == "true" ]]; then
        processing_cmd+=" --clean_plots"
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
    #   0 - Success (core analysis completed, visualization may have failed)
    #   1 - Failure (core analysis failed)
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 3.1: CORRELATOR CALCULATIONS"
    echo "==================================================================="
    echo "Calculating PCAC mass and pion effective mass..."
    
    log_info "=== STAGE 3.1: CORRELATOR CALCULATIONS ==="
    log_info "Executing correlator calculations script"
    
    # Build command
    local cmd="$CORRELATOR_CALCULATIONS_SCRIPT"
    cmd+=" -hdf5_jack \"${output_directory}/${JACKKNIFE_HDF5_FILENAME}\""
    cmd+=" -o \"$output_directory\""
    cmd+=" -log_dir \"$log_directory\""
    
    local viz_enabled=false
    if [[ -n "$plots_directory" ]]; then
        cmd+=" -p \"$plots_directory\" --enable-viz"
        viz_enabled=true
    fi
    
    if $skip_checks; then
        cmd+=" --skip_checks"
    fi
    
    if $skip_summaries; then
        cmd+=" --skip_summaries"
    fi

    if [[ "$clean_plots" == "true" ]]; then
        cmd+=" --clean_plots"
    fi
    
    log_info "Command: $cmd"
    
    # Execute and capture result
    if eval "$cmd"; then
        echo "✓ Stage 3.1 completed successfully"
        log_info "Stage 3.1 completed successfully"
        return 0
    else
        # Check if core outputs exist despite failure
        local pcac_output="${output_directory}/${PCAC_MASS_CSV_FILENAME}"
        local pion_output="${output_directory}/${PION_MASS_CSV_FILENAME}"
        
        if [[ -f "$pcac_output" ]] || [[ -f "$pion_output" ]]; then
            # Core analysis succeeded, likely visualization failed
            echo "⚠ WARNING: Stage 3.1 core analysis succeeded but visualization may have failed"
            log_warning "Stage 3.1: Core analysis completed but script reported failure (likely visualization)"
            
            if $viz_enabled; then
                visualization_failures+=("Stage 3.1")
            fi
            
            echo "✓ Stage 3.1 core analysis completed"
            return 0
        else
            # Core analysis actually failed
            echo "ERROR: Stage 3.1 failed" >&2
            log_error "Stage 3.1 failed (core analysis)"
            return 1
        fi
    fi
}

function run_stage_3_2() {
    # Execute Stage 3.2: Plateau Extraction
    #
    # Returns:
    #   0 - Success (core analysis completed, visualization may have failed)
    #   1 - Failure (core analysis failed)
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 3.2: PLATEAU EXTRACTION"
    echo "==================================================================="
    echo "Extracting PCAC and pion mass plateaus..."
    
    log_info "=== STAGE 3.2: PLATEAU EXTRACTION ==="
    log_info "Executing plateau extraction script"
    
    # Build command
    local cmd="$PLATEAU_EXTRACTION_SCRIPT"
    cmd+=" -i_pcac \"${output_directory}/${PCAC_MASS_HDF5_FILENAME}\""
    cmd+=" -i_pion \"${output_directory}/${PION_MASS_HDF5_FILENAME}\""
    cmd+=" -o \"$output_directory\""
    cmd+=" -log_dir \"$log_directory\""
    
    local viz_enabled=false
    if [[ -n "$plots_directory" ]]; then
        cmd+=" -p \"$plots_directory\" --enable-viz"
        viz_enabled=true
    fi
    
    if $skip_checks; then
        cmd+=" --skip_checks"
    fi
    
    if $skip_summaries; then
        cmd+=" --skip_summaries"
    fi

    if [[ "$clean_plots" == "true" ]]; then
        cmd+=" --clean_plots"
    fi
    
    log_info "Command: $cmd"
    
    # Execute and capture result
    if eval "$cmd"; then
        echo "✓ Stage 3.2 completed successfully"
        log_info "Stage 3.2 completed successfully"
        return 0
    else
        # Check if core outputs exist despite failure
        local pcac_output="${output_directory}/${PCAC_PLATEAU_CSV_FILENAME}"
        local pion_output="${output_directory}/${PION_PLATEAU_CSV_FILENAME}"
        
        if [[ -f "$pcac_output" ]] || [[ -f "$pion_output" ]]; then
            # Core analysis succeeded, likely visualization failed
            echo "⚠ WARNING: Stage 3.2 core analysis succeeded but visualization may have failed"
            log_warning "Stage 3.2: Core analysis completed but script reported failure (likely visualization)"
            
            if $viz_enabled; then
                visualization_failures+=("Stage 3.2")
            fi
            
            echo "✓ Stage 3.2 core analysis completed"
            return 0
        else
            # Core analysis actually failed
            echo "ERROR: Stage 3.2 failed" >&2
            log_error "Stage 3.2 failed (core analysis)"
            return 1
        fi
    fi
}

function run_stage_3_3() {
    # Execute Stage 3.3: Critical Mass Extrapolation
    #
    # Returns:
    #   0 - Success (core analysis completed, visualization may have failed)
    #   1 - Hard failure (core analysis failed)
    #   2 - Graceful skip (insufficient data for extrapolation)
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 3.3: CRITICAL MASS EXTRAPOLATION"
    echo "==================================================================="
    echo "Extrapolating critical bare mass..."
    
    log_info "=== STAGE 3.3: CRITICAL MASS EXTRAPOLATION ==="
    log_info "Executing critical mass extraction script"
    
    # Build command
    local cmd="$CRITICAL_MASS_SCRIPT"
    cmd+=" -i_pcac \"${output_directory}/${PCAC_PLATEAU_CSV_FILENAME}\""
    cmd+=" -i_pion \"${output_directory}/${PION_PLATEAU_CSV_FILENAME}\""
    cmd+=" -o \"$output_directory\""
    cmd+=" -log_dir \"$log_directory\""
    
    if [[ -n "$plots_directory" ]]; then
        cmd+=" -p \"$plots_directory\" --enable-viz"
    fi
    
    if $skip_checks; then
        cmd+=" --skip_checks"
    fi
    
    if $skip_summaries; then
        cmd+=" --skip_summaries"
    fi

    if [[ "$clean_plots" == "true" ]]; then
        cmd+=" --clean_plots"
    fi
    
    log_info "Command: $cmd"
    
    # Execute and capture exit code
    eval "$cmd"
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        # Complete success
        echo "✓ Stage 3.3 completed successfully"
        log_info "Stage 3.3 completed successfully"
        return 0
        
    elif [[ $exit_code -eq 2 ]]; then
        # Graceful skip - insufficient data (from run_critical_mass.sh)
        echo ""
        echo "⚠ WARNING: Insufficient data for critical mass extrapolation"
        echo "  → Not enough bare mass variation across parameter groups"
        echo "  → Stage 3.3 skipped gracefully"
        log_warning "Stage 3.3: Insufficient data for critical mass extrapolation"
        log_info "Stage 3.3 skipped gracefully (data limitation, not error)"
        return 2
        
    else
        # Hard failure
        echo "ERROR: Stage 3.3 failed" >&2
        log_error "Stage 3.3 failed (hard error)"
        return 1
    fi
}

function run_stage_3_4() {
    # Execute Stage 3.4: Cost Extrapolation
    #
    # Returns:
    #   0 - Success (core analysis completed, visualization may have failed)
    #   1 - Hard failure (core analysis failed)
    #   2 - Graceful skip (insufficient data for extrapolation)
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 3.4: COST EXTRAPOLATION"
    echo "==================================================================="
    echo "Extrapolating computational costs..."
    
    log_info "=== STAGE 3.4: COST EXTRAPOLATION ==="
    log_info "Executing cost extrapolation script"
    
    # Build command
    local cmd="$COST_EXTRAPOLATION_SCRIPT"
    cmd+=" -i_pcac \"${output_directory}/${PCAC_PLATEAU_CSV_FILENAME}\""
    cmd+=" -i_pion \"${output_directory}/${PION_PLATEAU_CSV_FILENAME}\""
    cmd+=" -i_cost \"${output_directory}/${PROCESSED_CSV_FILENAME}\""
    cmd+=" -o \"$output_directory\""
    cmd+=" -log_dir \"$log_directory\""
    
    local viz_enabled=false
    if [[ -n "$plots_directory" ]]; then
        cmd+=" -p \"$plots_directory\" --enable-viz"
        viz_enabled=true
    fi
    
    if $skip_checks; then
        cmd+=" --skip_checks"
    fi
    
    if $skip_summaries; then
        cmd+=" --skip_summaries"
    fi

    if [[ "$clean_plots" == "true" ]]; then
        cmd+=" --clean_plots"
    fi
    
    log_info "Command: $cmd"
    
    # Execute and capture output and exit code
    local temp_log=$(mktemp)
    eval "$cmd" 2>&1 | tee "$temp_log"
    local exit_code=${PIPESTATUS[0]}
    
    if [[ $exit_code -eq 0 ]]; then
        # Complete success
        rm -f "$temp_log"
        echo "✓ Stage 3.4 completed successfully"
        log_info "Stage 3.4 completed successfully"
        return 0
    else
        # Check if graceful skip due to insufficient data
        if grep -q "Cost extrapolation produced no results" "$temp_log"; then
            # Graceful skip - insufficient data
            rm -f "$temp_log"
            echo ""
            echo "⚠ WARNING: Insufficient data for cost extrapolation"
            echo "  → Not enough bare mass variation for power law fit"
            echo "  → Stage 3.4 skipped gracefully"
            log_warning "Stage 3.4: Insufficient data for cost extrapolation"
            log_info "Stage 3.4 skipped gracefully (data limitation, not error)"
            return 2
        fi
        
        # Check if core outputs exist despite failure
        local pcac_output="${output_directory}/${COST_PCAC_CSV_FILENAME}"
        local pion_output="${output_directory}/${COST_PION_CSV_FILENAME}"
        
        if [[ -f "$pcac_output" ]] || [[ -f "$pion_output" ]]; then
            # Core analysis succeeded, likely visualization failed
            rm -f "$temp_log"
            echo "⚠ WARNING: Stage 3.4 core analysis succeeded but visualization may have failed"
            log_warning "Stage 3.4: Core analysis completed but script reported failure (likely visualization)"
            
            if $viz_enabled; then
                visualization_failures+=("Stage 3.4")
            fi
            
            echo "✓ Stage 3.4 core analysis completed"
            return 0
        else
            # Hard failure - core analysis failed
            rm -f "$temp_log"
            echo "ERROR: Stage 3.4 failed" >&2
            log_error "Stage 3.4 failed (hard error)"
            return 1
        fi
    fi
}

function run_analysis_stage() {
    # Execute Stage 3: Complete analysis pipeline (all substages)
    #
    # Returns:
    #   0 - Success (all substages completed or gracefully skipped)
    #   1 - Failure (hard error in any substage)
    
    echo ""
    echo "==================================================================="
    echo "   STAGE 3: ANALYSIS"
    echo "==================================================================="
    echo "Running complete analysis pipeline (up to 4 substages)..."
    
    log_info "=== STAGE 3: ANALYSIS STAGE ==="
    log_info "Executing all analysis substages"
    
    # Stage 3.1: Correlator Calculations
    if ! run_stage_3_1; then
        return 1
    fi
    
    # Stage 3.2: Plateau Extraction
    if ! run_stage_3_2; then
        return 1
    fi
    
    # Stage 3.3: Critical Mass Extrapolation
    run_stage_3_3
    local stage_3_3_result=$?
    
    if [[ $stage_3_3_result -eq 1 ]]; then
        # Hard failure in Stage 3.3
        return 1
    elif [[ $stage_3_3_result -eq 2 ]]; then
        # Graceful skip in Stage 3.3 - don't run Stage 3.4
        echo ""
        echo "○ Stage 3.3 completed with graceful skip"
        echo "○ Stage 3.4 skipped (requires Stage 3.3 results)"
        log_info "Stage 3 completed with Stages 3.1-3.2 successful, 3.3-3.4 skipped"
        return 0
    fi
    
    # Stage 3.3 succeeded, proceed to Stage 3.4
    run_stage_3_4
    local stage_3_4_result=$?
    
    if [[ $stage_3_4_result -eq 1 ]]; then
        # Hard failure in Stage 3.4
        return 1
    elif [[ $stage_3_4_result -eq 2 ]]; then
        # Graceful skip in Stage 3.4
        echo ""
        echo "○ Stage 3.4 completed with graceful skip"
        log_info "Stage 3 completed with Stages 3.1-3.3 successful, 3.4 skipped"
        return 0
    fi
    
    # All substages completed successfully
    echo "✓ Stage 3 completed successfully"
    log_info "Stage 3 completed successfully - all substages executed"
    return 0
}

function organize_auxiliary_files() {
    # Move log and summary files to auxiliary subdirectories
    #
    # Returns:
    #   0 - Success
    
    echo ""
    echo "=== ORGANIZING AUXILIARY FILES ==="
    
    local aux_dir="${output_directory}/${AUXILIARY_DIR_NAME}"
    local aux_logs="${aux_dir}/${AUXILIARY_LOGS_SUBDIR}"
    local aux_summaries="${aux_dir}/${AUXILIARY_SUMMARIES_SUBDIR}"
    
    # Ensure auxiliary directories exist
    mkdir -p "$aux_logs" "$aux_summaries"
    
    # Move log files (except main orchestrator log)
    local logs_moved=0
    for log_file in "${output_directory}"/*.log; do
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
filter_config=""
stages_to_run=""
skip_checks=false
skip_summaries=false
clean_plots=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input_directory)
            input_directory="$2"
            shift 2
            ;;
        -filter|--filter_config)
            filter_config="$2"
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
        --stages)
            stages_to_run=$(validate_stages_argument "$2") || exit 1
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
        --clean-plots|--clean_plots)
            clean_plots=true
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

# Validate filter config if provided
if [[ -n "$filter_config" ]]; then
    if [[ ! -f "$filter_config" ]]; then
        echo "ERROR: Filter config file not found: $filter_config" >&2
        log_error "Filter config file not found: $filter_config"
        exit 1
    fi
    filter_config="$(realpath "$filter_config")"
    echo "INFO: Filter configuration will be applied: $(get_display_path "$filter_config")"
    log_info "Filter configuration: $filter_config"
fi

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

# Only create output directory if running Stage 1 or if it doesn't exist
if should_run_stage 1 || [[ ! -d "$output_directory" ]]; then
    if [[ ! -d "$output_directory" ]]; then
        mkdir -p "$output_directory" || {
            echo "ERROR: Failed to create output directory: $output_directory" >&2
            exit 1
        }
        echo "INFO: Created output directory: $(get_display_path "$output_directory")"
    fi
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

if [[ -n "$stages_to_run" ]]; then
    log_info "Requested stages: $stages_to_run (selective execution)"
else
    log_info "Requested stages: 1,2,3 (full pipeline)"
fi

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

if [[ -n "$stages_to_run" ]]; then
    echo "Stages: $stages_to_run (selective execution)"
else
    echo "Stages: 1,2,3 (full pipeline)"
fi

if [[ -n "$plots_directory" ]]; then
    echo "Plots:  $(get_display_path "$plots_directory") (visualization enabled)"
fi
echo "==================================================================="

# Detect if data set has correlator files (determines pipeline path)
has_correlators=false
if detect_correlator_files "$input_directory"; then
    has_correlators=true
    echo "✓ Correlator files (.dat) detected"
    if [[ -z "$stages_to_run" ]]; then
        echo "  → Full pipeline will execute: Stages 1, 2A, 2B, 2C, 3.1, 3.2, 3.3, 3.4"
    fi
    log_info "Correlator files detected - full pipeline execution"
else
    echo "ℹ Only log files (.txt) detected"
    if [[ -z "$stages_to_run" ]]; then
        echo "  → Limited pipeline: Stages 1, 2A only"
    fi
    log_info "No correlator files detected - parameters-only pipeline"
fi

# Validate stage inputs for requested stages
echo ""
echo "=== VALIDATING STAGE INPUTS ==="
if should_run_stage 1; then
    validate_stage_inputs 1 || exit 1
fi

if should_run_stage 2; then
    # Only validate Stage 2 inputs if NOT running Stage 1
    # (Stage 1 will create the inputs)
    if ! should_run_stage 1; then
        validate_stage_inputs 2 || exit 1
    fi
fi

if should_run_stage 3; then
    # Only validate Stage 3 inputs if NOT running Stage 2
    # (Stage 2 will create the inputs)
    if ! should_run_stage 2; then
        validate_stage_inputs 3 || exit 1
    fi
fi

echo "✓ All required inputs validated"

# Validate prerequisites
echo ""
echo "=== VALIDATING PREREQUISITES ==="
if ! validate_prerequisites; then
    echo "ERROR: Prerequisites validation failed" >&2
    log_error "Prerequisites validation failed"
    exit 1
fi

# Execute pipeline stages
pipeline_success=true
visualization_failures=()  # Track which visualization stages failed

# Stage 1: Parsing
if should_run_stage 1; then
    if ! run_parsing_stage; then
        pipeline_success=false
        echo "ERROR: Pipeline failed at Stage 1 (Parsing)" >&2
        log_error "Pipeline failed at Stage 1"
        exit 1
    fi
fi

# Stage 2: Processing
if should_run_stage 2; then
    if ! run_processing_stage; then
        pipeline_success=false
        echo "ERROR: Pipeline failed at Stage 2 (Processing)" >&2
        log_error "Pipeline failed at Stage 2"
        exit 1
    fi
fi

# Stage 3: Analysis (only if correlators present and Stage 3 requested)
if should_run_stage 3; then
    if $has_correlators; then
        if ! run_analysis_stage; then
            pipeline_success=false
            echo "ERROR: Pipeline failed at Stage 3 (Analysis)" >&2
            log_error "Pipeline failed at Stage 3"
            exit 1
        fi
    else
        echo ""
        echo "○ Stage 3 skipped: No correlator data available"
        log_info "Stage 3 skipped - no correlator data"
    fi
fi

# Organize auxiliary files
if $pipeline_success; then
    organize_auxiliary_files
fi

# Final summary
echo ""
echo "==================================================================="
if [[ ${#visualization_failures[@]} -eq 0 ]]; then
    echo "   PIPELINE COMPLETED SUCCESSFULLY"
else
    echo "   PIPELINE COMPLETED WITH WARNINGS"
fi
echo "==================================================================="

if $has_correlators; then
    echo "Complete analysis pipeline executed:"
    
    if should_run_stage 1; then
        echo "  Stage 1:   Parsing ✓"
    fi
    
    if should_run_stage 2; then
        echo "  Stage 2A:  Processing parameters ✓"
        echo "  Stage 2B:  Jackknife analysis ✓"
        if [[ -n "$plots_directory" ]]; then
            if [[ " ${visualization_failures[@]} " =~ " Stage 2C " ]]; then
                echo "  Stage 2C:  Jackknife visualization ⚠ (failed)"
            else
                echo "  Stage 2C:  Jackknife visualization ✓"
            fi
        fi
    fi
    
    if should_run_stage 3; then
        if [[ " ${visualization_failures[@]} " =~ " Stage 3.1 " ]]; then
            echo "  Stage 3.1: Correlator calculations ✓ (visualization ⚠)"
        else
            echo "  Stage 3.1: Correlator calculations ✓"
        fi
        
        if [[ " ${visualization_failures[@]} " =~ " Stage 3.2 " ]]; then
            echo "  Stage 3.2: Plateau extraction ✓ (visualization ⚠)"
        else
            echo "  Stage 3.2: Plateau extraction ✓"
        fi
        
        # Check if Stage 3.3 ran
        if [[ -f "${output_directory}/${CRITICAL_PCAC_CSV_FILENAME}" || -f "${output_directory}/${CRITICAL_PION_CSV_FILENAME}" ]]; then
            if [[ " ${visualization_failures[@]} " =~ " Stage 3.3 " ]]; then
                echo "  Stage 3.3: Critical mass extrapolation ✓ (visualization ⚠)"
            else
                echo "  Stage 3.3: Critical mass extrapolation ✓"
            fi
            
            # Check if Stage 3.4 ran
            if [[ -f "${output_directory}/${COST_PCAC_CSV_FILENAME}" || -f "${output_directory}/${COST_PION_CSV_FILENAME}" ]]; then
                if [[ " ${visualization_failures[@]} " =~ " Stage 3.4 " ]]; then
                    echo "  Stage 3.4: Cost extrapolation ✓ (visualization ⚠)"
                else
                    echo "  Stage 3.4: Cost extrapolation ✓"
                fi
            else
                echo "  Stage 3.4: Cost extrapolation ○ (skipped - insufficient data)"
            fi
        else
            echo "  Stage 3.3: Critical mass extrapolation ○ (skipped - insufficient data)"
            echo "  Stage 3.4: Cost extrapolation ○ (skipped - no Stage 3.3 results)"
        fi
    fi
else
    echo "Parameters-only pipeline executed:"
    
    if should_run_stage 1; then
        echo "  Stage 1:  Parsing ✓"
    fi
    
    if should_run_stage 2; then
        echo "  Stage 2A: Processing parameters ✓"
    fi
fi

echo ""
echo "Output structure:"
echo "  Data files:    $(get_display_path "$output_directory")"
echo "  Auxiliary:"
echo "    - Logs:      $(get_display_path "${output_directory}/${AUXILIARY_DIR_NAME}/${AUXILIARY_LOGS_SUBDIR}")"
echo "    - Summaries: $(get_display_path "${output_directory}/${AUXILIARY_DIR_NAME}/${AUXILIARY_SUMMARIES_SUBDIR}")"
if [[ -n "$plots_directory" ]]; then
    echo "  Plots:         $(get_display_path "$plots_directory")"
fi
echo ""
echo "Main log file: $(get_display_path "$SCRIPT_LOG_FILE_PATH")"
echo "==================================================================="

# Report visualization failures if any
if [[ ${#visualization_failures[@]} -gt 0 ]]; then
    echo ""
    echo "⚠ WARNING: Visualization failed for the following stages:"
    for stage in "${visualization_failures[@]}"; do
        echo "  - $stage"
    done
    echo ""
    echo "Core analysis completed successfully, but some plots may be missing."
    echo "Check individual stage logs for details."
    log_warning "Pipeline completed with ${#visualization_failures[@]} visualization failure(s): ${visualization_failures[*]}"
fi

log_info "=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ==="

if [[ -n "$stages_to_run" ]]; then
    log_info "Selective execution completed: Stages $stages_to_run"
else
    if $has_correlators; then
        if [[ -f "${output_directory}/${CRITICAL_PCAC_CSV_FILENAME}" || -f "${output_directory}/${CRITICAL_PION_CSV_FILENAME}" ]]; then
            log_info "Full pipeline executed: All stages (1, 2A, 2B, 2C, 3.1-3.4) completed"
        else
            log_info "Partial pipeline executed: Stages 1, 2A, 2B, 2C, 3.1, 3.2 completed (3.3/3.4 skipped)"
        fi
    else
        log_info "Parameters-only pipeline executed: Stages 1, 2A completed"
    fi
fi

if [[ ${#visualization_failures[@]} -gt 0 ]]; then
    log_info "Visualization failures: ${#visualization_failures[@]} stage(s)"
fi

log_info "Auxiliary files organized successfully"

exit 0  # Always exit 0 if we got here (core pipeline succeeded)
