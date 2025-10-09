#!/bin/bash

################################################################################
# run_processing_pipeline.sh - Execute Stage 2 (Processing) of QPB data analysis
#
# DESCRIPTION:
# This script orchestrates the processing stage of the QPB data analysis pipeline.
# It executes up to three sub-stages with intelligent conditional logic:
#
# Stage 2A (ALWAYS RUNS):
#   - process_raw_parameters.py
#   - Processes and validates raw parameters from log files
#   - Input: CSV + HDF5 log parameters
#   - Output: processed_parameter_values.csv
#
# Stage 2B (CONDITIONAL - only if correlator data exists):
#   - apply_jackknife_analysis.py
#   - Applies jackknife resampling for error estimation
#   - Input: HDF5 correlators
#   - Output: correlators_jackknife_analysis.h5
#
# Stage 2C (OPTIONAL - only if correlators exist and visualization enabled):
#   - visualize_jackknife_samples.py
#   - Generates diagnostic plots
#   - Input: HDF5 jackknife results
#   - Output: jackknife_plots/
#
# USAGE:
#   ./run_processing_pipeline.sh -csv <csv_file> -hdf5_param <hdf5_file> [options]
#
# REQUIRED ARGUMENTS:
#   -csv, --input_csv_file           Path to single-valued parameters CSV
#   -hdf5_param, --input_hdf5_param  Path to multivalued parameters HDF5
#
# OPTIONAL ARGUMENTS:
#   -hdf5_corr, --input_hdf5_corr  Path to correlators HDF5 (enables stages 2B/2C)
#   -out_dir, --output_directory   Output directory (default: CSV file directory)
#   -plots_dir, --plots_directory  Plots directory (default: output_directory)
#   -log_dir, --log_directory      Log directory (default: output directory)
#   -viz, --enable_visualization   Enable Stage 2C visualization
#   --skip_checks                  Skip intermediate file validation
#   -h, --help                     Display this help message
#
# EXAMPLES:
#   # Parameters only (Stage 2A only)
#   ./run_processing_pipeline.sh \
#       -csv single_valued_parameters.csv \
#       -hdf5_param multivalued_parameters.h5
#
#   # Full processing with correlators (Stages 2A, 2B, 2C)
#   ./run_processing_pipeline.sh \
#       -csv single_valued_parameters.csv \
#       -hdf5_param multivalued_parameters.h5 \
#       -hdf5_corr correlators_raw_data.h5 \
#       -viz
#
#   # Custom output location with separate plots directory
#   ./run_processing_pipeline.sh \
#       -csv params.csv \
#       -hdf5_param arrays.h5 \
#       -hdf5_corr correlators.h5 \
#       -out_dir data/processed/ \
#       -plots_dir output/plots/ \
#       -viz
#
# DEPENDENCIES:
# - Python scripts: process_raw_parameters.py, apply_jackknife_analysis.py,
#                  visualize_jackknife_samples.py
# - Library scripts in bash_scripts/library/
#
# OUTPUT FILES:
# - processed_parameter_values.csv      : Processed parameters (Stage 2A)
# - correlators_jackknife_analysis.h5   : Jackknife results (Stage 2B, if correlators)
# - Jackknife_samples_visualization/    : Diagnostic plots (Stage 2C, if enabled)
#                                         (in plots_directory if specified)
# - run_processing_pipeline.log         : Execution log
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
PROCESSING_SCRIPTS_DIR="${PYTHON_SCRIPTS_DIRECTORY}/processing"
PROCESS_RAW_SCRIPT="${PROCESSING_SCRIPTS_DIR}/process_raw_parameters.py"
JACKKNIFE_SCRIPT="${PROCESSING_SCRIPTS_DIR}/apply_jackknife_analysis.py"
VISUALIZATION_SCRIPT="${PROCESSING_SCRIPTS_DIR}/visualize_jackknife_samples.py"

# Output filenames (using constants from constants.sh)
PROCESSED_CSV_FILENAME="$PROCESSING_CSV_PROCESSED"
JACKKNIFE_HDF5_FILENAME="$PROCESSING_HDF5_JACKKNIFE"

# Script log filename
SCRIPT_LOG_FILENAME="${SCRIPT_NAME%.sh}.log"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function usage() {
    cat << EOF
Usage: $SCRIPT_NAME -csv <csv_file> -hdf5_param <hdf5_param_file> [options]

REQUIRED ARGUMENTS:
  -csv, --input_csv_file           Single-valued parameters CSV file
  -hdf5_param, --input_hdf5_param  Multivalued parameters HDF5 file

OPTIONAL ARGUMENTS:
  -hdf5_corr, --input_hdf5_corr  Correlators HDF5 file (enables stages 2B/2C)
  -out_dir, --output_directory   Output directory for data files (default: CSV directory)
  -plots_dir, --plots_directory  Plots directory (default: output_directory)
  -log_dir, --log_directory      Log directory (default: output directory)
  -viz, --enable_visualization   Enable visualization (Stage 2C)
  --skip_checks                  Skip validation checks
  -h, --help                     Display this help message

EXAMPLES:
  # Parameters only
  $SCRIPT_NAME -csv params.csv -hdf5_param arrays.h5

  # With correlators and visualization
  $SCRIPT_NAME -csv params.csv -hdf5_param arrays.h5 -hdf5_corr corr.h5 -viz

  # Separate data and plots directories
  $SCRIPT_NAME -csv params.csv -hdf5_param arrays.h5 -hdf5_corr corr.h5 \\
               -out_dir data/processed/ -plots_dir output/plots/ -viz

EOF
    # Clear exit handlers before exiting
    trap - EXIT
    exit 0
}

function validate_prerequisites() {
    # Validate that required Python scripts exist
    #
    # Returns:
    #   0 - All prerequisites valid
    #   1 - Validation failed
    
    # Always validate Stage 2A script
    validate_python_script "$PROCESS_RAW_SCRIPT" -s || return 1
    
    # Validate Stage 2B/2C scripts if correlators will be processed
    if [[ "$has_correlators" == "true" ]]; then
        validate_python_script "$JACKKNIFE_SCRIPT" -s || return 1
        
        if [[ "$enable_visualization" == "true" ]]; then
            validate_python_script "$VISUALIZATION_SCRIPT" -s || return 1
        fi
    fi
    
    echo "All prerequisites validated successfully"
    log_info "Prerequisites validation completed successfully"
    return 0
}

# =============================================================================
# STAGE FUNCTIONS
# =============================================================================

function run_stage_2a() {
    # Execute Stage 2A: Process raw parameters (ALWAYS RUNS)
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    
    echo ""
    echo "=== STAGE 2A: PROCESSING RAW PARAMETERS ==="
    echo "Processing and validating raw parameters..."
    log_info "STAGE 2A: Starting raw parameter processing"
    
    execute_python_script "$PROCESS_RAW_SCRIPT" "process_raw_parameters" \
        --input_single_valued_csv_file_path "$input_csv_file" \
        --input_multivalued_hdf5_file_path "$input_hdf5_param" \
        --output_directory "$output_directory" \
        --output_csv_filename "$PROCESSED_CSV_FILENAME" \
        --enable_logging \
        --log_directory "$log_directory" \
        || {
            log_error "Stage 2A: Raw parameter processing failed"
            return 1
        }
    
    # Validate output
    local processed_csv_path="${output_directory}/${PROCESSED_CSV_FILENAME}"
    if ! $skip_checks; then
        validate_output_file "$processed_csv_path" "Processed CSV" || return 1
    fi
    
    echo "${PROGRESS_SUCCESS} Stage 2A completed: Raw parameters processed"
    echo "  - Output: $(basename "$processed_csv_path")"
    log_info "Stage 2A completed successfully"
    
    return 0
}

function run_stage_2b() {
    # Execute Stage 2B: Jackknife analysis (only if correlators exist)
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    
    echo ""
    echo "=== STAGE 2B: JACKKNIFE ANALYSIS ==="
    echo "Applying jackknife resampling to correlator data..."
    log_info "STAGE 2B: Starting jackknife analysis"
    
    execute_python_script "$JACKKNIFE_SCRIPT" "apply_jackknife_analysis" \
        --input_hdf5_file "$input_hdf5_corr" \
        --output_directory "$output_directory" \
        --output_hdf5_file "$JACKKNIFE_HDF5_FILENAME" \
        --enable_logging \
        --log_directory "$log_directory" \
        || {
            log_error "Stage 2B: Jackknife analysis failed"
            return 1
        }
    
    # Validate output
    local jackknife_hdf5_path="${output_directory}/${JACKKNIFE_HDF5_FILENAME}"
    if ! $skip_checks; then
        validate_output_file "$jackknife_hdf5_path" "Jackknife HDF5" || return 1
    fi
    
    echo "${PROGRESS_SUCCESS} Stage 2B completed: Jackknife analysis applied"
    echo "  - Output: $(basename "$jackknife_hdf5_path")"
    log_info "Stage 2B completed successfully"
    
    return 0
}

function run_stage_2c() {
    # Execute Stage 2C: Visualization (optional, only if correlators exist)
    #
    # Returns:
    #   0 - Success
    #   1 - Failure (but doesn't fail pipeline)
    
    echo ""
    echo "=== STAGE 2C: VISUALIZATION (OPTIONAL) ==="
    echo "Generating jackknife diagnostic plots..."
    log_info "STAGE 2C: Starting visualization"
    
    local jackknife_hdf5_path="${output_directory}/${JACKKNIFE_HDF5_FILENAME}"
    
    execute_python_script "$VISUALIZATION_SCRIPT" "visualize_jackknife_samples" \
        --input_hdf5_file "$jackknife_hdf5_path" \
        --output_directory "$plots_directory" \
        --enable_logging \
        --log_directory "$log_directory" \
        || {
            echo "WARNING: Stage 2C visualization failed (optional stage)" >&2
            log_warning "Stage 2C visualization failed but continuing"
            return 0  # Don't fail pipeline for optional visualization
        }
    
    echo "${PROGRESS_SUCCESS} Stage 2C completed: Visualizations generated"
    echo "  - Plots directory: $(get_display_path "$plots_directory")"
    log_info "Stage 2C completed successfully"
    log_info "  Plots directory: $plots_directory"
    
    return 0
}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

function main() {
    # Parse command line arguments
    local input_csv_file=""
    local input_hdf5_param=""
    local input_hdf5_corr=""
    local output_dir=""
    local plots_dir=""
    local log_dir=""
    local enable_visualization=false
    local skip_checks=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -csv|--input_csv_file)
                input_csv_file="$2"
                shift 2
                ;;
            -hdf5_param|--input_hdf5_param)
                input_hdf5_param="$2"
                shift 2
                ;;
            -hdf5_corr|--input_hdf5_corr)
                input_hdf5_corr="$2"
                shift 2
                ;;
            -out_dir|--output_directory)
                output_dir="$2"
                shift 2
                ;;
            -plots_dir|--plots_directory)
                plots_dir="$2"
                shift 2
                ;;
            -log_dir|--log_directory)
                log_dir="$2"
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
                usage
                ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$input_csv_file" ]]; then
        echo "ERROR: Input CSV file not specified. Use -csv <file>" >&2
        usage
    fi
    
    if [[ -z "$input_hdf5_param" ]]; then
        echo "ERROR: Input HDF5 parameters file not specified. Use -hdf5_param <file>" >&2
        usage
    fi
    
    # Validate required input files exist
    if [[ ! -f "$input_csv_file" ]]; then
        echo "ERROR: Input CSV file not found: $input_csv_file" >&2
        exit 1
    fi
    
    if [[ ! -f "$input_hdf5_param" ]]; then
        echo "ERROR: Input HDF5 parameters file not found: $input_hdf5_param" >&2
        exit 1
    fi
    
    # Check if correlator file provided and exists
    local has_correlators=false
    if [[ -n "$input_hdf5_corr" ]]; then
        if [[ -f "$input_hdf5_corr" ]]; then
            has_correlators=true
            input_hdf5_corr="$(realpath "$input_hdf5_corr")"
            echo "INFO: Correlator data provided - Stages 2B and 2C will be available"
        else
            echo "WARNING: Specified correlator file not found: $input_hdf5_corr" >&2
            echo "         Continuing with Stage 2A only"
        fi
    fi
    
    # Convert input files to absolute paths
    input_csv_file="$(realpath "$input_csv_file")"
    input_hdf5_param="$(realpath "$input_hdf5_param")"
    
    # Set default output directory to input file directory if not specified
    if [[ -z "$output_dir" ]]; then
        output_directory="$(dirname "$input_csv_file")"
        echo "INFO: Using input file directory as output directory"
    else
        output_directory="$output_dir"
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
    
    # Set default plots directory to output directory if not specified
    if [[ -z "$plots_dir" ]]; then
        plots_directory="$output_directory"
        echo "INFO: Using output directory for plots"
    else
        plots_directory="$plots_dir"
        # Ensure plots directory exists
        if [[ ! -d "$plots_directory" ]]; then
            mkdir -p "$plots_directory" || {
                echo "ERROR: Failed to create plots directory: $plots_directory" >&2
                exit 1
            }
            echo "INFO: Created plots directory: $plots_directory"
        fi
        plots_directory="$(realpath "$plots_directory")"
        echo "INFO: Using separate plots directory: $(get_display_path "$plots_directory")"
    fi
    
    # Set default log directory to output directory if not specified
    if [[ -z "$log_dir" ]]; then
        log_directory="$output_directory"
    else
        log_directory="$log_dir"
        if [[ ! -d "$log_directory" ]]; then
            mkdir -p "$log_directory" || {
                echo "ERROR: Failed to create log directory: $log_directory" >&2
                exit 1
            }
        fi
        log_directory="$(realpath "$log_directory")"
    fi
    
    # Initialize logging
    local log_file="${log_directory}/${SCRIPT_LOG_FILENAME}"
    export SCRIPT_LOG_FILE_PATH="$log_file"
    echo -e "\t\t$(echo "$SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') EXECUTION INITIATED\n" > "$SCRIPT_LOG_FILE_PATH"
    export SCRIPT_TERMINATION_MESSAGE="\n\t\t$(echo "$SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') EXECUTION TERMINATED"
    
    log_info "Script: $SCRIPT_NAME"
    log_info "Input CSV: $input_csv_file"
    log_info "Input HDF5 (params): $input_hdf5_param"
    if $has_correlators; then
        log_info "Input HDF5 (corr): $input_hdf5_corr"
    fi
    log_info "Output directory: $output_directory"
    log_info "Plots directory: $plots_directory"
    log_info "Log directory: $log_directory"
    log_info "Has correlators: $has_correlators"
    log_info "Visualization enabled: $enable_visualization"
    
    # Validate prerequisites
    echo ""
    echo "=== VALIDATING PREREQUISITES ==="
    validate_prerequisites || return 1
    
    # =========================================================================
    # STAGE 2A: PROCESS RAW PARAMETERS (ALWAYS RUNS)
    # =========================================================================
    
    run_stage_2a || {
        echo ""
        echo "ERROR: Processing pipeline failed at Stage 2A" >&2
        log_error "Pipeline terminated: Stage 2A failed"
        return 1
    }
    
    # =========================================================================
    # STAGE 2B: JACKKNIFE ANALYSIS (CONDITIONAL)
    # =========================================================================
    
    if $has_correlators; then
        run_stage_2b || {
            echo ""
            echo "ERROR: Processing pipeline failed at Stage 2B" >&2
            log_error "Pipeline terminated: Stage 2B failed"
            return 1
        }
        
        # =====================================================================
        # STAGE 2C: VISUALIZATION (OPTIONAL)
        # =====================================================================
        
        if $enable_visualization; then
            run_stage_2c
            # Note: This stage is optional and won't fail the pipeline
        else
            echo ""
            echo "${PROGRESS_SKIPPED} Stage 2C skipped: Visualization not enabled"
            log_info "Stage 2C skipped (visualization not enabled)"
        fi
    else
        echo ""
        echo "${PROGRESS_SKIPPED} Stages 2B & 2C skipped: No correlator data provided"
        log_info "Stages 2B and 2C skipped (no correlator data)"
    fi
    
    # =========================================================================
    # COMPLETION
    # =========================================================================
    
    echo ""
    echo "=== PROCESSING PIPELINE COMPLETED ==="
    echo "All applicable stages completed successfully!"
    echo ""
    echo "Output files:"
    echo "  - Processed CSV: $(get_display_path "${output_directory}/${PROCESSED_CSV_FILENAME}")"
    if $has_correlators; then
        echo "  - Jackknife HDF5: $(get_display_path "${output_directory}/${JACKKNIFE_HDF5_FILENAME}")"
        if $enable_visualization; then
            echo ""
            echo "Visualization:"
            echo "  - Plots directory: $(get_display_path "$plots_directory")"
        fi
    fi
    echo ""
    echo "Log file: $(get_display_path "$log_file")"
    
    # Final logging
    local stages_executed="2A"
    if $has_correlators; then
        stages_executed+=", 2B"
        if $enable_visualization; then
            stages_executed+=", 2C"
        fi
    fi
    
    log_info "PROCESSING PIPELINE COMPLETED SUCCESSFULLY"
    log_info "  Stages executed: $stages_executed"
    log_info "  Output directory: $output_directory"
    
    echo -e "$SCRIPT_TERMINATION_MESSAGE" >> "$SCRIPT_LOG_FILE_PATH"
    echo ""
    echo "${PROGRESS_SUCCESS} Processing pipeline execution completed successfully!"
    
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
