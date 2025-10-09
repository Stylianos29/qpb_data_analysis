#!/bin/bash

################################################################################
# run_processing_pipeline.sh - Execute Stage 2 (Processing) of QPB data analysis
#
# DESCRIPTION:
# This script orchestrates the processing stage of the QPB data analysis pipeline.
# It transforms parsed data into analysis-ready format through three sequential steps:
#
# 1. process_raw_parameters.py - Transform and validate extracted parameters
#    Output: processed_parameter_values.csv
#
# 2. apply_jackknife_analysis.py - Apply jackknife resampling to correlators
#    Output: correlators_jackknife_analysis.h5
#
# 3. visualize_jackknife_samples.py - Generate diagnostic visualizations (optional)
#    Output: jackknife_plots/ directory
#
# USAGE:
#   ./run_processing_pipeline.sh -csv <input_csv> -hdf5 <input_hdf5> [options]
#
# REQUIRED ARGUMENTS:
#   -csv,  --input_csv_file     Path to input CSV file (single-valued parameters)
#   -hdf5, --input_hdf5_file    Path to input HDF5 file (multivalued parameters)
#
# OPTIONAL ARGUMENTS:
#   -o,  --output_directory     Output directory for processed files
#                               (default: directory of input HDF5 file)
#   -log_dir, --log_directory   Directory for log files
#                               (default: output directory)
#   -viz, --enable_visualization Enable visualization step (Stage 3)
#   --skip_checks               Skip intermediate file validation
#   --skip_summaries            Skip generation of summary files
#   -h,  --help                 Display this help message
#
# EXAMPLES:
#   # Basic usage
#   ./run_processing_pipeline.sh \
#       -csv single_valued_parameters.csv \
#       -hdf5 multivalued_parameters.h5
#
#   # With visualization enabled
#   ./run_processing_pipeline.sh \
#       -csv params.csv \
#       -hdf5 arrays.h5 \
#       -viz
#
#   # Custom output location
#   ./run_processing_pipeline.sh \
#       -csv params.csv \
#       -hdf5 arrays.h5 \
#       -o results/ \
#       -log_dir logs/
#
#   # Skip validation checks for speed
#   ./run_processing_pipeline.sh \
#       -csv params.csv \
#       -hdf5 arrays.h5 \
#       --skip_checks
#
# DEPENDENCIES:
# - Python scripts: process_raw_parameters.py, apply_jackknife_analysis.py,
#                   visualize_jackknife_samples.py
# - Library scripts in bash_scripts/library/
# - Python environment with qpb_data_analysis package
#
# OUTPUT FILES:
# - processed_parameter_values.csv     : Transformed parameters
# - correlators_jackknife_analysis.h5  : Jackknife resampled correlators
# - jackknife_plots/                   : Diagnostic plots (if -viz enabled)
# - run_processing_pipeline.log        : Script execution log
#
# SUMMARY FILES (unless --skip_summaries):
# - processed_parameter_values_uniqueness_report.txt : CSV column summary
# - correlators_jackknife_analysis_tree.txt          : HDF5 structure tree
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

# Export termination message for logging utilities
export SCRIPT_TERMINATION_MESSAGE="\n\t\t$(echo "$SCRIPT_NAME" | tr '[:lower:]' '[:upper:]') EXECUTION TERMINATED"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function usage() {
    cat << EOF
Usage: $SCRIPT_NAME -csv <input_csv> -hdf5 <input_hdf5> [options]

REQUIRED ARGUMENTS:
  -csv,  --input_csv_file      Path to CSV file with single-valued parameters
  -hdf5, --input_hdf5_file     Path to HDF5 file with multivalued parameters

OPTIONAL ARGUMENTS:
  -o,  --output_directory      Output directory (default: input HDF5 dir)
  -log_dir, --log_directory    Log files directory (default: output dir)
  -viz, --enable_visualization Enable visualization step
  --skip_checks                Skip intermediate file validation
  --skip_summaries             Skip generation of summary files
  -h,  --help                  Display this help message

EXAMPLES:
  $SCRIPT_NAME -csv params.csv -hdf5 arrays.h5
  $SCRIPT_NAME -csv params.csv -hdf5 arrays.h5 -viz
  $SCRIPT_NAME -csv params.csv -hdf5 arrays.h5 -o results/ -log_dir logs/

EOF
    exit 0
}

function validate_prerequisites() {
    local input_csv="$1"
    local input_hdf5="$2"
    local enable_viz="$3"
    
    log_info "Validating prerequisites..."
    
    # Validate input files exist
    if [[ ! -f "$input_csv" ]]; then
        log_error "Input CSV file not found: $input_csv"
        echo "ERROR: Input CSV file not found: $input_csv" >&2
        return 1
    fi
    log_info "Found input CSV file: $(basename "$input_csv")"
    
    if [[ ! -f "$input_hdf5" ]]; then
        log_error "Input HDF5 file not found: $input_hdf5"
        echo "ERROR: Input HDF5 file not found: $input_hdf5" >&2
        return 1
    fi
    log_info "Found input HDF5 file: $(basename "$input_hdf5")"
    
    # Validate Python scripts exist using validation helper
    validate_python_script "$PROCESS_RAW_SCRIPT" -s || {
        echo "ERROR: process_raw_parameters.py not found" >&2
        return 1
    }
    
    validate_python_script "$JACKKNIFE_SCRIPT" -s || {
        echo "ERROR: apply_jackknife_analysis.py not found" >&2
        return 1
    }
    
    # Validate visualization script if needed
    if [[ "$enable_viz" == "true" ]]; then
        validate_python_script "$VISUALIZATION_SCRIPT" -s || {
            echo "ERROR: visualize_jackknife_samples.py not found" >&2
            return 1
        }
    fi
    
    log_info "All prerequisites validated successfully"
    echo "All prerequisites validated successfully"
    return 0
}

function check_intermediate_output() {
    local file_path="$1"
    local stage_name="$2"
    
    if [[ ! -f "$file_path" ]]; then
        local error_msg="$stage_name did not produce expected output: $file_path"
        log_error "$error_msg"
        echo "ERROR: $error_msg" >&2
        return 1
    fi
    
    # Check if file is not empty
    if [[ ! -s "$file_path" ]]; then
        local error_msg="$stage_name produced empty output: $file_path"
        log_error "$error_msg"
        echo "ERROR: $error_msg" >&2
        return 1
    fi
    
    log_info "Validated output: $(basename "$file_path")"
    return 0
}

function cleanup() {
    # Cleanup function called on exit via trap
    # Add any cleanup tasks here (e.g., removing temp files)
    log_info "Cleanup completed"
}

function generate_hdf5_tree() {
    local hdf5_file_path="$1"
    local stage_name="$2"
    
    if ! command -v h5glance &> /dev/null; then
        log_warning "h5glance not found - skipping HDF5 tree generation"
        echo "  $PROGRESS_WARNING h5glance not available - skipping tree generation"
        return 0
    fi
    
    local tree_file_path="${hdf5_file_path%.h5}_tree.txt"
    
    h5glance "$hdf5_file_path" > "$tree_file_path" 2>&1 || {
        log_warning "Failed to generate HDF5 tree for $(basename "$hdf5_file_path")"
        echo "  $PROGRESS_WARNING Failed to generate HDF5 tree"
        return 1
    }
    
    log_info "Generated HDF5 tree: $(basename "$tree_file_path")"
    echo "  → HDF5 tree: $(basename "$tree_file_path")"
    return 0
}

function generate_csv_summary() {
    local csv_file_path="$1"
    local output_directory="$2"
    
    local inspect_script="${PYTHON_SCRIPTS_DIRECTORY}/utils/inspect_csv_file.py"
    
    if [[ ! -f "$inspect_script" ]]; then
        log_warning "inspect_csv_file.py not found - skipping CSV summary"
        echo "  $PROGRESS_WARNING CSV inspection script not available"
        return 0
    fi
    
    python "$inspect_script" \
        --csv_file_path "$csv_file_path" \
        --output_directory "$output_directory" \
        --uniqueness_report \
        || {
            log_warning "Failed to generate CSV summary for $(basename "$csv_file_path")"
            echo "  $PROGRESS_WARNING Failed to generate CSV summary"
            return 1
        }
    
    log_info "Generated CSV summary for $(basename "$csv_file_path")"
    echo "  → CSV summary generated"
    return 0
}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

function main() {
    # Parse command line arguments
    local input_csv_file=""
    local input_hdf5_file=""
    local output_directory=""
    local log_directory=""
    local enable_visualization=false
    local skip_checks=false
    local skip_summaries=false
    
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
    if [[ -z "$input_csv_file" ]]; then
        echo "ERROR: Input CSV file not specified. Use -csv <file_path>" >&2
        usage
    fi
    
    if [[ -z "$input_hdf5_file" ]]; then
        echo "ERROR: Input HDF5 file not specified. Use -hdf5 <file_path>" >&2
        usage
    fi
    
    # Validate input files exist
    if [[ ! -f "$input_csv_file" ]]; then
        echo "ERROR: Input CSV file not found: $input_csv_file" >&2
        return 1
    fi
    
    if [[ ! -f "$input_hdf5_file" ]]; then
        echo "ERROR: Input HDF5 file not found: $input_hdf5_file" >&2
        return 1
    fi
    
    # Convert to absolute paths
    input_csv_file="$(realpath "$input_csv_file")"
    input_hdf5_file="$(realpath "$input_hdf5_file")"
    
    # Set default output directory to HDF5 file directory if not specified
    if [[ -z "$output_directory" ]]; then
        output_directory="$(dirname "$input_hdf5_file")"
        echo "INFO: Using input file directory as output directory"
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
    
    log_info "Script: $SCRIPT_NAME"
    log_info "Input CSV: $input_csv_file"
    log_info "Input HDF5: $input_hdf5_file"
    log_info "Output directory: $output_directory"
    log_info "Log directory: $log_directory"
    log_info "Visualization enabled: $enable_visualization"
    
    # Validate prerequisites
    echo ""
    echo "=== VALIDATING PREREQUISITES ==="
    validate_prerequisites "$input_csv_file" "$input_hdf5_file" "$enable_visualization" || return 1
    
    # Define output file paths
    local processed_csv_path="${output_directory}/${PROCESSED_CSV_FILENAME}"
    local jackknife_hdf5_path="${output_directory}/${JACKKNIFE_HDF5_FILENAME}"
    
    # =========================================================================
    # STAGE 1: PROCESS RAW PARAMETERS
    # =========================================================================
    
    echo ""
    echo "=== STAGE 1: PROCESSING RAW PARAMETERS ==="
    echo "Transforming and validating extracted parameters..."
    log_info "STAGE 1: Starting raw parameter processing"
    
    python "$PROCESS_RAW_SCRIPT" \
        --input_single_valued_csv_file_path "$input_csv_file" \
        --input_multivalued_hdf5_file_path "$input_hdf5_file" \
        --output_directory "$output_directory" \
        --output_csv_filename "$PROCESSED_CSV_FILENAME" \
        --enable_logging \
        --log_directory "$log_directory" \
        || {
            log_error "process_raw_parameters.py execution failed"
            echo "ERROR: Raw parameter processing failed" >&2
            return 1
        }
    
    # Validate output
    if ! $skip_checks; then
        check_intermediate_output "$processed_csv_path" "Raw parameter processing" || return 1
    fi
    
    log_info "STAGE 1 COMPLETED: Raw parameters processed successfully"
    echo "$PROGRESS_SUCCESS Stage 1 completed: Raw parameters processed"
    echo "  - Output: $PROCESSED_CSV_FILENAME"
    
    # Generate summary for Stage 1 output
    if ! $skip_summaries; then
        echo ""
        echo "Generating CSV summary..."
        generate_csv_summary "$processed_csv_path" "$output_directory"
    fi
    
    # =========================================================================
    # STAGE 2: APPLY JACKKNIFE ANALYSIS
    # =========================================================================
    
    echo ""
    echo "=== STAGE 2: APPLYING JACKKNIFE ANALYSIS ==="
    echo "Applying jackknife resampling to correlator data..."
    log_info "STAGE 2: Starting jackknife analysis"
    
    python "$JACKKNIFE_SCRIPT" \
        --input_hdf5_file "$input_hdf5_file" \
        --output_hdf5_file "$jackknife_hdf5_path" \
        --output_directory "$output_directory" \
        --enable_logging \
        --log_directory "$log_directory" \
        || {
            log_error "apply_jackknife_analysis.py execution failed"
            echo "ERROR: Jackknife analysis failed" >&2
            return 1
        }
    
    # Validate output
    if ! $skip_checks; then
        check_intermediate_output "$jackknife_hdf5_path" "Jackknife analysis" || return 1
    fi
    
    log_info "STAGE 2 COMPLETED: Jackknife analysis completed successfully"
    echo "$PROGRESS_SUCCESS Stage 2 completed: Jackknife analysis applied"
    echo "  - Output: $JACKKNIFE_HDF5_FILENAME"
    
    # Generate summary for Stage 2 output
    if ! $skip_summaries; then
        echo ""
        echo "Generating HDF5 tree..."
        generate_hdf5_tree "$jackknife_hdf5_path" "Stage 2"
    fi
    
    # =========================================================================
    # STAGE 3: VISUALIZATION (OPTIONAL)
    # =========================================================================
    
    if $enable_visualization; then
        echo ""
        echo "=== STAGE 3: GENERATING VISUALIZATIONS ==="
        echo "Creating jackknife sample diagnostic plots..."
        log_info "STAGE 3: Starting visualization generation"
        
        python "$VISUALIZATION_SCRIPT" \
            --input_hdf5_file "$jackknife_hdf5_path" \
            --output_directory "$output_directory" \
            --enable_logging \
            --log_directory "$log_directory" \
            || {
                log_error "visualize_jackknife_samples.py execution failed"
                echo "ERROR: Visualization failed" >&2
                return 1
            }
        
        # Check for visualization output directory
        local plots_dir="${output_directory}/jackknife_plots"
        if ! $skip_checks && [[ -d "$plots_dir" ]]; then
            log_info "STAGE 3 COMPLETED: Visualizations generated in $plots_dir"
            echo "$PROGRESS_SUCCESS Stage 3 completed: Visualizations generated"
            echo "  - Output directory: jackknife_plots/"
        elif ! $skip_checks; then
            log_warning "Visualization completed but no plots directory found"
            echo "$PROGRESS_WARNING Stage 3 completed with warnings"
        else
            echo "$PROGRESS_SUCCESS Stage 3 completed: Visualization script executed"
        fi
    else
        echo ""
        echo "$PROGRESS_SKIPPED Stage 3 skipped: Visualization not enabled"
        log_info "STAGE 3 SKIPPED: Visualization not requested"
    fi
    
    # =========================================================================
    # PIPELINE COMPLETION
    # =========================================================================
    
    echo ""
    echo "=== PROCESSING PIPELINE COMPLETED ==="
    echo "All processing stages completed successfully!"
    echo ""
    echo "Output files:"
    echo "  - Processed parameters: $processed_csv_path"
    echo "  - Jackknife analysis:   $jackknife_hdf5_path"
    if $enable_visualization; then
        echo "  - Visualizations:       ${output_directory}/jackknife_plots/"
    fi
    if ! $skip_summaries; then
        echo ""
        echo "Summary files:"
        echo "  - CSV summary:     ${processed_csv_path%.csv}_uniqueness_report.txt"
        echo "  - HDF5 tree:       ${jackknife_hdf5_path%.h5}_tree.txt"
    fi
    echo ""
    echo "Log file: $log_file"
    
    # Final logging
    local log_msg="PROCESSING PIPELINE COMPLETED SUCCESSFULLY"
    log_msg+="\n  - Input CSV: $input_csv_file"
    log_msg+="\n  - Input HDF5: $input_hdf5_file"
    log_msg+="\n  - Stages executed: $(( enable_visualization ? 3 : 2 ))"
    log_msg+="\n  - Validation: $(( skip_checks ? "SKIPPED" : "COMPLETED" ))"
    log_msg+="\n  - Summary files: $(( skip_summaries ? "SKIPPED" : "GENERATED" ))"
    log_info "$log_msg"
    
    # Terminate logging properly using close_logging
    log_info "Processing pipeline completed successfully"
    close_logging
    
    echo -e "\n${PROGRESS_SUCCESS} Processing pipeline execution completed successfully!"
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
