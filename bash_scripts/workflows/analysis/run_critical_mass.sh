#!/bin/bash

################################################################################
# run_critical_mass.sh - Execute Stage 3.3 (Critical Mass Extrapolation)
#
# DESCRIPTION:
# This script orchestrates Stage 3.3 of the QPB data analysis pipeline,
# calculating critical bare mass through linear extrapolation to the chiral
# limit. It processes CSV files from Stage 3.2 (plateau estimates), performs
# robust linear fits with comprehensive quality validation, and extrapolates
# to find the critical bare mass where PCAC/pion mass vanishes.
#
# The script executes these Python scripts:
# 1. calculate_critical_mass_from_pcac.py - PCAC critical mass (PCAC branch)
# 2. calculate_critical_mass_from_pion.py - Pion critical mass (Pion branch)
# 3. visualize_critical_mass_analysis.py - Optional visualization (both branches)
#
# USAGE:
#   ./run_critical_mass.sh -i_pcac <pcac_csv> -i_pion <pion_csv> [options]
#
# REQUIRED ARGUMENTS (at least one):
#   -i_pcac, --input_pcac_csv   Path to PCAC plateau estimates CSV file
#                               (from extract_plateau_PCAC_mass.py)
#   -i_pion, --input_pion_csv   Path to pion plateau estimates CSV file
#                               (from extract_plateau_pion_mass.py)
#
# Note: At least one input file must be provided. If only one is provided,
#       only that branch will be processed.
#
# OPTIONAL ARGUMENTS:
#   -o,  --output_directory     Output directory for results CSV files
#                               (default: parent directory of first input file)
#   -p,  --plots_directory      Directory for visualization plots
#                               (default: not created unless --enable-viz)
#   -log_dir, --log_directory   Directory for log files
#                               (default: output_directory/auxiliary/logs/)
#   --enable-viz                Generate critical mass visualization plots
#   --skip-checks               Skip intermediate file validation
#   --skip-summaries            Skip generation of summary files
#   -h,  --help                 Display this help message
#
# EXAMPLES:
#   # Process both branches
#   ./run_critical_mass.sh \
#       -i_pcac plateau_PCAC_mass_estimates.csv \
#       -i_pion plateau_pion_mass_estimates.csv
#
#   # With visualization for both branches
#   ./run_critical_mass.sh \
#       -i_pcac data/plateau_PCAC_mass_estimates.csv \
#       -i_pion data/plateau_pion_mass_estimates.csv \
#       -p plots/ --enable-viz
#
#   # PCAC branch only
#   ./run_critical_mass.sh \
#       -i_pcac plateau_PCAC_mass_estimates.csv \
#       -o output/ -p plots/ --enable-viz
#
#   # Pion branch only with custom output
#   ./run_critical_mass.sh \
#       -i_pion plateau_pion_mass_estimates.csv \
#       -o processed_data/
#
# DEPENDENCIES:
# - Python scripts: calculate_critical_mass_from_pcac.py,
#                   calculate_critical_mass_from_pion.py,
#                   visualize_critical_mass_analysis.py
# - Library scripts in bash_scripts/library/
# - Python environment with qpb_data_analysis package
#
# INPUT FILES:
# - plateau_PCAC_mass_estimates.csv (optional, from Stage 3.2)
#     Output from extract_plateau_PCAC_mass.py
#     Contains PCAC plateau mass values vs bare mass
#
# - plateau_pion_mass_estimates.csv (optional, from Stage 3.2)
#     Output from extract_plateau_pion_mass.py
#     Contains pion plateau mass values vs bare mass
#
# OUTPUT FILES:
# Data files (in output_directory):
# - critical_bare_mass_from_pcac.csv       : PCAC critical mass results
# - critical_bare_mass_from_pion.csv       : Pion critical mass results
#
# Summary files (in output_directory/auxiliary/summaries/):
# - critical_bare_mass_from_pcac_uniqueness_report.txt
# - critical_bare_mass_from_pion_uniqueness_report.txt
#
# Plots (if --enable-viz, in plots_directory):
# - Critical_mass_extrapolation_pcac/      : PCAC extrapolation plots
# - Critical_mass_extrapolation_pion/      : Pion extrapolation plots
#
# - run_critical_mass.log                  : Execution log
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
    LIBRARY_SCRIPTS_DIRECTORY_PATH="$(realpath "${SCRIPT_DIR}/../../library")"
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
    PYTHON_SCRIPTS_DIRECTORY="$(realpath "${SCRIPT_DIR}/../../../core/src")"
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
CRITICAL_MASS_SCRIPTS_DIR="${PYTHON_SCRIPTS_DIRECTORY}/analysis/critical_mass_extrapolation"
PCAC_CRITICAL_SCRIPT="${CRITICAL_MASS_SCRIPTS_DIR}/calculate_critical_mass_from_pcac.py"
PION_CRITICAL_SCRIPT="${CRITICAL_MASS_SCRIPTS_DIR}/calculate_critical_mass_from_pion.py"
VIZ_SCRIPT="${CRITICAL_MASS_SCRIPTS_DIR}/visualize_critical_mass_analysis.py"

# Output filenames (using constants from constants.sh)
PCAC_OUTPUT_CSV="$ANALYSIS_CSV_CRITICAL_PCAC"
PION_OUTPUT_CSV="$ANALYSIS_CSV_CRITICAL_PION"

# Input filenames (for reference and validation)
PCAC_INPUT_CSV="$ANALYSIS_CSV_PCAC_PLATEAU"
PION_INPUT_CSV="$ANALYSIS_CSV_PION_PLATEAU"

# Script log filename
SCRIPT_LOG_FILENAME="${SCRIPT_NAME%.sh}.log"

# Auxiliary directory structure
AUXILIARY_DIR_NAME="auxiliary"
AUXILIARY_LOGS_SUBDIR="logs"
AUXILIARY_SUMMARIES_SUBDIR="summaries"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function usage() {
    cat << EOF
Usage: $SCRIPT_NAME -i_pcac <pcac_csv> -i_pion <pion_csv> [options]

REQUIRED ARGUMENTS (at least one):
  -i_pcac, --input_pcac_csv       Path to PCAC plateau estimates CSV
  -i_pion, --input_pion_csv       Path to pion plateau estimates CSV

Note: Provide at least one input file. If only one is given, only that
      branch will be processed.

OPTIONAL ARGUMENTS:
  -o, --output_directory          Output directory for results
                                  (default: parent dir of first input)
  -p, --plots_directory           Directory for visualization plots
  -log_dir, --log_directory       Directory for log files
                                  (default: output_dir/auxiliary/logs/)
  
  VISUALIZATION:
  --enable-viz                    Generate critical mass plots
  
  OTHER OPTIONS:
  --skip-checks                   Skip intermediate validation
  --skip-summaries                Skip summary file generation
  -h, --help                      Display this help message

EXAMPLES:
  # Both branches
  $SCRIPT_NAME -i_pcac plateau_PCAC.csv -i_pion plateau_pion.csv

  # With visualization
  $SCRIPT_NAME -i_pcac plateau_PCAC.csv -i_pion plateau_pion.csv \\
      -p plots/ --enable-viz

  # PCAC only
  $SCRIPT_NAME -i_pcac plateau_PCAC_mass_estimates.csv -o output/

For more information, see the script header documentation.
EOF
    exit 0
}

function setup_auxiliary_directories() {
    # Create auxiliary directory structure
    #
    # Arguments:
    #   $1 - output_directory (base for auxiliary dirs)
    #
    # Creates:
    #   - auxiliary/logs/
    #   - auxiliary/summaries/
    
    local output_dir="$1"
    local auxiliary_dir="${output_dir}/${AUXILIARY_DIR_NAME}"
    
    # Create main auxiliary directory
    validate_output_directory "$auxiliary_dir" -c -s || return 1
    
    # Create subdirectories
    validate_output_directory "${auxiliary_dir}/${AUXILIARY_LOGS_SUBDIR}" -c -s || return 1
    validate_output_directory "${auxiliary_dir}/${AUXILIARY_SUMMARIES_SUBDIR}" -c -s || return 1
    
    return 0
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

function main() {
    # Initialize variables with defaults
    local input_pcac_csv=""
    local input_pion_csv=""
    local output_directory=""
    local plots_directory=""
    local log_directory=""
    
    # Control flags
    local enable_viz="false"
    local skip_checks="false"
    local skip_summaries="false"
    local clean_plots=false
    
    # Parse command-line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -i_pcac|--input_pcac_csv)
                input_pcac_csv="$2"
                shift 2
                ;;
            -i_pion|--input_pion_csv)
                input_pion_csv="$2"
                shift 2
                ;;
            -o|--output_directory)
                output_directory="$2"
                shift 2
                ;;
            -p|--plots_directory)
                plots_directory="$2"
                shift 2
                ;;
            -log_dir|--log_directory)
                log_directory="$2"
                shift 2
                ;;
            --enable-viz)
                enable_viz="true"
                shift
                ;;
            --skip-checks)
                skip_checks="true"
                shift
                ;;
            --skip-summaries)
                skip_summaries="true"
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
                echo "ERROR: Unknown argument: $1" >&2
                echo "Use -h or --help for usage information" >&2
                exit 1
                ;;
        esac
    done
    
    # Validate that at least one input file is provided
    if [[ -z "$input_pcac_csv" && -z "$input_pion_csv" ]]; then
        echo "ERROR: At least one input CSV file is required" >&2
        echo "Use: $SCRIPT_NAME -i_pcac <file> and/or -i_pion <file>" >&2
        echo "Use -h or --help for usage information" >&2
        exit 1
    fi
    
    # Validate and convert input files to absolute paths
    if [[ -n "$input_pcac_csv" ]]; then
        input_pcac_csv="$(realpath "$input_pcac_csv")"
        if [[ ! -f "$input_pcac_csv" ]]; then
            echo "ERROR: PCAC CSV file not found: $input_pcac_csv" >&2
            exit 1
        fi
    fi
    
    if [[ -n "$input_pion_csv" ]]; then
        input_pion_csv="$(realpath "$input_pion_csv")"
        if [[ ! -f "$input_pion_csv" ]]; then
            echo "ERROR: Pion CSV file not found: $input_pion_csv" >&2
            exit 1
        fi
    fi
    
    # Set default output directory: parent directory of first provided input file
    if [[ -z "$output_directory" ]]; then
        if [[ -n "$input_pcac_csv" ]]; then
            output_directory="$(dirname "$input_pcac_csv")"
        else
            output_directory="$(dirname "$input_pion_csv")"
        fi
    fi
    output_directory="$(realpath "$output_directory")"
    
    # Handle plots directory - use realpath -m to canonicalize even if it doesn't exist yet
    if [[ -n "$plots_directory" ]]; then
        plots_directory="$(realpath -m "$plots_directory")"
    fi
    
    # Default log directory
    if [[ -z "$log_directory" ]]; then
        log_directory="${output_directory}/${AUXILIARY_DIR_NAME}/${AUXILIARY_LOGS_SUBDIR}"
    fi
    # Use realpath -m to handle the path even if it doesn't exist yet
    log_directory="$(realpath -m "$log_directory")"
    
    # Validate output directory exists (user must create it)
    if [[ ! -d "$output_directory" ]]; then
        echo "ERROR: Output directory does not exist: $output_directory" >&2
        echo "Please create it before running this script." >&2
        exit 1
    fi
    
    # Setup auxiliary directories (auto-created if needed)
    setup_auxiliary_directories "$output_directory" || exit 1
    
    # Initialize logging
    local log_file="${log_directory}/${SCRIPT_LOG_FILENAME}"
    init_logging "$log_file" -c || {
        echo "ERROR: Failed to initialize logging" >&2
        exit 1
    }
    
    log_info "=== Stage 3.3: Critical Mass Extrapolation Started ==="
    [[ -n "$input_pcac_csv" ]] && log_info "Input PCAC CSV: $input_pcac_csv"
    [[ -n "$input_pion_csv" ]] && log_info "Input Pion CSV: $input_pion_csv"
    log_info "Output directory: $output_directory"
    [[ -n "$plots_directory" ]] && log_info "Plots directory: $plots_directory"
    log_info "Log directory: $log_directory"
    
    # Track success/failure
    local pcac_success="false"
    local pion_success="false"
    local pcac_skipped="false"
    local pion_skipped="false"
    local pcac_data_insufficient="false"
    local pion_data_insufficient="false"
    
    # Auxiliary directories
    local summary_dir="${output_directory}/${AUXILIARY_DIR_NAME}/${AUXILIARY_SUMMARIES_SUBDIR}"
    
    # =========================================================================
    # PCAC BRANCH
    # =========================================================================
    
    if [[ -n "$input_pcac_csv" ]]; then
        echo ""
        echo "=========================================="
        echo "PCAC Branch: Critical Mass Extrapolation"
        echo "=========================================="
        log_info "Starting PCAC branch"
        
        # Validate Python script
        if [[ ! -f "$PCAC_CRITICAL_SCRIPT" ]]; then
            echo "ERROR: PCAC critical mass script not found: $PCAC_CRITICAL_SCRIPT" >&2
            log_error "PCAC critical mass script not found: $PCAC_CRITICAL_SCRIPT"
            close_logging
            exit 1
        fi
        
        # Construct full output path
        local pcac_output_csv="${output_directory}/${PCAC_OUTPUT_CSV}"
        
        # Create temporary file to capture stderr
        local stderr_capture=$(mktemp)
        
        # Execute PCAC critical mass calculation with stderr capture
        echo "  → Running calculate_critical_mass_from_pcac.py..."
        execute_python_script "$PCAC_CRITICAL_SCRIPT" "calculate_critical_mass_from_pcac" \
            --input_csv "$input_pcac_csv" \
            --output_csv "$pcac_output_csv" \
            --enable_logging \
            --log_directory "$log_directory" \
            2> >(tee -a "$stderr_capture" >&2)
        
        local script_exit_code=$?
        
        if [[ $script_exit_code -eq 0 ]]; then
            pcac_success="true"
            echo "  ✓ PCAC critical mass calculation completed"
            log_info "PCAC critical mass calculation successful"
            
            # Validate output file
            if [[ "$skip_checks" != "true" ]]; then
                if [[ ! -f "$pcac_output_csv" ]]; then
                    echo "ERROR: PCAC output CSV not created: $pcac_output_csv" >&2
                    log_error "PCAC output CSV missing: $pcac_output_csv"
                    close_logging
                    exit 1
                fi
                echo "  ✓ Output file validated"
            fi
            
            # Generate CSV summary
            if [[ "$skip_summaries" != "true" ]]; then
                echo "  → Generating CSV summary..."
                
                generate_csv_summary "$pcac_output_csv" "$summary_dir"
                if [[ $? -eq 0 ]]; then
                    echo "  ✓ CSV summary generated"
                    log_info "PCAC CSV summary created in: $summary_dir"
                else
                    echo "  ⚠ Warning: CSV summary generation failed" >&2
                    log_warning "Failed to generate PCAC CSV summary"
                fi
            fi
            
            # Optional visualization
            if [[ "$enable_viz" == "true" ]]; then
                if [[ -z "$plots_directory" ]]; then
                    echo "  ⚠ Warning: Visualization requested but no plots directory specified" >&2
                    log_warning "PCAC visualization skipped: no plots directory"
                elif [[ ! -f "$VIZ_SCRIPT" ]]; then
                    echo "  ⚠ Warning: Visualization script not found: $VIZ_SCRIPT" >&2
                    log_warning "Visualization script not found"
                elif [[ ! -d "$plots_directory" ]]; then
                    echo "  ⚠ Warning: Plots directory does not exist: $plots_directory" >&2
                    echo "  → Please create it to enable visualization" >&2
                    log_warning "PCAC visualization skipped: plots directory does not exist"
                else
                    echo "  → Generating PCAC visualizations..."
                    
                    # Build visualization arguments
                    local viz_args=(
                        --analysis_type pcac
                        --results_csv "$pcac_output_csv"
                        --plateau_csv "$input_pcac_csv"
                        --plots_directory "$plots_directory"
                        --enable_logging
                        --log_directory "$log_directory"
                    )
                    
                    # Add clear_existing flag if requested
                    if [[ "$clean_plots" == "true" ]]; then
                        viz_args+=(--clear_existing)
                        echo "  → Clearing existing plots before regeneration..."
                        log_info "Clearing existing plots (--clean-plots enabled)"
                    fi
                    
                    # Execute with built arguments
                    execute_python_script "$VIZ_SCRIPT" "visualize_correlator_analysis" \
                        "${viz_args[@]}"
                    
                    if [[ $? -eq 0 ]]; then
                        echo "  ✓ PCAC visualizations generated"
                        log_info "PCAC visualization completed"
                    else
                        echo "  ⚠ Warning: PCAC visualization failed" >&2
                        log_warning "PCAC visualization failed"
                    fi
                fi
            fi
        else
            # Check if it's a data insufficiency error vs a real failure
            # Check the captured stderr for the specific error message
            if grep -q "No groups have sufficient data points for analysis" "$stderr_capture" 2>/dev/null; then
                # Graceful handling of insufficient data
                pcac_data_insufficient="true"
                echo "  ⚠ PCAC analysis skipped - insufficient bare mass variation" >&2
                echo "  → Critical mass extrapolation requires multiple bare mass values" >&2
                log_warning "PCAC critical mass skipped: insufficient data points for extrapolation (only one bare mass value)"
            else
                # Real error - should fail
                echo "ERROR: PCAC critical mass calculation failed" >&2
                log_error "PCAC critical mass calculation failed"
                rm -f "$stderr_capture"
                close_logging
                exit 1
            fi
        fi
        
        # Clean up temporary stderr capture file
        rm -f "$stderr_capture"
    else
        pcac_skipped="true"
        echo "○ PCAC branch skipped (no input file provided)"
        log_info "PCAC branch skipped: no input file"
    fi
    
    # =========================================================================
    # PION BRANCH
    # =========================================================================
    
    if [[ -n "$input_pion_csv" ]]; then
        echo ""
        echo "=========================================="
        echo "Pion Branch: Critical Mass Extrapolation"
        echo "=========================================="
        log_info "Starting Pion branch"
        
        # Validate Python script
        if [[ ! -f "$PION_CRITICAL_SCRIPT" ]]; then
            echo "ERROR: Pion critical mass script not found: $PION_CRITICAL_SCRIPT" >&2
            log_error "Pion critical mass script not found: $PION_CRITICAL_SCRIPT"
            close_logging
            exit 1
        fi
        
        # Construct full output path
        local pion_output_csv="${output_directory}/${PION_OUTPUT_CSV}"
        
        # Create temporary file to capture stderr
        local stderr_capture=$(mktemp)
        
        # Execute pion critical mass calculation with stderr capture
        echo "  → Running calculate_critical_mass_from_pion.py..."
        execute_python_script "$PION_CRITICAL_SCRIPT" "calculate_critical_mass_from_pion" \
            --input_csv "$input_pion_csv" \
            --output_csv "$pion_output_csv" \
            --enable_logging \
            --log_directory "$log_directory" \
            2> >(tee -a "$stderr_capture" >&2)
        
        local script_exit_code=$?
        
        if [[ $script_exit_code -eq 0 ]]; then
            pion_success="true"
            echo "  ✓ Pion critical mass calculation completed"
            log_info "Pion critical mass calculation successful"
            
            # Validate output file
            if [[ "$skip_checks" != "true" ]]; then
                if [[ ! -f "$pion_output_csv" ]]; then
                    echo "ERROR: Pion output CSV not created: $pion_output_csv" >&2
                    log_error "Pion output CSV missing: $pion_output_csv"
                    close_logging
                    exit 1
                fi
                echo "  ✓ Output file validated"
            fi
            
            # Generate CSV summary
            if [[ "$skip_summaries" != "true" ]]; then
                echo "  → Generating CSV summary..."
                
                generate_csv_summary "$pion_output_csv" "$summary_dir"
                if [[ $? -eq 0 ]]; then
                    echo "  ✓ CSV summary generated"
                    log_info "Pion CSV summary created in: $summary_dir"
                else
                    echo "  ⚠ Warning: CSV summary generation failed" >&2
                    log_warning "Failed to generate Pion CSV summary"
                fi
            fi
            
            # Optional visualization
            if [[ "$enable_viz" == "true" ]]; then
                if [[ -z "$plots_directory" ]]; then
                    echo "  ⚠ Warning: Visualization requested but no plots directory specified" >&2
                    log_warning "PCAC visualization skipped: no plots directory"
                elif [[ ! -f "$VIZ_SCRIPT" ]]; then
                    echo "  ⚠ Warning: Visualization script not found: $VIZ_SCRIPT" >&2
                    log_warning "Visualization script not found"
                elif [[ ! -d "$plots_directory" ]]; then
                    echo "  ⚠ Warning: Plots directory does not exist: $plots_directory" >&2
                    echo "  → Please create it to enable visualization" >&2
                    log_warning "PCAC visualization skipped: plots directory does not exist"
                else
                    echo "  → Generating PCAC visualizations..."
                    
                    # Build visualization arguments
                    local viz_args=(
                        --analysis_type pion
                        --results_csv "$pion_output_csv"
                        --plateau_csv "$input_pion_csv"
                        --plots_directory "$plots_directory"
                        --enable_logging
                        --log_directory "$log_directory"
                    )
                    
                    # Add clear_existing flag if requested
                    if [[ "$clean_plots" == "true" ]]; then
                        viz_args+=(--clear_existing)
                        echo "  → Clearing existing plots before regeneration..."
                        log_info "Clearing existing plots (--clean-plots enabled)"
                    fi
                    
                    # Execute with built arguments
                    execute_python_script "$VIZ_SCRIPT" "visualize_correlator_analysis" \
                        "${viz_args[@]}"
                    
                    if [[ $? -eq 0 ]]; then
                        echo "  ✓ PCAC visualizations generated"
                        log_info "PCAC visualization completed"
                    else
                        echo "  ⚠ Warning: PCAC visualization failed" >&2
                        log_warning "PCAC visualization failed"
                    fi
                fi
            fi
        else
            # Check if it's a data insufficiency error vs a real failure
            # Check the captured stderr for the specific error message
            if grep -q "No groups have sufficient data points for analysis" "$stderr_capture" 2>/dev/null; then
                # Graceful handling of insufficient data
                pion_data_insufficient="true"
                echo "  ⚠ Pion analysis skipped - insufficient bare mass variation" >&2
                echo "  → Critical mass extrapolation requires multiple bare mass values" >&2
                log_warning "Pion critical mass skipped: insufficient data points for extrapolation (only one bare mass value)"
            else
                # Real error - should fail
                echo "ERROR: Pion critical mass calculation failed" >&2
                log_error "Pion critical mass calculation failed"
                rm -f "$stderr_capture"
                close_logging
                exit 1
            fi
        fi
        
        # Clean up temporary stderr capture file
        rm -f "$stderr_capture"
    else
        pion_skipped="true"
        echo "○ Pion branch skipped (no input file provided)"
        log_info "Pion branch skipped: no input file"
    fi
    
    # =========================================================================
    # COMPLETION SUMMARY
    # =========================================================================
    
    echo ""
    echo "=========================================="
    echo "Stage 3.3 Completion Summary"
    echo "=========================================="
    
    if [[ "$pcac_success" == "true" ]]; then
        echo "✓ PCAC branch completed successfully"
    elif [[ "$pcac_data_insufficient" == "true" ]]; then
        echo "⚠ PCAC branch skipped (insufficient bare mass variation)"
    elif [[ "$pcac_skipped" == "true" ]]; then
        echo "○ PCAC branch skipped (no input file)"
    fi
    
    if [[ "$pion_success" == "true" ]]; then
        echo "✓ Pion branch completed successfully"
    elif [[ "$pion_data_insufficient" == "true" ]]; then
        echo "⚠ Pion branch skipped (insufficient bare mass variation)"
    elif [[ "$pion_skipped" == "true" ]]; then
        echo "○ Pion branch skipped (no input file)"
    fi
    
    echo ""
    echo "Output files location: $output_directory"
    [[ -n "$plots_directory" ]] && echo "Plots location: $plots_directory"
    echo "Log file: $log_file"
    
    log_info "=== Stage 3.3: Critical Mass Extrapolation Completed ==="
    close_logging
    
    echo ""
    # Provide a meaningful final status message based on what actually happened
    if [[ "$pcac_success" == "true" || "$pion_success" == "true" ]]; then
        echo "✓ Stage 3.3 (Critical Mass Extrapolation) completed successfully"
    elif [[ "$pcac_data_insufficient" == "true" || "$pion_data_insufficient" == "true" ]]; then
        echo "⚠ Stage 3.3 (Critical Mass Extrapolation) completed with warnings"
        echo "  (Insufficient bare mass variation for extrapolation)"
    else
        echo "○ Stage 3.3 (Critical Mass Extrapolation) completed (all branches skipped)"
    fi
    
    return 0
}

# Execute main function
main "$@"
