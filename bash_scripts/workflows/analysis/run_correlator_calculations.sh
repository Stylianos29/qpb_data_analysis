#!/bin/bash

################################################################################
# run_correlator_calculations.sh - Execute Stage 3.1 (Correlator Calculations)
#
# DESCRIPTION:
# This script orchestrates Stage 3.1 of the QPB data analysis pipeline,
# calculating PCAC mass and pion effective mass from jackknife correlators.
# It handles both PCAC and Pion branches independently, with intelligent
# input detection and optional visualization.
#
# The script executes these Python scripts:
# 1. calculate_PCAC_mass.py - PCAC mass calculation (PCAC branch)
# 2. calculate_effective_mass.py - Pion effective mass (Pion branch)
# 3. visualize_correlator_analysis.py - Optional visualization (both branches)
#
# USAGE:
#   ./run_correlator_calculations.sh -hdf5_jack <jackknife_hdf5_file> [options]
#
# REQUIRED ARGUMENTS:
#   -hdf5_jack, --input_hdf5_jackknife  Path to jackknife HDF5 file
#                                       (output from apply_jackknife_analysis.py:
#                                        correlators_jackknife_analysis.h5)
#
# OPTIONAL ARGUMENTS:
#   -o,  --output_directory     Output directory for analysis files
#                               (default: parent directory of input HDF5 file)
#   -p,  --plots_directory      Directory for visualization plots
#                               (default: not created unless visualization enabled)
#   -log_dir, --log_directory   Directory for log files
#                               (default: input_directory/auxiliary/logs/)
#   --enable-viz                Generate visualizations for both branches
#   --pcac-only                 Run only PCAC branch (skip pion)
#   --pion-only                 Run only Pion branch (skip PCAC)
#   --skip-checks               Skip intermediate file validation
#   --skip-summaries            Skip generation of summary files
#   -h,  --help                 Display this help message
#
# EXAMPLES:
#   # Basic usage - process both branches
#   ./run_correlator_calculations.sh -hdf5_jack correlators_jackknife_analysis.h5
#
#   # With visualization for both branches
#   ./run_correlator_calculations.sh \
#       -hdf5_jack data/correlators_jackknife_analysis.h5 \
#       -p plots/ --enable-viz
#
#   # PCAC branch only with visualization
#   ./run_correlator_calculations.sh \
#       -hdf5_jack correlators_jackknife_analysis.h5 \
#       -o processed_data/ -p plots/ \
#       --pcac-only --enable-viz
#
#   # Full path with all options
#   ./run_correlator_calculations.sh \
#       -hdf5_jack /path/to/correlators_jackknife_analysis.h5 \
#       -o /path/to/output/ -p /path/to/plots/ \
#       --enable-viz
#
# DEPENDENCIES:
# - Python scripts: calculate_PCAC_mass.py, calculate_effective_mass.py,
#                   visualize_correlator_analysis.py
# - Library scripts in bash_scripts/library/
# - Python environment with qpb_data_analysis package
#
# INPUT FILES:
# - correlators_jackknife_analysis.h5 (REQUIRED)
#     Output from Stage 2B (apply_jackknife_analysis.py)
#     Contains jackknife samples, means, and errors for g5-g5 and g4g5-g5 correlators
#     This single file is sufficient input for both PCAC and Pion branches
#
# OUTPUT FILES:
# - PCAC_mass_analysis.h5              : PCAC mass time series (PCAC branch)
# - pion_effective_mass_analysis.h5    : Pion effective mass (Pion branch)
# - PCAC_mass_analysis_tree.txt        : HDF5 structure summary (PCAC)
# - pion_effective_mass_analysis_tree.txt : HDF5 structure (Pion)
# - Correlator visualization plots     : Optional, in plots_directory
# - run_correlator_calculations.log    : Execution log
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
CORRELATOR_SCRIPTS_DIR="${PYTHON_SCRIPTS_DIRECTORY}/analysis/correlator_calculations"
PCAC_MASS_SCRIPT="${CORRELATOR_SCRIPTS_DIR}/calculate_PCAC_mass.py"
PION_MASS_SCRIPT="${CORRELATOR_SCRIPTS_DIR}/calculate_effective_mass.py"
VIZ_SCRIPT="${CORRELATOR_SCRIPTS_DIR}/visualize_correlator_analysis.py"

# Output filenames (using constants from constants.sh)
PCAC_OUTPUT_HDF5="$ANALYSIS_HDF5_PCAC_MASS"
PION_OUTPUT_HDF5="$ANALYSIS_HDF5_PION_MASS"

# Script log filename
SCRIPT_LOG_FILENAME="${SCRIPT_NAME%.sh}.log"

# Auxiliary directory structure
AUXILIARY_DIR_NAME="auxiliary"
AUXILIARY_LOGS_SUBDIR="logs"
AUXILIARY_SUMMARIES_SUBDIR="summaries"
AUXILIARY_VIZ_SUBDIR="visualization"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function usage() {
    cat << EOF
Usage: $SCRIPT_NAME -hdf5_jack <jackknife_hdf5_file> [options]

REQUIRED ARGUMENTS:
  -hdf5_jack, --input_hdf5_jackknife  Path to jackknife HDF5 file

OPTIONAL ARGUMENTS:
  -o, --output_directory          Output directory for analysis files
                                  (default: parent dir of input HDF5)
  -p, --plots_directory           Directory for visualization plots
  -log_dir, --log_directory       Directory for log files
                                  (default: output_dir/auxiliary/logs/)
  
  VISUALIZATION:
  --enable-viz                    Generate visualizations for both branches
  
  BRANCH CONTROL:
  --pcac-only                     Run only PCAC branch (skip pion)
  --pion-only                     Run only Pion branch (skip PCAC)
  
  OTHER OPTIONS:
  --skip-checks                   Skip intermediate validation
  --skip-summaries                Skip summary file generation
  -h, --help                      Display this help message

EXAMPLES:
  # Both branches
  $SCRIPT_NAME -hdf5_jack correlators_jackknife_analysis.h5

  # With visualization
  $SCRIPT_NAME -hdf5_jack data.h5 -p plots/ --enable-viz

  # PCAC only
  $SCRIPT_NAME -hdf5_jack data.h5 --pcac-only

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
    #   - auxiliary/visualization/
    
    local output_dir="$1"
    local auxiliary_dir="${output_dir}/${AUXILIARY_DIR_NAME}"
    
    # Create main auxiliary directory
    validate_output_directory "$auxiliary_dir" -c -s || return 1
    
    # Create subdirectories
    validate_output_directory "${auxiliary_dir}/${AUXILIARY_LOGS_SUBDIR}" -c -s || return 1
    validate_output_directory "${auxiliary_dir}/${AUXILIARY_SUMMARIES_SUBDIR}" -c -s || return 1
    validate_output_directory "${auxiliary_dir}/${AUXILIARY_VIZ_SUBDIR}" -c -s || return 1
    
    return 0
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

function main() {
    # Initialize variables with defaults
    local input_hdf5_jackknife=""
    local output_directory=""
    local plots_directory=""
    local log_directory=""
    
    # Branch control flags
    local enable_viz="false"
    local pcac_only="false"
    local pion_only="false"
    
    # Other flags
    local skip_checks="false"
    local skip_summaries="false"
    local clean_plots=false
    
    # Parse command-line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -hdf5_jack|--input_hdf5_jackknife)
                input_hdf5_jackknife="$2"
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
            --pcac-only)
                pcac_only="true"
                shift
                ;;
            --pion-only)
                pion_only="true"
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
    
    # Validate required arguments
    if [[ -z "$input_hdf5_jackknife" ]]; then
        echo "ERROR: Jackknife HDF5 file is required" >&2
        echo "Use: $SCRIPT_NAME -hdf5_jack <file>" >&2
        echo "Use -h or --help for usage information" >&2
        exit 1
    fi
    
    # Convert input file to absolute path and validate it exists
    input_hdf5_jackknife="$(realpath "$input_hdf5_jackknife")"
    if [[ ! -f "$input_hdf5_jackknife" ]]; then
        echo "ERROR: Jackknife HDF5 file not found: $input_hdf5_jackknife" >&2
        exit 1
    fi
    
    # Set defaults for optional arguments
    # Default output directory: parent directory of input HDF5 file
    if [[ -z "$output_directory" ]]; then
        output_directory="$(dirname "$input_hdf5_jackknife")"
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
    
    # Validate conflicting flags
    if [[ "$pcac_only" == "true" && "$pion_only" == "true" ]]; then
        echo "ERROR: Cannot specify both --pcac-only and --pion-only" >&2
        exit 1
    fi
    
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
    
    log_info "=== Stage 3.1: Correlator Calculations Started ==="
    log_info "Input jackknife HDF5: $input_hdf5_jackknife"
    log_info "Output directory: $output_directory"
    [[ -n "$plots_directory" ]] && log_info "Plots directory: $plots_directory"
    log_info "Log directory: $log_directory"
    
    # Track success/failure
    local pcac_success="false"
    local pion_success="false"
    local pcac_skipped="false"
    local pion_skipped="false"
    
    # Auxiliary directories
    local summary_dir="${output_directory}/${AUXILIARY_DIR_NAME}/${AUXILIARY_SUMMARIES_SUBDIR}"
    
    # =========================================================================
    # PCAC BRANCH
    # =========================================================================
    
    if [[ "$pion_only" != "true" ]]; then
        echo ""
        echo "=========================================="
        echo "PCAC Branch: Calculating PCAC Mass"
        echo "=========================================="
        log_info "Starting PCAC branch"
        
        # Validate Python script
        if [[ ! -f "$PCAC_MASS_SCRIPT" ]]; then
            echo "ERROR: PCAC mass calculation script not found: $PCAC_MASS_SCRIPT" >&2
            log_error "PCAC mass script not found: $PCAC_MASS_SCRIPT"
            close_logging
            exit 1
        fi
        
        # Construct output path
        local pcac_output_hdf5="${output_directory}/${PCAC_OUTPUT_HDF5}"
        
        # Execute PCAC mass calculation
        echo "  → Running calculate_PCAC_mass.py..."
        execute_python_script "$PCAC_MASS_SCRIPT" "calculate_PCAC_mass" \
            --input_hdf5_file "$input_hdf5_jackknife" \
            --output_hdf5_file "$pcac_output_hdf5" \
            --enable_logging \
            --log_directory "$log_directory"
        
        if [[ $? -eq 0 ]]; then
            pcac_success="true"
            echo "  ✓ PCAC mass calculation completed"
            log_info "PCAC mass calculation successful"
            
            # Validate output file
            if [[ "$skip_checks" != "true" ]]; then
                if [[ ! -f "$pcac_output_hdf5" ]]; then
                    echo "ERROR: PCAC HDF5 file not created: $pcac_output_hdf5" >&2
                    log_error "PCAC HDF5 file missing: $pcac_output_hdf5"
                    close_logging
                    exit 1
                fi
                echo "  ✓ Output file validated"
            fi
            
            # Generate HDF5 tree summary
            if [[ "$skip_summaries" != "true" ]]; then
                echo "  → Generating HDF5 tree summary..."
                generate_hdf5_summary "$pcac_output_hdf5" "$summary_dir"
                if [[ $? -eq 0 ]]; then
                    echo "  ✓ HDF5 tree summary generated"
                    log_info "PCAC HDF5 tree summary created in: $summary_dir"
                else
                    echo "  ⚠ Warning: HDF5 tree summary generation failed" >&2
                    log_warning "Failed to generate PCAC HDF5 tree summary"
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
                        --analysis_type pcac_mass
                        --input_hdf5_file "$pcac_output_hdf5"
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
            echo "ERROR: PCAC mass calculation failed" >&2
            log_error "PCAC mass calculation failed"
            close_logging
            exit 1
        fi
    else
        pcac_skipped="true"
        echo "○ PCAC branch skipped (--pion-only specified)"
        log_info "PCAC branch skipped: pion-only mode"
    fi
    
    # =========================================================================
    # PION BRANCH
    # =========================================================================
    
    if [[ "$pcac_only" != "true" ]]; then
        echo ""
        echo "=========================================="
        echo "Pion Branch: Calculating Effective Mass"
        echo "=========================================="
        log_info "Starting Pion branch"
        
        # Validate Python script
        if [[ ! -f "$PION_MASS_SCRIPT" ]]; then
            echo "ERROR: Pion effective mass script not found: $PION_MASS_SCRIPT" >&2
            log_error "Pion effective mass script not found: $PION_MASS_SCRIPT"
            close_logging
            exit 1
        fi
        
        # Construct output path
        local pion_output_hdf5="${output_directory}/${PION_OUTPUT_HDF5}"
        
        # Execute pion effective mass calculation
        echo "  → Running calculate_effective_mass.py..."
        execute_python_script "$PION_MASS_SCRIPT" "calculate_effective_mass" \
            --input_hdf5_file "$input_hdf5_jackknife" \
            --output_hdf5_file "$pion_output_hdf5" \
            --enable_logging \
            --log_directory "$log_directory"
        
        if [[ $? -eq 0 ]]; then
            pion_success="true"
            echo "  ✓ Pion effective mass calculation completed"
            log_info "Pion effective mass calculation successful"
            
            # Validate output file
            if [[ "$skip_checks" != "true" ]]; then
                if [[ ! -f "$pion_output_hdf5" ]]; then
                    echo "ERROR: Pion HDF5 file not created: $pion_output_hdf5" >&2
                    log_error "Pion HDF5 file missing: $pion_output_hdf5"
                    close_logging
                    exit 1
                fi
                echo "  ✓ Output file validated"
            fi
            
            # Generate HDF5 tree summary
            if [[ "$skip_summaries" != "true" ]]; then
                echo "  → Generating HDF5 tree summary..."
                generate_hdf5_summary "$pion_output_hdf5" "$summary_dir"
                if [[ $? -eq 0 ]]; then
                    echo "  ✓ HDF5 tree summary generated"
                    log_info "Pion HDF5 tree summary created in: $summary_dir"
                else
                    echo "  ⚠ Warning: HDF5 tree summary generation failed" >&2
                    log_warning "Failed to generate Pion HDF5 tree summary"
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
                        --analysis_type effective_mass
                        --input_hdf5_file "$pion_output_hdf5"
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
            echo "ERROR: Pion effective mass calculation failed" >&2
            log_error "Pion effective mass calculation failed"
            close_logging
            exit 1
        fi
    else
        pion_skipped="true"
        echo "○ Pion branch skipped (--pcac-only specified)"
        log_info "Pion branch skipped: pcac-only mode"
    fi
    
    # =========================================================================
    # COMPLETION SUMMARY
    # =========================================================================
    
    echo ""
    echo "=========================================="
    echo "Stage 3.1 Completion Summary"
    echo "=========================================="
    
    if [[ "$pcac_success" == "true" ]]; then
        echo "✓ PCAC branch completed successfully"
    elif [[ "$pcac_skipped" == "true" ]]; then
        echo "○ PCAC branch skipped"
    fi
    
    if [[ "$pion_success" == "true" ]]; then
        echo "✓ Pion branch completed successfully"
    elif [[ "$pion_skipped" == "true" ]]; then
        echo "○ Pion branch skipped"
    fi
    
    echo ""
    echo "Output files location: $output_directory"
    [[ -n "$plots_directory" ]] && echo "Plots location: $plots_directory"
    echo "Log file: $log_file"
    
    log_info "=== Stage 3.1: Correlator Calculations Completed ==="
    close_logging
    
    echo ""
    echo "✓ Stage 3.1 (Correlator Calculations) completed successfully"
    
    return 0
}

# Execute main function
main "$@"
