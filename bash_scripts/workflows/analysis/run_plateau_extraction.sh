#!/bin/bash

################################################################################
# run_plateau_extraction.sh - Execute Stage 3.2 (Plateau Extraction)
#
# DESCRIPTION:
# This script orchestrates Stage 3.2 of the QPB data analysis pipeline,
# extracting plateau regions from PCAC mass and pion effective mass time series.
# It processes HDF5 files from Stage 3.1, detects plateau regions using
# sophisticated algorithms, and exports results to CSV and HDF5 formats.
#
# The script executes these Python scripts:
# 1. extract_plateau_PCAC_mass.py - PCAC plateau extraction (PCAC branch)
# 2. extract_plateau_pion_mass.py - Pion plateau extraction (Pion branch)
# 3. visualize_plateau_extraction.py - Optional visualization (both branches)
#
# USAGE:
#   ./run_plateau_extraction.sh -i_pcac <pcac_hdf5> -i_pion <pion_hdf5> [options]
#
# REQUIRED ARGUMENTS (at least one):
#   -i_pcac, --input_pcac_hdf5  Path to PCAC mass analysis HDF5 file
#                               (from calculate_PCAC_mass.py)
#   -i_pion, --input_pion_hdf5  Path to pion effective mass analysis HDF5 file
#                               (from calculate_effective_mass.py)
#
# Note: At least one input file must be provided. If only one is provided,
#       only that branch will be processed.
#
# OPTIONAL ARGUMENTS:
#   -o,  --output_directory     Output directory for CSV files
#                               (default: parent directory of first input file)
#   -p,  --plots_directory      Directory for visualization plots
#                               (default: not created unless --enable-viz)
#   -log_dir, --log_directory   Directory for log files
#                               (default: output_directory/auxiliary/logs/)
#   --enable-viz                Generate plateau extraction visualization plots
#   --skip-checks               Skip intermediate file validation
#   --skip-summaries            Skip generation of summary files
#   -h,  --help                 Display this help message
#
# EXAMPLES:
#   # Process both branches
#   ./run_plateau_extraction.sh \
#       -i_pcac PCAC_mass_analysis.h5 \
#       -i_pion pion_effective_mass_analysis.h5
#
#   # With visualization for both branches
#   ./run_plateau_extraction.sh \
#       -i_pcac data/PCAC_mass_analysis.h5 \
#       -i_pion data/pion_effective_mass_analysis.h5 \
#       -p plots/ --enable-viz
#
#   # PCAC branch only
#   ./run_plateau_extraction.sh \
#       -i_pcac PCAC_mass_analysis.h5 \
#       -o output/ -p plots/ --enable-viz
#
#   # Pion branch only with custom output
#   ./run_plateau_extraction.sh \
#       -i_pion pion_effective_mass_analysis.h5 \
#       -o processed_data/
#
# DEPENDENCIES:
# - Python scripts: extract_plateau_PCAC_mass.py, extract_plateau_pion_mass.py,
#                   visualize_plateau_extraction.py
# - Library scripts in bash_scripts/library/
# - Python environment with qpb_data_analysis package
#
# INPUT FILES:
# - PCAC_mass_analysis.h5 (optional, from Stage 3.1)
#     Output from calculate_PCAC_mass.py
#     Contains PCAC mass time series with jackknife samples
#
# - pion_effective_mass_analysis.h5 (optional, from Stage 3.1)
#     Output from calculate_effective_mass.py
#     Contains pion effective mass time series with jackknife samples
#
# OUTPUT FILES:
# Data files (in output_directory):
# - plateau_PCAC_mass_estimates.csv        : PCAC plateau values (PCAC branch)
# - plateau_pion_mass_estimates.csv        : Pion plateau values (Pion branch)
#
# Visualization files (in output_directory/auxiliary/visualization/):
# - plateau_PCAC_mass_estimates.h5         : PCAC plateau data for viz
# - plateau_pion_mass_estimates.h5         : Pion plateau data for viz
#
# Summary files (in output_directory/auxiliary/summaries/):
# - plateau_PCAC_mass_estimates_tree.txt   : HDF5 structure summary
# - plateau_PCAC_mass_estimates_uniqueness_report.txt : CSV summary
# - (similar files for Pion branch)
#
# Plots (if --enable-viz, in plots_directory):
# - Plateau_extraction_pcac/               : PCAC plateau plots
# - Plateau_extraction_pion/               : Pion plateau plots
#
# - run_plateau_extraction.log             : Execution log
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
PLATEAU_SCRIPTS_DIR="${PYTHON_SCRIPTS_DIRECTORY}/analysis/plateau_extraction"
PCAC_PLATEAU_SCRIPT="${PLATEAU_SCRIPTS_DIR}/extract_plateau_PCAC_mass.py"
PION_PLATEAU_SCRIPT="${PLATEAU_SCRIPTS_DIR}/extract_plateau_pion_mass.py"
VIZ_SCRIPT="${PLATEAU_SCRIPTS_DIR}/visualize_plateau_extraction.py"

# Output filenames (using constants from constants.sh)
# These are explicitly passed to Python scripts to ensure consistency
PCAC_OUTPUT_CSV="$ANALYSIS_CSV_PCAC_PLATEAU"
PCAC_OUTPUT_HDF5="$ANALYSIS_HDF5_PCAC_PLATEAU"
PION_OUTPUT_CSV="$ANALYSIS_CSV_PION_PLATEAU"
PION_OUTPUT_HDF5="$ANALYSIS_HDF5_PION_PLATEAU"

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
Usage: $SCRIPT_NAME -i_pcac <pcac_hdf5> -i_pion <pion_hdf5> [options]

REQUIRED ARGUMENTS (at least one):
  -i_pcac, --input_pcac_hdf5      Path to PCAC mass analysis HDF5 file
  -i_pion, --input_pion_hdf5      Path to pion effective mass HDF5 file

Note: Provide at least one input file. If only one is given, only that
      branch will be processed.

OPTIONAL ARGUMENTS:
  -o, --output_directory          Output directory for CSV files
                                  (default: parent dir of first input)
  -p, --plots_directory           Directory for visualization plots
  -log_dir, --log_directory       Directory for log files
                                  (default: output_dir/auxiliary/logs/)
  
  VISUALIZATION:
  --enable-viz                    Generate plateau visualization plots
  
  OTHER OPTIONS:
  --skip-checks                   Skip intermediate validation
  --skip-summaries                Skip summary file generation
  -h, --help                      Display this help message

EXAMPLES:
  # Both branches
  $SCRIPT_NAME -i_pcac PCAC_mass_analysis.h5 -i_pion pion_effective_mass_analysis.h5

  # With visualization
  $SCRIPT_NAME -i_pcac PCAC.h5 -i_pion pion.h5 -p plots/ --enable-viz

  # PCAC only
  $SCRIPT_NAME -i_pcac PCAC_mass_analysis.h5 -o output/

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
    local input_pcac_hdf5=""
    local input_pion_hdf5=""
    local output_directory=""
    local plots_directory=""
    local log_directory=""
    
    # Control flags
    local enable_viz="false"
    local skip_checks="false"
    local skip_summaries="false"
    
    # Parse command-line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -i_pcac|--input_pcac_hdf5)
                input_pcac_hdf5="$2"
                shift 2
                ;;
            -i_pion|--input_pion_hdf5)
                input_pion_hdf5="$2"
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
    if [[ -z "$input_pcac_hdf5" && -z "$input_pion_hdf5" ]]; then
        echo "ERROR: At least one input HDF5 file is required" >&2
        echo "Use: $SCRIPT_NAME -i_pcac <file> and/or -i_pion <file>" >&2
        echo "Use -h or --help for usage information" >&2
        exit 1
    fi
    
    # Validate and convert input files to absolute paths
    if [[ -n "$input_pcac_hdf5" ]]; then
        input_pcac_hdf5="$(realpath "$input_pcac_hdf5")"
        if [[ ! -f "$input_pcac_hdf5" ]]; then
            echo "ERROR: PCAC HDF5 file not found: $input_pcac_hdf5" >&2
            exit 1
        fi
    fi
    
    if [[ -n "$input_pion_hdf5" ]]; then
        input_pion_hdf5="$(realpath "$input_pion_hdf5")"
        if [[ ! -f "$input_pion_hdf5" ]]; then
            echo "ERROR: Pion HDF5 file not found: $input_pion_hdf5" >&2
            exit 1
        fi
    fi
    
    # Set default output directory: parent directory of first provided input file
    if [[ -z "$output_directory" ]]; then
        if [[ -n "$input_pcac_hdf5" ]]; then
            output_directory="$(dirname "$input_pcac_hdf5")"
        else
            output_directory="$(dirname "$input_pion_hdf5")"
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
    
    log_info "=== Stage 3.2: Plateau Extraction Started ==="
    [[ -n "$input_pcac_hdf5" ]] && log_info "Input PCAC HDF5: $input_pcac_hdf5"
    [[ -n "$input_pion_hdf5" ]] && log_info "Input Pion HDF5: $input_pion_hdf5"
    log_info "Output directory: $output_directory"
    [[ -n "$plots_directory" ]] && log_info "Plots directory: $plots_directory"
    log_info "Log directory: $log_directory"
    
    # Track success/failure
    local pcac_success="false"
    local pion_success="false"
    local pcac_skipped="false"
    local pion_skipped="false"
    
    # Auxiliary directories
    local viz_dir="${output_directory}/${AUXILIARY_DIR_NAME}/${AUXILIARY_VIZ_SUBDIR}"
    local summary_dir="${output_directory}/${AUXILIARY_DIR_NAME}/${AUXILIARY_SUMMARIES_SUBDIR}"
    
    # =========================================================================
    # PCAC BRANCH
    # =========================================================================
    
    if [[ -n "$input_pcac_hdf5" ]]; then
        echo ""
        echo "=========================================="
        echo "PCAC Branch: Extracting Plateau Regions"
        echo "=========================================="
        log_info "Starting PCAC branch"
        
        # Validate Python script
        if [[ ! -f "$PCAC_PLATEAU_SCRIPT" ]]; then
            echo "ERROR: PCAC plateau extraction script not found: $PCAC_PLATEAU_SCRIPT" >&2
            log_error "PCAC plateau script not found: $PCAC_PLATEAU_SCRIPT"
            close_logging
            exit 1
        fi
        
        # Execute PCAC plateau extraction
        echo "  → Running extract_plateau_PCAC_mass.py..."
        execute_python_script "$PCAC_PLATEAU_SCRIPT" "extract_plateau_PCAC_mass" \
            --input_hdf5_file "$input_pcac_hdf5" \
            --output_directory "$output_directory" \
            --output_hdf5_filename "$PCAC_OUTPUT_HDF5" \
            --output_csv_filename "$PCAC_OUTPUT_CSV" \
            --enable_logging \
            --log_directory "$log_directory"
        
        if [[ $? -eq 0 ]]; then
            pcac_success="true"
            echo "  ✓ PCAC plateau extraction completed"
            log_info "PCAC plateau extraction successful"
            
            # Validate output files
            local pcac_csv_file="${output_directory}/${PCAC_OUTPUT_CSV}"
            local pcac_hdf5_file="${output_directory}/${PCAC_OUTPUT_HDF5}"
            
            if [[ "$skip_checks" != "true" ]]; then
                if [[ ! -f "$pcac_csv_file" ]]; then
                    echo "ERROR: PCAC CSV file not created: $pcac_csv_file" >&2
                    log_error "PCAC CSV file missing: $pcac_csv_file"
                    close_logging
                    exit 1
                fi
                if [[ ! -f "$pcac_hdf5_file" ]]; then
                    echo "ERROR: PCAC HDF5 file not created: $pcac_hdf5_file" >&2
                    log_error "PCAC HDF5 file missing: $pcac_hdf5_file"
                    close_logging
                    exit 1
                fi
                echo "  ✓ Output files validated"
            fi
            
            # Move HDF5 file to visualization directory
            echo "  → Moving HDF5 to visualization directory..."
            mv "$pcac_hdf5_file" "$viz_dir/" || {
                echo "ERROR: Failed to move PCAC HDF5 to visualization directory" >&2
                log_error "Failed to move PCAC HDF5 file"
                close_logging
                exit 1
            }
            local pcac_hdf5_viz="${viz_dir}/${PCAC_OUTPUT_HDF5}"
            echo "  ✓ HDF5 file relocated for visualization"
            log_info "PCAC HDF5 moved to: $pcac_hdf5_viz"
            
            # Generate summaries
            if [[ "$skip_summaries" != "true" ]]; then
                echo "  → Generating summaries..."
                
                # CSV summary
                generate_csv_summary "$pcac_csv_file" "$summary_dir"
                if [[ $? -eq 0 ]]; then
                    echo "  → CSV summary generated"
                    log_info "PCAC CSV summary created in: $summary_dir"
                else
                    echo "  ⚠ Warning: CSV summary generation failed" >&2
                    log_warning "Failed to generate PCAC CSV summary"
                fi
                
                # HDF5 tree summary
                generate_hdf5_tree_summary "$pcac_hdf5_viz" "$summary_dir"
                if [[ $? -eq 0 ]]; then
                    echo "  → HDF5 tree summary generated"
                    log_info "PCAC HDF5 tree summary created in: $summary_dir"
                else
                    echo "  ⚠ Warning: HDF5 tree summary generation failed" >&2
                    log_warning "Failed to generate PCAC HDF5 tree summary"
                fi
                
                echo "  ✓ Summaries generated"
            fi
            
            # Optional visualization
            if [[ "$enable_viz" == "true" ]]; then
                if [[ -z "$plots_directory" ]]; then
                    echo "  ⚠ Warning: Visualization requested but no plots directory specified" >&2
                    log_warning "PCAC visualization skipped: no plots directory"
                elif [[ ! -f "$VIZ_SCRIPT" ]]; then
                    echo "  ⚠ Warning: Visualization script not found: $VIZ_SCRIPT" >&2
                    log_warning "Visualization script not found"
                else
                    echo "  → Generating PCAC plateau visualizations..."
                    
                    # Validate plots directory exists (user must create it)
                    if [[ ! -d "$plots_directory" ]]; then
                        echo "  ⚠ Warning: Plots directory does not exist: $plots_directory" >&2
                        echo "  → Please create it to enable visualization" >&2
                        log_warning "PCAC visualization skipped: plots directory does not exist"
                    else
                        execute_python_script "$VIZ_SCRIPT" "visualize_plateau_extraction" \
                            --analysis_type pcac \
                            --results_hdf5 "$pcac_hdf5_viz" \
                            --plots_directory "$plots_directory" \
                            --enable_logging \
                            --log_directory "$log_directory"
                        
                        if [[ $? -eq 0 ]]; then
                            echo "  ✓ PCAC visualizations generated"
                            log_info "PCAC visualization completed"
                        else
                            echo "  ⚠ Warning: PCAC visualization failed" >&2
                            log_warning "PCAC visualization failed"
                        fi
                    fi
                fi
            fi
        else
            echo "ERROR: PCAC plateau extraction failed" >&2
            log_error "PCAC plateau extraction failed"
            close_logging
            exit 1
        fi
    else
        pcac_skipped="true"
        echo "○ PCAC branch skipped (no input file provided)"
        log_info "PCAC branch skipped: no input file"
    fi
    
    # =========================================================================
    # PION BRANCH
    # =========================================================================
    
    if [[ -n "$input_pion_hdf5" ]]; then
        echo ""
        echo "=========================================="
        echo "Pion Branch: Extracting Plateau Regions"
        echo "=========================================="
        log_info "Starting Pion branch"
        
        # Validate Python script
        if [[ ! -f "$PION_PLATEAU_SCRIPT" ]]; then
            echo "ERROR: Pion plateau extraction script not found: $PION_PLATEAU_SCRIPT" >&2
            log_error "Pion plateau script not found: $PION_PLATEAU_SCRIPT"
            close_logging
            exit 1
        fi
        
        # Execute pion plateau extraction
        echo "  → Running extract_plateau_pion_mass.py..."
        execute_python_script "$PION_PLATEAU_SCRIPT" "extract_plateau_pion_mass" \
            --input_hdf5_file "$input_pion_hdf5" \
            --output_directory "$output_directory" \
            --output_hdf5_filename "$PION_OUTPUT_HDF5" \
            --output_csv_filename "$PION_OUTPUT_CSV" \
            --enable_logging \
            --log_directory "$log_directory"
        
        if [[ $? -eq 0 ]]; then
            pion_success="true"
            echo "  ✓ Pion plateau extraction completed"
            log_info "Pion plateau extraction successful"
            
            # Validate output files
            local pion_csv_file="${output_directory}/${PION_OUTPUT_CSV}"
            local pion_hdf5_file="${output_directory}/${PION_OUTPUT_HDF5}"
            
            if [[ "$skip_checks" != "true" ]]; then
                if [[ ! -f "$pion_csv_file" ]]; then
                    echo "ERROR: Pion CSV file not created: $pion_csv_file" >&2
                    log_error "Pion CSV file missing: $pion_csv_file"
                    close_logging
                    exit 1
                fi
                if [[ ! -f "$pion_hdf5_file" ]]; then
                    echo "ERROR: Pion HDF5 file not created: $pion_hdf5_file" >&2
                    log_error "Pion HDF5 file missing: $pion_hdf5_file"
                    close_logging
                    exit 1
                fi
                echo "  ✓ Output files validated"
            fi
            
            # Move HDF5 file to visualization directory
            echo "  → Moving HDF5 to visualization directory..."
            mv "$pion_hdf5_file" "$viz_dir/" || {
                echo "ERROR: Failed to move Pion HDF5 to visualization directory" >&2
                log_error "Failed to move Pion HDF5 file"
                close_logging
                exit 1
            }
            local pion_hdf5_viz="${viz_dir}/${PION_OUTPUT_HDF5}"
            echo "  ✓ HDF5 file relocated for visualization"
            log_info "Pion HDF5 moved to: $pion_hdf5_viz"
            
            # Generate summaries
            if [[ "$skip_summaries" != "true" ]]; then
                echo "  → Generating summaries..."
                
                # CSV summary
                generate_csv_summary "$pion_csv_file" "$summary_dir"
                if [[ $? -eq 0 ]]; then
                    echo "  → CSV summary generated"
                    log_info "Pion CSV summary created in: $summary_dir"
                else
                    echo "  ⚠ Warning: CSV summary generation failed" >&2
                    log_warning "Failed to generate Pion CSV summary"
                fi
                
                # HDF5 tree summary
                generate_hdf5_tree_summary "$pion_hdf5_viz" "$summary_dir"
                if [[ $? -eq 0 ]]; then
                    echo "  → HDF5 tree summary generated"
                    log_info "Pion HDF5 tree summary created in: $summary_dir"
                else
                    echo "  ⚠ Warning: HDF5 tree summary generation failed" >&2
                    log_warning "Failed to generate Pion HDF5 tree summary"
                fi
                
                echo "  ✓ Summaries generated"
            fi
            
            # Optional visualization
            if [[ "$enable_viz" == "true" ]]; then
                if [[ -z "$plots_directory" ]]; then
                    echo "  ⚠ Warning: Visualization requested but no plots directory specified" >&2
                    log_warning "Pion visualization skipped: no plots directory"
                elif [[ ! -f "$VIZ_SCRIPT" ]]; then
                    echo "  ⚠ Warning: Visualization script not found: $VIZ_SCRIPT" >&2
                    log_warning "Visualization script not found"
                else
                    echo "  → Generating pion plateau visualizations..."
                    
                    # Validate plots directory exists (user must create it)
                    if [[ ! -d "$plots_directory" ]]; then
                        echo "  ⚠ Warning: Plots directory does not exist: $plots_directory" >&2
                        echo "  → Please create it to enable visualization" >&2
                        log_warning "Pion visualization skipped: plots directory does not exist"
                    else
                        execute_python_script "$VIZ_SCRIPT" "visualize_plateau_extraction" \
                            --analysis_type pion \
                            --results_hdf5 "$pion_hdf5_viz" \
                            --plots_directory "$plots_directory" \
                            --enable_logging \
                            --log_directory "$log_directory"
                        
                        if [[ $? -eq 0 ]]; then
                            echo "  ✓ Pion visualizations generated"
                            log_info "Pion visualization completed"
                        else
                            echo "  ⚠ Warning: Pion visualization failed" >&2
                            log_warning "Pion visualization failed"
                        fi
                    fi
                fi
            fi
        else
            echo "ERROR: Pion plateau extraction failed" >&2
            log_error "Pion plateau extraction failed"
            close_logging
            exit 1
        fi
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
    echo "Stage 3.2 Completion Summary"
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
    
    log_info "=== Stage 3.2: Plateau Extraction Completed ==="
    close_logging
    
    echo ""
    echo "✓ Stage 3.2 (Plateau Extraction) completed successfully"
    
    return 0
}

# Execute main function
main "$@"
