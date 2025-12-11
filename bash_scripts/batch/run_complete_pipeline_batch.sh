#!/bin/bash

################################################################################
# run_complete_pipeline_batch.sh - Batch execution of complete pipeline on 
# multiple data file sets
#
# DESCRIPTION:
# Master batch orchestrator that runs the complete QPB data analysis pipeline
# on multiple data file sets. Supports both automatic discovery and targeted
# processing with intelligent timestamp-based caching.
#
# HYBRID INPUT APPROACH:
# - If -i specified: Process all data sets found under that directory tree
# - If -i omitted:   Default to processing all sets under ../data_files/raw/
#
# TIMESTAMP-BASED CACHING:
# - Tracks processing status per data set using timestamp files
# - Skips data sets that haven't been modified since last successful run
# - Use --all flag to force reprocessing all data sets
# - Use --force flag (synonym for --all) for compatibility
#
# SELECTIVE STAGE EXECUTION:
# - Use --stages flag to run only specific pipeline stages (1, 2, or 3)
# - Gracefully skips data sets that lack required inputs for requested stages
# - In batch mode, missing inputs cause skip (not error) for robustness
#
# PIPELINE BEHAVIOR:
# For each data set discovered:
# - Detects data set type (correlators vs. parameters-only)
# - Executes appropriate pipeline stages via run_complete_pipeline.sh
# - Updates timestamp on successful completion
# - Continues to next data set on failure (non-blocking)
#
# DIRECTORY STRUCTURE:
# The script expects data organized as:
#   input_base_dir/
#   ├── main_program_1/
#   │   ├── data_set_A/
#   │   │   ├── *.txt (log files)
#   │   │   └── *.dat (optional correlator files)
#   │   └── data_set_B/
#   │       └── ...
#   └── main_program_2/
#       └── ...
#
# USAGE:
#   ./run_complete_pipeline_batch.sh [-i <directory>] [options]
#
# For detailed usage, run with -h or --help
#
################################################################################

# =============================================================================
# SECTION 1: BASIC PATH SETUP
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

# Single-set pipeline script
SINGLE_SET_SCRIPT="$(realpath "${SCRIPT_DIR}/../single_set/run_complete_pipeline.sh")"

# Default directories (relative to script location)
DEFAULT_INPUT_BASE_DIR="$(realpath "${SCRIPT_DIR}/../../data_files/raw")"
DEFAULT_OUTPUT_BASE_DIR="$(realpath "${SCRIPT_DIR}/../../data_files/processed")"
DEFAULT_PLOTS_BASE_DIR="$(realpath "${SCRIPT_DIR}/../../output/plots")"

# Auxiliary directory structure
AUXILIARY_TIMESTAMPS_SUBDIR="timestamps"

# Timestamp script identifier
TIMESTAMP_SCRIPT_NAME="run_complete_pipeline"

# Expected output filenames (for stage validation)
PARSED_CSV_FILENAME="single_valued_parameters.csv"
PARSED_HDF5_LOG_FILENAME="multivalued_parameters.h5"
PROCESSED_CSV_FILENAME="processed_parameter_values.csv"
JACKKNIFE_HDF5_FILENAME="correlators_jackknife_analysis.h5"

# =============================================================================
# SECTION 4: FUNCTION DEFINITIONS
# =============================================================================

function usage() {
    cat << EOF
Usage: $SCRIPT_NAME [-i <input_base_directory>] [options]

DESCRIPTION:
  Batch execution of the complete QPB data analysis pipeline on multiple
  data file sets. Automatically discovers and processes all data sets under
  the specified input directory (or default raw data directory).

OPTIONAL ARGUMENTS:
  -i, --input_base_directory   Base directory containing data sets
                               (default: ../../data_files/raw/)
                               Can be set to any level:
                               - All data:     ../data_files/raw/
                               - One program:  ../data_files/raw/invert/
                               - One set:      ../data_files/raw/invert/experiment1/

  -o, --output_base_directory  Base output directory for processed data
                               (default: ../../data_files/processed/)
                               Output structure mirrors input structure

  -plots_dir, --plots_base_directory
                               Base directory for plots
                               (default: ../../output/plots/ - AUTO-ENABLED)
                               Visualization is automatically enabled in batch mode

  --no-plots                   Disable visualization (faster processing)

  -log_dir, --log_base_directory
                               Base directory for log files
                               (default: uses output directories)

  --stages <1,2,3>            Comma-separated stages to execute (default: all)
                               Examples: --stages 2,3 (rerun processing+analysis)
                                        --stages 3 (rerun analysis only)
                               Note: In batch mode, data sets lacking required
                               inputs are skipped gracefully (not errors)

  --all                        Force reprocessing of ALL data sets,
                               bypassing timestamp checks

  --force                      Synonym for --all (force reprocessing)

  --skip_checks                Skip intermediate file validation
                               (passed to single-set pipeline)

  --skip_summaries             Skip generation of summary files
                               (passed to single-set pipeline)

  --clean-plots                Remove existing plots before generating new ones
                               (passed to single-set pipeline)

  -h, --help                   Display this help message

TIMESTAMP CACHING:
  The script tracks the processing status of each data set using timestamp
  files stored in the output directory structure:
    output_dir/data_set_name/auxiliary/run_complete_pipeline.timestamp

  Behavior:
  - Data sets are skipped if unmodified since last successful run
  - Failed runs don't update timestamps (will retry on next batch run)
  - Use --all or --force to ignore timestamps and reprocess everything

SELECTIVE STAGE EXECUTION:
  When using --stages flag:
  - Stage 1: Requires raw .txt files in input directory
  - Stage 2: Requires Stage 1 outputs (single_valued_parameters.csv, etc.)
  - Stage 3: Requires Stage 2 outputs + correlators (jackknife HDF5)
  
  Batch mode behavior:
  - Data sets lacking required inputs are SKIPPED (not errors)
  - Allows mixed processing across heterogeneous data sets
  - Example: --stages 3 skips parameter-only data sets automatically

EXAMPLES:
  # Process all data sets under default raw directory
  $SCRIPT_NAME

  # Process all data sets under a specific directory
  $SCRIPT_NAME -i /path/to/raw/data/

  # Process only data sets under the 'invert' program directory
  $SCRIPT_NAME -i ../data_files/raw/invert/

  # Rerun only processing and analysis (skip parsing)
  $SCRIPT_NAME --stages 2,3

  # Rerun only analysis (requires existing Stage 2 outputs)
  $SCRIPT_NAME --stages 3

  # Force reprocess all data sets
  $SCRIPT_NAME --all

  # Disable visualization for faster batch processing
  $SCRIPT_NAME --no-plots

  # Fast batch processing (no checks, summaries, or plots)
  $SCRIPT_NAME --no-plots --skip_checks --skip_summaries

NOTES:
  - The script is non-blocking: failures in one data set don't stop others
  - Each data set runs independently via run_complete_pipeline.sh
  - Progress is reported for each data set processed
  - Summary statistics are displayed at the end
  - Visualization is auto-enabled by default (use --no-plots to disable)

EOF
    exit 0
}

function validate_stages_argument() {
    # Validate and normalize the --stages argument
    #
    # Arguments:
    #   $1 - stages : Comma-separated stage numbers
    #
    # Returns:
    #   Prints normalized stages string (sorted, deduplicated)
    #   Exit code 0 on success, 1 on validation failure
    
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

function discover_data_sets() {
    # Discover all data set directories under input base directory
    #
    # Traverses the directory tree to find valid data set directories.
    # A valid data set directory is a leaf directory (no subdirectories)
    # or a directory containing .txt or .dat files.
    #
    # Arguments:
    #   $1 - input_base_dir      : Base directory to search
    #   $2 - discovered_sets_arr : Name of array to store results
    #
    # Returns:
    #   Populates the named array with discovered data set paths
    
    local input_base_dir="$1"
    local -n discovered_sets_arr="$2"
    
    # Clear the output array
    discovered_sets_arr=()
    
    # Check if input directory exists
    if [[ ! -d "$input_base_dir" ]]; then
        return 1
    fi
    
    # Find all directories containing .txt or .dat files
    while IFS= read -r -d '' dir; do
        # Check if directory contains .txt or .dat files
        if find "$dir" -maxdepth 1 -type f \( -name "*.txt" -o -name "*.dat" \) -print -quit | grep -q .; then
            discovered_sets_arr+=("$dir")
        fi
    done < <(find "$input_base_dir" -type d -print0)
    
    return 0
}

function should_process_data_set() {
    # Determine if a data set should be processed based on timestamps
    #
    # Arguments:
    #   $1 - data_set_dir   : Data set directory
    #   $2 - timestamp_file : Path to timestamp file
    #
    # Returns:
    #   0 - Should process (modified or force mode)
    #   1 - Should skip (not modified)
    
    local data_set_dir="$1"
    local timestamp_file="$2"
    
    # Force mode: always process
    if $force_all; then
        return 0
    fi
    
    # No timestamp file: must process
    if [[ ! -f "$timestamp_file" ]]; then
        return 0
    fi
    
    # Check if directory has been modified since timestamp
    if check_directory_for_changes "$data_set_dir" "$timestamp_file"; then
        return 0  # Modified - should process
    else
        return 1  # Not modified - skip
    fi
}

function compute_output_directory() {
    # Compute output directory path by mirroring input structure
    #
    # Arguments:
    #   $1 - data_set_dir       : Input data set directory
    #   $2 - input_base_dir     : Input base directory
    #   $3 - output_base_dir    : Output base directory
    #
    # Returns:
    #   Prints the computed output directory path
    
    local data_set_dir="$1"
    local input_base_dir="$2"
    local output_base_dir="$3"
    
    # Use replace_parent_directory to mirror structure
    local output_dir
    output_dir=$(replace_parent_directory \
        "$data_set_dir" \
        "$input_base_dir" \
        "$output_base_dir")
    
    echo "$output_dir"
}

function can_process_stage() {
    # Check if a data set can satisfy the requirements for a given stage
    # Used in batch mode for graceful skipping of incompatible data sets
    #
    # Arguments:
    #   $1 - stage      : Stage number (1, 2, or 3)
    #   $2 - output_dir : Output directory for the data set
    #
    # Returns:
    #   0 - Can process this stage
    #   1 - Cannot process (missing inputs)
    
    local stage="$1"
    local output_dir="$2"
    
    case "$stage" in
        1)
            # Stage 1 always valid (needs raw files, checked elsewhere)
            return 0
            ;;
        2)
            # Stage 2 needs Stage 1 outputs
            local csv="${output_dir}/${PARSED_CSV_FILENAME}"
            local hdf5="${output_dir}/${PARSED_HDF5_LOG_FILENAME}"
            
            if [[ -f "$csv" && -f "$hdf5" ]]; then
                return 0
            else
                return 1
            fi
            ;;
        3)
            # Stage 3 needs Stage 2 outputs + correlators
            local csv="${output_dir}/${PROCESSED_CSV_FILENAME}"
            local hdf5="${output_dir}/${JACKKNIFE_HDF5_FILENAME}"
            
            if [[ -f "$csv" && -f "$hdf5" ]]; then
                return 0
            else
                return 1
            fi
            ;;
        *)
            return 1
            ;;
    esac
}

function process_single_data_set() {
    # Process a single data set by calling run_complete_pipeline.sh
    #
    # Arguments:
    #   $1 - data_set_dir     : Input data set directory
    #   $2 - output_dir       : Output directory for this data set
    #   $3 - plots_base_dir   : Base plots directory (optional)
    #   $4 - log_base_dir     : Base log directory (optional)
    #   $5 - stages_to_run    : Comma-separated stages (optional)
    #   $6 - skip_checks      : Boolean flag (true/false)
    #   $7 - skip_summaries   : Boolean flag (true/false)
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    
    local data_set_dir="$1"
    local output_dir="$2"
    local plots_base_dir="$3"
    local log_base_dir="$4"
    local stages_to_run="$5"
    local skip_checks="$6"
    local skip_summaries="$7"
    
    # Build command
    local cmd="$SINGLE_SET_SCRIPT"
    cmd+=" -i \"$data_set_dir\""
    cmd+=" -o \"$output_dir\""
    
    # Add plots directory if specified
    if [[ -n "$plots_base_dir" ]]; then
        # Mirror the structure for plots as well
        local plots_dir
        plots_dir=$(replace_parent_directory \
            "$data_set_dir" \
            "$input_base_dir" \
            "$plots_base_dir")
        cmd+=" -plots_dir \"$plots_dir\""
    fi
    
    # Add log directory if specified
    if [[ -n "$log_base_dir" ]]; then
        cmd+=" -log_dir \"$output_dir\""
    fi
    
    # Add stages flag if specified
    if [[ -n "$stages_to_run" ]]; then
        cmd+=" --stages \"$stages_to_run\""
    fi
    
    # Add skip flags
    if [[ "$skip_checks" == "true" ]]; then
        cmd+=" --skip_checks"
    fi
    
    if [[ "$skip_summaries" == "true" ]]; then
        cmd+=" --skip_summaries"
    fi
    
    if [[ "$clean_plots" == "true" ]]; then
        cmd+=" --clean-plots"
    fi

    # Execute the command
    if eval "$cmd"; then
        return 0
    else
        return 1
    fi
}

# =============================================================================
# SECTION 5: ARGUMENT PARSING
# =============================================================================

# Initialize variables with defaults
input_base_dir=""
output_base_dir=""
plots_base_dir=""
log_base_dir=""
stages_to_run=""
force_all=false
skip_checks=false
skip_summaries=false
disable_plots=false
clean_plots=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input_base_directory)
            input_base_dir="$2"
            shift 2
            ;;
        -o|--output_base_directory)
            output_base_dir="$2"
            shift 2
            ;;
        -plots_dir|--plots_base_directory)
            plots_base_dir="$2"
            shift 2
            ;;
        --no-plots)
            disable_plots=true
            shift
            ;;
        -log_dir|--log_base_directory)
            log_base_dir="$2"
            shift 2
            ;;
        --stages)
            stages_to_run=$(validate_stages_argument "$2") || exit 1
            shift 2
            ;;
        --all|--force)
            force_all=true
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
# SECTION 6: INPUT VALIDATION AND SETUP
# =============================================================================

# Set default input directory if not specified
if [[ -z "$input_base_dir" ]]; then
    input_base_dir="$DEFAULT_INPUT_BASE_DIR"
    echo "INFO: Using default input directory: $(get_display_path "$input_base_dir")"
fi

# Validate and resolve input directory
if [[ ! -d "$input_base_dir" ]]; then
    echo "ERROR: Input directory does not exist: $input_base_dir" >&2
    exit 1
fi
input_base_dir="$(realpath "$input_base_dir")"

# Set default output directory if not specified
if [[ -z "$output_base_dir" ]]; then
    # Try to mirror input structure
    if [[ "$input_base_dir" == *"/data_files/raw"* ]]; then
        output_base_dir="${input_base_dir/\/raw/\/processed}"
        echo "INFO: Auto-detected output directory: $(get_display_path "$output_base_dir")"
    else
        output_base_dir="$DEFAULT_OUTPUT_BASE_DIR"
        echo "INFO: Using default output directory: $(get_display_path "$output_base_dir")"
    fi
fi

# Ensure output base directory exists
if [[ ! -d "$output_base_dir" ]]; then
    mkdir -p "$output_base_dir" || {
        echo "ERROR: Failed to create output directory: $output_base_dir" >&2
        exit 1
    }
    echo "INFO: Created output directory: $(get_display_path "$output_base_dir")"
fi
output_base_dir="$(realpath "$output_base_dir")"

# Handle plots directory (auto-enable unless --no-plots specified)
if ! $disable_plots; then
    if [[ -z "$plots_base_dir" ]]; then
        plots_base_dir="$DEFAULT_PLOTS_BASE_DIR"
        echo "INFO: Auto-enabling visualization with plots directory: $(get_display_path "$plots_base_dir")"
    fi
    
    # Ensure plots base directory exists
    if [[ ! -d "$plots_base_dir" ]]; then
        mkdir -p "$plots_base_dir" || {
            echo "ERROR: Failed to create plots directory: $plots_base_dir" >&2
            exit 1
        }
        echo "INFO: Created plots directory: $(get_display_path "$plots_base_dir")"
    fi
    plots_base_dir="$(realpath "$plots_base_dir")"
else
    plots_base_dir=""  # Explicitly disable
    echo "INFO: Visualization disabled (--no-plots)"
fi

# Validate single-set script exists
if [[ ! -f "$SINGLE_SET_SCRIPT" ]]; then
    echo "ERROR: Single-set pipeline script not found: $SINGLE_SET_SCRIPT" >&2
    exit 1
fi

# =============================================================================
# SECTION 7: MAIN EXECUTION
# =============================================================================

# Display banner
echo ""
echo "==================================================================="
echo "   QPB DATA ANALYSIS - BATCH PIPELINE EXECUTION"
echo "==================================================================="
echo "Input base:  $(get_display_path "$input_base_dir")"
echo "Output base: $(get_display_path "$output_base_dir")"

if [[ -n "$plots_base_dir" ]]; then
    echo "Plots base:  $(get_display_path "$plots_base_dir")"
    echo "Visualization: ENABLED (automatic in batch mode)"
else
    echo "Visualization: DISABLED (--no-plots)"
fi

if [[ -n "$stages_to_run" ]]; then
    echo "Stages: $stages_to_run (selective execution)"
else
    echo "Stages: 1,2,3 (full pipeline)"
fi

if [[ "$force_all" == "true" ]]; then
    echo "Mode: FORCE ALL (ignoring timestamps)"
else
    echo "Mode: INCREMENTAL (using timestamps)"
fi
echo "==================================================================="
echo ""

# Discover all data sets
echo "Discovering data sets..."
data_sets=()
if ! discover_data_sets "$input_base_dir" data_sets; then
    echo "ERROR: Failed to discover data sets" >&2
    exit 1
fi

total_sets=${#data_sets[@]}
if [[ $total_sets -eq 0 ]]; then
    echo "WARNING: No data sets found under $(get_display_path "$input_base_dir")"
    echo "Data sets should contain .txt or .dat files"
    exit 0
fi

echo "Found $total_sets data set(s) to process"
echo ""

# Initialize counters
processed_count=0
skipped_count=0
success_count=0
failure_count=0

# Process each data set
set_index=0
for data_set_dir in "${data_sets[@]}"; do
    ((set_index++))
    data_set_name=$(basename "$data_set_dir")
    
    # Compute output directory
    output_dir=$(compute_output_directory "$data_set_dir" "$input_base_dir" "$output_base_dir")
    
    # Ensure output directory exists
    check_if_directory_exists "$output_dir" -c -s
    
    # Setup auxiliary directory for timestamps
    auxiliary_dir="${output_dir}/auxiliary"
    check_if_directory_exists "$auxiliary_dir" -c -s
        
    # Setup timestamps subdirectory
    timestamps_dir="${auxiliary_dir}/${AUXILIARY_TIMESTAMPS_SUBDIR}"
    check_if_directory_exists "$timestamps_dir" -c -s

    # Get timestamp file path
    timestamp_file=$(get_timestamp_file_path \
        "$data_set_dir" \
        "$timestamps_dir" \
        "$TIMESTAMP_SCRIPT_NAME")
    
    # Ensure timestamp file exists
    check_if_file_exists "$timestamp_file" -c -s
    
    # Check if should process based on timestamps
    if ! should_process_data_set "$data_set_dir" "$timestamp_file"; then
        echo "⊘ SKIPPING: $data_set_name (no changes detected)"
        ((skipped_count++))
        continue
    fi
    
    # If selective stages requested, check if this data set can satisfy them
    if [[ -n "$stages_to_run" ]]; then
        can_process=true
        
        # Check each requested stage
        if should_run_stage 2; then
            if ! can_process_stage 2 "$output_dir"; then
                echo "⊘ SKIPPING: $data_set_name (Stage 2 requested but missing Stage 1 outputs)"
                ((skipped_count++))
                continue
            fi
        fi
        
        if should_run_stage 3; then
            if ! can_process_stage 3 "$output_dir"; then
                echo "⊘ SKIPPING: $data_set_name (Stage 3 requested but missing Stage 2 outputs or no correlators)"
                ((skipped_count++))
                continue
            fi
        fi
    fi
    
    # Process this data set
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ PROCESSING [$set_index/$total_sets]: $data_set_name"
    echo "  Input:  $(get_display_path "$data_set_dir")"
    echo "  Output: $(get_display_path "$output_dir")"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    ((processed_count++))
    
    # Execute single-set pipeline
    if process_single_data_set \
        "$data_set_dir" \
        "$output_dir" \
        "$plots_base_dir" \
        "$log_base_dir" \
        "$stages_to_run" \
        "$skip_checks" \
        "$skip_summaries"; then
        
        # Success - update timestamp
        update_timestamp "$data_set_dir" "$timestamp_file"
        ((success_count++))
        echo ""
        echo "✓ SUCCESS: $data_set_name completed successfully"
        echo ""
    else
        # Failure - don't update timestamp
        ((failure_count++))
        echo ""
        echo "✗ FAILED: $data_set_name encountered errors"
        echo ""
        # Continue to next data set (non-blocking)
    fi
done

# Display final summary
echo ""
echo "==================================================================="
echo "   BATCH PROCESSING COMPLETE"
echo "==================================================================="
echo "Total data sets found:    $total_sets"
echo "Processed:                $processed_count"
echo "Skipped (no changes):     $skipped_count"
echo "Successful:               $success_count"
echo "Failed:                   $failure_count"
echo "==================================================================="

if [[ $success_count -eq $processed_count && $processed_count -gt 0 ]]; then
    echo "INFO: All processed data sets completed successfully"
    exit 0
elif [[ $failure_count -gt 0 ]]; then
    echo "WARNING: Some data sets failed processing"
    exit 1
else
    echo "INFO: Batch processing complete"
    exit 0
fi
