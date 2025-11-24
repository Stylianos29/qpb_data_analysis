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

# Timestamp script identifier
TIMESTAMP_SCRIPT_NAME="run_complete_pipeline"

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
                               Base directory for plots (enables visualization)
                               (default: none - visualization disabled)

  -log_dir, --log_base_directory
                               Base directory for log files
                               (default: uses output directories)

  --all                        Force reprocessing of ALL data sets,
                               bypassing timestamp checks

  --force                      Synonym for --all (force reprocessing)

  --skip_checks                Skip intermediate file validation
                               (passed to single-set pipeline)

  --skip_summaries             Skip generation of summary files
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

EXAMPLES:
  # Process all data sets under default raw directory
  $SCRIPT_NAME

  # Process all data sets under a specific directory
  $SCRIPT_NAME -i /path/to/raw/data/

  # Process only data sets under the 'invert' program directory
  $SCRIPT_NAME -i ../data_files/raw/invert/

  # Process single data set (batch mode with one set)
  $SCRIPT_NAME -i ../data_files/raw/invert/my_experiment/

  # Force reprocess all data sets
  $SCRIPT_NAME --all

  # Process with visualization enabled
  $SCRIPT_NAME -plots_dir ../output/plots/

  # Process with custom output location
  $SCRIPT_NAME -i ../raw_data/ -o ../processed_data/

  # Fast processing (skip checks and summaries)
  $SCRIPT_NAME --skip_checks --skip_summaries

NOTES:
  - The script is non-blocking: failures in one data set don't stop others
  - Each data set runs independently via run_complete_pipeline.sh
  - Progress is reported for each data set processed
  - Summary statistics are displayed at the end

EOF
    exit 0
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
        echo "ERROR: Input directory does not exist: $input_base_dir" >&2
        return 1
    fi
    
    # Find all directories that contain .txt or .dat files
    # This identifies actual data set directories vs. organizational directories
    while IFS= read -r -d '' dir; do
        # Check if directory contains .txt or .dat files
        if find "$dir" -maxdepth 1 -type f \( -name "*.txt" -o -name "*.dat" \) -print -quit | grep -q .; then
            discovered_sets_arr+=("$dir")
        fi
    done < <(find "$input_base_dir" -type d -print0)
    
    return 0
}


function should_process_data_set() {
    # Determine if a data set should be processed based on timestamp
    #
    # Arguments:
    #   $1 - data_set_dir       : Path to data set directory
    #   $2 - timestamp_file     : Path to timestamp file
    #   $3 - force_flag         : Boolean flag (true/false) to bypass timestamp
    #
    # Returns:
    #   0 - Should process (modified or force flag set)
    #   1 - Should skip (unmodified and no force flag)
    
    local data_set_dir="$1"
    local timestamp_file="$2"
    local force_flag="$3"
    
    # If force flag set, always process
    if [[ "$force_flag" == "true" ]]; then
        return 0
    fi
    
    # If timestamp file doesn't exist, must process
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


function process_single_data_set() {
    # Process a single data set by calling run_complete_pipeline.sh
    #
    # Arguments:
    #   $1 - data_set_dir     : Input data set directory
    #   $2 - output_dir       : Output directory for this data set
    #   $3 - plots_base_dir   : Base plots directory (optional)
    #   $4 - log_base_dir     : Base log directory (optional)
    #   $5 - skip_checks      : Boolean flag (true/false)
    #   $6 - skip_summaries   : Boolean flag (true/false)
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    
    local data_set_dir="$1"
    local output_dir="$2"
    local plots_base_dir="$3"
    local log_base_dir="$4"
    local skip_checks="$5"
    local skip_summaries="$6"
    
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
        # For logs, we can just use the output directory
        # or create a mirrored structure
        cmd+=" -log_dir \"$output_dir\""
    fi
    
    # Add skip flags
    if [[ "$skip_checks" == "true" ]]; then
        cmd+=" --skip_checks"
    fi
    
    if [[ "$skip_summaries" == "true" ]]; then
        cmd+=" --skip_summaries"
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
force_all=false
skip_checks=false
skip_summaries=false

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
        -log_dir|--log_base_directory)
            log_base_dir="$2"
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

# Handle plots directory if specified
if [[ -n "$plots_base_dir" ]]; then
    if [[ ! -d "$plots_base_dir" ]]; then
        mkdir -p "$plots_base_dir" || {
            echo "ERROR: Failed to create plots directory: $plots_base_dir" >&2
            exit 1
        }
        echo "INFO: Created plots directory: $(get_display_path "$plots_base_dir")"
    fi
    plots_base_dir="$(realpath "$plots_base_dir")"
    echo "INFO: Visualization enabled - plots will be saved to $(get_display_path "$plots_base_dir")"
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
for data_set_dir in "${data_sets[@]}"; do
    data_set_name=$(basename "$data_set_dir")
    
    # Compute output directory
    output_dir=$(compute_output_directory "$data_set_dir" "$input_base_dir" "$output_base_dir")
    
    # Ensure output directory exists
    check_if_directory_exists "$output_dir" -c -s
    
    # Setup auxiliary directory for timestamps
    auxiliary_dir="${output_dir}/auxiliary"
    check_if_directory_exists "$auxiliary_dir" -c -s
    
    # Get timestamp file path
    timestamp_file=$(get_timestamp_file_path \
        "$data_set_dir" \
        "$auxiliary_dir" \
        "$TIMESTAMP_SCRIPT_NAME")
    
    # Ensure timestamp file exists
    check_if_file_exists "$timestamp_file" -c -s
    
    # Check if should process
    if ! should_process_data_set "$data_set_dir" "$timestamp_file" "$force_all"; then
        echo "⊘ SKIPPING: $data_set_name (no changes detected)"
        ((skipped_count++))
        continue
    fi
    
    # Process the data set
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ PROCESSING [$((processed_count + 1))/$total_sets]: $data_set_name"
    echo "  Input:  $(get_display_path "$data_set_dir")"
    echo "  Output: $(get_display_path "$output_dir")"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    ((processed_count++))
    
    # Process the data set
    if process_single_data_set \
        "$data_set_dir" \
        "$output_dir" \
        "$plots_base_dir" \
        "$log_base_dir" \
        "$skip_checks" \
        "$skip_summaries"; then
        
        # Success - update timestamp
        update_timestamp "$data_set_dir" "$timestamp_file"
        ((success_count++))
        echo ""
        echo "✓ SUCCESS: $data_set_name completed successfully"
    else
        # Failure - don't update timestamp (will retry next time)
        ((failure_count++))
        echo ""
        echo "✗ FAILURE: $data_set_name failed (will retry on next batch run)"
    fi
    
    echo ""
done

# =============================================================================
# SECTION 8: SUMMARY REPORT
# =============================================================================

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

# Exit with appropriate code
if [[ $failure_count -gt 0 ]]; then
    echo "WARNING: Some data sets failed processing"
    exit 1
elif [[ $processed_count -eq 0 ]]; then
    echo "INFO: No data sets required processing"
    exit 0
else
    echo "INFO: All processed data sets completed successfully"
    exit 0
fi
