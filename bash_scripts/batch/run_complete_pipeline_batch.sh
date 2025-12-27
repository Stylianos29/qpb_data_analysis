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
# DATA FILE SET FILTERING:
# - Use --filter flag to selectively include/exclude data file sets
# - Filter configuration via JSON file (default: run_complete_pipeline_batch.json)
# - Supports both allowlist (include) and blocklist (exclude) modes
# - Filter paths are relative to input_base_dir
#
# PIPELINE BEHAVIOR:
# For each data file set discovered:
# - Detects data file set type (correlators vs. parameters-only)
# - Executes appropriate pipeline stages via run_complete_pipeline.sh
# - Updates timestamp on successful completion
# - Continues to next data file set on failure (non-blocking)
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

# Default filter configuration file
DEFAULT_FILTER_FILENAME="run_complete_pipeline_batch.json"
DEFAULT_FILTER_FILE="$(realpath "${SCRIPT_DIR}/${DEFAULT_FILTER_FILENAME}")"

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

  --filter [filter_file]       Enable data file set filtering (REQUIRES explicit flag)
                               - No argument: use default file ($DEFAULT_FILTER_FILENAME)
                               - With argument: use specified JSON file
                               Filter file must exist and be readable
                               Paths in filter are relative to input_base_directory

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

FILTER CONFIGURATION:
  The filter JSON file controls which data file sets to process:
  
  Structure:
    {
      "filter_type": "data-file-set",
      "include": [],
      "exclude": ["path/to/data_file_set1", "path/to/data_file_set2"]
    }
  
  Behavior:
  - If "include" is non-empty: Process ONLY listed data file sets (allowlist)
  - If "include" is empty and "exclude" is non-empty: Process all EXCEPT listed (blocklist)
  - If both empty: No filtering (same as not using --filter)
  - If both non-empty: "include" takes precedence (allowlist mode)
  
  Notes:
  - Paths are relative to input_base_directory
  - Exact string matching (no wildcards)
  - Filter must be explicitly enabled with --filter flag

TIMESTAMP CACHING:
  The script tracks the processing status of each data file set using timestamp
  files stored in the output directory structure:
    output_dir/data_file_set_name/auxiliary/run_complete_pipeline.timestamp

  Behavior:
  - Data file sets are skipped if unmodified since last successful run
  - Failed runs don't update timestamps (will retry on next batch run)
  - Use --all or --force to ignore timestamps and reprocess everything

SELECTIVE STAGE EXECUTION:
  When using --stages flag:
  - Stage 1: Requires raw .txt files in input directory
  - Stage 2: Requires Stage 1 outputs (single_valued_parameters.csv, etc.)
  - Stage 3: Requires Stage 2 outputs + correlators (jackknife HDF5)
  
  Batch mode behavior:
  - Data file sets lacking required inputs are SKIPPED (not errors)
  - Allows mixed processing across heterogeneous data file sets
  - Example: --stages 3 skips parameter-only data file sets automatically

EXAMPLES:
  # Process all data file sets under default raw directory
  $SCRIPT_NAME

  # Process all data file sets under a specific directory
  $SCRIPT_NAME -i /path/to/raw/data/

  # Process only data file sets under the 'invert' program directory
  $SCRIPT_NAME -i ../data_files/raw/invert/

  # Use default filter to exclude certain data file sets
  $SCRIPT_NAME --filter

  # Use custom filter configuration
  $SCRIPT_NAME --filter my_custom_filter.json

  # Rerun only processing and analysis (skip parsing)
  $SCRIPT_NAME --stages 2,3

  # Rerun only analysis (requires existing Stage 2 outputs)
  $SCRIPT_NAME --stages 3

  # Force reprocess all data file sets
  $SCRIPT_NAME --all

  # Disable visualization for faster batch processing
  $SCRIPT_NAME --no-plots

  # Fast batch processing (no checks, summaries, or plots)
  $SCRIPT_NAME --no-plots --skip_checks --skip_summaries

  # Filtered processing with custom settings
  $SCRIPT_NAME --filter custom.json --stages 2,3 --all

NOTES:
  - The script is non-blocking: failures in one data file set don't stop others
  - Each data file set runs independently via run_complete_pipeline.sh
  - Progress is reported for each data file set processed
  - Summary statistics are displayed at the end
  - Visualization is auto-enabled by default (use --no-plots to disable)
  - Filtering is only active when --filter flag is explicitly used

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
    # Discover all data file set directories under input base directory
    #
    # A valid data file set directory is a leaf directory (no subdirectories)
    # or a directory containing .txt or .dat files.
    #
    # Arguments:
    #   $1 - input_base_dir      : Base directory to search
    #   $2 - discovered_sets_arr : Name of array to store results
    #
    # Returns:
    #   Populates the named array with discovered data file set paths
    
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
    # Check if a data file set should be processed based on timestamp
    #
    # Arguments:
    #   $1 - data_set_dir   : Data file set directory
    #   $2 - timestamp_file : Timestamp file path
    #
    # Returns:
    #   0 - Should process
    #   1 - Should skip
    
    local data_set_dir="$1"
    local timestamp_file="$2"
    
    # If force_all is enabled, always process
    if [[ "$force_all" == "true" ]]; then
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

function can_process_stage() {
    # Check if a data file set can satisfy the requirements for a given stage
    # Used in batch mode for graceful skipping of incompatible data file sets
    #
    # Arguments:
    #   $1 - stage      : Stage number (1, 2, or 3)
    #   $2 - output_dir : Output directory for the data file set
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

function load_filter_config() {
    # Load and validate filter configuration from JSON file
    #
    # Arguments:
    #   $1 - filter_file : Path to filter JSON file
    #   $2 - include_var : Name of array variable to store include list
    #   $3 - exclude_var : Name of array variable to store exclude list
    #
    # Returns:
    #   0 - Success
    #   1 - Failure (file error or invalid JSON)
    
    local filter_file="$1"
    local include_var="$2"
    local exclude_var="$3"
    
    # Check if jq is available
    if ! command -v jq &> /dev/null; then
        echo "ERROR: jq is required for filter configuration parsing but not found" >&2
        echo "  Please install jq: sudo apt-get install jq" >&2
        return 1
    fi
    
    # Validate file exists and is readable
    if [[ ! -f "$filter_file" ]]; then
        echo "ERROR: Filter file not found: $filter_file" >&2
        return 1
    fi
    
    if [[ ! -r "$filter_file" ]]; then
        echo "ERROR: Filter file not readable: $filter_file" >&2
        return 1
    fi
    
    # Validate JSON structure
    if ! jq -e . "$filter_file" > /dev/null 2>&1; then
        echo "ERROR: Invalid JSON in filter file: $filter_file" >&2
        return 1
    fi
    
    # Check for required fields
    if ! jq -e '.filter_type' "$filter_file" > /dev/null 2>&1; then
        echo "ERROR: Missing 'filter_type' field in filter file" >&2
        return 1
    fi
    
    if ! jq -e '.include' "$filter_file" > /dev/null 2>&1; then
        echo "ERROR: Missing 'include' field in filter file" >&2
        return 1
    fi
    
    if ! jq -e '.exclude' "$filter_file" > /dev/null 2>&1; then
        echo "ERROR: Missing 'exclude' field in filter file" >&2
        return 1
    fi
    
    # Validate filter_type
    local filter_type
    filter_type=$(jq -r '.filter_type' "$filter_file")
    if [[ "$filter_type" != "data-file-set" ]]; then
        echo "ERROR: Invalid filter_type '$filter_type' (expected 'data-file-set')" >&2
        return 1
    fi
    
    # Parse include list
    local include_count
    include_count=$(jq '.include | length' "$filter_file")
    eval "$include_var=()"
    
    if [[ $include_count -gt 0 ]]; then
        while IFS= read -r path; do
            eval "$include_var+=(\"\$path\")"
        done < <(jq -r '.include[]' "$filter_file")
    fi
    
    # Parse exclude list
    local exclude_count
    exclude_count=$(jq '.exclude | length' "$filter_file")
    eval "$exclude_var=()"
    
    if [[ $exclude_count -gt 0 ]]; then
        while IFS= read -r path; do
            eval "$exclude_var+=(\"\$path\")"
        done < <(jq -r '.exclude[]' "$filter_file")
    fi
    
    return 0
}

function get_relative_path() {
    # Get relative path of data file set from input_base_dir
    #
    # Arguments:
    #   $1 - data_file_set_path : Absolute path to data file set
    #   $2 - input_base_dir     : Absolute path to input base directory
    #
    # Returns:
    #   Prints the relative path
    
    local data_file_set_path="$1"
    local input_base_dir="$2"
    
    # Remove base directory prefix and leading slash
    local rel_path="${data_file_set_path#"$input_base_dir"}"
    rel_path="${rel_path#/}"
    
    echo "$rel_path"
}

function should_process_data_file_set_filter() {
    # Check if a data file set should be processed based on filter configuration
    #
    # Arguments:
    #   $1 - data_file_set_path : Absolute path to data file set
    #   $2 - input_base_dir     : Absolute path to input base directory
    #   $3 - include_array      : Name of include array variable
    #   $4 - exclude_array      : Name of exclude array variable
    #   $5 - reason_var         : Name of variable to store skip reason (output)
    #
    # Returns:
    #   0 - Should process
    #   1 - Should skip (reason stored in reason_var)
    
    local data_file_set_path="$1"
    local input_base_dir="$2"
    local include_array="$3"
    local exclude_array="$4"
    local reason_var="$5"
    
    # Get relative path
    local rel_path
    rel_path=$(get_relative_path "$data_file_set_path" "$input_base_dir")
    
    # Get array sizes
    local include_size=0
    local exclude_size=0
    
    eval "include_size=\${#${include_array}[@]}"
    eval "exclude_size=\${#${exclude_array}[@]}"
    
    # If both lists are empty, no filtering
    if [[ $include_size -eq 0 && $exclude_size -eq 0 ]]; then
        return 0
    fi
    
    # If include list is non-empty, use allowlist mode
    if [[ $include_size -gt 0 ]]; then
        # Check if data file set is in include list
        local found=false
        local i
        for ((i=0; i<include_size; i++)); do
            local include_path
            eval "include_path=\${${include_array}[$i]}"
            if [[ "$rel_path" == "$include_path" ]]; then
                found=true
                break
            fi
        done
        
        if [[ "$found" == "true" ]]; then
            return 0  # In allowlist, process
        else
            eval "$reason_var='not in include list (allowlist mode)'"
            return 1  # Not in allowlist, skip
        fi
    fi
    
    # If only exclude list is non-empty, use blocklist mode
    if [[ $exclude_size -gt 0 ]]; then
        # Check if data file set is in exclude list
        local i
        for ((i=0; i<exclude_size; i++)); do
            local exclude_path
            eval "exclude_path=\${${exclude_array}[$i]}"
            if [[ "$rel_path" == "$exclude_path" ]]; then
                eval "$reason_var='in exclude list (blocklist mode)'"
                return 1  # In blocklist, skip
            fi
        done
        
        return 0  # Not in blocklist, process
    fi
    
    return 0
}

function validate_filter_paths() {
    # Validate that filter paths match discovered data file sets
    # Warns about paths that don't match any discovered data file sets
    #
    # Arguments:
    #   $1 - discovered_array : Name of array with discovered data file set paths
    #   $2 - input_base_dir   : Absolute path to input base directory
    #   $3 - include_array    : Name of include array variable
    #   $4 - exclude_array    : Name of exclude array variable
    #
    # Returns:
    #   Prints warnings for unmatched paths (always returns 0)
    
    local discovered_array="$1"
    local input_base_dir="$2"
    local include_array="$3"
    local exclude_array="$4"
    
    # Build set of discovered relative paths
    local -A discovered_set
    local discovered_size
    eval "discovered_size=\${#${discovered_array}[@]}"
    
    local i
    for ((i=0; i<discovered_size; i++)); do
        local data_file_set_path
        eval "data_file_set_path=\${${discovered_array}[$i]}"
        local rel_path
        rel_path=$(get_relative_path "$data_file_set_path" "$input_base_dir")
        discovered_set["$rel_path"]=1
    done
    
    # Check include list
    local include_size
    eval "include_size=\${#${include_array}[@]}"
    
    if [[ $include_size -gt 0 ]]; then
        for ((i=0; i<include_size; i++)); do
            local include_path
            eval "include_path=\${${include_array}[$i]}"
            if [[ ! -v discovered_set["$include_path"] ]]; then
                echo "WARNING: Filter include path does not match any discovered data file set: $include_path" >&2
            fi
        done
    fi
    
    # Check exclude list
    local exclude_size
    eval "exclude_size=\${#${exclude_array}[@]}"
    
    if [[ $exclude_size -gt 0 ]]; then
        for ((i=0; i<exclude_size; i++)); do
            local exclude_path
            eval "exclude_path=\${${exclude_array}[$i]}"
            if [[ ! -v discovered_set["$exclude_path"] ]]; then
                echo "WARNING: Filter exclude path does not match any discovered data file set: $exclude_path" >&2
            fi
        done
    fi
    
    return 0
}

function process_single_data_set() {
    # Process a single data file set by calling run_complete_pipeline.sh
    #
    # Arguments:
    #   $1 - data_set_dir     : Input data file set directory
    #   $2 - output_dir       : Output directory for this data file set
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
use_filter=false
filter_file=""

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
        --stages)
            stages_to_run="$2"
            shift 2
            ;;
        --filter)
            use_filter=true
            # Check if next argument is a file path or another flag
            if [[ $# -gt 1 && "$2" != -* ]]; then
                filter_file="$2"
                shift 2
            else
                filter_file="$DEFAULT_FILTER_FILE"
                shift 1
            fi
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
        --no-plots)
            disable_plots=true
            shift
            ;;
        --clean-plots)
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

# =============================================================================
# SECTION 6: VALIDATION AND SETUP
# =============================================================================

# Validate and normalize --stages argument if provided
if [[ -n "$stages_to_run" ]]; then
    if ! stages_to_run=$(validate_stages_argument "$stages_to_run"); then
        exit 1
    fi
fi

# Setup input base directory
if [[ -z "$input_base_dir" ]]; then
    input_base_dir="$DEFAULT_INPUT_BASE_DIR"
fi

# Ensure input directory exists
if [[ ! -d "$input_base_dir" ]]; then
    echo "ERROR: Input base directory not found: $input_base_dir" >&2
    exit 1
fi
input_base_dir="$(realpath "$input_base_dir")"

# Setup output base directory
if [[ -z "$output_base_dir" ]]; then
    output_base_dir="$DEFAULT_OUTPUT_BASE_DIR"
fi

# Ensure output directory exists
if [[ ! -d "$output_base_dir" ]]; then
    mkdir -p "$output_base_dir" || {
        echo "ERROR: Failed to create output directory: $output_base_dir" >&2
        exit 1
    }
    echo "INFO: Created output directory: $(get_display_path "$output_base_dir")"
fi
output_base_dir="$(realpath "$output_base_dir")"

# Handle plots directory
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

# Load filter configuration if requested
filter_include=()
filter_exclude=()
filter_mode=""

if [[ "$use_filter" == "true" ]]; then
    echo "INFO: Loading filter configuration from: $(get_display_path "$filter_file")"
    
    if ! load_filter_config "$filter_file" filter_include filter_exclude; then
        echo "ERROR: Failed to load filter configuration" >&2
        exit 1
    fi
    
    # Determine filter mode
    if [[ ${#filter_include[@]} -gt 0 ]]; then
        filter_mode="ALLOWLIST"
        echo "INFO: Filter mode: ALLOWLIST (include) with ${#filter_include[@]} data file set(s)"
        if [[ ${#filter_exclude[@]} -gt 0 ]]; then
            echo "WARNING: Both 'include' and 'exclude' lists are non-empty; using ALLOWLIST mode (include takes precedence)"
        fi
    elif [[ ${#filter_exclude[@]} -gt 0 ]]; then
        filter_mode="BLOCKLIST"
        echo "INFO: Filter mode: BLOCKLIST (exclude) with ${#filter_exclude[@]} data file set(s)"
    else
        filter_mode="NONE"
        echo "WARNING: Filter file loaded but both 'include' and 'exclude' are empty (no filtering will occur)"
    fi
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

if [[ "$use_filter" == "true" ]]; then
    echo "Filtering: ENABLED ($filter_mode mode)"
else
    echo "Filtering: DISABLED"
fi

echo "==================================================================="
echo ""

# Discover all data file sets
echo "Discovering data file sets..."
data_sets=()
if ! discover_data_sets "$input_base_dir" data_sets; then
    echo "ERROR: Failed to discover data file sets" >&2
    exit 1
fi

total_sets=${#data_sets[@]}
if [[ $total_sets -eq 0 ]]; then
    echo "WARNING: No data file sets found under $(get_display_path "$input_base_dir")"
    echo "Data file sets should contain .txt or .dat files"
    exit 0
fi

echo "Found $total_sets data file set(s)"

# Validate filter paths if filtering is enabled
if [[ "$use_filter" == "true" && "$filter_mode" != "NONE" ]]; then
    validate_filter_paths data_sets "$input_base_dir" filter_include filter_exclude
fi

echo ""

# Initialize counters
processed_count=0
skipped_timestamp_count=0
skipped_filter_count=0
skipped_stage_count=0
success_count=0
failure_count=0
failed_sets=()  # Array to store relative paths of failed data sets

# Process each data set
set_index=0
for data_set_dir in "${data_sets[@]}"; do
    ((set_index++))
    data_set_name=$(basename "$data_set_dir")
    
    # Get relative path for display
    rel_path=$(get_relative_path "$data_set_dir" "$input_base_dir")
    
    # Check filter first (if enabled)
    if [[ "$use_filter" == "true" && "$filter_mode" != "NONE" ]]; then
        skip_reason=""
        if ! should_process_data_file_set_filter "$data_set_dir" "$input_base_dir" filter_include filter_exclude skip_reason; then
            echo "⊘ SKIPPING [$set_index/$total_sets]: $rel_path (filter: $skip_reason)"
            ((skipped_filter_count++))
            continue
        fi
    fi
    
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
        echo "⊘ SKIPPING [$set_index/$total_sets]: $rel_path (timestamp: no changes detected)"
        ((skipped_timestamp_count++))
        continue
    fi
    
    # If selective stages requested, check if this data set can satisfy them
    if [[ -n "$stages_to_run" ]]; then
        can_process=true
        
        # Check each requested stage
        if should_run_stage 2; then
            if ! can_process_stage 2 "$output_dir"; then
                echo "⊘ SKIPPING [$set_index/$total_sets]: $rel_path (stage validation: Stage 2 requested but missing Stage 1 outputs)"
                ((skipped_stage_count++))
                continue
            fi
        fi
        
        if should_run_stage 3; then
            if ! can_process_stage 3 "$output_dir"; then
                echo "⊘ SKIPPING [$set_index/$total_sets]: $rel_path (stage validation: Stage 3 requested but missing Stage 2 outputs or no correlators)"
                ((skipped_stage_count++))
                continue
            fi
        fi
    fi
    
    # Process this data set
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ PROCESSING [$set_index/$total_sets]: $rel_path"
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
        echo "✓ SUCCESS [$set_index/$total_sets]: $rel_path completed successfully"
        echo ""
    else
        # Failure - don't update timestamp
        ((failure_count++))
        failed_sets+=("$rel_path")
        echo ""
        echo "✗ FAILED [$set_index/$total_sets]: $rel_path encountered errors"
        echo ""
        # Continue to next data set (non-blocking)
    fi
done

# Calculate total skipped
total_skipped=$((skipped_timestamp_count + skipped_filter_count + skipped_stage_count))

# Display final summary
echo ""
echo "==================================================================="
echo "   BATCH PROCESSING COMPLETE"
echo "==================================================================="
echo "Total data file sets found:    $total_sets"
echo "Processed:                     $processed_count"
echo "Skipped (total):               $total_skipped"
if [[ $skipped_timestamp_count -gt 0 ]]; then
    echo "  - Timestamp (no changes):    $skipped_timestamp_count"
fi
if [[ $skipped_filter_count -gt 0 ]]; then
    echo "  - Filter:                    $skipped_filter_count"
fi
if [[ $skipped_stage_count -gt 0 ]]; then
    echo "  - Stage validation:          $skipped_stage_count"
fi
echo "Successful:                    $success_count"
echo "Failed:                        $failure_count"
echo "==================================================================="

if [[ $success_count -eq $processed_count && $processed_count -gt 0 ]]; then
    echo "INFO: All processed data file sets completed successfully"
    exit 0
elif [[ $failure_count -gt 0 ]]; then
    echo "WARNING - Some data file sets failed processing:"
    for failed_set in "${failed_sets[@]}"; do
        echo "  ✗ $failed_set"
    done
    exit 1
else
    echo "INFO: Batch processing complete"
    exit 0
fi
