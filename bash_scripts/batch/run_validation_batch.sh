#!/bin/bash

################################################################################
# run_validation_batch.sh - Batch execution of validation on multiple data 
# file sets
#
# DESCRIPTION:
# Master batch orchestrator that runs the QPB data files validation on multiple
# data file sets. Supports both automatic discovery and targeted processing
# with intelligent timestamp-based caching.
#
# HYBRID INPUT APPROACH:
# - If -i specified: Process all data sets found under that directory tree
# - If -i omitted:   Default to processing all sets under ../data_files/raw/
#
# TIMESTAMP-BASED CACHING:
# - Tracks validation status per data set using timestamp files
# - Skips data sets that haven't been modified since last successful run
# - Use --all flag to force revalidating all data sets
# - Use --force flag (synonym for --all) for compatibility
#
# DATA FILE SET FILTERING:
# - Use --filter flag to selectively include/exclude data file sets
# - Filter configuration via JSON file (default: run_validation_batch.json)
# - Supports both allowlist (include) and blocklist (exclude) modes
# - Filter paths are relative to input_base_dir
#
# VALIDATION BEHAVIOR:
# For each data file set discovered:
# - Validates raw QPB data files (.txt log files, .dat correlator files)
# - Checks file naming conventions and structure
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
#   ./run_validation_batch.sh [-i <directory>] [options]
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

# Single-set validation script
SINGLE_SET_SCRIPT="$(realpath "${SCRIPT_DIR}/../single_set/run_validation.sh")"

# Default directories (relative to script location)
DEFAULT_INPUT_BASE_DIR="$(realpath "${SCRIPT_DIR}/../../data_files/raw")"
DEFAULT_OUTPUT_BASE_DIR="$(realpath "${SCRIPT_DIR}/../../data_files/processed")"

# Default filter configuration file
DEFAULT_FILTER_FILENAME="run_validation_batch.json"
DEFAULT_FILTER_FILE="$(realpath "${SCRIPT_DIR}/${DEFAULT_FILTER_FILENAME}")"

# Auxiliary directory structure
AUXILIARY_DIR_NAME="auxiliary"
AUXILIARY_TIMESTAMPS_SUBDIR="timestamps"

# Timestamp script identifier
TIMESTAMP_SCRIPT_NAME="run_validation"

# =============================================================================
# SECTION 4: FUNCTION DEFINITIONS
# =============================================================================

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

function should_process_data_file_set_filter() {
    # Check if a data file set should be processed based on filter configuration
    #
    # Arguments:
    #   $1 - data_file_set_path : Absolute path to data file set
    #   $2 - input_base_dir     : Absolute path to input base directory
    #   $3 - include_array      : Name of include array variable
    #   $4 - exclude_array      : Name of exclude array variable
    #   $5 - reason_var         : Name of variable to store skip reason
    #
    # Returns:
    #   0 - Should process
    #   1 - Should skip (filtered out)
    
    local data_file_set_path="$1"
    local input_base_dir="$2"
    local include_array="$3"
    local exclude_array="$4"
    local reason_var="$5"
    
    # Get relative path for comparison
    local rel_path
    rel_path=$(get_relative_path "$data_file_set_path" "$input_base_dir")
    
    # Get array sizes
    local include_size exclude_size
    eval "include_size=\${#${include_array}[@]}"
    eval "exclude_size=\${#${exclude_array}[@]}"
    
    # ALLOWLIST mode: include list has entries
    if [[ $include_size -gt 0 ]]; then
        local i
        for ((i=0; i<include_size; i++)); do
            local include_path
            eval "include_path=\${${include_array}[$i]}"
            if [[ "$rel_path" == "$include_path" ]]; then
                return 0  # Found in allowlist, process
            fi
        done
        
        eval "$reason_var='not in include list (allowlist mode)'"
        return 1  # Not in allowlist, skip
    fi
    
    # BLOCKLIST mode: only exclude list has entries
    if [[ $exclude_size -gt 0 ]]; then
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

function usage() {
    cat << EOF
Usage: $SCRIPT_NAME [-i <input_base_directory>] [options]

DESCRIPTION:
  Batch execution of QPB data files validation on multiple data file sets.
  Automatically discovers and validates all data sets under the specified
  input directory (or default raw data directory).

OPTIONAL ARGUMENTS:
  -i, --input_base_directory   Base directory containing data sets
                               (default: ../../data_files/raw/)
                               Can be set to any level:
                               - All data:     ../data_files/raw/
                               - One program:  ../data_files/raw/invert/
                               - One set:      ../data_files/raw/invert/experiment1/

  -o, --output_base_directory  Base output directory for auxiliary files
                               (default: ../../data_files/processed/)
                               Output structure mirrors input structure

  --filter [filter_file]       Enable data file set filtering (REQUIRES explicit flag)
                               - No argument: use default file ($DEFAULT_FILTER_FILENAME)
                               - With argument: use specified JSON file
                               Filter file must exist and be readable
                               Paths in filter are relative to input_base_directory

  --all                        Force revalidation of ALL data sets,
                               bypassing timestamp checks

  --force                      Synonym for --all (force revalidation)

  --disable-cache              Disable timestamp-based caching
                               (passed to single-set validation)

  -h, --help                   Display this help message

FILTER CONFIGURATION:
  The filter JSON file controls which data file sets to validate:
  
  Structure:
    {
      "filter_type": "data-file-set",
      "include": [],
      "exclude": ["path/to/data_file_set1", "path/to/data_file_set2"]
    }
  
  Behavior:
  - If "include" is non-empty: Validate ONLY listed data file sets (allowlist)
  - If "include" is empty and "exclude" is non-empty: Validate all EXCEPT listed (blocklist)
  - If both empty: No filtering (same as not using --filter)
  - If both non-empty: "include" takes precedence (allowlist mode)
  
  Notes:
  - Paths are relative to input_base_directory
  - Exact string matching (no wildcards)
  - Filter must be explicitly enabled with --filter flag

TIMESTAMP CACHING:
  The script tracks the validation status of each data file set using timestamp
  files stored in the output directory structure:
    output_dir/data_file_set_name/auxiliary/timestamps/run_validation.timestamp

  Behavior:
  - Data file sets are skipped if unmodified since last successful run
  - Failed runs don't update timestamps (will retry on next batch run)
  - Use --all or --force to ignore timestamps and revalidate everything

EXAMPLES:
  # Validate all data sets (using defaults)
  $SCRIPT_NAME

  # Validate all data sets under specific directory
  $SCRIPT_NAME -i ../data_files/raw/invert/

  # Force revalidation of all data sets
  $SCRIPT_NAME --all

  # Validate with filtering
  $SCRIPT_NAME --filter my_filter.json

  # Validate with custom output directory
  $SCRIPT_NAME -i ../raw/ -o ../processed/

EOF
    exit 0
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

function validate_single_data_set() {
    # Validate a single data file set by calling run_validation.sh
    #
    # Arguments:
    #   $1 - data_set_dir     : Input data file set directory
    #   $2 - auxiliary_dir    : Auxiliary directory for this data file set
    #   $3 - disable_cache    : Boolean flag (true/false)
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    
    local data_set_dir="$1"
    local auxiliary_dir="$2"
    local disable_cache="$3"
    
    # Build command
    local cmd="$SINGLE_SET_SCRIPT"
    cmd+=" -i \"$data_set_dir\""
    cmd+=" -aux_dir \"$auxiliary_dir\""
    
    # Add disable-cache flag if specified
    if [[ "$disable_cache" == "true" ]]; then
        cmd+=" --disable-cache"
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
force_all=false
disable_cache=false
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
        --disable-cache)
            disable_cache=true
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

# Validate single-set script exists
if [[ ! -f "$SINGLE_SET_SCRIPT" ]]; then
    echo "ERROR: Single-set validation script not found: $SINGLE_SET_SCRIPT" >&2
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
echo "   QPB DATA FILES - BATCH VALIDATION"
echo "==================================================================="
echo "Input base:  $(get_display_path "$input_base_dir")"
echo "Output base: $(get_display_path "$output_base_dir")"

if [[ "$force_all" == "true" ]]; then
    echo "Mode: FORCE ALL (ignoring timestamps)"
else
    echo "Mode: INCREMENTAL (using timestamps)"
fi

if [[ "$disable_cache" == "true" ]]; then
    echo "Cache: DISABLED (--disable-cache)"
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
    echo "WARNING: No data file sets found under: $(get_display_path "$input_base_dir")"
    echo "INFO: A data file set is a directory containing .txt or .dat files"
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
success_count=0
failure_count=0
skipped_timestamp_count=0
skipped_filter_count=0
set_index=0

# Process each data file set
for data_set_dir in "${data_sets[@]}"; do
    ((set_index++))
    data_set_name=$(basename "$data_set_dir")
    
    # Get relative path for display and filtering
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
    
    # Compute output directory (mirrors input structure)
    output_dir=$(compute_output_directory "$data_set_dir" "$input_base_dir" "$output_base_dir")
    
    # Ensure output directory exists
    check_if_directory_exists "$output_dir" -c -s
    
    # Construct auxiliary directory path
    auxiliary_dir="${output_dir}/${AUXILIARY_DIR_NAME}"
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
    
    # Check if this data set needs processing based on timestamps
    if ! should_process_data_set "$data_set_dir" "$timestamp_file"; then
        echo "⊘ SKIPPING [$set_index/$total_sets]: $rel_path (timestamp: no changes detected)"
        ((skipped_timestamp_count++))
        continue
    fi
    
    # Process this data set
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ VALIDATING [$set_index/$total_sets]: $rel_path"
    echo "  Input:     $(get_display_path "$data_set_dir")"
    echo "  Auxiliary: $(get_display_path "$auxiliary_dir")"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    ((processed_count++))
    
    # Execute single-set validation
    if validate_single_data_set \
        "$data_set_dir" \
        "$auxiliary_dir" \
        "$disable_cache"; then
        
        ((success_count++))
        echo ""
        echo "✓ SUCCESS [$set_index/$total_sets]: $rel_path validated successfully"
        echo ""
    else
        # Failure - don't update timestamp (validation script handles this)
        ((failure_count++))
        echo ""
        echo "✗ FAILED [$set_index/$total_sets]: $rel_path validation failed"
        echo ""
        # Continue to next data set (non-blocking)
    fi
done

# Calculate total skipped
total_skipped=$((skipped_timestamp_count + skipped_filter_count))

# Display final summary
echo ""
echo "==================================================================="
echo "   BATCH VALIDATION COMPLETE"
echo "==================================================================="
echo "Total data file sets found:    $total_sets"
echo "Validated:                     $processed_count"
echo "Skipped (total):               $total_skipped"
if [[ $skipped_timestamp_count -gt 0 ]]; then
    echo "  - Timestamp (no changes):    $skipped_timestamp_count"
fi
if [[ $skipped_filter_count -gt 0 ]]; then
    echo "  - Filter:                    $skipped_filter_count"
fi
echo "Successful:                    $success_count"
echo "Failed:                        $failure_count"
echo "==================================================================="

if [[ $success_count -eq $processed_count && $processed_count -gt 0 ]]; then
    echo "INFO: All validated data file sets passed successfully"
    exit 0
elif [[ $failure_count -gt 0 ]]; then
    echo "WARNING: Some data file sets failed validation"
    exit 1
else
    echo "INFO: Batch validation complete"
    exit 0
fi
