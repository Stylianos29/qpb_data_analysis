#!/bin/bash

################################################################################
# validation.sh - Validation and timestamp management library
#
# DESCRIPTION:
# Provides functions for input/output validation, timestamp-based caching,
# file pattern matching, and tool availability checking. Supports:
# - Directory and file validation with automatic creation
# - Timestamp-based change detection for caching
# - QPB log file pattern matching
# - Python script and external tool validation
#
# USAGE:
# Source this file in your script:
#   source "${LIBRARY_SCRIPTS_DIRECTORY_PATH}/validation.sh"
#
# FUNCTIONS:
# Timestamp Management:
# - get_timestamp_file_path()      : Construct timestamp file path
# - check_directory_for_changes()  : Check if directory modified since last run
# - update_timestamp()             : Update timestamp file after processing
#
# File Pattern Matching:
# - find_matching_qpb_log_files()  : Find QPB log files by pattern
#
# Validation Helpers (New):
# - validate_input_directory()     : Validate input directory exists
# - validate_output_directory()    : Validate/create output directory
# - validate_log_directory()       : Validate/create log directory
# - validate_python_script()       : Check Python script exists and is executable
# - validate_required_tools()      : Check for required external tools
#
################################################################################

# MULTIPLE SOURCING GUARD

# Prevent multiple sourcing of this script by exiting if VALIDATION_SH_INCLUDED
# is already set. Otherwise, set VALIDATION_SH_INCLUDED to mark it as sourced.
[[ -n "${VALIDATION_SH_INCLUDED}" ]] && return
VALIDATION_SH_INCLUDED=1

# =============================================================================
# TIMESTAMP MANAGEMENT FUNCTIONS
# =============================================================================

function get_timestamp_file_path() {
    # Construct the path to a timestamp file for a data files set directory
    #
    # Creates a timestamp file path based on the data files set directory and
    # optional script name. Timestamp files are used for caching to avoid
    # reprocessing unmodified data sets.
    #
    # Arguments:
    #   $1 - DATA_FILES_SET_DIRECTORY : Path to the data files set directory
    #   $2 - TIMESTAMP_DIRECTORY      : Directory where timestamp file will be stored
    #   $3 - SCRIPT_NAME              : Optional script name for timestamp file naming
    #
    # Returns:
    #   Prints the timestamp file path to stdout
    #   Return code: 0 on success, 1 on failure
    #
    # Output:
    #   If SCRIPT_NAME provided: TIMESTAMP_DIRECTORY/SCRIPT_NAME.timestamp
    #   Otherwise:              TIMESTAMP_DIRECTORY/DIRNAME.timestamp
    #
    # Examples:
    #   # With script name
    #   timestamp_file=$(get_timestamp_file_path \
    #       "/data/raw/experiment1" \
    #       "/data/processed/experiment1" \
    #       "process_raw_data")
    #   echo "$timestamp_file"  # /data/processed/experiment1/process_raw_data.timestamp
    #
    #   # Without script name (uses directory basename)
    #   timestamp_file=$(get_timestamp_file_path \
    #       "/data/raw/experiment1" \
    #       "/data/processed/experiment1")
    #   echo "$timestamp_file"  # /data/processed/experiment1/experiment1.timestamp
    
    local DATA_FILES_SET_DIRECTORY="$1"
    local TIMESTAMP_DIRECTORY="$2"
    local SCRIPT_NAME="$3"

    # Validate input directories
    check_if_directory_exists "$DATA_FILES_SET_DIRECTORY" -s || {
        echo "ERROR: Invalid data files set directory." >&2
        return 1
    }
    check_if_directory_exists "$TIMESTAMP_DIRECTORY" -s || {
        echo "ERROR: Invalid timestamp directory." >&2
        return 1
    }
    
    # Construct the timestamp file path
    local TIMESTAMP_FILE="$TIMESTAMP_DIRECTORY/"
    if [[ -n "$SCRIPT_NAME" ]]; then
        TIMESTAMP_FILE+="${SCRIPT_NAME}.timestamp"
    else
        TIMESTAMP_FILE+="$(basename "$DATA_FILES_SET_DIRECTORY").timestamp"
    fi

    echo "$TIMESTAMP_FILE"
}


function check_directory_for_changes() {
    # Check if a directory has been modified since last timestamp
    #
    # Compares the directory's current modification time with the stored
    # timestamp to determine if reprocessing is needed. Used for intelligent
    # caching in batch processing scripts.
    #
    # Arguments:
    #   $1 - DATA_FILES_SET_DIRECTORY : Path to the directory to check
    #   $2 - TIMESTAMP_FILE           : Path to the timestamp file
    #
    # Returns:
    #   0 - Directory has changed (needs reprocessing)
    #   1 - Directory unchanged (can skip)
    #
    # Examples:
    #   # Basic usage in batch processing
    #   if check_directory_for_changes "$data_dir" "$timestamp_file"; then
    #       echo "Directory modified - processing..."
    #       process_directory "$data_dir"
    #   else
    #       echo "Directory unchanged - skipping"
    #   fi
    #
    #   # With --all flag override
    #   if [[ "$PROCESS_ALL" == "true" ]] || \
    #      check_directory_for_changes "$data_dir" "$timestamp_file"; then
    #       process_directory "$data_dir"
    #   fi

    local DATA_FILES_SET_DIRECTORY="$1"
    local TIMESTAMP_FILE="$2"

    # Validate inputs
    check_if_directory_exists "$DATA_FILES_SET_DIRECTORY" -s || {
        echo "ERROR: Invalid data files set directory." >&2
        return 1
    }
    check_if_file_exists "$TIMESTAMP_FILE" -s || {
        echo "ERROR: Invalid timestamp file." >&2
        return 1
    }

    # Get the current modification time of the data files directory
    local CURRENT_TIMESTAMP
    CURRENT_TIMESTAMP=$(stat -c %Y "$DATA_FILES_SET_DIRECTORY")

    # Extract the stored timestamp
    local STORED_TIMESTAMP
    STORED_TIMESTAMP=$(cat "$TIMESTAMP_FILE")

    # Check if the current timestamp is less than or equal to the stored one
    if [[ "$CURRENT_TIMESTAMP" -le "$STORED_TIMESTAMP" ]]; then
        # No changes detected, skip processing
        return 1
    fi

    # Changes detected, proceed with processing
    return 0
}


function update_timestamp() {
    # Update timestamp file with current directory modification time
    #
    # Writes the current modification time of a directory to its timestamp file.
    # Called after successful processing to mark the directory as up-to-date.
    #
    # Arguments:
    #   $1 - DATA_FILES_SET_DIRECTORY : Path to the directory that was processed
    #   $2 - TIMESTAMP_FILE           : Path to the timestamp file to update
    #
    # Returns:
    #   0 - Success
    #   1 - Failure (invalid directory or file)
    #
    # Examples:
    #   # After successful processing
    #   process_directory "$data_dir"
    #   update_timestamp "$data_dir" "$timestamp_file"
    #
    #   # In error handling context
    #   if process_directory "$data_dir"; then
    #       update_timestamp "$data_dir" "$timestamp_file"
    #       echo "Processing completed and timestamp updated"
    #   else
    #       echo "Processing failed - timestamp not updated"
    #   fi

    local DATA_FILES_SET_DIRECTORY="$1"
    local TIMESTAMP_FILE="$2"

    # Validate inputs
    check_if_directory_exists "$DATA_FILES_SET_DIRECTORY" -s || {
        echo "ERROR: Invalid data files set directory." >&2
        return 1
    }
    check_if_file_exists "$TIMESTAMP_FILE" -s || {
        echo "ERROR: Invalid timestamp file." >&2
        return 1
    }

    # Get the current modification time of the data files directory
    local CURRENT_TIMESTAMP
    CURRENT_TIMESTAMP=$(stat -c %Y "$DATA_FILES_SET_DIRECTORY")

    # Update the timestamp file
    echo "${CURRENT_TIMESTAMP}" > "$TIMESTAMP_FILE"
}

# =============================================================================
# FILE PATTERN MATCHING
# =============================================================================

function find_matching_qpb_log_files() {
    # Find QPB log files matching or excluding specific patterns
    #
    # Searches through an array of QPB log file paths and filters them based
    # on a search pattern and match mode. Useful for finding files with specific
    # success/failure flags or other markers.
    #
    # Arguments:
    #   $1 - log_file_paths_array     : Name of array containing log file paths
    #   $2 - search_pattern           : Search pattern (regex for grep -E)
    #   $3 - match_mode               : "include" (find matching) or "exclude" (find non-matching)
    #   $4 - matching_log_files_array : Name of array to store results
    #
    # Returns:
    #   0 - Success
    #   1 - Failure (invalid match mode)
    #
    # Examples:
    #   # Find files with success flag
    #   log_files=("file1.txt" "file2.txt" "file3.txt")
    #   success_files=()
    #   find_matching_qpb_log_files log_files "SUCCESS" "include" success_files
    #   echo "Found ${#success_files[@]} successful runs"
    #
    #   # Find files without error flag
    #   error_free_files=()
    #   find_matching_qpb_log_files log_files "ERROR" "exclude" error_free_files
    #
    #   # Pattern matching with regex
    #   convergence_pattern="converged.*iteration"
    #   converged_files=()
    #   find_matching_qpb_log_files log_files "$convergence_pattern" "include" converged_files

    local -n log_file_paths_array="$1"
    local search_pattern="$2"
    local match_mode="$3"
    local -n matching_log_files_array="$4"

    # Ensure the result array is empty before populating
    matching_log_files_array=()

    # Check the match mode and validate
    if [[ "$match_mode" != "include" && "$match_mode" != "exclude" ]]; then
        echo "ERROR: Invalid match mode. Use 'include' or 'exclude'." >&2
        return 1
    fi

    # Iterate through the log files
    for qpb_log_file in "${log_file_paths_array[@]}"; do
        if [[ "$match_mode" == "include" ]]; then
            # Include files matching the search pattern
            if grep -Eq "$search_pattern" "$qpb_log_file"; then
                matching_log_files_array+=("$qpb_log_file")
            fi
        elif [[ "$match_mode" == "exclude" ]]; then
            # Exclude files matching the search pattern
            if ! grep -Eq "$search_pattern" "$qpb_log_file"; then
                matching_log_files_array+=("$qpb_log_file")
            fi
        fi
    done
}

# =============================================================================
# VALIDATION HELPER FUNCTIONS (NEW)
# =============================================================================

function validate_input_directory() {
    # Validate that an input directory exists and is readable
    #
    # High-level validation for input directories. Ensures directory exists
    # and is accessible. Does not create the directory if missing.
    #
    # Arguments:
    #   $1 - directory : Path to input directory to validate
    #
    # Options:
    #   -s|--silent : Suppress output messages
    #
    # Returns:
    #   0 - Success (directory exists and is readable)
    #   1 - Failure (directory missing or not readable)
    #
    # Examples:
    #   # Basic validation
    #   validate_input_directory "$input_dir" || exit 1
    #
    #   # Silent validation in conditional
    #   if validate_input_directory "$raw_data_dir" -s; then
    #       process_data "$raw_data_dir"
    #   fi
    #
    #   # With error message
    #   validate_input_directory "$input_dir" || {
    #       echo "Please provide valid input directory"
    #       exit 1
    #   }

    local directory=""
    local silent="false"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -s|--silent)
                silent="true"
                shift
                ;;
            *)
                directory="$1"
                shift
                ;;
        esac
    done

    # Validate directory exists
    if [[ "$silent" == "true" ]]; then
        check_if_directory_exists "$directory" -s
    else
        check_if_directory_exists "$directory"
    fi
}


function validate_output_directory() {
    # Validate output directory and optionally create it
    #
    # High-level validation for output directories. Can create the directory
    # if it doesn't exist, making it suitable for processing scripts.
    #
    # Arguments:
    #   $1 - directory : Path to output directory to validate
    #
    # Options:
    #   -c|--create : Create directory if it doesn't exist
    #   -s|--silent : Suppress output messages
    #
    # Returns:
    #   0 - Success (directory exists or was created)
    #   1 - Failure (directory missing and --create not specified, or creation failed)
    #
    # Examples:
    #   # Ensure output directory exists (create if needed)
    #   validate_output_directory "$output_dir" -c || exit 1
    #
    #   # Silent validation with creation
    #   validate_output_directory "$results_dir" -c -s
    #
    #   # Check without creating
    #   if ! validate_output_directory "$output_dir"; then
    #       echo "Output directory doesn't exist. Create it? (y/n)"
    #       # ... handle user input ...
    #   fi

    local directory=""
    local create="false"
    local silent="false"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -c|--create)
                create="true"
                shift
                ;;
            -s|--silent)
                silent="true"
                shift
                ;;
            *)
                directory="$1"
                shift
                ;;
        esac
    done

    # Build check_if_directory_exists arguments
    local check_args=("$directory")
    [[ "$create" == "true" ]] && check_args+=("-c")
    [[ "$silent" == "true" ]] && check_args+=("-s")

    check_if_directory_exists "${check_args[@]}"
}


function validate_log_directory() {
    # Validate log directory and optionally create it
    #
    # Specialized validation for log directories. Always creates directory
    # if missing (logs should always be writable).
    #
    # Arguments:
    #   $1 - directory : Path to log directory to validate
    #
    # Options:
    #   -s|--silent : Suppress output messages
    #
    # Returns:
    #   0 - Success (directory exists or was created)
    #   1 - Failure (creation failed)
    #
    # Examples:
    #   # Ensure log directory exists
    #   validate_log_directory "$log_dir" || exit 1
    #
    #   # Silent validation
    #   validate_log_directory "$log_dir" -s
    #
    #   # Initialize logging after validation
    #   validate_log_directory "$log_dir"
    #   init_logging "${log_dir}/script.log" -c

    local directory=""
    local silent="false"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -s|--silent)
                silent="true"
                shift
                ;;
            *)
                directory="$1"
                shift
                ;;
        esac
    done

    # Always create log directories if they don't exist
    if [[ "$silent" == "true" ]]; then
        check_if_directory_exists "$directory" -c -s
    else
        check_if_directory_exists "$directory" -c
    fi
}


function validate_python_script() {
    # Validate that a Python script exists and is executable
    #
    # Checks that the specified Python script file exists and has execute
    # permissions. Optionally makes the script executable if needed.
    #
    # Arguments:
    #   $1 - script_path : Path to Python script to validate
    #
    # Options:
    #   -x|--make-executable : Make script executable if it isn't
    #   -s|--silent          : Suppress output messages
    #
    # Returns:
    #   0 - Success (script exists and is executable)
    #   1 - Failure (script missing or not executable)
    #
    # Examples:
    #   # Basic validation
    #   validate_python_script "$PARSE_LOG_FILES_SCRIPT" || exit 1
    #
    #   # Make executable if needed
    #   validate_python_script "$script_path" -x
    #
    #   # Silent validation before execution
    #   if validate_python_script "$script" -s; then
    #       python "$script" "$@"
    #   fi

    local script_path=""
    local make_executable="false"
    local silent="false"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -x|--make-executable)
                make_executable="true"
                shift
                ;;
            -s|--silent)
                silent="true"
                shift
                ;;
            *)
                script_path="$1"
                shift
                ;;
        esac
    done

    # Check if script exists
    if [[ ! -f "$script_path" ]]; then
        [[ "$silent" == "false" ]] && {
            echo "ERROR: Python script not found: $script_path" >&2
        }
        return 1
    fi

    # Check if script is executable
    if [[ ! -x "$script_path" ]]; then
        if [[ "$make_executable" == "true" ]]; then
            chmod +x "$script_path" 2>/dev/null || {
                [[ "$silent" == "false" ]] && {
                    echo "ERROR: Failed to make script executable: $script_path" >&2
                }
                return 1
            }
            [[ "$silent" == "false" ]] && {
                echo "INFO: Made script executable: $(basename "$script_path")"
            }
        else
            [[ "$silent" == "false" ]] && {
                echo "WARNING: Script not executable: $script_path" >&2
            }
            # Don't fail - Python can still execute it
        fi
    fi

    return 0
}


function validate_required_tools() {
    # Check if required external tools are available
    #
    # Validates that specified command-line tools are installed and available
    # in PATH. Useful for checking dependencies before script execution.
    #
    # Arguments:
    #   $@ - tool_names : List of tool names to check
    #
    # Options:
    #   -s|--silent : Suppress output messages
    #
    # Returns:
    #   0 - All tools available
    #   1 - One or more tools missing
    #
    # Examples:
    #   # Check single tool
    #   validate_required_tools python || exit 1
    #
    #   # Check multiple tools
    #   validate_required_tools python h5glance git || {
    #       echo "Please install missing tools"
    #       exit 1
    #   }
    #
    #   # Silent check with custom message
    #   if ! validate_required_tools python h5py -s; then
    #       echo "This script requires Python and h5py"
    #       exit 1
    #   fi
    #
    #   # Optional tool check
    #   if ! validate_required_tools h5glance -s; then
    #       echo "Note: h5glance not found - HDF5 trees will be skipped"
    #   fi

    local silent="false"
    local tools=()

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -s|--silent)
                silent="true"
                shift
                ;;
            *)
                tools+=("$1")
                shift
                ;;
        esac
    done

    local missing_tools=()

    # Check each tool
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    # Report results
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        [[ "$silent" == "false" ]] && {
            echo "ERROR: Required tools not found: ${missing_tools[*]}" >&2
        }
        return 1
    fi

    return 0
}

# =============================================================================
# END OF FILE
# =============================================================================