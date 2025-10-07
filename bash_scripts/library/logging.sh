#!/bin/bash

################################################################################
# logging.sh - Logging utilities library
#
# DESCRIPTION:
# Provides centralized logging functionality with severity levels, automatic
# timestamping, and message formatting. Supports:
# - Structured logging with severity levels (DEBUG, INFO, WARNING, ERROR)
# - Automatic timestamp generation
# - Message wrapping at 80 characters
# - Log file management
#
# USAGE:
# Source this file in your script:
#   source "${LIBRARY_SCRIPTS_DIRECTORY_PATH}/logging.sh"
#
# Initialize logging (optional but recommended):
#   init_logging "/path/to/logfile.log"
#
# Or set SCRIPT_LOG_FILE_PATH manually:
#   export SCRIPT_LOG_FILE_PATH="/path/to/logfile.log"
#
# FUNCTIONS:
# - log()           : Main logging function with severity levels
# - init_logging()  : Initialize logging for a script
# - log_info()      : Convenience wrapper for INFO level
# - log_warning()   : Convenience wrapper for WARNING level
# - log_error()     : Convenience wrapper for ERROR level
# - log_debug()     : Convenience wrapper for DEBUG level
# - close_logging() : Clean up logging resources
#
################################################################################

# MULTIPLE SOURCING GUARD

# Prevent multiple sourcing of this script by returning if already included
[[ -n "${LOGGING_SH_INCLUDED}" ]] && return
LOGGING_SH_INCLUDED=1

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Default date format for timestamps
readonly LOG_DATE_FORMAT="%Y-%m-%d %H:%M:%S"

# Default message wrap width
readonly LOG_WRAP_WIDTH=80

# =============================================================================
# CORE LOGGING FUNCTIONS
# =============================================================================

function log() {
    # Main logging function with severity levels and automatic timestamping
    #
    # Writes formatted log messages to the file specified by SCRIPT_LOG_FILE_PATH.
    # Messages are automatically timestamped and wrapped at 80 characters for
    # readability.
    #
    # Arguments:
    #   $1 - log_level : Severity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    #   $2 - message   : Log message to write
    #
    # Environment Variables (Required):
    #   SCRIPT_LOG_FILE_PATH : Path to the log file (must be set before calling)
    #
    # Returns:
    #   0 - Success (message logged)
    #   1 - Failure (SCRIPT_LOG_FILE_PATH not set)
    #
    # Output Format:
    #   YYYY-MM-DD HH:MM:SS [LEVEL] : Message text wrapped at 80 characters
    #
    # Examples:
    #   # Direct usage
    #   export SCRIPT_LOG_FILE_PATH="/path/to/script.log"
    #   log "INFO" "Script execution started"
    #   log "WARNING" "Configuration file not found, using defaults"
    #   log "ERROR" "Failed to connect to database"
    #
    #   # Multi-line messages
    #   log_message="Processing completed successfully"
    #   log_message+="\n  - Files processed: 42"
    #   log_message+="\n  - Total time: 3.5 seconds"
    #   log "INFO" "$log_message"
    
    local log_level="$1"
    local message="$2"

    # Validate that log file path has been set
    if [[ -z "$SCRIPT_LOG_FILE_PATH" ]]; then
        echo "ERROR: No log file path has been provided." >&2
        echo "Set SCRIPT_LOG_FILE_PATH before calling log()." >&2
        return 1
    fi

    # Format message with timestamp and severity level
    # Use fold to wrap the message at configured width
    local wrapped_message
    wrapped_message=$(echo -e \
        "$(date +"$LOG_DATE_FORMAT") [$log_level] : $message" | fold -sw "$LOG_WRAP_WIDTH")
    
    # Write to log file
    echo -e "$wrapped_message" >> "$SCRIPT_LOG_FILE_PATH"
}


function init_logging() {
    # Initialize logging for a script with optional log file creation
    #
    # Sets up the logging environment by exporting SCRIPT_LOG_FILE_PATH and
    # optionally creating/clearing the log file with an initialization message.
    #
    # Arguments:
    #   $1 - log_file_path : Path to the log file
    #
    # Options:
    #   -c|--create : Create log file and write initialization header
    #   -a|--append : Append to existing log file instead of overwriting
    #
    # Environment Variables (Exported):
    #   SCRIPT_LOG_FILE_PATH : Set to the provided log file path
    #
    # Returns:
    #   0 - Success
    #   1 - Failure (no log file path provided or directory doesn't exist)
    #
    # Examples:
    #   # Initialize with new log file (overwrites if exists)
    #   init_logging "/path/to/script.log" -c
    #
    #   # Initialize and append to existing log
    #   init_logging "/path/to/script.log" -a
    #
    #   # Just set the path without creating file
    #   init_logging "/path/to/script.log"
    #
    #   # Full initialization example
    #   SCRIPT_NAME="my_script.sh"
    #   LOG_FILE="${LOG_DIR}/${SCRIPT_NAME%.sh}.log"
    #   init_logging "$LOG_FILE" -c
    #   log "INFO" "Script execution started"

    local log_file_path=""
    local create_file="false"
    local append_mode="false"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -c|--create)
                create_file="true"
                shift
                ;;
            -a|--append)
                append_mode="true"
                shift
                ;;
            *)
                log_file_path="$1"
                shift
                ;;
        esac
    done

    # Validate log file path provided
    if [[ -z "$log_file_path" ]]; then
        echo "ERROR: No log file path provided to init_logging()" >&2
        return 1
    fi

    # Ensure parent directory exists
    local log_dir
    log_dir=$(dirname "$log_file_path")
    if [[ ! -d "$log_dir" ]]; then
        echo "ERROR: Log directory does not exist: $log_dir" >&2
        return 1
    fi

    # Export log file path for use by other functions
    export SCRIPT_LOG_FILE_PATH="$log_file_path"

    # Create/initialize log file if requested
    if [[ "$create_file" == "true" ]]; then
        if [[ "$append_mode" == "true" ]]; then
            # Append mode - just ensure file exists
            touch "$log_file_path" 2>/dev/null || {
                echo "ERROR: Failed to create/access log file: $log_file_path" >&2
                return 1
            }
            # Add separator for new session
            echo "" >> "$log_file_path"
            echo "=== NEW LOGGING SESSION: $(date) ===" >> "$log_file_path"
        else
            # Overwrite mode - create fresh log file with header
            {
                echo "=== LOGGING INITIALIZED: $(date) ==="
                echo ""
            } > "$log_file_path" 2>/dev/null || {
                echo "ERROR: Failed to create log file: $log_file_path" >&2
                return 1
            }
        fi
    fi

    return 0
}


function close_logging() {
    # Clean up logging resources and write termination message
    #
    # Writes a termination message to the log file and optionally unsets
    # the SCRIPT_LOG_FILE_PATH variable.
    #
    # Options:
    #   -u|--unset : Unset SCRIPT_LOG_FILE_PATH after closing
    #
    # Environment Variables (Used):
    #   SCRIPT_LOG_FILE_PATH : Path to the log file
    #   SCRIPT_TERMINATION_MESSAGE : Optional custom termination message
    #
    # Returns:
    #   0 - Success
    #   1 - Failure (SCRIPT_LOG_FILE_PATH not set)
    #
    # Examples:
    #   # Basic closure
    #   close_logging
    #
    #   # Close and unset variable
    #   close_logging -u
    #
    #   # With custom termination message
    #   export SCRIPT_TERMINATION_MESSAGE="MY_SCRIPT.SH EXECUTION TERMINATED"
    #   close_logging

    local unset_var="false"

    # Parse options
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -u|--unset)
                unset_var="true"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done

    # Check if logging is active
    if [[ -z "$SCRIPT_LOG_FILE_PATH" ]]; then
        echo "WARNING: SCRIPT_LOG_FILE_PATH not set, nothing to close" >&2
        return 1
    fi

    # Write termination message
    if [[ -n "$SCRIPT_TERMINATION_MESSAGE" ]]; then
        echo -e "$SCRIPT_TERMINATION_MESSAGE" >> "$SCRIPT_LOG_FILE_PATH"
    else
        echo "=== LOGGING TERMINATED: $(date) ===" >> "$SCRIPT_LOG_FILE_PATH"
    fi

    # Unset variable if requested
    if [[ "$unset_var" == "true" ]]; then
        unset SCRIPT_LOG_FILE_PATH
    fi

    return 0
}

# =============================================================================
# CONVENIENCE WRAPPER FUNCTIONS
# =============================================================================

function log_info() {
    # Convenience wrapper for logging INFO level messages
    #
    # Arguments:
    #   $1 - message : Log message to write
    #
    # Returns:
    #   Return code from log() function
    #
    # Example:
    #   log_info "Processing started"
    #   log_info "Found 42 files to process"

    log "INFO" "$1"
}


function log_warning() {
    # Convenience wrapper for logging WARNING level messages
    #
    # Arguments:
    #   $1 - message : Log message to write
    #
    # Returns:
    #   Return code from log() function
    #
    # Example:
    #   log_warning "Configuration file not found, using defaults"
    #   log_warning "Disk space running low: 5% remaining"

    log "WARNING" "$1"
}


function log_error() {
    # Convenience wrapper for logging ERROR level messages
    #
    # Arguments:
    #   $1 - message : Log message to write
    #
    # Returns:
    #   Return code from log() function
    #
    # Example:
    #   log_error "Failed to connect to database"
    #   log_error "Invalid input file format"

    log "ERROR" "$1"
}


function log_debug() {
    # Convenience wrapper for logging DEBUG level messages
    #
    # Arguments:
    #   $1 - message : Log message to write
    #
    # Returns:
    #   Return code from log() function
    #
    # Example:
    #   log_debug "Variable value: x=$x"
    #   log_debug "Entering function: process_data()"

    log "DEBUG" "$1"
}

# =============================================================================
# END OF FILE
# =============================================================================