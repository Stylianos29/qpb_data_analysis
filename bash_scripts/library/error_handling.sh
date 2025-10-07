#!/bin/bash

################################################################################
# error_handling.sh - Error handling and termination library
#
# DESCRIPTION:
# Provides functions for graceful error handling, script termination, and
# cleanup operations. Supports:
# - Error and success exit handling with logging
# - Python script failure management
# - Termination message formatting
# - Signal handling (e.g., Ctrl+C interrupts)
# - Cleanup operations on exit
#
# USAGE:
# Source this file in your script:
#   source "${LIBRARY_SCRIPTS_DIRECTORY_PATH}/error_handling.sh"
#
# Set up error handling:
#   trap 'handle_interrupt' INT TERM
#   trap 'cleanup_on_exit' EXIT
#
# FUNCTIONS:
# - termination_output()           : Error termination with logging (legacy)
# - set_script_termination_message() : Set termination message (helper)
# - failed_python_script()         : Handle Python script failures
# - error_exit()                   : Generic error exit with message
# - success_exit()                 : Successful exit with message
# - handle_interrupt()             : Handle interrupt signals (Ctrl+C)
# - cleanup_on_exit()              : Cleanup function for trap
#
################################################################################

# MULTIPLE SOURCING GUARD

# Prevent multiple sourcing of this script by returning if already included
[[ -n "${ERROR_HANDLING_SH_INCLUDED}" ]] && return
ERROR_HANDLING_SH_INCLUDED=1

# =============================================================================
# LEGACY ERROR HANDLING FUNCTIONS
# =============================================================================

function set_script_termination_message() {
    # Set or use default termination message for script exit
    #
    # Helper function that sets a termination message variable. If the provided
    # variable is empty, it uses SCRIPT_TERMINATION_MESSAGE if set, or falls
    # back to a default message.
    #
    # Arguments:
    #   $1 - termination_message_ref : Name of variable to set (passed by reference)
    #
    # Environment Variables (Used):
    #   SCRIPT_TERMINATION_MESSAGE : Optional custom termination message
    #
    # Returns:
    #   0 - Always succeeds
    #
    # Example:
    #   local my_message=""
    #   set_script_termination_message my_message
    #   echo "$my_message"  # Will contain default or SCRIPT_TERMINATION_MESSAGE
    
    local -n termination_message_ref="$1"
    
    # Check if the argument variable is empty
    if [[ -z "$termination_message_ref" ]]; then
        # Use the global variable if set, or default message otherwise
        if [[ -n "$SCRIPT_TERMINATION_MESSAGE" ]]; then
            termination_message_ref="$SCRIPT_TERMINATION_MESSAGE"
        else
            termination_message_ref="\n\t\t SCRIPT EXECUTION TERMINATED"
        fi
    fi
}


function termination_output() {
    # Legacy error termination function with logging
    #
    # This function is provided for backward compatibility with existing scripts.
    # New scripts should use error_exit() instead, which provides clearer semantics.
    #
    # Outputs error message to stderr, logs the error if logging is active,
    # and writes a termination message to the log file.
    #
    # Arguments:
    #   $1 - error_message             : Error message to display and log
    #   $2 - script_termination_message : Optional custom termination message
    #
    # Environment Variables (Used):
    #   SCRIPT_LOG_FILE_PATH       : Path to log file (if logging active)
    #   SCRIPT_TERMINATION_MESSAGE : Default termination message
    #
    # Returns:
    #   Does not return (but doesn't exit - caller must exit)
    #
    # Example:
    #   termination_output "Configuration file not found"
    #   exit 1
    
    local error_message="$1"
    local script_termination_message="$2"
    
    # Set termination message if not provided
    set_script_termination_message script_termination_message

    # Output error to stderr
    echo -e "ERROR: $error_message" >&2
    echo "Exiting..." >&2

    # Log error if logging is active
    if [[ -n "$SCRIPT_LOG_FILE_PATH" ]]; then
        # Use log function if available, otherwise write directly
        if declare -f log &>/dev/null; then
            log "ERROR" "$error_message"
        else
            echo -e "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] : $error_message" \
                >> "$SCRIPT_LOG_FILE_PATH"
        fi
        echo -e "$script_termination_message" >> "$SCRIPT_LOG_FILE_PATH"
    fi
}


function failed_python_script() {
    # Handle Python script execution failures
    #
    # Specialized error handler for Python script failures. Formats an error
    # message with the script name, calls termination_output(), and exits.
    #
    # Arguments:
    #   $1 - python_script_path : Full path to the failed Python script
    #
    # Returns:
    #   Does not return (exits with code 1)
    #
    # Examples:
    #   # Direct usage after Python script failure
    #   python "$PARSE_LOG_FILES_SCRIPT" "$@" || failed_python_script "$PARSE_LOG_FILES_SCRIPT"
    #
    #   # With variable
    #   python_script_path="${PYTHON_SCRIPTS_DIRECTORY}/analysis/script.py"
    #   python "$python_script_path" "$@" || failed_python_script "$python_script_path"

    local python_script_path="$1"

    local error_message
    error_message="!! Executing '$(basename "$python_script_path")' failed!"
    
    termination_output "$error_message"

    exit 1
}

# =============================================================================
# MODERN ERROR HANDLING FUNCTIONS
# =============================================================================

function error_exit() {
    # Generic error exit function with logging and customizable exit code
    #
    # Modern replacement for termination_output(). Provides clearer semantics
    # and supports custom exit codes. Logs error, writes termination message,
    # and exits with specified code.
    #
    # Arguments:
    #   $1 - error_message : Error message to display and log
    #   $2 - exit_code     : Optional exit code (default: 1)
    #
    # Environment Variables (Used):
    #   SCRIPT_LOG_FILE_PATH       : Path to log file (if logging active)
    #   SCRIPT_TERMINATION_MESSAGE : Optional custom termination message
    #
    # Returns:
    #   Does not return (exits with specified code)
    #
    # Examples:
    #   # Basic error exit
    #   error_exit "Configuration file not found"
    #
    #   # With custom exit code
    #   error_exit "Invalid arguments provided" 2
    #
    #   # In conditional
    #   [[ -f "$config_file" ]] || error_exit "Config file missing"
    #
    #   # With detailed message
    #   error_exit "Database connection failed: timeout after 30 seconds" 3

    local error_message="$1"
    local exit_code="${2:-1}"  # Default to exit code 1

    # Output error to stderr
    echo "ERROR: $error_message" >&2

    # Log error if logging is active
    if [[ -n "$SCRIPT_LOG_FILE_PATH" ]]; then
        if declare -f log &>/dev/null; then
            log "ERROR" "$error_message"
        else
            echo -e "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] : $error_message" \
                >> "$SCRIPT_LOG_FILE_PATH"
        fi
        
        # Write termination message
        if [[ -n "$SCRIPT_TERMINATION_MESSAGE" ]]; then
            echo -e "$SCRIPT_TERMINATION_MESSAGE" >> "$SCRIPT_LOG_FILE_PATH"
        else
            echo "=== SCRIPT TERMINATED WITH ERROR: $(date) ===" \
                >> "$SCRIPT_LOG_FILE_PATH"
        fi
    fi

    exit "$exit_code"
}


function success_exit() {
    # Successful script exit with logging and message
    #
    # Logs successful completion, writes termination message, and exits
    # with code 0. Provides clean script termination for successful execution.
    #
    # Arguments:
    #   $1 - message : Optional success message (default: "Script completed successfully")
    #
    # Environment Variables (Used):
    #   SCRIPT_LOG_FILE_PATH       : Path to log file (if logging active)
    #   SCRIPT_TERMINATION_MESSAGE : Optional custom termination message
    #
    # Returns:
    #   Does not return (exits with code 0)
    #
    # Examples:
    #   # Basic success exit
    #   success_exit
    #
    #   # With custom message
    #   success_exit "All files processed successfully"
    #
    #   # At end of main()
    #   main() {
    #       # ... processing ...
    #       success_exit "Pipeline completed: 42 files processed"
    #   }

    local message="${1:-Script completed successfully}"

    # Log success if logging is active
    if [[ -n "$SCRIPT_LOG_FILE_PATH" ]]; then
        if declare -f log &>/dev/null; then
            log "INFO" "$message"
        else
            echo -e "$(date '+%Y-%m-%d %H:%M:%S') [INFO] : $message" \
                >> "$SCRIPT_LOG_FILE_PATH"
        fi
        
        # Write termination message
        if [[ -n "$SCRIPT_TERMINATION_MESSAGE" ]]; then
            echo -e "$SCRIPT_TERMINATION_MESSAGE" >> "$SCRIPT_LOG_FILE_PATH"
        else
            echo "=== SCRIPT COMPLETED SUCCESSFULLY: $(date) ===" \
                >> "$SCRIPT_LOG_FILE_PATH"
        fi
    fi

    exit 0
}

# =============================================================================
# SIGNAL HANDLING AND CLEANUP
# =============================================================================

function handle_interrupt() {
    # Handle interrupt signals (SIGINT, SIGTERM)
    #
    # Trap handler for graceful shutdown on Ctrl+C or termination signals.
    # Logs the interruption, performs cleanup if available, and exits.
    #
    # This function is designed to be used with trap:
    #   trap 'handle_interrupt' INT TERM
    #
    # Environment Variables (Used):
    #   SCRIPT_LOG_FILE_PATH : Path to log file (if logging active)
    #
    # Returns:
    #   Does not return (exits with code 130)
    #
    # Examples:
    #   # Set up interrupt handling
    #   trap 'handle_interrupt' INT TERM
    #
    #   # In script with cleanup
    #   cleanup() {
    #       rm -f "$temp_file"
    #   }
    #   trap 'handle_interrupt' INT TERM
    #   trap 'cleanup' EXIT
    #
    #   # Full example
    #   main() {
    #       trap 'handle_interrupt' INT TERM
    #       # ... long-running operations ...
    #   }

    echo "" >&2  # New line after ^C
    echo "Script interrupted by user" >&2

    # Log interruption if logging is active
    if [[ -n "$SCRIPT_LOG_FILE_PATH" ]]; then
        if declare -f log &>/dev/null; then
            log "WARNING" "Script execution interrupted by user"
        else
            echo -e "$(date '+%Y-%m-%d %H:%M:%S') [WARNING] : Script interrupted" \
                >> "$SCRIPT_LOG_FILE_PATH"
        fi
        echo "=== SCRIPT INTERRUPTED: $(date) ===" >> "$SCRIPT_LOG_FILE_PATH"
    fi

    # Call cleanup function if it exists
    if declare -f cleanup &>/dev/null; then
        cleanup
    fi

    exit 130  # Standard exit code for SIGINT
}


function cleanup_on_exit() {
    # Generic cleanup function for EXIT trap
    #
    # Performs cleanup operations when script exits (normal or error).
    # Can be customized per-script by defining a cleanup() function.
    #
    # This function is designed to be used with trap:
    #   trap 'cleanup_on_exit' EXIT
    #
    # Environment Variables (Used):
    #   SCRIPT_LOG_FILE_PATH : Path to log file (if logging active)
    #
    # Returns:
    #   0 - Always succeeds
    #
    # Examples:
    #   # Basic usage - define custom cleanup
    #   cleanup() {
    #       rm -f "$temp_file"
    #       kill $background_pid 2>/dev/null
    #   }
    #   trap 'cleanup_on_exit' EXIT
    #
    #   # Automatic logging of cleanup
    #   trap 'cleanup_on_exit' EXIT
    #   # Cleanup activities logged automatically
    #
    #   # Combined with interrupt handling
    #   trap 'handle_interrupt' INT TERM
    #   trap 'cleanup_on_exit' EXIT

    local exit_code=$?

    # Call script-specific cleanup function if defined
    if declare -f cleanup &>/dev/null; then
        cleanup
        
        # Log cleanup if logging is active
        if [[ -n "$SCRIPT_LOG_FILE_PATH" ]]; then
            if declare -f log &>/dev/null; then
                log "INFO" "Cleanup completed"
            fi
        fi
    fi

    return 0
}

# =============================================================================
# END OF FILE
# =============================================================================