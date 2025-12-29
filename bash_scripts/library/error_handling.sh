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
# PYTHON SCRIPT EXECUTION FUNCTIONS
# =============================================================================

function execute_python_script() {
    # Execute a Python script with standardized error handling and logging
    #
    # This function provides a consistent interface for executing Python scripts
    # in the pipeline with proper error handling, logging, and validation.
    #
    # Arguments:
    #   $1 - script_path     : Path to the Python script to execute
    #   $2 - script_name     : Human-readable name for logging
    #   $@ - script_args     : Additional arguments to pass to the Python script
    #
    # Returns:
    #   0 - Script executed successfully
    #   1 - Script execution failed (hard error)
    #   2 - Script completed with graceful skip (insufficient data)
    #   N - Other exit codes are passed through as-is
    #
    # Exit Code Convention:
    #   Scripts using this function should follow this convention:
    #     0 - Success: Analysis completed, results exported
    #     1 - Error: Unexpected failure (file errors, malformed data, etc.)
    #     2 - Graceful skip: Insufficient data for analysis (not an error)
    #
    # Examples:
    #   # Simple execution
    #   execute_python_script "$PARSE_LOG_FILES_SCRIPT" "parse_log_files" \
    #       --input_directory "$input_dir" \
    #       --output_directory "$output_dir" \
    #       || return 1
    #
    #   # With logging enabled
    #   execute_python_script "$PROCESS_RAW_SCRIPT" "process_raw_parameters" \
    #       --input_csv "$csv_file" \
    #       --output_directory "$output_dir" \
    #       --enable_logging \
    #       --log_directory "$log_dir" \
    #       || exit 1
    #
    #   # Handling graceful skip (exit code 2)
    #   execute_python_script "$CRITICAL_MASS_SCRIPT" "calculate_critical_mass" \
    #       --input_csv "$csv_file" \
    #       --output_directory "$output_dir"
    #   local exit_code=$?
    #   if [[ $exit_code -eq 0 ]]; then
    #       echo "Success"
    #   elif [[ $exit_code -eq 2 ]]; then
    #       echo "Graceful skip - insufficient data"
    #   else
    #       echo "Error"
    #       return 1
    #   fi
    #
    # Usage Pattern:
    #   local script_path="$1"
    #   local script_name="$2"
    #   shift 2  # Remove first two arguments
    #   local script_args=("$@")  # Remaining args are for the Python script
    #
    # Output:
    #   Prints execution status to stdout
    #   Prints errors to stderr
    #   Logs execution details if logging is initialized
    
    local script_path="$1"
    local script_name="$2"
    shift 2
    local script_args=("$@")
    
    # Validate Python script exists
    if [[ ! -f "$script_path" ]]; then
        echo "ERROR: Python script not found: $script_path" >&2
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_error "Python script not found: $script_path"
        return 1
    fi
    
    # Log execution start
    [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_info "Executing Python script: $script_name"
    [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_info "Command: python $script_path ${script_args[*]}"
    
    # Execute Python script
    python "$script_path" "${script_args[@]}"
    local exit_code=$?
    
    # Check execution result
    if [[ $exit_code -eq 0 ]]; then
        # Success
        echo "  ✓ $script_name completed successfully"
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_info "$script_name execution successful"
        return 0
    elif [[ $exit_code -eq 2 ]]; then
        # Graceful skip - insufficient data (not an error)
        # Don't print ERROR - let the calling script handle the messaging
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_warning "$script_name completed with graceful skip (exit code 2)"
        return 2
    else
        # Hard error
        echo "ERROR: $script_name failed with exit code $exit_code" >&2
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_error "$script_name failed with exit code $exit_code"
        return $exit_code
    fi
}

function execute_python_script_with_command_building() {
    # Execute a Python script by building command string (alternative approach)
    #
    # This is an alternative to execute_python_script() that builds a command
    # string first, which can be useful for logging the exact command or when
    # you need to use eval for complex command construction.
    #
    # Arguments:
    #   $1 - script_path     : Path to the Python script
    #   $2 - script_name     : Human-readable name
    #   $3 - command_string  : Pre-built command string (without 'python' prefix)
    #
    # Returns:
    #   0 - Success
    #   1 - Failure
    #
    # Example:
    #   local cmd="\"$script_path\" --input \"$input\" --output \"$output\""
    #   execute_python_script_with_command_building \
    #       "$script_path" "my_script" "$cmd" || return 1
    #
    # Note: This function is less commonly needed but provided for compatibility
    #       with existing code patterns that build command strings.
    
    local script_path="$1"
    local script_name="$2"
    local command_string="$3"
    
    # Validate Python script exists
    if [[ ! -f "$script_path" ]]; then
        echo "ERROR: Python script not found: $script_path" >&2
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_error "Python script not found: $script_path"
        return 1
    fi
    
    # Build full command
    local full_command="python $command_string"
    
    # Log execution
    [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_info "Executing: $script_name"
    [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_info "Command: $full_command"
    
    # Execute using eval to handle quoted arguments
    eval "$full_command"
    local exit_code=$?
    
    # Check result
    if [[ $exit_code -eq 0 ]]; then
        echo "  ✓ $script_name completed successfully"
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_info "$script_name execution successful"
        return 0
    else
        echo "ERROR: $script_name failed with exit code $exit_code" >&2
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_error "$script_name failed with exit code $exit_code"
        return 1
    fi
}

# =============================================================================
# END OF FILE
# =============================================================================