#!/bin/bash

# MULTIPLE SOURCING GUARD

# Prevent multiple sourcing of this script by exiting if UTILITIES_SH_INCLUDED
# is already set. Otherwise, set UTILITIES_SH_INCLUDED to mark it as sourced.
[[ -n "${UTILITIES_SH_INCLUDED}" ]] && return
UTILITIES_SH_INCLUDED=1

# CUSTOM FUNCTIONS DEFINITIONS

check_directory_exists() {
    # Function to check if directory exists
    
    local directory="$1"
    
    if [ ! -d "$directory" ]; then
        echo "ERROR: Directory '$directory' does not exist."
        exit 1
    fi
}


replace_parent_directory() {
    # Function to replace the parent directory of a subdirectory with a new
    # parent directory
    # 
    # Arguments:
    #   - original_subdirectory: the full path to the subdirectory
    #   - original_parent_directory: the parent directory to be replaced
    #   - new_parent_directory: the new parent directory that will replace the
    # old one
    # 
    # Returns:
    #   - the updated path with the new parent directory

    local subdirectory_path="$1"
    local old_parent_directory="$2"
    local new_parent_directory="$3"
    
    # Perform the replacement
    echo "${subdirectory_path/$old_parent_directory/$new_parent_directory}"
}


log()
{
    local log_level="$1"
    local message="$2"

    # Log only if the global variable of the log file path has been set properly
    if [ ! -z "$SCRIPT_LOG_FILE_PATH" ]; then
        # Use fold to wrap the message at 80 characters
        wrapped_message=$(echo -e \
           "$(date '+%Y-%m-%d %H:%M:%S') [$log_level] : $message" | fold -sw 80)
        echo -e "$wrapped_message" >> "$SCRIPT_LOG_FILE_PATH"
    else
        # Otherwise exit with error
        echo "No current script's log file path has been provided." >&2
        echo "Exiting..." >&2
        return 1
    fi
}


set_script_termination_message() {
    # Accepts a variable name as an argument
    local -n termination_message_ref="$1"
    
    # Check if the argument variable is empty
    if [ -z "$termination_message_ref" ]; then
        # Use the global variable if set, or default message otherwise
        if [ -n "$SCRIPT_TERMINATION_MESSAGE" ]; then
            termination_message_ref="$SCRIPT_TERMINATION_MESSAGE"
        else
            termination_message_ref="\n\t\t SCRIPT EXECUTION TERMINATED"
        fi
    fi
}


termination_output()
{
    local error_message="$1"
    local script_termination_message="$2"
    set_script_termination_message script_termination_message

    echo -e "ERROR: $error_message" >&2
    echo "Exiting..."  >&2

    if [ -n "$SCRIPT_LOG_FILE_PATH" ]; then
      log "ERROR" "$error_message"
      echo -e "$script_termination_message" >> "$SCRIPT_LOG_FILE_PATH"
    fi
}
