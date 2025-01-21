#!/bin/bash

# MULTIPLE SOURCING GUARD

# Prevent multiple sourcing of this script by exiting if UTILITIES_SH_INCLUDED
# is already set. Otherwise, set UTILITIES_SH_INCLUDED to mark it as sourced.
[[ -n "${UTILITIES_SH_INCLUDED}" ]] && return
UTILITIES_SH_INCLUDED=1

# CUSTOM FUNCTIONS DEFINITIONS

check_if_directory_exists() {
    # Function to check if a directory exists with additional options.
    # Supports:
    #   -s|--silent : Suppress all output
    #   -c|--create : Create the directory if it doesn't exist
    #   -r|--remove : Remove all files and subdirectories in the directory if it exists

    local directory=""
    local silent="false"
    local create="false"
    local remove="false"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -s|--silent)
                silent="true"
                shift
                ;;
            -c|--create)
                create="true"
                shift
                ;;
            -r|--remove)
                remove="true"
                shift
                ;;
            *)
                directory="$1"
                shift
                ;;
        esac
    done

    # Validate that a directory was provided
    if [[ -z "$directory" ]]; then
        if [[ "$silent" == "false" ]]; then
            echo "ERROR: No directory specified."
        fi
        return 1
    fi

    # Handle remove flag
    if [[ "$remove" == "true" && -d "$directory" ]]; then
        rm -rf "$directory"/*
        if [[ $? -eq 0 ]]; then
            [[ "$silent" == "false" ]] && {
                echo "INFO: Contents of '$directory' have been removed."
                }
        else
            [[ "$silent" == "false" ]] && {
                echo "ERROR: Failed to remove contents of '$directory'."
                }
            return 1
        fi
    fi

    # Check if the directory exists
    if [ ! -d "$directory" ]; then
        if [[ "$create" == "true" ]]; then
            # Create the directory if it doesn't exist
            mkdir -p "$directory"
            if [[ $? -eq 0 ]]; then
                [[ "$silent" == "false" ]] && {
                    echo "INFO: Directory '$directory' has been created."
                    }
            else
                [[ "$silent" == "false" ]] && {
                    echo "ERROR: Failed to create directory '$directory'."
                    }
                return 1
            fi
        else
            # Report error if the directory doesn't exist and create flag isn't set
            [[ "$silent" == "false" ]] && {
                echo "ERROR: Directory '$directory' does not exist."
                }
            return 1
        fi
    fi

    return 0
}


check_if_file_exists() {
    # Function to check if a given file is a regular file.
    # Supports -s (silent) and -c (create) flags in any order.

    local file=""
    local silent="false"
    local create="false"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -s|--silent)
                silent="true"
                shift
                ;;
            -c|--create)
                create="true"
                shift
                ;;
            *)
                file="$1"
                shift
                ;;
        esac
    done

    # Validate that a file was provided
    if [[ -z "$file" ]]; then
        [[ "$silent" == "false" ]] && echo "ERROR: No file specified."
        return 1
    fi

    # Check if the argument is a regular file
    if [ ! -f "$file" ]; then
        [[ "$silent" == "false" ]] && {
            echo "ERROR: '$file' is not a valid regular file."
            }

        # If create flag is set, create the file
        if [[ "$create" == "true" ]]; then
            touch "$file"  # Create the file (if possible)
            if [[ "$silent" == "false" ]]; then
                echo "INFO: '$file' has been created."
            fi
        fi

        return 1
    fi

    return 0
}


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


termination_output() {
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


failed_python_script() {
    local python_script_path="$1"

    error_message="!! Executing '$(basename $python_script_path)' failed!"
    termination_output "$error_message"

    exit 1
}
