#!/bin/bash

################################################################################
# file_operations.sh - File and directory operations library
#
# DESCRIPTION:
# Provides utility functions for file system operations including:
# - Directory existence checking with creation and cleanup options
# - File existence checking with creation options
# - Path manipulation utilities
#
# USAGE:
# Source this file in your script:
#   source "${LIBRARY_SCRIPTS_DIRECTORY_PATH}/file_operations.sh"
#
# FUNCTIONS:
# - check_if_directory_exists()  : Enhanced directory validation with flags
# - check_if_file_exists()       : Enhanced file validation with flags
# - check_directory_exists()     : DEPRECATED - Legacy directory check
# - replace_parent_directory()   : Path manipulation utility
#
################################################################################

# MULTIPLE SOURCING GUARD

# Prevent multiple sourcing of this script by returning if already included
[[ -n "${FILE_OPERATIONS_SH_INCLUDED}" ]] && return
FILE_OPERATIONS_SH_INCLUDED=1

# =============================================================================
# DIRECTORY OPERATIONS
# =============================================================================

function check_if_directory_exists() {
    # Check if a directory exists with optional creation, cleanup, and silent modes
    #
    # This is the primary directory validation function. It provides flexible
    # options for handling directories that don't exist or need cleanup.
    #
    # Arguments:
    #   $1 - directory : Path to the directory to check
    #
    # Options:
    #   -s|--silent  : Suppress all output messages
    #   -c|--create  : Create the directory if it doesn't exist (with mkdir -p)
    #   -r|--remove  : Remove all contents of the directory if it exists
    #
    # Returns:
    #   0 - Success (directory exists or was created)
    #   1 - Failure (directory doesn't exist and --create not specified,
    #                or creation/removal failed)
    #
    # Examples:
    #   # Basic check
    #   check_if_directory_exists "/path/to/dir" || exit 1
    #
    #   # Create directory if it doesn't exist
    #   check_if_directory_exists "/path/to/dir" -c
    #
    #   # Silent check with creation
    #   check_if_directory_exists "/path/to/dir" -s -c
    #
    #   # Clear directory contents before use
    #   check_if_directory_exists "/path/to/dir" -r
    
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
            echo "ERROR: No directory specified." >&2
        fi
        return 1
    fi

    # Handle remove flag - clear directory contents if it exists
    if [[ "$remove" == "true" && -d "$directory" ]]; then
        rm -rf "$directory"/* 2>/dev/null
        if [[ $? -eq 0 ]]; then
            [[ "$silent" == "false" ]] && {
                echo "INFO: Contents of '$directory' have been removed."
            }
        else
            [[ "$silent" == "false" ]] && {
                echo "ERROR: Failed to remove contents of '$directory'." >&2
            }
            return 1
        fi
    fi

    # Check if the directory exists
    if [[ ! -d "$directory" ]]; then
        if [[ "$create" == "true" ]]; then
            # Create the directory if it doesn't exist
            mkdir -p "$directory" 2>/dev/null
            if [[ $? -eq 0 ]]; then
                [[ "$silent" == "false" ]] && {
                    echo "INFO: Directory '$directory' has been created."
                }
            else
                [[ "$silent" == "false" ]] && {
                    echo "ERROR: Failed to create directory '$directory'." >&2
                }
                return 1
            fi
        else
            # Report error if directory doesn't exist and create flag not set
            [[ "$silent" == "false" ]] && {
                echo "ERROR: Directory '$directory' does not exist." >&2
            }
            return 1
        fi
    fi

    return 0
}


function check_directory_exists() {
    # DEPRECATED: Legacy directory existence check
    #
    # This function is provided for backward compatibility with existing scripts.
    # New scripts should use check_if_directory_exists() instead.
    #
    # Arguments:
    #   $1 - directory : Path to the directory to check
    #
    # Returns:
    #   Exits with code 1 if directory doesn't exist
    #
    # Deprecation Notice:
    #   This function will be removed in a future version.
    #   Use check_if_directory_exists() for new code.
    #
    # Example:
    #   check_directory_exists "/path/to/dir"
    
    local directory="$1"
    
    if [[ ! -d "$directory" ]]; then
        echo "ERROR: Directory '$directory' does not exist." >&2
        exit 1
    fi
}

# =============================================================================
# FILE OPERATIONS
# =============================================================================

function check_if_file_exists() {
    # Check if a file exists with optional creation and silent modes
    #
    # This is the primary file validation function. It provides flexible
    # options for handling files that don't exist.
    #
    # Arguments:
    #   $1 - file : Path to the file to check
    #
    # Options:
    #   -s|--silent  : Suppress all output messages
    #   -c|--create  : Create an empty file if it doesn't exist (with touch)
    #
    # Returns:
    #   0 - Success (file exists)
    #   1 - Failure (file doesn't exist and --create not specified,
    #                or creation failed, or path is not a regular file)
    #
    # Examples:
    #   # Basic check
    #   check_if_file_exists "/path/to/file.txt" || exit 1
    #
    #   # Create file if it doesn't exist
    #   check_if_file_exists "/path/to/file.log" -c
    #
    #   # Silent check with creation
    #   check_if_file_exists "/path/to/file.txt" -s -c
    #
    #   # Use in conditional
    #   if check_if_file_exists "$config_file" -s; then
    #       echo "Config file found"
    #   fi

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
        [[ "$silent" == "false" ]] && {
            echo "ERROR: No file specified." >&2
        }
        return 1
    fi

    # Check if the path is a regular file
    if [[ ! -f "$file" ]]; then
        [[ "$silent" == "false" ]] && {
            echo "ERROR: '$file' is not a valid regular file." >&2
        }

        # If create flag is set, create the file
        if [[ "$create" == "true" ]]; then
            touch "$file" 2>/dev/null
            if [[ $? -eq 0 ]]; then
                [[ "$silent" == "false" ]] && {
                    echo "INFO: '$file' has been created."
                }
                return 0
            else
                [[ "$silent" == "false" ]] && {
                    echo "ERROR: Failed to create '$file'." >&2
                }
                return 1
            fi
        fi

        return 1
    fi

    return 0
}

# =============================================================================
# PATH MANIPULATION
# =============================================================================

function replace_parent_directory() {
    # Replace the parent directory portion of a path with a new parent
    #
    # This utility function helps convert paths between different directory
    # structures, commonly used when mirroring directory hierarchies (e.g.,
    # converting raw data paths to processed data paths).
    #
    # Arguments:
    #   $1 - subdirectory_path        : Full path to the subdirectory
    #   $2 - old_parent_directory     : Parent directory to be replaced
    #   $3 - new_parent_directory     : New parent directory
    #
    # Returns:
    #   Prints the updated path to stdout
    #   Return code: 0 (always succeeds if arguments provided)
    #
    # Examples:
    #   # Convert raw data path to processed path
    #   old_path="/data/raw/experiment1/file.txt"
    #   new_path=$(replace_parent_directory "$old_path" "/data/raw" "/data/processed")
    #   echo "$new_path"  # Output: /data/processed/experiment1/file.txt
    #
    #   # Convert between different directory structures
    #   input_dir="/home/user/input/project/data"
    #   output_dir=$(replace_parent_directory "$input_dir" "/home/user/input" "/mnt/output")
    #   echo "$output_dir"  # Output: /mnt/output/project/data
    #
    #   # Use in variable assignment
    #   processed_path=$(replace_parent_directory \
    #       "$raw_data_files_set_directory" \
    #       "$RAW_DATA_FILES_DIRECTORY" \
    #       "$PROCESSED_DATA_FILES_DIRECTORY")

    local subdirectory_path="$1"
    local old_parent_directory="$2"
    local new_parent_directory="$3"
    
    # Perform the replacement using parameter expansion
    echo "${subdirectory_path/$old_parent_directory/$new_parent_directory}"
}

# =============================================================================
# END OF FILE
# =============================================================================