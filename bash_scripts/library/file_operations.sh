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


function get_display_path() {
    # Convert absolute path to display-friendly relative path
    #
    # Shortens absolute paths by stripping the prefix up to and including
    # the project base directory parent. This makes console output cleaner
    # and more readable.
    #
    # The function searches for "qpb_data_analysis" in the path and returns
    # everything from that point onwards. If the project directory is not
    # found in the path, returns the original path unchanged.
    #
    # Arguments:
    #   $1 - path : Absolute path to shorten
    #
    # Returns:
    #   Prints shortened path to stdout
    #   Return code: 0 (always succeeds)
    #
    # Examples:
    #   # Long absolute path
    #   long_path="/home/user/projects/qpb_data_analysis/data_files/raw/experiment1"
    #   short_path=$(get_display_path "$long_path")
    #   echo "$short_path"  
    #   # Output: qpb_data_analysis/data_files/raw/experiment1
    #
    #   # Path without project directory (returns unchanged)
    #   other_path="/tmp/some/other/path"
    #   display_path=$(get_display_path "$other_path")
    #   echo "$display_path"
    #   # Output: /tmp/some/other/path
    #
    # Usage in scripts:
    #   echo "Input:  $(get_display_path "$input_directory")"
    #   echo "Output: $(get_display_path "$output_directory")"
    
    local path="$1"
    local project_dir="qpb_data_analysis"
    
    # Check if path contains the project directory
    if [[ "$path" == *"$project_dir"* ]]; then
        # Extract everything from project_dir onwards using sed
        echo "$path" | sed "s|.*\(${project_dir}/.*\)|\1|"
    else
        # Return original path if project directory not found
        echo "$path"
    fi
}

# =============================================================================
# SUMMARY FILE GENERATION FUNCTIONS
# =============================================================================

function generate_hdf5_tree() {
    # Generate HDF5 file structure tree using h5glance
    #
    # Creates a text file showing the hierarchical structure of an HDF5 file,
    # including groups, datasets, and attributes. Uses the h5glance tool if
    # available.
    #
    # Arguments:
    #   $1 - hdf5_file_path   : Path to the HDF5 file to analyze
    #   $2 - output_directory : Directory where tree file will be saved
    #
    # Optional Arguments (via flags):
    #   -s, --silent          : Suppress output messages
    #
    # Returns:
    #   0 - Tree generated successfully
    #   1 - Generation failed or h5glance not available
    #
    # Output:
    #   Creates: <hdf5_filename>_tree.txt in output_directory
    #
    # Examples:
    #   generate_hdf5_tree "$hdf5_path" "$output_dir"
    #   generate_hdf5_tree "$hdf5_path" "$output_dir" --silent
    #
    # Dependencies:
    #   - h5glance Python package
    
    local hdf5_file_path=""
    local output_directory=""
    local silent=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--silent)
                silent=true
                shift
                ;;
            *)
                if [[ -z "$hdf5_file_path" ]]; then
                    hdf5_file_path="$1"
                elif [[ -z "$output_directory" ]]; then
                    output_directory="$1"
                fi
                shift
                ;;
        esac
    done
    
    # Validate inputs
    if [[ -z "$hdf5_file_path" ]] || [[ -z "$output_directory" ]]; then
        echo "ERROR: generate_hdf5_tree requires HDF5 file path and output directory" >&2
        return 1
    fi
    
    if [[ ! -f "$hdf5_file_path" ]]; then
        $silent || echo "ERROR: HDF5 file not found: $hdf5_file_path" >&2
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_error "HDF5 file not found: $hdf5_file_path"
        return 1
    fi
    
    # Check if h5glance is available
    if ! command -v h5glance &> /dev/null; then
        $silent || echo "WARNING: h5glance not available - skipping HDF5 tree generation" >&2
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_warning "h5glance not available"
        return 1
    fi
    
    # Determine output file path
    local hdf5_basename="$(basename "$hdf5_file_path" .h5)"
    local tree_file="${output_directory}/${hdf5_basename}_tree.txt"
    
    # Generate tree
    h5glance "$hdf5_file_path" > "$tree_file" 2>/dev/null
    
    if [[ $? -eq 0 ]] && [[ -s "$tree_file" ]]; then
        $silent || echo "  → HDF5 tree: $(basename "$tree_file")"
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_info "Generated HDF5 tree: $tree_file"
        return 0
    else
        $silent || echo "WARNING: Failed to generate HDF5 tree" >&2
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_warning "Failed to generate HDF5 tree for $hdf5_file_path"
        return 1
    fi
}

function generate_csv_summary() {
    # Generate CSV summary report using inspect_csv_file.py
    #
    # Creates a text file summarizing the CSV structure, including column
    # uniqueness, data types, and value distributions. Uses the project's
    # inspect_csv_file.py utility script.
    #
    # Arguments:
    #   $1 - csv_file_path    : Path to the CSV file to analyze
    #   $2 - output_directory : Directory where summary will be saved
    #
    # Optional Arguments (via flags):
    #   -s, --silent          : Suppress output messages
    #
    # Returns:
    #   0 - Summary generated successfully
    #   1 - Generation failed or inspection script not available
    #
    # Output:
    #   Creates: <csv_filename>_uniqueness_report.txt in output_directory
    #
    # Examples:
    #   generate_csv_summary "$csv_path" "$output_dir"
    #   generate_csv_summary "$csv_path" "$output_dir" --silent
    #
    # Dependencies:
    #   - inspect_csv_file.py script in Python scripts directory
    
    local csv_file_path=""
    local output_directory=""
    local silent=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--silent)
                silent=true
                shift
                ;;
            *)
                if [[ -z "$csv_file_path" ]]; then
                    csv_file_path="$1"
                elif [[ -z "$output_directory" ]]; then
                    output_directory="$1"
                fi
                shift
                ;;
        esac
    done
    
    # Validate inputs
    if [[ -z "$csv_file_path" ]] || [[ -z "$output_directory" ]]; then
        echo "ERROR: generate_csv_summary requires CSV file path and output directory" >&2
        return 1
    fi
    
    if [[ ! -f "$csv_file_path" ]]; then
        $silent || echo "ERROR: CSV file not found: $csv_file_path" >&2
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_error "CSV file not found: $csv_file_path"
        return 1
    fi
    
    # Locate inspect_csv_file.py script
    local inspect_script=""
    if [[ -n "$PYTHON_SCRIPTS_DIRECTORY" ]]; then
        inspect_script="${PYTHON_SCRIPTS_DIRECTORY}/utils/inspect_csv_file.py"
    fi
    
    # Check if script exists
    if [[ ! -f "$inspect_script" ]]; then
        $silent || echo "WARNING: inspect_csv_file.py not found - skipping CSV summary" >&2
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_warning "inspect_csv_file.py not available"
        return 1
    fi
    
    # Execute inspection script
    python "$inspect_script" \
        --csv_file_path "$csv_file_path" \
        --output_directory "$output_directory" \
        --uniqueness_report \
        &>/dev/null
    
    if [[ $? -eq 0 ]]; then
        $silent || echo "  → CSV summary generated"
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_info "Generated CSV summary for $(basename "$csv_file_path")"
        return 0
    else
        $silent || echo "WARNING: Failed to generate CSV summary" >&2
        [[ -n "$SCRIPT_LOG_FILE_PATH" ]] && log_warning "Failed to generate CSV summary for $csv_file_path"
        return 1
    fi
}

function generate_summaries() {
    # Generate both HDF5 tree and CSV summary for output files
    #
    # Convenience function that generates summary files for common pipeline
    # outputs. Can handle CSV files, HDF5 files, or both.
    #
    # Arguments:
    #   $1 - output_directory : Directory containing files to summarize
    #   $2 - csv_file_path    : Optional CSV file path
    #   $3 - hdf5_file_path   : Optional HDF5 file path
    #
    # Optional Arguments (via flags):
    #   -s, --silent          : Suppress output messages
    #   --csv-only            : Only generate CSV summary
    #   --hdf5-only           : Only generate HDF5 tree
    #
    # Returns:
    #   0 - At least one summary generated successfully
    #   1 - All summary generation failed
    #
    # Examples:
    #   # Generate summaries for both files
    #   generate_summaries "$output_dir" "$csv_path" "$hdf5_path"
    #
    #   # Only CSV summary
    #   generate_summaries "$output_dir" "$csv_path" "" --csv-only
    #
    #   # Silent mode
    #   generate_summaries "$output_dir" "$csv_path" "$hdf5_path" --silent
    
    local output_directory=""
    local csv_file_path=""
    local hdf5_file_path=""
    local silent=false
    local csv_only=false
    local hdf5_only=false
    
    # Parse arguments
    local positional_count=0
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--silent)
                silent=true
                shift
                ;;
            --csv-only)
                csv_only=true
                shift
                ;;
            --hdf5-only)
                hdf5_only=true
                shift
                ;;
            *)
                case $positional_count in
                    0) output_directory="$1" ;;
                    1) csv_file_path="$1" ;;
                    2) hdf5_file_path="$1" ;;
                esac
                ((positional_count++))
                shift
                ;;
        esac
    done
    
    local success_count=0
    
    # Generate CSV summary if requested and file provided
    if [[ ! "$hdf5_only" == "true" ]] && [[ -n "$csv_file_path" ]]; then
        if generate_csv_summary "$csv_file_path" "$output_directory" $([ "$silent" == "true" ] && echo "--silent"); then
            ((success_count++))
        fi
    fi
    
    # Generate HDF5 tree if requested and file provided
    if [[ ! "$csv_only" == "true" ]] && [[ -n "$hdf5_file_path" ]]; then
        if generate_hdf5_tree "$hdf5_file_path" "$output_directory" $([ "$silent" == "true" ] && echo "--silent"); then
            ((success_count++))
        fi
    fi
    
    # Return success if at least one summary was generated
    [[ $success_count -gt 0 ]] && return 0 || return 1
}

# =============================================================================
# END OF FILE
# =============================================================================