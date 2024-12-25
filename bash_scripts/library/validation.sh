#!/bin/bash

# MULTIPLE SOURCING GUARD

# Prevent multiple sourcing of this script by exiting if VALIDATION_SH_INCLUDED
# is already set. Otherwise, set VALIDATION_SH_INCLUDED to mark it as sourced.
[[ -n "${VALIDATION_SH_INCLUDED}" ]] && return
VALIDATION_SH_INCLUDED=1

# CUSTOM FUNCTIONS DEFINITIONS

check_directory_for_changes() {
    # Function to check if a directory has changed since the last validation.
    # Arguments:
    #   $1: DATA_SET_DIRECTORY - Directory to check.
    #   $2: TIMESTAMP_DIRECTORY - Directory where timestamp files are stored.

    local DATA_SET_DIRECTORY="$1"
    local TIMESTAMP_DIRECTORY="$2"
    
    local TIMESTAMP_FILE="$TIMESTAMP_DIRECTORY/"
    TIMESTAMP_FILE+="$(basename "$DATA_SET_DIRECTORY").timestamp"

    # Ensure the timestamp directory exists
    mkdir -p "$TIMESTAMP_DIRECTORY"

    # Get the current modification time of the data files directory
    local CURRENT_TIMESTAMP
    CURRENT_TIMESTAMP=$(stat -c %Y "$DATA_SET_DIRECTORY")

    # Check if a stored timestamp file exists
    if [[ -f "$TIMESTAMP_FILE" ]]; then
        local STORED_TIMESTAMP
        STORED_TIMESTAMP=$(cat "$TIMESTAMP_FILE")

        if [[ "$CURRENT_TIMESTAMP" -le "$STORED_TIMESTAMP" ]]; then
            # No changes detected, skip validation
            return 1
        fi
    fi

    # Update the timestamp file with the latest modification time
    echo "$CURRENT_TIMESTAMP" > "$TIMESTAMP_FILE"
    # Changes detected, proceed with validation
    return 0
}


find_matching_qpb_log_files() {
    # Function to find qpb log files matching specific patterns
    # Parameters: 
    #   $1: Name of the array containing paths to qpb log files
    #   $2: Search pattern string (success or failure flags)
    #   $3: Match mode ("include" to find matching files, "exclude" to find non-matching files)
    #   $4: Name of the array to store matching log files (modified by the function)

    local -n log_file_paths_array="$1"
    local search_pattern="$2"
    local match_mode="$3"
    local -n matching_log_files_array="$4"

    # Ensure the result array is empty before populating
    matching_log_files_array=()

    # Check the match mode and validate
    if [[ "$match_mode" != "include" && "$match_mode" != "exclude" ]]; then
        echo "Error: Invalid match mode. Use 'include' or 'exclude'."
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
