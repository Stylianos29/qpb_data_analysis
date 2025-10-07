#!/bin/bash

# MULTIPLE SOURCING GUARD

# Prevent multiple sourcing of this script by exiting if VALIDATION_SH_INCLUDED
# is already set. Otherwise, set VALIDATION_SH_INCLUDED to mark it as sourced.
[[ -n "${VALIDATION_SH_INCLUDED}" ]] && return
VALIDATION_SH_INCLUDED=1

# CUSTOM FUNCTIONS DEFINITIONS

function get_timestamp_file_path() {
    
    local DATA_FILES_SET_DIRECTORY="$1"
    local TIMESTAMP_DIRECTORY="$2"
    local SCRIPT_NAME="$3"

    # Validate input
    check_if_directory_exists $DATA_FILES_SET_DIRECTORY || { 
                            echo "Invalid data files set directory."; exit 1;}
    check_if_directory_exists $TIMESTAMP_DIRECTORY || { 
                            echo "Invalid timestamp directory."; exit 1;}
    
    # Construct the timestamp file path
    local TIMESTAMP_FILE="$TIMESTAMP_DIRECTORY/"
    if [ -n "$SCRIPT_NAME" ]; then
        TIMESTAMP_FILE+="${SCRIPT_NAME}.timestamp"
    else
        TIMESTAMP_FILE+="$(basename "$DATA_FILES_SET_DIRECTORY").timestamp"
    fi

    echo "$TIMESTAMP_FILE"
}

function check_directory_for_changes() {

    local DATA_FILES_SET_DIRECTORY="$1"
    local TIMESTAMP_FILE="$2"

    # Validate input
    check_if_directory_exists "$DATA_FILES_SET_DIRECTORY" || { 
        echo "ERROR: Invalid data files set directory."; exit 1; 
    }
    check_if_file_exists "$TIMESTAMP_FILE" -s || { 
        echo "ERROR: Invalid timestamp file."; exit 1; 
    }

    # Get the current modification time of the data files directory
    local CURRENT_TIMESTAMP
    CURRENT_TIMESTAMP=$(stat -c %Y "$DATA_FILES_SET_DIRECTORY")

    # Extract the stored timestamp, removing the script name prefix if provided
    local STORED_TIMESTAMP
    STORED_TIMESTAMP=$(cat "$TIMESTAMP_FILE")

    # Check if the current timestamp is less than or equal to the stored one
    if [[ "$CURRENT_TIMESTAMP" -le "$STORED_TIMESTAMP" ]]; then
        # No changes detected, skip validation
        return 1
    fi

    # Changes detected, proceed with validation
    return 0
}


function update_timestamp() {
    # TODO: description

    local DATA_FILES_SET_DIRECTORY="$1"
    local TIMESTAMP_FILE="$2"

    # Validate input
    check_if_directory_exists "$DATA_FILES_SET_DIRECTORY" || { 
        echo "ERROR: Invalid data files set directory."; exit 1; 
    }
    check_if_file_exists "$TIMESTAMP_FILE" -s || { 
        echo "ERROR: Invalid timestamp file."; exit 1; 
    }

    # Get the current modification time of the data files directory
    local CURRENT_TIMESTAMP
    CURRENT_TIMESTAMP=$(stat -c %Y "$DATA_FILES_SET_DIRECTORY")

    # Update the timestamp file
    echo "${CURRENT_TIMESTAMP}" > "$TIMESTAMP_FILE"
}



function find_matching_qpb_log_files() {
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
