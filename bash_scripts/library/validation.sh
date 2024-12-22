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
