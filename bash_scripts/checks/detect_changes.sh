#!/bin/bash

# Base directory containing data files subdirectories
DATA_DIRECTORY="./data"

# Directory to store timestamp records
TIMESTAMP_RECORD_DIRECTORY="./timestamps"

# Ensure the timestamp record directory exists
mkdir -p "$TIMESTAMP_RECORD_DIRECTORY"

# Loop over each subdirectory in the data directory
for subdirectory in "$DATA_DIRECTORY"/*; do

    # Skip if the entry is not a directory
    if [ ! -d "$subdirectory" ]; then
        continue
    fi

    # Get the name of the current subdirectory
    subdirectory_name=$(basename "$subdirectory")

    # Path to the timestamp record file for this subdirectory
    timestamp_file="$TIMESTAMP_RECORD_DIRECTORY/$subdirectory_name.timestamp"

    # Find the latest modification time of all files in the subdirectory
    latest_modification_time=$(find "$subdirectory" -type f -printf "%T@\n" | sort -n | tail -1)

    # If no files exist, skip processing this subdirectory
    if [ -z "$latest_modification_time" ]; then
        echo "No files found in subdirectory: $subdirectory"
        continue
    fi

    # If no previous timestamp exists, assume the subdirectory is new
    if [ ! -f "$timestamp_file" ]; then
        echo "Processing new subdirectory: $subdirectory"
        echo "$latest_modification_time" > "$timestamp_file"

        # Add your processing logic here
        # For example: python process_data.py --input "$subdirectory"
        continue
    fi

    # Read the stored timestamp
    previous_timestamp=$(cat "$timestamp_file")

    # Compare the latest and previous timestamps
    if (( $(echo "$latest_modification_time > $previous_timestamp" | bc -l) )); then
        echo "Changes detected in subdirectory: $subdirectory"
        echo "$latest_modification_time" > "$timestamp_file"

        # Add your processing logic here
        # For example: python process_data.py --input "$subdirectory"
    else
        echo "No changes in subdirectory: $subdirectory. Skipping."
    fi

done
