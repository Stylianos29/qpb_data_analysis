#!/bin/bash

# Function to check if a file is empty
is_empty_file() {
    local file="$1"
    # Check if the file exists and if its size is 0
    if [ -f "$file" ] && [ ! -s "$file" ]; then
        echo "$file"
    fi
}

# Main script
if [ -z "$1" ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

directory_path="$1"

# Check if the provided path is a directory
if [ ! -d "$directory_path" ]; then
    echo "Error: '$directory_path' is not a valid directory."
    exit 1
fi

# Find all files in the directory and its subdirectories
# and check if they are empty
find "$directory_path" -type f | while read -r file; do
    is_empty_file "$file"
done
