#!/bin/bash

################################################################################
# find_empty_files.sh - Script for finding and optionally deleting all empty 
# files within a given directory and its subdirectories.
#
# Functionalities:
# - Takes a directory path as input.
# - Optionally removes empty files when the `-r` or `--remove` flag is provided.
# - Uses the `find` command to locate all files in the specified directory and
#   its subdirectories.
# - Checks if each file is empty (i.e., file size is 0 bytes).
# - Prints the paths of all empty files.
#
# Input:
# - A directory path as the first argument (e.g., ./find_empty_files.sh 
#   /path/to/directory).
# - Optional flag `-r` or `--remove` to delete empty files.
#
# Output:
# - The script outputs the list of empty files within the provided directory.
# - If `-r` or `--remove` is specified, the script also deletes these files.
################################################################################

# Function to check if a file is empty
is_empty_file() {
    local file="$1"
    # Check if the file exists and if its size is 0
    if [ -f "$file" ] && [ ! -s "$file" ]; then
        echo "$file"
    fi
}

# Usage function
usage() {
    echo "Usage: $0 [-r|--remove] <directory_path>"
    echo "  -r, --remove   Delete the empty files found."
    exit 1
}

# Parse arguments
remove_empty_files=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--remove)
            remove_empty_files=true
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            directory_path="$1"
            shift
            ;;
    esac
done

# Validate directory path
if [ -z "$directory_path" ]; then
    echo "Error: Directory path is required."
    usage
fi

if [ ! -d "$directory_path" ]; then
    echo "Error: '$directory_path' is not a valid directory."
    exit 1
fi

# Find and optionally delete empty files
find "$directory_path" -type f | while read -r file; do
    empty_file=$(is_empty_file "$file")
    if [ -n "$empty_file" ]; then
        echo "$empty_file"
        if [ "$remove_empty_files" = true ]; then
            rm "$empty_file"
            echo "Deleted: $empty_file"
        fi
    fi
done
