#!/bin/bash


################################################################################
# find_empty_files.sh -Script for finding and listing all empty files within a
# given directory and its subdirectories.
#
# Functionalities:
# - Takes a directory path as input.
# - Uses the `find` command to locate all files in the specified directory and
#   its subdirectories.
# - Checks if each file is empty (i.e., file size is 0 bytes).
# - Prints the paths of all empty files.
#
# Input:
# - A directory path as the first argument (e.g., ./find_empty_files.sh
#   /path/to/directory).
#
# Output:
# - The script outputs the list of empty files within the provided directory.
################################################################################


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
