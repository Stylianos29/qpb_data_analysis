#!/bin/bash

################################################################################
# find_empty_files.sh - Script for finding and optionally deleting all empty
# files within a given directory and its subdirectories.
#
# Functionalities:
# - Takes a directory path as input using the `-d` or `--directory` flag.
# - Optionally removes empty files when the `-r` or `--remove` flag is provided.
# - Optionally prints usage information when the `-u` or `--usage` flag is
#   provided.
# - Uses the `find` command to locate all files in the specified directory and
#   its subdirectories.
# - Checks if each file is empty (i.e., file size is 0 bytes).
# - Prints the paths of all empty files.
#
# Input:
# - A directory path as the first argument, or via the `-d` or `--directory`
#   flag (e.g., ./find_empty_files.sh -d /path/to/directory).
# - Optional flag `-r` or `--remove` to delete empty files.
# - Optional flag `-u` or `--usage` to print the usage instructions.
#
# Output:
# - The script outputs the list of empty files within the provided directory.
# - If `-r` or `--remove` is specified, the script also deletes these files.
# - If `-u` or `--usage` is specified, the script prints the usage instructions.
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
    echo "Usage: $0 [-r|--remove] [-d|--directory <directory_path>] [-u|--usage]"
    echo "  -r, --remove       Delete the empty files found."
    echo "  -d, --directory    Specify the directory path to search for empty files."
    echo "  -u, --usage        Print the usage instructions."
    exit 1
}

# Parse arguments
remove_empty_files=false
directory_path=""
print_usage=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--remove)
            remove_empty_files=true
            shift
            ;;
        -d|--directory)
            directory_path="$2"
            shift 2
            ;;
        -u|--usage)
            print_usage=true
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Print usage and exit if the -u or --usage flag was provided
if [ "$print_usage" = true ]; then
    usage
fi

# Validate directory path if provided
if [ -z "$directory_path" ]; then
    echo "ERROR: Directory path is required. Use -d or --directory to specify it."
    usage
fi

if [ ! -d "$directory_path" ]; then
    echo "ERROR: '$directory_path' is not a valid directory."
    exit 1
fi

# Flag to track if any empty files are found
empty_files_found=false

echo # Print new line

# Find and optionally delete empty files
find "$directory_path" -type f | while read -r file; do
    empty_file=$(is_empty_file "$file")
    if [ -n "$empty_file" ]; then
        echo "$empty_file"
        empty_files_found=true
        if [ "$remove_empty_files" = true ]; then
            rm "$empty_file"
            echo "Deleted: $empty_file"
        fi
    fi
done

# If no empty files were found, print a message
if [ "$empty_files_found" = false ]; then
    echo "No empty files were found in '$(basename $directory_path)/' directory."
fi
