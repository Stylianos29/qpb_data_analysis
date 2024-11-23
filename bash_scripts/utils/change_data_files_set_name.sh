#!/bin/bash

################################################################################
# change_data_files_set_name.sh - Script for modifying data file set names 
# within a specified directory.
#
# Functionalities:
# - Accepts the path to a directory as an argument using the -d or 
#   --directory_path flag.
# - Provides usage information using the -u or --usage flag.
#
# Usage:
#   ./change_data_files_set_name.sh -d <directory_path>
#   ./change_data_files_set_name.sh --directory_path <directory_path>
#   ./change_data_files_set_name.sh -u
#   ./change_data_files_set_name.sh --usage
#
# Notes:
# - The provided directory path must exist and be accessible.
################################################################################

# Function to display usage information
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -d, --directory_path    Path to the directory containing data files."
    echo "  -u, --usage             Display this usage information and exit."
    echo ""
    echo "Examples:"
    echo "  $0 -d /path/to/directory"
    echo "  $0 --directory_path /path/to/directory"
    echo "  $0 -u"
    echo "  $0 --usage"
    exit 0
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -d|--directory_path)
            directory_path="$2"
            shift 2
            ;;
        -u|--usage)
            usage
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            echo "Unexpected argument: $1"
            usage
            ;;
    esac
done

# Validate the directory path argument if provided
if [ -z "$directory_path" ]; then
    echo "Error: Directory path not provided."
    usage
fi

if [ ! -d "$directory_path" ]; then
    echo "Error: Directory path '$directory_path' does not exist or is not a directory."
    exit 1
fi

# Inform the user of the received directory path
echo "Directory path provided: $directory_path"

# Add further script functionalities here
