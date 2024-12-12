#!/bin/bash

################################################################################
# err_file_cleaner.sh
#
# Description: This script checks for ".err" files in a specified directory and
# provides an option to delete them if they exist. If no ".err" files are found,
# the script continues silently to the next step and validates that remaining
# files come in pairs of ".txt" and ".dat" with matching base names.
#
# Purpose:
# - To manage "raw files" directories by optionally removing unnecessary ".err"
#   files.
# - To ensure that remaining files are properly paired for further processing.
#
# Usage:
# - ./err_file_cleaner.sh -p <directory_path>
#
# Flags: -p, --path  Specify the directory to analyze for ".err" files and
# validate pairs.
#
# Note:
# - The script will exit with an error message if the directory path is invalid
#   or not provided.
################################################################################

# Function to display usage information
usage() {
    echo "Usage: $0 -p <directory_path>"
    echo "  -p, --path   Specify the directory containing raw files"
    exit 1
}

# Function to check if directory exists
check_directory_exists() {
    local directory="$1"
    if [ ! -d "$directory" ]; then
        echo "Error: Directory '$directory' does not exist."
        exit 1
    fi
}

# Parse input arguments
directory_path=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--path)
            directory_path="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            usage
            ;;
    esac
done

# Ensure a directory path is provided
if [ -z "$directory_path" ]; then
    echo "Error: No directory path specified."
    usage
fi

# Verify the directory exists
check_directory_exists "$directory_path"

# Check for .err files
err_files=("$directory_path"/*.err)
if [ -e "${err_files[0]}" ]; then
    # echo "Found the following .err files:"
    # for file in "${err_files[@]}"; do
    #     echo "  - $file"
    # done
    echo -e "\n'.err' files found."

    # Ask the user whether to delete the .err files
    read -p "Do you want to delete these files? (y/n): " user_response
    case "$user_response" in
        [Yy]*)
            for file in "${err_files[@]}"; do
                rm "$file"
            done
            echo "All '.err' files were deleted."
            ;;
        # [Nn]*)
        #     echo "No files were deleted."
        #     ;;
        *)
            echo "Invalid response. No files were deleted."
            ;;
    esac
# else
#     echo "No .err files found. Moving to the next step..."
fi

# VALIDATE THAT REMAINING FILES COME IN PAIRS OF .TXT AND .DAT

remaining_txt_files=("$directory_path"/*.txt)
remaining_dat_files=("$directory_path"/*.dat)

# Extract base names (without extensions)
base_txt_files=()
for txt_file in "${remaining_txt_files[@]}"; do
    if [ -e "$txt_file" ]; then
        base_txt_files+=("$(basename "$txt_file" .txt)")
    fi
done

base_dat_files=()
for dat_file in "${remaining_dat_files[@]}"; do
    if [ -e "$dat_file" ]; then
        base_dat_files+=("$(basename "$dat_file" .dat)")
    fi
done

# Compare base names
missing_pairs=false
for base_name in "${base_txt_files[@]}"; do
    if [[ ! " ${base_dat_files[@]} " =~ " $base_name " ]]; then
        echo -e "\nWarning: Missing .dat file for base name: $base_name"
        missing_pairs=true
    fi
done

for base_name in "${base_dat_files[@]}"; do
    if [[ ! " ${base_txt_files[@]} " =~ " $base_name " ]]; then
        echo -e "\nWarning: Missing .txt file for base name: $base_name"
        missing_pairs=true
    fi
done

if [ "$missing_pairs" = false ]; then
    echo -e "\nAll files are properly paired."
fi
