#!/bin/bash

# Usage function
usage() {
    echo "Usage: $0 -d <directory_path> [-r|--remove]"
    echo "Checks if any .txt file in the given directory does not contain the line '(per stochastic source)'."
    echo "If the -r or --remove flag is provided, the files missing the line will be deleted."
    exit 1
}

# Parse arguments
REMOVE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--directory)
            DIRECTORY_PATH="$2"
            shift 2
            ;;
        -r|--remove)
            REMOVE=true
            shift
            ;;
        *)
            usage
            ;;
    esac
done

# Check if directory path is provided
if [[ -z "$DIRECTORY_PATH" ]]; then
    echo "Error: Directory path is required."
    usage
fi

# Check if the provided directory exists
if [[ ! -d "$DIRECTORY_PATH" ]]; then
    echo "Error: Directory '$DIRECTORY_PATH' does not exist."
    exit 1
fi

# Check .txt files in the directory
found_flag=false
for file in "$DIRECTORY_PATH"/*.txt; do
    # Skip if no .txt files are found
    [[ ! -e "$file" ]] && continue

    # Check if the line "(per stochastic source)" exists in the file
    if ! grep -q "per stochastic source" "$file"; then
        found_flag=true
        if [[ "$REMOVE" = true ]]; then
            echo "Deleting file: $file"
            rm "$file"
        else
            echo "File missing 'per stochastic source': $file"
        fi
    fi
done

# If no files were missing the line, notify user
if [[ "$found_flag" = false ]]; then
    echo "All .txt files contain 'per stochastic source'."
else
    [[ "$REMOVE" = false ]] && echo -e "\nUse -r or --remove to delete these files."
fi

# TODO: Analyze .err files for error or suspicious behavior.