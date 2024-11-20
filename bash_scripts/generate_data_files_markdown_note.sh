#!/bin/bash


################################################################################
# generate_data_files_markdown_note.sh - Script for automating the generation of
# markdown files containing information about the raw data file sets as stored
# in specific directories, allowing for the addition of further notes.
#
# Functionalities:
# - Takes a directory path as input.
# - Loops through all directories within the specified raw data directory.
# - For each subdirectory, it generates a markdown file with details about the
#   data files and their naming conventions.
# - If the markdown file already exists, it preserves any existing notes under
#   the "### Notes" section.
# - If the corresponding directory does not exist in the processed data
#   directory, it creates it.
#
# Input:
# - A raw data directory path defined in the script (e.g., ../data_files/raw).
#
# Output:
# - Markdown files generated in the corresponding processed data subdirectories
#   with details about the data file sets, both automatically generated and
#   user-added notes.
################################################################################



RAW_DATA_FILES_DIRECTORY="../data_files/raw"
PROCESSED_DATA_FILES_DIRECTORY="../data_files/processed"

# Loop over all directories of depth 1, excluding 'pion_correlator_data_files'
for data_files_main_program_directory in "$RAW_DATA_FILES_DIRECTORY"/*; do

    # Check if "data_files_main_program_directory" is a directory
    if [ ! -d "$data_files_main_program_directory" ]; then
        continue
    fi

    # Loop over all directories of depth 2
    for data_files_project_directory in "$data_files_main_program_directory"/*; do

        # Check if "data_files_project_directory" is a directory
        if [ ! -d "$data_files_project_directory" ]; then
            continue
        fi

        # CREATE THE MARKDOWN FILE

        # Get the name of the directory of depth 2
        data_files_project_directory_name=$(basename "$data_files_project_directory")

        # Construct markdown file name and its full path
        markdown_file_full_path="${data_files_project_directory/$RAW_DATA_FILES_DIRECTORY/$PROCESSED_DATA_FILES_DIRECTORY}"
        
        # Verify if a directory with the same name exists in the processed files directory
        if [ ! -d "$markdown_file_full_path" ]; then
            echo "There is no '${markdown_file_full_path}' directory."
            mkdir "$markdown_file_full_path"
            echo "The directory has been created."
            continue
        fi

        markdown_file_full_path+="/${data_files_project_directory_name}.md"

        # Preserve existing notes if the markdown file already exists
        existing_notes=""
        if [ -f "$markdown_file_full_path" ]; then
            existing_notes=$(awk '/^### Notes/ {found=1; next} found' "$markdown_file_full_path")
        fi

        # EXTRACT USEFUL INFORMATION

        operator_method="${data_files_project_directory_name%%_*}"
        data_files_main_program=$(basename "$data_files_main_program_directory")

        # FILL IN THE MARKDOWN FILE

        note_section_title="### Notes"

        {
            # First line: Directory name as header
            echo "# $data_files_project_directory_name"
            echo
            # Second line: "## Overview"
            echo "## Overview"
            echo "This directory contains log files generated by the '$data_files_main_program' $operator_method main program. They are used for plotting the change of the $data_files_category with varying ? values for several ?."
            echo
            echo "## Naming Conventions"
            echo "* Directory:  \`<operator_method>_several_<list of parameters that acquire several specific values>_varying_<list of parameters with varying values>.txt\`"
            echo "* Log files: \`<operator_method>_<operator_type>_rho_cSW_EpsCG_n_config_mu.txt\`"
            echo "**Example:**"
            echo
            echo "$note_section_title"
            # Add preserved notes if any
            if [ -n "$existing_notes" ]; then
                echo "$existing_notes"
            else
                echo
            fi
        } > "$markdown_file_full_path"

    done
done
