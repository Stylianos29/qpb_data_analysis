#!/bin/bash

# Define directories
SOURCE_DIRECTORY="${HOME}/scratch/raw_qpb_data_files/"
BACKUP_DIRECTORY="/onyx/qdata/cy22sg1/raw_qpb_data_files/"
LOG_DIRECTORY="${HOME}/qpb_data_analysis/archive"
LOG_FILE="${LOG_DIRECTORY}/backup_raw_qpb_data_files.log"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIRECTORY}"

# Start logging
echo "Starting backup at $(date)" > "${LOG_FILE}"
echo "Source directory: ${SOURCE_DIRECTORY}" >> "${LOG_FILE}"
echo "Backup directory: ${BACKUP_DIRECTORY}" >> "${LOG_FILE}"
echo "----------------------------------------" >> "${LOG_FILE}"

# Check if source directory exists
if [ ! -d "${SOURCE_DIRECTORY}" ]; then
    echo "ERROR: Source directory does not exist!" >> "${LOG_FILE}"
    exit 1
fi

# Check if backup directory exists
if [ ! -d "${BACKUP_DIRECTORY}" ]; then
    echo "ERROR: Backup directory does not exist!" >> "${LOG_FILE}"
    mkdir -p "${BACKUP_DIRECTORY}"
    echo "Created backup directory: ${BACKUP_DIRECTORY}" >> "${LOG_FILE}"
fi

# Get list of subdirectories in source (depth 1)
source_subdirectories=()
while IFS= read -r dir; do
    if [ -d "$dir" ]; then
        source_subdirectories+=("$dir")
    fi
done < <(find "${SOURCE_DIRECTORY}" -mindepth 1 -maxdepth 1 -type d)

# Clean up backup directory - remove subdirectories that don't exist in source
while IFS= read -r backup_subdir; do
    backup_subdir_name=$(basename "${backup_subdir}")
    source_subdir="${SOURCE_DIRECTORY}${backup_subdir_name}"
    
    if [ ! -d "${source_subdir}" ]; then
        echo "Removing backup subdirectory that doesn't exist in source: ${backup_subdir}" >> "${LOG_FILE}"
        rm -rf "${backup_subdir}"
    fi
done < <(find "${BACKUP_DIRECTORY}" -mindepth 1 -maxdepth 1 -type d)

# Counter for total files backed up
total_files_backed_up=0
total_archives_created=0

# Process each source subdirectory
for subdir in "${source_subdirectories[@]}"; do
    subdir_name=$(basename "${subdir}")
    echo "Processing subdirectory: ${subdir_name}" >> "${LOG_FILE}"
    
    # Create corresponding backup subdirectory
    backup_subdir="${BACKUP_DIRECTORY}${subdir_name}"
    mkdir -p "${backup_subdir}"
    
    # Get depth-2 subdirectories
    depth2_dirs=()
    while IFS= read -r dir2; do
        if [ -d "$dir2" ]; then
            depth2_dirs+=("$dir2")
        fi
    done < <(find "${subdir}" -mindepth 1 -maxdepth 1 -type d)
    
    # Process each depth-2 subdirectory
    for depth2_dir in "${depth2_dirs[@]}"; do
        depth2_name=$(basename "${depth2_dir}")
        echo "  Processing depth-2 subdirectory: ${depth2_name}" >> "${LOG_FILE}"
        
        # Create corresponding backup subdirectory for depth-2
        backup_depth2_dir="${backup_subdir}/${depth2_name}"
        mkdir -p "${backup_depth2_dir}"
        
        # Count files to be backed up
        file_count=$(find "${depth2_dir}" -type f | wc -l)
        
        if [ "${file_count}" -gt 0 ]; then
            # Create tar.gz file directly in the backup location
            tar_file="${backup_depth2_dir}/${depth2_name}.tar.gz"
            
            # Compress all files from the source depth-2 directory directly to the backup location
            tar -czf "${tar_file}" -C "${depth2_dir}" .
            
            # Check if tar command was successful
            if [ $? -eq 0 ]; then
                echo "  ✓ Backed up ${file_count} files from ${depth2_name} to ${tar_file}" >> "${LOG_FILE}"
                
                # Update counters
                total_files_backed_up=$((total_files_backed_up + file_count))
                total_archives_created=$((total_archives_created + 1))
            else
                echo "  ✗ Failed to create backup for ${depth2_name}" >> "${LOG_FILE}"
            fi
        else
            echo "  ✗ No files found in ${depth2_name}" >> "${LOG_FILE}"
        fi
    done
done

# Finalize log
echo "----------------------------------------" >> "${LOG_FILE}"
echo "Backup completed at $(date)" >> "${LOG_FILE}"
echo "Total files backed up: ${total_files_backed_up}" >> "${LOG_FILE}"
echo "Total .tar.gz archives created: ${total_archives_created}" >> "${LOG_FILE}"
echo "Backup status: SUCCESS" >> "${LOG_FILE}"

# Print completion message to console
echo "Backup completed successfully. Log file: ${LOG_FILE}"