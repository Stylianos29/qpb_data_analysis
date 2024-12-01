#!/bin/bash

SOURCE_DIR="/nvme/h/cy22sg1/Data_analysis/data_files"
DEST_DIR="/onyx/qdata/cy22sg1/data_files"
LOG_FILE="/nvme/h/cy22sg1/Data_analysis/rsync.log"

# Step 1: Compress subdirectories (depth-2) into .tar.gz files inside their
# respective depth-1 subdirectory
find "$SOURCE_DIR" -mindepth 2 -maxdepth 2 -type d | while read subdirDepth2; do
    # Get the parent directory (depth-1 subdirectory)
    parent_dir=$(dirname "$subdirDepth2")
    
    # Ensure the subdirectory is correctly compressed into a tar.gz file
    # inside the parent directory
    tar -czf "$parent_dir/$(basename "$subdirDepth2").tar.gz" -C "$parent_dir" \
                                                "$(basename "$subdirDepth2")"
done

# Step 2: Rsync to backup files (only .tar.gz and markdown files, maintaining
# depth-1 structure, and excluding depth-2 directories entirely)
rsync -av --partial --delete --log-file="$LOG_FILE" \
    --include="/*/" --include="*.tar.gz" --include="*.md" \
    --exclude="/*/*/" --exclude="*" \
    "$SOURCE_DIR/" "$DEST_DIR/"

# Step 3: Clean up by deleting the .tar.gz files from the source directory
# (keeping the subdirectories intact)
find "$SOURCE_DIR" -type f -name "*.tar.gz" -exec rm -f {} \;
