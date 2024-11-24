#!/bin/bash

SOURCE_DIR="/nvme/h/cy22sg1/Data_analysis/data_files"
DEST_DIR="/onyx/qdata/cy22sg1/data_files"
LOG_FILE="/nvme/h/cy22sg1/Data_analysis/rsync.log"

# Step 1: Compress subdirectories into .tar.gz files (without deleting the
# subdirectories)
find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d | while read subdir; do
    # Create the tar.gz archive
    tar -czf "$subdir.tar.gz" -C "$SOURCE_DIR" "$(basename "$subdir")"
done

# Step 2: Rsync to backup the files (both .tar.gz files and markdown files)
rsync -av --partial --delete --log-file="$LOG_FILE" "$SOURCE_DIR"/* "$DEST_DIR/"

# Step 3: Clean up by deleting the .tar.gz files in the source directory
# (keeping subdirectories intact)
find "$SOURCE_DIR" -type f -name "*.tar.gz" -exec rm -f {} \;
