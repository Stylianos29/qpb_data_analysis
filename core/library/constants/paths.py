"""File paths and directory constants."""

from pathlib import Path

# Define the root directory of the project
ROOT = Path(__file__).resolve().parents[4]

# Use project-relative paths for portability
RAW_DATA_FILES_DIRECTORY = str(ROOT / "data_files" / "raw")
PROCESSED_DATA_FILES_DIRECTORY = str(ROOT / "data_files" / "processed")
