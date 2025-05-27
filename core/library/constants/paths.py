"""File paths and directory constants."""

from pathlib import Path

# Define the root directory of the project
ROOT = Path(__file__).resolve().parents[3]

# TODO: Revisit its usefulness
RAW_DATA_FILES_DIRECTORY = "/nvme/h/cy22sg1/scratch/raw_qpb_data_files/"
PROCESSED_DATA_FILES_DIRECTORY = "../data_files/processed/"
# TODO: Validate provided constant directories. Relative paths do not make sense
