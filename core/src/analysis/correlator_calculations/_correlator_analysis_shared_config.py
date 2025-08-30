#!/usr/bin/env python3
"""
Minimal shared configuration for correlator analysis.
"""

# Shared constants
MIN_JACKKNIFE_SAMPLES = 10

# Metadata datasets to copy
METADATA_DATASETS = [
    "gauge_configuration_labels",
    "mpi_geometry_values",
    "qpb_log_filenames",
    "Number_of_gauge_configurations",
]


# Validation
def validate_shared_config():
    """Validate shared configuration."""
    # Check MIN_JACKKNIFE_SAMPLES
    if not isinstance(MIN_JACKKNIFE_SAMPLES, int) or MIN_JACKKNIFE_SAMPLES < 2:
        raise ValueError(
            f"MIN_JACKKNIFE_SAMPLES must be integer >= 2, got {MIN_JACKKNIFE_SAMPLES}"
        )
    if MIN_JACKKNIFE_SAMPLES > 1000:  # Catch obvious typos
        raise ValueError(
            f"MIN_JACKKNIFE_SAMPLES too large: {MIN_JACKKNIFE_SAMPLES} > 1000"
        )

    # Check METADATA_DATASETS structure
    if not isinstance(METADATA_DATASETS, list):
        raise ValueError(
            f"METADATA_DATASETS must be list, got {type(METADATA_DATASETS)}"
        )
    if len(METADATA_DATASETS) != 4:
        raise ValueError(
            f"METADATA_DATASETS must have exactly 4 elements, got {len(METADATA_DATASETS)}"
        )

    # Check metadata dataset names
    for i, dataset_name in enumerate(METADATA_DATASETS):
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError(
                f"METADATA_DATASETS[{i}] must be non-empty string, got {repr(dataset_name)}"
            )
        if " " in dataset_name:
            raise ValueError(
                f"METADATA_DATASETS[{i}] should not contain spaces: {repr(dataset_name)}"
            )
