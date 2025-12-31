#!/usr/bin/env python3
"""
Minimal shared configuration for correlator analysis.
"""

# Shared constants
REPRESENTATIVE_GROUP_INDEX = 0  # Index of group to use for file-level validation
# Note: Using index 0 is recommended for safety - ensures validation
# uses the first available group

# Metadata datasets to copy
METADATA_DATASETS = [
    "gauge_configuration_labels",
    "mpi_geometry_values",
    "qpb_log_filenames",
    "Number_of_gauge_configurations",
    "average_core_hours_per_spinor",
]


# Validation
def validate_shared_config():
    """Validate shared configuration."""
    # Check REPRESENTATIVE_GROUP_INDEX
    if (
        not isinstance(REPRESENTATIVE_GROUP_INDEX, int)
        or REPRESENTATIVE_GROUP_INDEX < 0
    ):
        raise ValueError(
            "REPRESENTATIVE_GROUP_INDEX must be non-negative integer, "
            f"got {REPRESENTATIVE_GROUP_INDEX}"
        )

    # Check METADATA_DATASETS structure
    if not isinstance(METADATA_DATASETS, list):
        raise ValueError(
            f"METADATA_DATASETS must be list, got {type(METADATA_DATASETS)}"
        )
    if len(METADATA_DATASETS) != 5:
        raise ValueError(
            "METADATA_DATASETS must have exactly 5 elements, "
            f"got {len(METADATA_DATASETS)}"
        )

    # Check metadata dataset names
    for i, dataset_name in enumerate(METADATA_DATASETS):
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError(
                f"METADATA_DATASETS[{i}] must be non-empty string, "
                f"got {repr(dataset_name)}"
            )
        if " " in dataset_name:
            raise ValueError(
                f"METADATA_DATASETS[{i}] should not contain spaces: "
                f"{repr(dataset_name)}"
            )
