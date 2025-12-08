"""
Filter configuration handling for jackknife analysis.

This module provides functions to load and apply filename-based filters
for jackknife analysis, allowing users to include or exclude specific
files from processing.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional
import logging


def load_filter_config(filter_config_path: str, logger) -> Dict:
    """
    Load and validate filter configuration from JSON file.

    Args:
        filter_config_path: Path to JSON filter configuration file
        logger: Logger instance for reporting

    Returns:
        Dictionary with filter configuration

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If file doesn't exist
    """
    logger.info(f"Loading filter configuration from: {filter_config_path}")

    # Load JSON file
    try:
        with open(filter_config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        error_msg = f"Filter configuration file not found: {filter_config_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in filter configuration: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate structure
    if "filter_type" not in config:
        error_msg = "Filter configuration must contain 'filter_type' field"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if config["filter_type"] != "filename":
        error_msg = f"Unsupported filter_type: {config['filter_type']}. Only 'filename' is supported."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Ensure include and exclude lists exist (can be empty)
    if "include" not in config:
        config["include"] = []
        logger.warning("No 'include' list in filter config, assuming empty list")

    if "exclude" not in config:
        config["exclude"] = []
        logger.warning("No 'exclude' list in filter config, assuming empty list")

    # Validate that lists contain strings
    if not isinstance(config["include"], list):
        error_msg = "'include' must be a list"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not isinstance(config["exclude"], list):
        error_msg = "'exclude' must be a list"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Check if both lists are empty
    if not config["include"] and not config["exclude"]:
        logger.warning(
            "Both 'include' and 'exclude' lists are empty - no filtering will be applied"
        )

    # Log configuration summary
    logger.info("=" * 80)
    logger.info("FILTER CONFIGURATION LOADED")
    logger.info("=" * 80)
    logger.info(f"Filter type: {config['filter_type']}")
    logger.info(f"Include list: {len(config['include'])} files")
    logger.info(f"Exclude list: {len(config['exclude'])} files")

    if config["include"]:
        logger.info("Filter mode: INCLUDE (only listed files will be processed)")
        logger.info(f"First few files to include: {config['include'][:3]}")
        if len(config["include"]) > 3:
            logger.info(f"  ... and {len(config['include']) - 3} more")
    elif config["exclude"]:
        logger.info("Filter mode: EXCLUDE (listed files will be skipped)")
        logger.info(f"First few files to exclude: {config['exclude'][:3]}")
        if len(config["exclude"]) > 3:
            logger.info(f"  ... and {len(config['exclude']) - 3} more")

    logger.info("=" * 80)

    return config


def apply_filename_filter(
    filenames: List[str], filter_config: Dict, logger
) -> tuple[List[str], Dict[str, List[str]]]:
    """
    Apply filename-based filtering to a list of filenames.

    Logic:
    - If 'include' list has entries: only keep filenames in the include list
    - Otherwise, if 'exclude' list has entries: remove filenames in the exclude list
    - If both are empty: return all filenames unchanged

    Args:
        filenames: List of filenames (with or without extensions)
        filter_config: Filter configuration dictionary
        logger: Logger instance for reporting

    Returns:
        Tuple of (filtered_filenames, statistics_dict)
        where statistics_dict contains:
            - 'kept': list of kept filenames
            - 'removed': list of removed filenames
            - 'not_found_in_data': list of filter entries not found in data
    """
    # Strip extensions from filenames for comparison
    filenames_no_ext = [Path(f).stem for f in filenames]

    # Get filter lists
    include_list = filter_config.get("include", [])
    exclude_list = filter_config.get("exclude", [])

    # IMPORTANT: Strip extensions from filter entries as well
    # This allows the JSON to have entries with or without extensions
    include_list_no_ext = [Path(f).stem for f in include_list] if include_list else []
    exclude_list_no_ext = [Path(f).stem for f in exclude_list] if exclude_list else []

    # Convert to sets for efficient lookup
    include_set = set(include_list_no_ext) if include_list_no_ext else set()
    exclude_set = set(exclude_list_no_ext) if exclude_list_no_ext else set()

    # Statistics tracking
    stats = {"kept": [], "removed": [], "not_found_in_data": []}

    filtered_filenames = []

    # INCLUDE mode takes precedence
    if include_set:
        logger.info("Applying INCLUDE filter...")

        for orig_filename, filename_no_ext in zip(filenames, filenames_no_ext):
            if filename_no_ext in include_set:
                filtered_filenames.append(orig_filename)
                stats["kept"].append(filename_no_ext)

        # Check for include entries that weren't found
        found_set = set(stats["kept"])
        not_found = include_set - found_set
        stats["not_found_in_data"] = list(not_found)

        # Determine removed files
        stats["removed"] = [f for f in filenames_no_ext if f not in found_set]

        # Log warnings for not found entries
        if stats["not_found_in_data"]:
            logger.warning(
                f"WARNING: {len(stats['not_found_in_data'])} files in INCLUDE list were not found in data:"
            )
            for filename in stats["not_found_in_data"][:10]:
                logger.warning(f"  - {filename}")
            if len(stats["not_found_in_data"]) > 10:
                logger.warning(f"  ... and {len(stats['not_found_in_data']) - 10} more")

    # EXCLUDE mode (only if include list is empty)
    elif exclude_set:
        logger.info("Applying EXCLUDE filter...")

        for orig_filename, filename_no_ext in zip(filenames, filenames_no_ext):
            if filename_no_ext not in exclude_set:
                filtered_filenames.append(orig_filename)
                stats["kept"].append(filename_no_ext)
            else:
                stats["removed"].append(filename_no_ext)

        # Check for exclude entries that weren't found
        found_removed = set(stats["removed"])
        not_found = exclude_set - found_removed
        stats["not_found_in_data"] = list(not_found)

        # Log warnings for not found entries
        if stats["not_found_in_data"]:
            logger.warning(
                f"WARNING: {len(stats['not_found_in_data'])} files in EXCLUDE list were not found in data:"
            )
            for filename in stats["not_found_in_data"][:10]:
                logger.warning(f"  - {filename}")
            if len(stats["not_found_in_data"]) > 10:
                logger.warning(f"  ... and {len(stats['not_found_in_data']) - 10} more")

    # NO FILTER mode
    else:
        logger.info("No filtering applied (both include and exclude lists are empty)")
        filtered_filenames = filenames
        stats["kept"] = filenames_no_ext

    # Log summary
    logger.info("=" * 80)
    logger.info("FILTER RESULTS")
    logger.info("=" * 80)
    logger.info(f"Files before filtering: {len(filenames)}")
    logger.info(f"Files after filtering: {len(filtered_filenames)}")
    logger.info(f"Files kept: {len(stats['kept'])}")
    logger.info(f"Files removed: {len(stats['removed'])}")
    if stats["not_found_in_data"]:
        logger.warning(
            f"Filter entries not found in data: {len(stats['not_found_in_data'])}"
        )
    logger.info("=" * 80)

    return filtered_filenames, stats
