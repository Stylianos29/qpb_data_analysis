#!/usr/bin/env python3
"""
Unified Plateau Extraction Visualization Script

This script creates high-quality multi-panel visualizations of plateau
extraction results from both PCAC mass and pion effective mass analyses. 

It reads extraction results from HDF5 files and generates plots showing
individual jackknife samples with their detected plateau regions,
matching the output quality of the original extract_plateau_PCAC_mass.py
script.

Usage:
    python visualize_plateau_extraction.py \
        --analysis_type pcac_mass \
        -i plateau_PCAC_mass_extraction.h5 \
        -p plots_dir
        
    python visualize_plateau_extraction.py \
        --analysis_type pion_mass \
        -i plateau_pion_mass_extraction.h5 \
        -p plots_dir
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import click
import h5py

# Import library components
from library.constants.paths import ROOT
from library.constants.labels import TITLE_LABELS_BY_COLUMN_NAME
from library.visualization.managers.file_manager import PlotFileManager
from library.visualization.builders.title_builder import PlotTitleBuilder
from library.utils.logging_utilities import create_script_logger
from library.validation.click_validators import (
    hdf5_file,
    directory,
    validate_log_filename,
)
from library.data.hdf5_analyzer import HDF5Analyzer

# Import visualization components
from src.analysis.plateau_extraction._plateau_visualization_config import (
    get_analysis_config,
    validate_analysis_type,
    validate_visualization_config,
)
from src.analysis.plateau_extraction._plateau_visualization_core import (
    process_group_visualization,
)


@click.command()
@click.option(
    "--analysis_type",
    required=True,
    type=click.Choice(["pcac_mass", "pion_mass"], case_sensitive=False),
    help="Type of analysis: 'pcac_mass' or 'pion_mass'",
)
@click.option(
    "-i",
    "--input_hdf5_file",
    required=True,
    callback=hdf5_file.input,
    help="Path to input HDF5 file containing plateau extraction results.",
)
@click.option(
    "-p",
    "--plots_directory",
    required=True,
    callback=directory.can_create,
    help="Directory for output plots.",
)
@click.option(
    "-clear",
    "--clear_existing_plots",
    is_flag=True,
    default=False,
    help="Clear existing plots before creating new ones.",
)
@click.option(
    "-log_on",
    "--enable_logging",
    is_flag=True,
    default=False,
    help="Enable logging to file.",
)
@click.option(
    "-log_dir",
    "--log_directory",
    default=None,
    callback=directory.can_create,
    help="Directory for log files. Default: same as plots directory.",
)
@click.option(
    "-log_name",
    "--log_filename",
    callback=validate_log_filename,
    help="Custom log filename. Default: auto-generated.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose console output.",
)
def main(
    analysis_type: str,
    input_hdf5_file: str,
    plots_directory: str,
    clear_existing_plots: bool,
    enable_logging: bool,
    log_directory: Optional[str],
    log_filename: Optional[str],
    verbose: bool,
) -> None:
    """
    Create high-quality visualizations of plateau extraction results.

    This script processes plateau extraction results from HDF5 files and
    creates multi-panel plots showing individual jackknife samples with
    their detected plateau regions.

    The script handles both PCAC mass and pion effective mass analyses,
    automatically adapting the visualization style and parameters based
    on the analysis type.
    """
    # Validate configuration
    try:
        validate_analysis_type(analysis_type)
        validate_visualization_config()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        sys.exit(1)

    # Set up logging
    log_dir = (
        log_directory
        if log_directory
        else os.path.dirname(input_hdf5_file) if enable_logging else None
    )

    logger = create_script_logger(
        log_directory=log_dir,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=verbose,
    )

    logger.log_script_start("Plateau extraction visualization")

    try:
        # Log input parameters
        logger.info(f"Analysis type: {analysis_type}")
        logger.info(f"Input HDF5 file: {input_hdf5_file}")
        logger.info(f"Plots directory: {plots_directory}")
        logger.info(f"Clear existing plots: {clear_existing_plots}")

        # Get analysis-specific configuration
        analysis_config = get_analysis_config(analysis_type)
        logger.info(f"Configured for {analysis_config['description']} analysis")

        # Prepare output directories and managers
        file_manager, title_builder = _prepare_visualization_tools(
            plots_directory, analysis_config, clear_existing_plots, logger
        )

        # Process HDF5 file
        visualization_results = _process_hdf5_file(
            input_hdf5_file,
            analysis_config,
            file_manager,
            title_builder,
            logger,
            verbose,
        )

        # Report final statistics
        _report_final_statistics(visualization_results, logger)

        logger.log_script_end("Plateau extraction visualization completed successfully")
        click.echo(
            f"✅ Visualization complete! Created {len(visualization_results)} plot sets."
        )
        click.echo(f"   Plots saved to: {Path(plots_directory).relative_to(ROOT)}")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.log_script_end("Plateau extraction visualization failed")
        click.echo(f"❌ ERROR: {e}", err=True)
        sys.exit(1)


def _prepare_visualization_tools(
    plots_directory: str,
    analysis_config: Dict,
    clear_existing_plots: bool,
    logger,
) -> tuple[PlotFileManager, PlotTitleBuilder]:
    """
    Prepare visualization tools and output directories.

    Args:
        plots_directory: Base plots directory
        analysis_config: Analysis-specific configuration
        clear_existing_plots: Whether to clear existing plots
        logger: Logger instance

    Returns:
        Tuple of (PlotFileManager, PlotTitleBuilder)
    """
    logger.info("Setting up visualization tools...")

    # Create plot file manager
    file_manager = PlotFileManager(plots_directory)

    # Prepare analysis-specific subdirectory
    plot_subdir = analysis_config["plot_subdirectory"]
    subdir_path = file_manager.prepare_subdirectory(
        plot_subdir,
        clear_existing=clear_existing_plots,
        confirm_clear=False,  # Don't require confirmation in script
    )

    if clear_existing_plots:
        logger.info(f"Cleared existing plots in {plot_subdir}")

    logger.info(f"Plot output directory: {subdir_path}")

    # Create title builder with standard labels
    title_builder = PlotTitleBuilder(
        title_labels=TITLE_LABELS_BY_COLUMN_NAME,
        title_number_format=".2f",
    )

    logger.info("Visualization tools prepared successfully")
    return file_manager, title_builder


def _process_hdf5_file(
    input_hdf5_file: str,
    analysis_config: Dict,
    file_manager: PlotFileManager,
    title_builder: PlotTitleBuilder,
    logger,
    verbose: bool,
) -> List[Dict]:
    """
    Process HDF5 file and create visualizations for all groups.

    Args:
        input_hdf5_file: Path to input HDF5 file
        analysis_config: Analysis-specific configuration
        file_manager: PlotFileManager instance
        title_builder: PlotTitleBuilder instance
        logger: Logger instance
        verbose: Whether to show verbose output

    Returns:
        List of visualization results for each group
    """
    logger.info(f"Processing HDF5 file: {input_hdf5_file}")

    # Validate input file exists and is readable
    if not os.path.exists(input_hdf5_file):
        raise FileNotFoundError(f"Input HDF5 file not found: {input_hdf5_file}")

    try:
        with h5py.File(input_hdf5_file, "r") as hdf5_file:
            # Find groups with extraction results
            valid_groups = _find_extraction_groups(hdf5_file, analysis_config, logger)

            if not valid_groups:
                logger.warning("No valid extraction groups found")
                return []

            logger.info(f"Found {len(valid_groups)} groups to visualize")

            # Process each group
            visualization_results = []
            for group_idx, group_path in enumerate(valid_groups):
                group_name = os.path.basename(group_path)

                if verbose:
                    click.echo(
                        f"  Processing group {group_idx + 1}/{len(valid_groups)}: {group_name}"
                    )

                logger.info(f"Processing group: {group_path}")

                try:
                    group = hdf5_file[group_path]
                    # Validate that group is indeed an HDF5 group
                    if not isinstance(group, h5py.Group):
                        raise ValueError(f"Expected h5py.Group, got {type(group)}")
                    plot_paths = process_group_visualization(
                        group,
                        group_name,
                        analysis_config,
                        title_builder,
                        file_manager,
                        logger,
                    )

                    result = {
                        "success": len(plot_paths) > 0,
                        "group_name": group_name,
                        "group_path": group_path,
                        "n_plots_created": len(plot_paths),
                        "plot_paths": plot_paths,
                    }

                    if result["success"]:
                        logger.info(
                            f"✅ Group {group_name}: Created {len(plot_paths)} plots"
                        )
                    else:
                        logger.warning(f"⚠️ Group {group_name}: No plots created")

                except Exception as e:
                    logger.error(f"❌ Failed to process group {group_name}: {e}")
                    result = {
                        "success": False,
                        "group_name": group_name,
                        "group_path": group_path,
                        "error_message": str(e),
                        "n_plots_created": 0,
                        "plot_paths": [],
                    }

                visualization_results.append(result)

        return visualization_results

    except Exception as e:
        logger.error(f"Failed to process HDF5 file: {e}")
        raise


def _find_extraction_groups(
    hdf5_file: h5py.File,
    analysis_config: Dict,
    logger,
) -> List[str]:
    """
    Find all groups containing plateau extraction results.

    Args:
        hdf5_file: HDF5 file object
        analysis_config: Analysis-specific configuration
        logger: Logger instance

    Returns:
        List of group paths containing extraction results
    """
    logger.info("Searching for extraction result groups...")

    try:
        analyzer = HDF5Analyzer(hdf5_file.filename)

        datasets = analysis_config["input_datasets"]
        required_datasets = [
            datasets["time_series"],
            datasets["plateau_estimates"],
            datasets["sigma_thresholds"],
        ]

        valid_groups = []
        for group_path in analyzer.active_groups:
            try:
                group = hdf5_file[group_path]
                if not isinstance(group, h5py.Group):
                    continue

                # Check for required datasets
                has_all_datasets = all(
                    dataset in group for dataset in required_datasets
                )

                # Check for extraction success attribute
                extraction_success = group.attrs.get(
                    "plateau_extraction_success", False
                )

                if has_all_datasets and extraction_success:
                    valid_groups.append(group_path)
                    logger.debug(f"Valid extraction group: {group_path}")
                else:
                    missing = [ds for ds in required_datasets if ds not in group]
                    if missing:
                        logger.debug(f"Group {group_path} missing datasets: {missing}")
                    if not extraction_success:
                        logger.debug(
                            f"Group {group_path} has extraction_success={extraction_success}"
                        )

            except Exception as e:
                logger.debug(f"Error checking group {group_path}: {e}")
                continue

        analyzer.close()
        logger.info(f"Found {len(valid_groups)} valid extraction groups")
        return valid_groups

    except Exception as e:
        logger.error(f"Error finding extraction groups: {e}")
        return []


def _report_final_statistics(visualization_results: List[Dict], logger) -> None:
    """
    Report final statistics and summary.

    Args:
        visualization_results: List of visualization results
        logger: Logger instance
    """
    if not visualization_results:
        logger.warning("No visualization results to report")
        return

    n_total = len(visualization_results)
    n_successful = sum(1 for result in visualization_results if result["success"])
    n_failed = n_total - n_successful
    total_plots = sum(
        result.get("n_plots_created", 0) for result in visualization_results
    )

    logger.info("=" * 60)
    logger.info("VISUALIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total groups processed: {n_total}")
    logger.info(f"Successful groups: {n_successful}")
    logger.info(f"Failed groups: {n_failed}")
    logger.info(f"Total plots created: {total_plots}")

    if n_failed > 0:
        logger.info("\nFailed groups:")
        for result in visualization_results:
            if not result["success"]:
                group_name = result["group_name"]
                error_msg = result.get("error_message", "Unknown error")
                logger.info(f"  - {group_name}: {error_msg}")

    if n_successful > 0:
        logger.info("\nSuccessful groups:")
        for result in visualization_results:
            if result["success"]:
                group_name = result["group_name"]
                n_plots = result["n_plots_created"]
                logger.info(f"  - {group_name}: {n_plots} plots")

    success_rate = (n_successful / n_total) * 100 if n_total > 0 else 0
    logger.info(f"\nSuccess rate: {success_rate:.1f}% ({n_successful}/{n_total})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
