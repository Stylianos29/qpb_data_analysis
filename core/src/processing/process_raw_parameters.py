#!/usr/bin/env python3
"""
QPB log file parameter processing script.

This script processes extracted parameter values from QPB log files
using a configuration-driven, modular approach. It replaces the original
hardcoded transformation logic with systematic, maintainable processing
classes.

Key improvements: 
    - Configuration-driven transformations 
    - Modular processor architecture 
    - Proper use of library classes (DataFrameAnalyzer, HDF5Analyzer) 
    - Clear separation of concerns 
    - Comprehensive logging and error handling
"""

import os
from typing import Optional

import click

# Import library components
from library import ROOT
from library.validation.click_validators import (
    hdf5_file,
    csv_file,
    directory,
    validate_log_filename,
)
from library.utils.logging_utilities import create_script_logger

# Import from auxiliary module
from src.processing._qpb_parameter_processor import QPBParameterProcessor


@click.command()
@click.option(
    "-in_param_csv",
    "--input_single_valued_csv_file_path",
    required=True,
    callback=csv_file.input,
    help="Path to .csv file containing extracted single-valued parameter values from qpb log files.",
)
@click.option(
    "-in_param_hdf5",
    "--input_multivalued_hdf5_file_path",
    required=True,
    callback=hdf5_file.input,
    help="Path to .hdf5 file containing extracted multivalued parameter values from qpb log files.",
)
@click.option(
    "-out_dir",
    "--output_directory",
    default=None,
    callback=directory.must_exist,
    help="Directory for output files. If not specified, uses input file directory.",
)
@click.option(
    "-out_csv_name",
    "--output_csv_filename",
    default="processed_qpb_log_files_extracted_values.csv",
    callback=csv_file.output,
    help="Specific name for the output .csv file containing processed values.",
)
@click.option(
    "-log_on",
    "--enable_logging",
    is_flag=True,
    default=False,
    help="Enable detailed logging to file.",
)
@click.option(
    "-log_dir",
    "--log_directory",
    default=None,
    callback=directory.can_create,
    help="Directory for log files. Default: output directory",
)
@click.option(
    "-log_name",
    "--log_filename",
    default=None,
    callback=validate_log_filename,
    help="Specific name for the script's log file.",
)
def main(
    input_single_valued_csv_file_path: str,
    input_multivalued_hdf5_file_path: str,
    output_directory: Optional[str],
    output_csv_filename: str,
    enable_logging: bool,
    log_directory: Optional[str],
    log_filename: Optional[str],
) -> None:
    """
    Process extracted QPB log file parameters using configuration-driven
    approach.

    This script transforms raw extracted parameters into analysis-ready
    format using systematic, maintainable processing classes.
    """

    # Handle default output directory
    if output_directory is None:
        output_directory = os.path.dirname(input_single_valued_csv_file_path)

    # Setup logging
    logger = create_script_logger(
        log_directory=log_directory,
        log_filename=log_filename,
        enable_file_logging=enable_logging,
        enable_console_logging=False,  # Keep console output separate via click.echo
        verbose=False,
    )

    logger.log_script_start("QPB parameter processing")

    try:
        # Create and run processor
        processor = QPBParameterProcessor(
            single_valued_csv_path=input_single_valued_csv_file_path,
            multivalued_hdf5_path=input_multivalued_hdf5_file_path,
            output_directory=output_directory,
            output_filename=output_csv_filename,
        )

        logger.info(f"Initialized processor for {input_single_valued_csv_file_path}")
        logger.info(
            f"Output target: {os.path.join(output_directory, output_csv_filename)}"
        )

        # Execute processing pipeline
        logger.info("Starting parameter processing pipeline")
        result_dataframe = processor.process_all_parameters()
        logger.info(
            f"Processing completed: {result_dataframe.shape[0]} rows, {result_dataframe.shape[1]} columns"
        )

        click.echo(
            "✓ Processing extracted values from QPB log files completed successfully."
        )
        full_path = os.path.join(output_directory, output_csv_filename)
        relative_path = os.path.relpath(full_path, ROOT)
        click.echo(f"✓ Results saved to: {relative_path}")
        click.echo(
            f"✓ Final dataset: {result_dataframe.shape[0]} rows, "
            f"{result_dataframe.shape[1]} columns"
        )

    except Exception as e:
        logger.log_script_error(e)
        click.echo(f"✗ Processing failed: {e}")
        raise

    finally:
        # Proper logging cleanup
        logger.log_script_end("QPB parameter processing completed")
        logger.close()


if __name__ == "__main__":
    main()
