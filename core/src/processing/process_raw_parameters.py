#!/usr/bin/env python3
"""
Refactored QPB log file parameter processing script.

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
import logging
from typing import Optional

import click
import pandas as pd

from library import (
    filesystem_utilities,
    validate_input_directory,
    validate_input_script_log_filename,
)
from library.data import DataFrameAnalyzer, HDF5Analyzer, load_csv
from src.preprocessing._param_transform_engine import (
    ParameterTransformationEngine,
    HDF5ParameterProcessor,
    AnalysisCaseProcessor,
)


class QPBParameterProcessor:
    """
    Main orchestrator for QPB parameter processing.

    This class coordinates all aspects of parameter processing using the
    modular, configuration-driven approach.
    """

    def __init__(
        self,
        single_valued_csv_path: str,
        multivalued_hdf5_path: str,
        output_directory: str,
        output_filename: str,
    ):
        """
        Initialize the processor with input and output paths.

        Args:
            single_valued_csv_path: Path to CSV with single-valued
            parameters multivalued_hdf5_path: Path to HDF5 with
            multivalued parameters output_directory: Directory for
            output files output_filename: Name of output CSV file
        """
        self.single_valued_csv_path = single_valued_csv_path
        self.multivalued_hdf5_path = multivalued_hdf5_path
        self.output_directory = output_directory
        self.output_filename = output_filename

        self.logger = logging.getLogger(__name__)

        # Initialize data containers
        self.dataframe: Optional[pd.DataFrame] = None
        self.dataframe_analyzer: Optional[DataFrameAnalyzer] = None
        self.hdf5_analyzer: Optional[HDF5Analyzer] = None

    def process_all_parameters(self) -> pd.DataFrame:
        """
        Execute the complete parameter processing pipeline.

        Returns:
            Processed DataFrame ready for export
        """
        self.logger.info("Starting QPB parameter processing pipeline")

        try:
            # Step 1: Load and validate input data
            self._load_input_data()

            # Step 2: Process single-valued parameters using
            # transformation engine
            self._process_single_valued_parameters()

            # Step 3: Process multivalued parameters from HDF5
            self._process_multivalued_parameters()

            # Step 4: Apply analysis-case-specific calculations
            self._process_analysis_cases()

            # Step 5: Final validation and cleanup
            self._finalize_processing()

            # Step 6: Export results
            self._export_results()

            if self.dataframe is None:
                raise RuntimeError("Pipeline completed but dataframe is None")

            self.logger.info("QPB parameter processing pipeline completed successfully")
            return self.dataframe

        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {e}")
            raise

    def _load_input_data(self) -> None:
        """Load and validate input data using library functions."""
        self.logger.info("Loading input data")

        # Load single-valued parameters CSV using library function
        loaded_dataframe = load_csv(
            self.single_valued_csv_path, encoding="utf-8", apply_categorical=True
        )

        if loaded_dataframe is None:
            raise ValueError(f"Failed to load CSV file: {self.single_valued_csv_path}")

        self.dataframe = loaded_dataframe

        # Initialize DataFrameAnalyzer for systematic data understanding
        self.dataframe_analyzer = DataFrameAnalyzer(self.dataframe)

        # Log data structure insights
        self.logger.info(
            f"Loaded CSV with {len(self.dataframe)} rows and {len(self.dataframe.columns)} columns"
        )
        self.logger.info(
            f"Identified {len(self.dataframe_analyzer.list_of_single_valued_column_names)} single-valued columns"
        )
        self.logger.info(
            f"Identified {len(self.dataframe_analyzer.list_of_multivalued_column_names)} multivalued columns"
        )

        # Initialize HDF5Analyzer for multivalued parameter access
        self.hdf5_analyzer = HDF5Analyzer(self.multivalued_hdf5_path)

        # Log HDF5 structure insights
        self.logger.info(
            f"Loaded HDF5 with {len(self.hdf5_analyzer.active_groups)} data groups"
        )
        self.logger.info(
            f"Found {len(self.hdf5_analyzer.list_of_output_quantity_names_from_hdf5)} output quantities"
        )

        # Validate data compatibility
        self._validate_data_compatibility()

    def _validate_data_compatibility(self) -> None:
        """Validate that CSV and HDF5 data are compatible."""

        # Ensure dataframe is loaded
        if self.dataframe is None:
            raise RuntimeError(
                "Cannot validate data compatibility: CSV data not loaded"
            )

        # Check that CSV has 'Filename' column for HDF5 mapping
        if "Filename" not in self.dataframe.columns:
            raise ValueError("CSV must contain 'Filename' column for HDF5 data mapping")

        # Additional validation logic could be added here
        csv_filenames = set(self.dataframe["Filename"].unique())
        self.logger.info(f"CSV contains {len(csv_filenames)} unique filenames")

    def _process_single_valued_parameters(self) -> None:
        """Process single-valued parameters using the transformation
        engine."""
        self.logger.info("Processing single-valued parameters")

        # Create and run transformation engine
        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is None before single-valued parameter processing"
            )
        transformation_engine = ParameterTransformationEngine(self.dataframe)
        self.dataframe = transformation_engine.apply_all_transformations()

        # Update analyzer with transformed data
        self.dataframe_analyzer = DataFrameAnalyzer(self.dataframe)

        self.logger.info("Single-valued parameter processing completed")

    def _process_multivalued_parameters(self) -> None:
        """Process multivalued parameters from HDF5 file."""
        self.logger.info("Processing multivalued parameters from HDF5")

        # Create and run HDF5 processor
        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is None before multivalued parameter processing"
            )
        hdf5_processor = HDF5ParameterProcessor(self.hdf5_analyzer, self.dataframe)
        self.dataframe = hdf5_processor.process_all_hdf5_parameters()

        # Update analyzer with HDF5-enriched data
        self.dataframe_analyzer = DataFrameAnalyzer(self.dataframe)

        self.logger.info("Multivalued parameter processing completed")

    def _process_analysis_cases(self) -> None:
        """Apply analysis-case-specific calculations."""
        self.logger.info("Processing analysis case calculations")

        # Create and run analysis case processor
        if self.dataframe is None:
            raise RuntimeError("DataFrame is None before analysis case processing")
        analysis_processor = AnalysisCaseProcessor(self.dataframe)
        self.dataframe = analysis_processor.process_analysis_cases()

        self.logger.info("Analysis case processing completed")

    def _finalize_processing(self) -> None:
        """Perform final validation and cleanup operations."""
        self.logger.info("Finalizing processing")

        # Final data validation
        if self.dataframe is None:
            raise RuntimeError("Cannot finalize processing: DataFrame is None")
        if self.dataframe.empty:
            raise ValueError("Final DataFrame is empty - processing failed")

        # Log final data statistics
        final_analyzer = DataFrameAnalyzer(self.dataframe)
        self.logger.info(
            f"Final DataFrame: {len(self.dataframe)} rows, {len(self.dataframe.columns)} columns"
        )
        self.logger.info(
            f"Final single-valued columns: {len(final_analyzer.list_of_single_valued_column_names)}"
        )
        self.logger.info(
            f"Final multivalued columns: {len(final_analyzer.list_of_multivalued_column_names)}"
        )

        # Optional: Generate processing report
        self._generate_processing_report(final_analyzer)

    def _generate_processing_report(self, final_analyzer: DataFrameAnalyzer) -> None:
        """Generate a summary report of the processing results."""
        report_lines = [
            "=== QPB Parameter Processing Report ===",
            f"Input CSV: {self.single_valued_csv_path}",
            f"Input HDF5: {self.multivalued_hdf5_path}",
            f"Output: {os.path.join(self.output_directory, self.output_filename)}",
            "",
            f"Final data shape: {self.dataframe.shape if self.dataframe is not None else 'DataFrame is None'}",
            f"Single-valued parameters: {len(final_analyzer.list_of_single_valued_column_names)}",
            f"Multivalued parameters: {len(final_analyzer.list_of_multivalued_column_names)}",
            "",
            "Column summary:",
        ]

        # Add column categorization
        for col in sorted(final_analyzer.list_of_dataframe_column_names):
            col_type = (
                "single"
                if col in final_analyzer.list_of_single_valued_column_names
                else "multi"
            )
            param_type = (
                "tunable"
                if col in final_analyzer.list_of_tunable_parameter_names_from_dataframe
                else "output"
            )
            report_lines.append(f"  {col}: {col_type}-valued {param_type}")

        report_content = "\n".join(report_lines)

        # Write report to file
        report_path = os.path.join(self.output_directory, "processing_report.txt")
        with open(report_path, "w") as f:
            f.write(report_content)

        self.logger.info(f"Processing report saved to {report_path}")

    def _export_results(self) -> None:
        """Export the processed DataFrame to CSV."""
        output_path = os.path.join(self.output_directory, self.output_filename)

        self.logger.info(f"Exporting results to {output_path}")
        if self.dataframe is not None:
            self.dataframe.to_csv(output_path, index=False)
            self.logger.info("Export completed successfully")
        else:
            self.logger.error("Export failed: DataFrame is None")
            raise RuntimeError("Cannot export results: DataFrame is None")


@click.command()
@click.option(
    "-in_param_csv",
    "--input_single_valued_csv_file_path",
    "input_single_valued_csv_file_path",
    required=True,
    callback=filesystem_utilities.validate_input_csv_file,
    help="Path to .csv file containing extracted single-valued parameter values from qpb log files.",
)
@click.option(
    "-in_param_hdf5",
    "--input_multivalued_hdf5_file_path",
    "input_multivalued_hdf5_file_path",
    required=True,
    callback=filesystem_utilities.validate_input_HDF5_file,
    help="Path to .hdf5 file containing extracted multivalued parameter values from qpb log files.",
)
@click.option(
    "-out_dir",
    "--output_files_directory",
    "output_files_directory",
    default=None,
    callback=validate_input_directory,
    help="Path to directory where all output files will be stored.",
)
@click.option(
    "-out_csv_name",
    "--output_csv_filename",
    "output_csv_filename",
    default="processed_qpb_log_files_extracted_values.csv",
    callback=filesystem_utilities.validate_output_csv_filename,
    help="Specific name for the output .csv file containing processed values.",
)
@click.option(
    "-log_on",
    "--enable_logging",
    "enable_logging",
    is_flag=True,
    default=False,
    help="Enable logging.",
)
@click.option(
    "-log_file_dir",
    "--log_file_directory",
    "log_file_directory",
    default=None,
    callback=filesystem_utilities.validate_script_log_file_directory,
    help="Directory where the script's log file will be stored.",
)
@click.option(
    "-log_name",
    "--log_filename",
    "log_filename",
    default=None,
    callback=validate_input_script_log_filename,
    help="Specific name for the script's log file.",
)
def main(
    input_single_valued_csv_file_path: str,
    input_multivalued_hdf5_file_path: str,
    output_files_directory: Optional[str],
    output_csv_filename: str,
    enable_logging: bool,
    log_file_directory: Optional[str],
    log_filename: Optional[str],
) -> None:
    """
    Process extracted QPB log file parameters using configuration-driven
    approach.

    This script transforms raw extracted parameters into analysis-ready
    format using systematic, maintainable processing classes.
    """

    # Handle default output directory
    if output_files_directory is None:
        output_files_directory = os.path.dirname(input_single_valued_csv_file_path)

    # Setup logging
    logger_wrapper = filesystem_utilities.LoggingWrapper(
        log_file_directory, log_filename, enable_logging
    )
    logger_wrapper.initiate_script_logging()

    try:
        # Create and run processor
        processor = QPBParameterProcessor(
            single_valued_csv_path=input_single_valued_csv_file_path,
            multivalued_hdf5_path=input_multivalued_hdf5_file_path,
            output_directory=output_files_directory,
            output_filename=output_csv_filename,
        )

        # Execute processing pipeline
        result_dataframe = processor.process_all_parameters()

        click.echo(
            "✓ Processing extracted values from QPB log files completed successfully."
        )
        click.echo(
            f"✓ Results saved to: {os.path.join(output_files_directory, output_csv_filename)}"
        )
        click.echo(
            f"✓ Final dataset: {result_dataframe.shape[0]} rows, "
            f"{result_dataframe.shape[1]} columns"
        )

    except Exception as e:
        click.echo(f"✗ Processing failed: {e}")
        raise

    finally:
        # Terminate logging
        logger_wrapper.terminate_script_logging()


if __name__ == "__main__":
    main()
