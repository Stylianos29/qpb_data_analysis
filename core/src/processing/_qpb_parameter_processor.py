import os
import logging
from typing import Optional

import pandas as pd

# Import library components
from library.data import DataFrameAnalyzer, HDF5Analyzer, load_csv

# Import parameter transformation components
from src.processing._param_transform_engine import (
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
        logger=None,
    ):
        """
        Initialize the processor with input and output paths.

        Args:
            - single_valued_csv_path: Path to CSV with single-valued
              parameters
            - multivalued_hdf5_path: Path to HDF5 with multivalued
              parameters
            - output_directory: Directory for output files
            - output_filename: Name of output CSV file
            - logger: Optional logger instance (if None, creates
              default)
        """
        self.single_valued_csv_path = single_valued_csv_path
        self.multivalued_hdf5_path = multivalued_hdf5_path
        self.output_directory = output_directory
        self.output_filename = output_filename

        # Use provided logger or create default
        self.logger = logger if logger is not None else logging.getLogger(__name__)

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
        self.logger.info("=" * 70)
        self.logger.info("=== STAGE: LOADING INPUT DATA ===")
        self.logger.info("=" * 70)

        # Load single-valued parameters CSV using library function
        self.logger.info(
            f"Loading CSV: {os.path.basename(self.single_valued_csv_path)}"
        )
        loaded_dataframe = load_csv(self.single_valued_csv_path, encoding="utf-8")

        if loaded_dataframe is None:
            raise ValueError(f"Failed to load CSV file: {self.single_valued_csv_path}")

        self.dataframe = loaded_dataframe

        # Initialize DataFrameAnalyzer for systematic data understanding
        self.dataframe_analyzer = DataFrameAnalyzer(self.dataframe)

        # Log data structure insights
        self.logger.info(
            f"✓ CSV loaded: {len(self.dataframe)} rows × {len(self.dataframe.columns)} columns"
        )
        self.logger.info(
            f"  - Single-valued columns: {len(self.dataframe_analyzer.list_of_single_valued_column_names)}"
        )
        self.logger.info(
            f"  - Multivalued columns: {len(self.dataframe_analyzer.list_of_multivalued_column_names)}"
        )

        # Initialize HDF5Analyzer for multivalued parameter access
        self.logger.info(
            f"Loading HDF5: {os.path.basename(self.multivalued_hdf5_path)}"
        )
        self.hdf5_analyzer = HDF5Analyzer(self.multivalued_hdf5_path)

        # Log HDF5 structure insights
        self.logger.info(
            f"✓ HDF5 loaded: {len(self.hdf5_analyzer.active_groups)} data groups"
        )
        self.logger.info(
            f"  - Output quantities: {len(self.hdf5_analyzer.list_of_output_quantity_names_from_hdf5)}"
        )

        # Validate data compatibility
        self._validate_data_compatibility()

        self.logger.info("Input data loading completed successfully")

    def _validate_data_compatibility(self) -> None:
        """Validate that CSV and HDF5 data are compatible."""
        self.logger.info("Validating CSV-HDF5 data compatibility...")

        # Ensure dataframe is loaded
        if self.dataframe is None:
            raise RuntimeError(
                "Cannot validate data compatibility: CSV data not loaded"
            )

        # Check that CSV has 'Filename' column for HDF5 mapping
        if "Filename" not in self.dataframe.columns:
            raise ValueError("CSV must contain 'Filename' column for HDF5 data mapping")

        # Additional validation logic
        csv_filenames = set(self.dataframe["Filename"].unique())
        self.logger.info(f"✓ CSV contains {len(csv_filenames)} unique filenames")
        self.logger.info(f"✓ Data compatibility validated")

    def _process_single_valued_parameters(self) -> None:
        """Process single-valued parameters using the transformation
        engine."""
        self.logger.info("=" * 70)
        self.logger.info("=== STAGE: PROCESSING SINGLE-VALUED PARAMETERS ===")
        self.logger.info("=" * 70)

        # Create and run transformation engine
        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is None before single-valued parameter processing"
            )

        transformation_engine = ParameterTransformationEngine(
            self.dataframe, logger=self.logger
        )
        self.dataframe = transformation_engine.apply_all_transformations()

        # Update analyzer with transformed data
        self.dataframe_analyzer = DataFrameAnalyzer(self.dataframe)

        self.logger.info("✓ Single-valued parameter processing completed")
        self.logger.info(
            f"  Final shape: {self.dataframe.shape[0]} rows × {self.dataframe.shape[1]} columns"
        )

    def _process_multivalued_parameters(self) -> None:
        """Process multivalued parameters from HDF5 file."""
        self.logger.info("=" * 70)
        self.logger.info("=== STAGE: PROCESSING MULTIVALUED PARAMETERS ===")
        self.logger.info("=" * 70)

        # Create and run HDF5 processor
        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is None before multivalued parameter processing"
            )

        hdf5_processor = HDF5ParameterProcessor(
            self.hdf5_analyzer, self.dataframe, logger=self.logger
        )

        # Log number of columns before
        cols_before = len(self.dataframe.columns)

        self.dataframe = hdf5_processor.process_all_hdf5_parameters()

        # Update analyzer with HDF5-enriched data
        self.dataframe_analyzer = DataFrameAnalyzer(self.dataframe)

        # Log number of columns added
        cols_after = len(self.dataframe.columns)
        cols_added = cols_after - cols_before

        self.logger.info(f"✓ Multivalued parameter processing completed")
        self.logger.info(f"  Added {cols_added} new columns from HDF5")
        self.logger.info(
            f"  Final shape: {self.dataframe.shape[0]} rows × {cols_after} columns"
        )

    def _process_analysis_cases(self) -> None:
        """Apply analysis-case-specific calculations."""
        self.logger.info("=" * 70)
        self.logger.info("=== STAGE: PROCESSING ANALYSIS CASES ===")
        self.logger.info("=" * 70)

        # Create and run analysis case processor
        if self.dataframe is None:
            raise RuntimeError("DataFrame is None before analysis case processing")

        analysis_processor = AnalysisCaseProcessor(self.dataframe, logger=self.logger)
        self.dataframe = analysis_processor.process_analysis_cases()

        self.logger.info("✓ Analysis case processing completed")

    def _finalize_processing(self) -> None:
        """Perform final validation and cleanup operations."""
        self.logger.info("=" * 70)
        self.logger.info("=== STAGE: FINALIZATION ===")
        self.logger.info("=" * 70)

        # Apply categorical dtypes after all transformations
        categorical_config = {
            "Kernel_operator_type": {
                "categories": ["Wilson", "Brillouin"],
                "ordered": False,
            },
            "Overlap_operator_method": {
                "categories": ["Chebyshev", "KL", "Bare", "Neuberger", "Zolotarev"],
                "ordered": False,
            },
        }

        # Ensure dataframe exists
        assert self.dataframe is not None, "DataFrame is None in finalization"

        from library.data.processing import apply_categorical_dtypes

        self.logger.info("Applying categorical dtypes...")
        self.dataframe = apply_categorical_dtypes(self.dataframe, categorical_config)
        self.logger.info("✓ Applied categorical dtypes to 2 columns")

        # Final data validation
        if self.dataframe is None:
            raise RuntimeError("Cannot finalize processing: DataFrame is None")
        if self.dataframe.empty:
            raise ValueError("Final DataFrame is empty - processing failed")

        # Log final data statistics
        final_analyzer = DataFrameAnalyzer(self.dataframe)
        self.logger.info("Final dataset summary:")
        self.logger.info(
            f"  Shape: {len(self.dataframe)} rows × {len(self.dataframe.columns)} columns"
        )
        self.logger.info(
            f"  Single-valued parameters: {len(final_analyzer.list_of_single_valued_column_names)}"
        )
        self.logger.info(
            f"  Multivalued parameters: {len(final_analyzer.list_of_multivalued_column_names)}"
        )

        # Log parameter categorization
        tunable_params = final_analyzer.list_of_tunable_parameter_names_from_dataframe
        output_params = [
            col
            for col in self.dataframe.columns
            if col not in tunable_params and col != "Filename"
        ]

        self.logger.info(f"  Tunable parameters: {len(tunable_params)}")
        self.logger.info(f"  Output parameters: {len(output_params)}")

        self.logger.info("✓ Finalization completed successfully")

    def _export_results(self) -> None:
        """Export the processed DataFrame to CSV."""
        output_path = os.path.join(self.output_directory, self.output_filename)

        self.logger.info("=" * 70)
        self.logger.info("=== EXPORTING RESULTS ===")
        self.logger.info("=" * 70)
        self.logger.info(f"Exporting to: {output_path}")

        if self.dataframe is not None:
            self.dataframe.to_csv(output_path, index=False)
            file_size = os.path.getsize(output_path) / 1024  # KB
            self.logger.info(f"✓ Export completed successfully ({file_size:.1f} KB)")
        else:
            self.logger.error("Export failed: DataFrame is None")
            raise RuntimeError("Cannot export results: DataFrame is None")
