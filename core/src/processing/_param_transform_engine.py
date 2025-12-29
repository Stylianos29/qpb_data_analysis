"""
Enhanced Parameter Transformation Engine with Solver Resolution.

This module extends the transformation engine to include systematic
resolution of ambiguous solver parameters into canonical names.
"""

import logging
from typing import Dict, Optional
from typing import cast

import numpy as np
import pandas as pd

from src.processing._param_transform_config import (
    RATIONAL_ORDER_RESOLUTION_MAPPING,
    SOLVER_PARAMETER_RESOLUTION_RULES,
    RAW_SOLVER_PARAMETER_NAMES,
    STRING_TRANSFORMATIONS,
    MATH_TRANSFORMATIONS,
    EXTRACTION_RULES,
    COLUMN_OPERATIONS,
    HDF5_PROCESSING_RULES,
    ANALYSIS_CASES,
    TIME_COST_CALCULATIONS,
    FORMATTER_FUNCTIONS,
    PARSER_FUNCTIONS,
    CALCULATION_FUNCTIONS,
)


class ParameterTransformationEngine:
    """
    Enhanced transformation engine with solver parameter resolution.

    This class orchestrates all parameter transformations including the
    new solver parameter resolution step that disambiguates Inner/Outer/
    Generic solver parameters into canonical CG/MSCG names.
    """

    def __init__(self, dataframe: pd.DataFrame, logger=None):
        """
        Initialize the transformation engine.

        Args:
            dataframe: The DataFrame containing single-valued parameters
            logger: Optional logger instance
        """
        self.dataframe = dataframe.copy()
        self.original_dataframe = dataframe.copy()
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def apply_all_transformations(self) -> pd.DataFrame:
        """
        Apply all configured transformations in the correct dependency
        order.

        Order of operations:
            1. String transformations (formatting, replacements)
            2. Extraction rules (compound parameter splitting)
            3. **Solver parameter resolution (NEW)**
            4. Column operations (additions, default values, derived
               calcs)
            5. Mathematical transformations (sqrt, ratios, type
               conversions)

        Returns:
            Transformed DataFrame
        """
        self.logger.info("Starting parameter transformation pipeline")

        # Apply transformations in dependency order
        self._apply_string_transformations()
        self._apply_extraction_rules()
        self._resolve_rational_order()
        self._resolve_solver_parameters()
        self._apply_column_operations()
        self._apply_math_transformations()

        self.logger.info("Parameter transformation pipeline completed")
        return self.dataframe.copy()

    def _resolve_rational_order(self) -> None:
        """
        Resolve generic Rational_order to method-specific parameter
        names.

        **Problem**: The Rational_order parameter parsed from filenames
        is generic and needs to be mapped to the specific parameter name
        used by each overlap operator method.

        **Resolution Mapping**:
            - KL method: Rational_order → KL_diagonal_order
            - Neuberger method: Rational_order → Neuberger_order
            - Zolotarev method: Rational_order → Zolotarev_order
            - Bare/Chebyshev: No resolution needed (doesn't use rational
              approximation)

        **Process**:
            1. Check if Rational_order column exists
            2. For each row, get the Overlap_operator_method
            3. Look up the appropriate target parameter name
            4. Create new column with method-specific name
            5. Original Rational_order is preserved for later removal

        **Modifies**:
            - self.dataframe: Adds method-specific order columns
            (e.g., Zolotarev_order, KL_diagonal_order)

        **Logs**:
            - Summary of resolution (e.g., "Resolved Rational_order →
              Zolotarev_order for 201 rows")

        **Example**:
            Input: Overlap_operator_method='Zolotarev',
            Rational_order=10 Output: Creates Zolotarev_order=10 column

        **Notes**:
            - The generic Rational_order column is removed later by
              _apply_column_operations()
            - Only applies to methods that use rational approximations
              (KL, Neuberger, Zolotarev)
        """

        if "Rational_order" not in self.dataframe.columns:
            return
        if "Overlap_operator_method" not in self.dataframe.columns:
            return

        self.logger.info("Resolving Rational_order to method-specific names")

        resolved_count = 0

        for idx in self.dataframe.index:
            method = self.dataframe.loc[idx, "Overlap_operator_method"]

            # Type check: ensure method is a string
            if not isinstance(method, str):
                continue

            if method in RATIONAL_ORDER_RESOLUTION_MAPPING:
                target_col = RATIONAL_ORDER_RESOLUTION_MAPPING[method]
                self.dataframe.loc[idx, target_col] = self.dataframe.loc[
                    idx, "Rational_order"
                ]
                resolved_count += 1

        if resolved_count > 0:
            method = self.dataframe["Overlap_operator_method"].iloc[0]

            # Type check here too
            if isinstance(method, str):
                target = RATIONAL_ORDER_RESOLUTION_MAPPING.get(method, "unknown")
                self.logger.info(
                    f"✓ Resolved Rational_order → {target} for {resolved_count} rows"
                )

    def _resolve_solver_parameters(self) -> None:
        """
        Resolve ambiguous solver parameters to canonical names based on
        context.

        **Problem**: QPB log files use generic parameter names
        (Inner_solver_*, Outer_solver_*) whose meaning depends on the
        overlap operator method and program type. This method
        disambiguates them into canonical names.

        **Resolution Strategy**:

        For Zolotarev/KL/Neuberger overlap operators:
            - Inner_solver_* → MSCG_* (multi-shift CG for sign function)
            - Outer_solver_* → CG_* (conjugate gradient for full
              inversion)

        For Chebyshev overlap operator:
            - Generic_solver_* → MSCG_* (forward applications)
            - Generic_solver_* → CG_* (inversion programs)

        For Bare operator:
            - Generic_solver_* → CG_* (no multi-shift solver)

        **Process**:
            1. Check each row's Overlap_operator_method and
               Main_program_type
            2. Look up the appropriate parameter mapping
            3. Create new columns with canonical names (MSCG_epsilon,
               CG_epsilon, etc.)
            4. Original raw parameters are preserved for later removal

        **Modifies**:
            - self.dataframe: Adds new columns with canonical parameter
              names

        **Logs**:
            - List of canonical parameters created
            - List of raw parameters that will be removed later

        **Example**:
            Input columns: Inner_solver_epsilon, Outer_solver_epsilon
            Output columns (added): MSCG_epsilon, CG_epsilon

        **Notes**:
            - Raw parameters (Inner_solver_*, Outer_solver_*,
              Generic_solver_*)
            are removed later in the pipeline by
            _apply_column_operations() - This resolution is necessary
            for correct downstream analysis
        """
        self.logger.info("Resolving solver parameters to canonical names")

        if "Overlap_operator_method" not in self.dataframe.columns:
            self.logger.warning(
                "Overlap_operator_method not found - skipping solver resolution"
            )
            return

        resolved_params_set = set()

        # Iterate over index directly instead of using iterrows()
        for idx in self.dataframe.index:
            overlap_method = self.dataframe.loc[idx, "Overlap_operator_method"]
            main_program_type = (
                self.dataframe.loc[idx, "Main_program_type"]
                if "Main_program_type" in self.dataframe.columns
                else ""
            )

            if not isinstance(overlap_method, str) or not overlap_method:
                continue

            is_invert = bool(main_program_type == "invert")
            mapping_key = (overlap_method, is_invert)

            if mapping_key not in SOLVER_PARAMETER_RESOLUTION_RULES:
                continue

            resolution_mapping = SOLVER_PARAMETER_RESOLUTION_RULES[mapping_key]

            for raw_param, canonical_param in resolution_mapping.items():
                if raw_param in self.dataframe.columns:
                    # Now idx has the correct type from the index
                    self.dataframe.loc[idx, canonical_param] = self.dataframe.loc[
                        idx, raw_param
                    ]
                    resolved_params_set.add(canonical_param)

        if resolved_params_set:
            self.logger.info(
                f"✓ Created canonical solver parameters: {sorted(list(resolved_params_set))}"
            )

        unresolved_params = [
            col for col in RAW_SOLVER_PARAMETER_NAMES if col in self.dataframe.columns
        ]
        if unresolved_params:
            self.logger.info(
                f"  Raw solver parameters will be removed: {unresolved_params}"
            )

    def _apply_string_transformations(self) -> None:
        """Apply all string-based transformations."""
        self.logger.info("Applying string transformations")

        for column_name, config in STRING_TRANSFORMATIONS.items():
            if column_name not in self.dataframe.columns:
                continue

            # Apply value replacements
            if "replacements" in config:
                self.dataframe[column_name] = self.dataframe[column_name].replace(
                    config["replacements"]
                )
                self.logger.info(f"Applied replacements to {column_name}")

            # Apply formatting
            if "formatter" in config:
                formatter_name = config["formatter"]
                formatter_func = FORMATTER_FUNCTIONS.get(formatter_name)

                if formatter_func is not None:
                    # Get formatter parameters
                    formatter_params = {
                        k: v for k, v in config.items() if k not in ["formatter"]
                    }

                    # Assign to local variable for type narrowing in
                    # lambda
                    func = formatter_func

                    self.dataframe[column_name] = self.dataframe[column_name].apply(
                        lambda x: func(x, **formatter_params)
                    )

    def _apply_extraction_rules(self) -> None:
        """Apply compound parameter extraction rules."""
        self.logger.info("Applying extraction rules")

        for source_column, config in EXTRACTION_RULES.items():
            if source_column not in self.dataframe.columns:
                continue

            # Parse the source column if needed
            parser_name = config.get("parser")
            if parser_name:
                parser_func = PARSER_FUNCTIONS.get(parser_name)
                if parser_func:
                    self.dataframe[source_column] = self.dataframe[source_column].apply(
                        parser_func
                    )

            # Handle extraction
            if "extract_to" in config:
                for target_column, extraction_config in config["extract_to"].items():
                    index = extraction_config["index"]
                    target_type = extraction_config["type"]

                    self.dataframe[target_column] = self.dataframe[source_column].apply(
                        lambda x: (
                            target_type(x[index])
                            if isinstance(x, (list, tuple))
                            else None
                        )
                    )
                    self.logger.info(f"Extracted {target_column} from {source_column}")

            # Handle custom transformations
            if "transform" in config:
                transform_name = config["transform"]
                if transform_name == "remove_first_element_and_stringify":
                    self.dataframe[source_column] = self.dataframe[source_column].apply(
                        lambda x: (
                            str(tuple(x[1:]))
                            if isinstance(x, (list, tuple)) and len(x) > 1
                            else str(x)
                        )
                    )
                    self.logger.info(f"Applied {transform_name} to {source_column}")

            # Remove source column if specified
            if config.get("remove_source", False):
                self.dataframe.drop(columns=[source_column], inplace=True)
                self.logger.info(f"Removed source column {source_column}")

    def _apply_column_operations(self) -> None:
        """Apply column addition, removal, and derived calculations."""
        self.logger.info("Applying column operations")

        # Apply column additions
        for result_column, source_columns in COLUMN_OPERATIONS.get(
            "additions", {}
        ).items():
            existing_sources = [
                col for col in source_columns if col in self.dataframe.columns
            ]

            if len(existing_sources) > 1:
                # Sum the columns
                self.dataframe[result_column] = self.dataframe[existing_sources].sum(
                    axis=1
                )
                self.logger.info(f"Added columns {existing_sources} → {result_column}")
            elif len(existing_sources) == 1:
                # Just rename/keep the single column
                if result_column != existing_sources[0]:
                    self.dataframe[result_column] = self.dataframe[existing_sources[0]]

        # Apply default values for missing columns
        for column_name, default_value in COLUMN_OPERATIONS.get(
            "default_values", {}
        ).items():
            if column_name not in self.dataframe.columns:
                self.dataframe[column_name] = default_value
                self.logger.info(
                    f"Set default value for {column_name} = {default_value}"
                )

        # Apply derived calculations
        for result_column, config in COLUMN_OPERATIONS.get(
            "derived_calculations", {}
        ).items():
            formula_name = config["formula"]
            input_columns = config["inputs"]

            # Check if all input columns exist
            if all(col in self.dataframe.columns for col in input_columns):
                formula_func = CALCULATION_FUNCTIONS.get(formula_name)

                if formula_func is not None:
                    # Assign to local variable for type narrowing in
                    # lambda
                    func = formula_func

                    try:
                        self.dataframe[result_column] = self.dataframe.apply(
                            lambda row: func(*[row[col] for col in input_columns]),
                            axis=1,
                        )
                        self.logger.info(
                            f"Calculated {result_column} using {formula_name}"
                        )
                    except Exception as e:
                        self.logger.error(f"Error calculating {result_column}: {e}")

        # Remove specified columns
        columns_to_remove = COLUMN_OPERATIONS.get("columns_to_remove", [])
        existing_columns_to_remove = [
            col for col in columns_to_remove if col in self.dataframe.columns
        ]

        if existing_columns_to_remove:
            self.dataframe.drop(columns=existing_columns_to_remove, inplace=True)
            self.logger.info(f"Removed columns: {existing_columns_to_remove}")

    def _apply_math_transformations(self) -> None:
        """Apply mathematical transformations."""
        self.logger.info("Applying mathematical transformations")

        # Apply square root transformations
        for column_name in MATH_TRANSFORMATIONS.get("sqrt_parameters", []):
            if column_name in self.dataframe.columns:
                self.dataframe[column_name] = np.sqrt(
                    self.dataframe[column_name].apply(float)
                )
                self.logger.info(f"Applied square root to {column_name}")

        # Apply ratio calculations
        for result_column, (numerator, denominator) in MATH_TRANSFORMATIONS.get(
            "ratio_calculations", {}
        ).items():
            if (
                numerator in self.dataframe.columns
                and denominator in self.dataframe.columns
            ):
                self.dataframe[result_column] = self.dataframe[numerator].apply(
                    float
                ) / self.dataframe[denominator].apply(float)
                self.logger.info(
                    f"Calculated {result_column} as {numerator}/{denominator}"
                )

        # Apply type conversions
        for column_name, target_type in MATH_TRANSFORMATIONS.get(
            "type_conversions", {}
        ).items():
            if column_name in self.dataframe.columns:
                if target_type == int:
                    self.dataframe[column_name] = self.dataframe[column_name].astype(
                        int
                    )
                    self.logger.info(f"Converted {column_name} to integer type")


class HDF5ParameterProcessor:
    """
    Processor for extracting and transforming parameters from HDF5
    files.

    This class handles the multivalued parameter processing using the
    configuration-driven approach.
    """

    def __init__(self, hdf5_analyzer, dataframe: pd.DataFrame, logger=None):
        """
        Initialize the HDF5 processor.

        Args:
            - hdf5_analyzer: HDF5Analyzer instance for data access
            - dataframe: DataFrame to merge results into
            - logger: Optional logger instance
        """
        self.hdf5_analyzer = hdf5_analyzer
        self.dataframe = dataframe
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def process_all_hdf5_parameters(self) -> pd.DataFrame:
        self.logger.info("Processing HDF5 parameters...")

        self._handle_vector_spinor_columns()
        self._validate_vector_spinor_counts()

        cols_before = len(self.dataframe.columns)  # Track before

        for dataset_name, config in HDF5_PROCESSING_RULES.items():
            self._process_hdf5_dataset(dataset_name, config)

        cols_after = len(self.dataframe.columns)  # Track after
        cols_added = cols_after - cols_before

        self.logger.info(f"✓ Processed {cols_added} HDF5 datasets")  # Use actual count
        return self.dataframe

    def _handle_vector_spinor_columns(self) -> None:
        """
        Handle Number_of_vectors and Number_of_spinors columns with
        fallback logic.

        This method implements the business logic for these special
        columns that's too complex for configuration-driven processing.
        """
        # Safely get Main_program_type with proper column existence
        # check
        main_program_type = None
        if "Main_program_type" in self.dataframe.columns and len(self.dataframe) > 0:
            main_program_type = self.dataframe["Main_program_type"].iloc[0]

        # Handle Number_of_spinors (only for invert cases)
        if main_program_type == "invert":
            if "Number_of_spinors" not in self.dataframe.columns:
                # Try to deduce from HDF5 dataset lengths
                spinor_count = self._deduce_spinor_count_from_hdf5()
                if spinor_count is not None:
                    self.dataframe["Number_of_spinors"] = spinor_count
                    self.logger.info(
                        f"Deduced Number_of_spinors = {spinor_count} from HDF5"
                    )
                else:
                    self.dataframe["Number_of_spinors"] = 12  # Default
                    self.logger.info("Using default Number_of_spinors = 12")

        # Handle Number_of_vectors
        if "Number_of_vectors" not in self.dataframe.columns:
            if main_program_type != "invert":
                # Try to deduce from HDF5
                vector_count = self._deduce_vector_count_from_hdf5()
                if vector_count is not None:
                    self.dataframe["Number_of_vectors"] = vector_count
                    self.logger.info(
                        f"Deduced Number_of_vectors = {vector_count} from HDF5"
                    )
                else:
                    self.dataframe["Number_of_vectors"] = 1  # Default
                    self.logger.info("Using default Number_of_vectors = 1")
            else:
                # For invert cases, default to 1
                self.dataframe["Number_of_vectors"] = 1
                self.logger.info("Using default Number_of_vectors = 1 for invert case")

    def _deduce_spinor_count_from_hdf5(self) -> Optional[int]:
        """Try to deduce spinor count from HDF5 dataset lengths."""
        test_datasets = [
            "CG_total_calculation_time_per_spinor",
            "Total_number_of_CG_iterations_per_spinor",
        ]

        for dataset_name in test_datasets:
            if (
                dataset_name
                in self.hdf5_analyzer.list_of_output_quantity_names_from_hdf5
            ):
                try:
                    data_values = self.hdf5_analyzer.dataset_values(
                        dataset_name, return_gvar=False
                    )
                    if data_values and len(data_values) > 0:
                        return len(data_values[0])  # Length of first group's data
                except:
                    continue
        return None

    def _deduce_vector_count_from_hdf5(self) -> Optional[int]:
        """Try to deduce vector count from HDF5 dataset lengths."""
        if (
            "Calculation_result_per_vector"
            in self.hdf5_analyzer.list_of_output_quantity_names_from_hdf5
        ):
            try:
                data_values = self.hdf5_analyzer.dataset_values(
                    "Calculation_result_per_vector", return_gvar=False
                )
                if data_values and len(data_values) > 0:
                    return len(data_values[0])  # Length of first group's data
            except:
                pass
        return None

    def _add_mean_with_error_column(
        self, dataset_dict: Dict, output_column: str, config: Dict
    ) -> None:
        """Add column with mean values and errors where applicable."""
        result_dict = {}

        for filename, dataset_values in dataset_dict.items():
            if not isinstance(dataset_values, np.ndarray):
                dataset_values = np.array(dataset_values)

            if len(dataset_values) > 1:
                mean_val = float(np.mean(dataset_values))
                error_val = float(
                    np.std(dataset_values, ddof=1) / np.sqrt(len(dataset_values))
                )
                result_dict[filename] = (mean_val, error_val)
            elif len(dataset_values) == 1 and "fallback" in config:
                # Use fallback for single values
                result_dict[filename] = float(dataset_values[0])

        # Map results to DataFrame
        self.dataframe[output_column] = self.dataframe["Filename"].map(result_dict)

        # Count how many had errors vs single values
        has_error = sum(1 for v in result_dict.values() if isinstance(v, tuple))
        single_val = len(result_dict) - has_error

        self.logger.info(
            f"    Added: {len(result_dict)} values ({has_error} with errors, {single_val} single)"
        )

    def _add_single_value_column(self, dataset_dict: Dict, output_column: str) -> None:
        """Add column extracting single values from datasets."""
        result_dict = {}

        for filename, dataset_values in dataset_dict.items():
            if not isinstance(dataset_values, np.ndarray):
                dataset_values = np.array(dataset_values)

            if len(dataset_values) == 1:
                result_dict[filename] = float(dataset_values[0])

        self.dataframe[output_column] = self.dataframe["Filename"].map(result_dict)
        self.logger.info(f"    Added: {len(result_dict)} single values")

    def _add_mean_column(self, dataset_dict: Dict, output_column: str) -> None:
        """Add column with mean values (no error)."""
        result_dict = {}

        for filename, dataset_values in dataset_dict.items():
            if not isinstance(dataset_values, np.ndarray):
                dataset_values = np.array(dataset_values)

            result_dict[filename] = float(np.mean(dataset_values))

        self.dataframe[output_column] = self.dataframe["Filename"].map(result_dict)
        self.logger.info(f"    Added: {len(result_dict)} mean values")

    def _add_sum_divided_column(
        self, dataset_dict: Dict, output_column: str, config: Dict
    ) -> None:
        """Add column with summed values divided by unit."""
        result_dict = {}

        for filename, dataset_values in dataset_dict.items():
            if not isinstance(dataset_values, np.ndarray):
                dataset_values = np.array(dataset_values)

            total = float(np.sum(dataset_values))
            result_dict[filename] = total

        # Apply unit division
        result_dict = self._apply_unit_division(result_dict, config)

        self.dataframe[output_column] = self.dataframe["Filename"].map(result_dict)

        # Log divisor used
        divisor_info = self._get_divisor_info(config)
        self.logger.info(
            f"    Added: {len(result_dict)} summed values (divided by {divisor_info})"
        )

    def _add_sum_plus_length_divided_column(
        self, dataset_dict: Dict, output_column: str, config: Dict
    ) -> None:
        """Add column with summed values plus length, divided by
        unit."""
        result_dict = {}

        for filename, dataset_values in dataset_dict.items():
            if not isinstance(dataset_values, np.ndarray):
                dataset_values = np.array(dataset_values)

            # Sum + length (for kernel applications with initial
            # application)
            total = int(np.sum(dataset_values)) + len(dataset_values)
            result_dict[filename] = total

        # Apply unit division
        result_dict = self._apply_unit_division(result_dict, config)

        self.dataframe[output_column] = self.dataframe["Filename"].map(result_dict)

        divisor_info = self._get_divisor_info(config)
        self.logger.info(
            f"    Added: {len(result_dict)} sum+length values (divided by {divisor_info})"
        )

    def _add_unique_values_column(
        self, dataset_dict: Dict, output_column: str, config: Dict
    ) -> None:
        """Add column with unique values as tuples."""
        result_dict = {}
        data_type = config.get("data_type", "float")

        for filename, dataset_values in dataset_dict.items():
            if not isinstance(dataset_values, np.ndarray):
                dataset_values = np.array(dataset_values)

            if data_type == "float":
                unique_vals = tuple(float(val) for val in np.unique(dataset_values))
            else:
                unique_vals = tuple(np.unique(dataset_values))
            result_dict[filename] = unique_vals

        self.dataframe[output_column] = self.dataframe["Filename"].map(result_dict)

        # Log how many unique values per entry (average)
        avg_unique = np.mean([len(v) for v in result_dict.values()])
        self.logger.info(
            f"    Added: {len(result_dict)} entries (avg {avg_unique:.1f} unique values each)"
        )

    def _get_divisor_info(self, config: Dict) -> str:
        """Get human-readable divisor information for logging."""
        unit_mapping = config.get("unit_mapping", {})

        if (
            "has_Number_of_spinors" in unit_mapping
            and "Number_of_spinors" in self.dataframe.columns
        ):
            divisor_col = unit_mapping["has_Number_of_spinors"]["divisor"]
            divisor_val = self.dataframe[divisor_col].iloc[0]
            return f"{divisor_col}={divisor_val}"
        elif "default" in unit_mapping:
            divisor_col = unit_mapping["default"]["divisor"]
            if divisor_col in self.dataframe.columns:
                divisor_val = self.dataframe[divisor_col].iloc[0]
                return f"{divisor_col}={divisor_val}"

        return "unit"

    def _log_column_statistics(self, column_name: str) -> None:
        """Log statistics about a newly added column."""
        if column_name not in self.dataframe.columns:
            return

        col_data = self.dataframe[column_name]

        # Handle both numeric and tuple columns
        if col_data.dtype == object:
            # Could be tuples (mean with error) or other objects
            sample_value = (
                col_data.dropna().iloc[0] if not col_data.dropna().empty else None
            )

            if isinstance(sample_value, tuple) and len(sample_value) == 2:
                # Mean with error format: extract means
                means = col_data.apply(lambda x: x[0] if isinstance(x, tuple) else x)
                self.logger.info(
                    f"    Range (means): [{means.min():.6g}, {means.max():.6g}]"
                )
            else:
                self.logger.info(
                    f"    Non-numeric data (type: {type(sample_value).__name__})"
                )
        else:
            # Numeric column
            self.logger.info(f"    Range: [{col_data.min():.6g}, {col_data.max():.6g}]")
            self.logger.info(
                f"    Mean: {col_data.mean():.6g}, Std: {col_data.std():.6g}"
            )

    def _process_hdf5_dataset(self, dataset_name: str, config: dict) -> None:
        """
        Process a single HDF5 dataset according to its configuration.

        Args:
            dataset_name: Name of the dataset in HDF5 config: Processing
            configuration dictionary
        """
        # Check if dataset exists in HDF5
        if (
            dataset_name
            not in self.hdf5_analyzer.list_of_output_quantity_names_from_hdf5
        ):
            self.logger.debug(f"  Skipping {dataset_name} (not in HDF5)")
            return

        try:
            # Extract dataset values using HDF5Analyzer
            dataset_dict = self._extract_dataset_to_dict(dataset_name)

            if not dataset_dict:
                self.logger.info(f"  No data found for {dataset_name}")
                return

            # Skip if all arrays are empty
            if all(len(arr) == 0 for arr in dataset_dict.values()):
                self.logger.info(f"  Skipping {dataset_name}: all arrays are empty")
                return

            # Determine output column name
            output_pattern = config["output_pattern"]
            output_column = self._resolve_output_pattern(output_pattern, config)

            self.logger.info(f"  Processing: {dataset_name} → {output_column}")

            # Apply aggregation method based on type
            aggregation_method = config["aggregation_method"]

            if aggregation_method == "mean_with_error":
                self._add_mean_with_error_column(dataset_dict, output_column, config)
            elif aggregation_method == "single_value":
                self._add_single_value_column(dataset_dict, output_column)
            elif aggregation_method == "mean":
                self._add_mean_column(dataset_dict, output_column)
            elif aggregation_method == "sum_then_divide":
                self._add_sum_divided_column(dataset_dict, output_column, config)
            elif aggregation_method == "sum_plus_length_then_divide":
                self._add_sum_plus_length_divided_column(
                    dataset_dict, output_column, config
                )
            elif aggregation_method == "unique_values_as_list":
                self._add_unique_values_column(dataset_dict, output_column, config)
            else:
                self.logger.warning(
                    f"  Unknown aggregation method '{aggregation_method}' for {dataset_name}"
                )
                return

            # Log statistics about the added column
            self._log_column_statistics(output_column)

        except Exception as e:
            self.logger.error(f"  Error processing {dataset_name}: {e}")

    def _apply_unit_division(self, processed_dict: Dict, config: Dict) -> Dict:
        """Apply unit-based division to processed values."""
        unit_mapping = config.get("unit_mapping", {})

        # Determine divisor column based on DataFrame contents
        divisor_column = None
        if (
            "has_Number_of_spinors" in unit_mapping
            and "Number_of_spinors" in self.dataframe.columns
        ):
            divisor_column = unit_mapping["has_Number_of_spinors"]["divisor"]
        elif "default" in unit_mapping:
            divisor_column = unit_mapping["default"]["divisor"]

        if divisor_column and divisor_column in self.dataframe.columns:
            # Get divisor value (assuming it's the same for all rows)
            divisor_value = self.dataframe[divisor_column].iloc[0]

            # Apply division to all values
            for filename in processed_dict:
                processed_dict[filename] = processed_dict[filename] / divisor_value

        return processed_dict

    def _extract_dataset_to_dict(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Extract dataset values to a dictionary mapping filenames to
        arrays."""
        dataset_dict = {}

        try:
            if (
                dataset_name
                not in self.hdf5_analyzer.list_of_output_quantity_names_from_hdf5
            ):
                self.logger.warning(f"Dataset {dataset_name} not found in HDF5 file")
                return dataset_dict

            # Get all active groups
            active_groups = list(self.hdf5_analyzer.active_groups)

            # Fetch data for each group individually
            for group_path in active_groups:
                # Extract filename
                filename = self._extract_filename_from_group_path(group_path)
                if not filename:
                    continue

                # Fetch data directly for this specific group
                try:
                    # Access the already-open HDF5 file
                    f = self.hdf5_analyzer._file
                    if group_path in f and dataset_name in f[group_path]:
                        dataset = f[group_path][dataset_name]
                        dataset_dict[filename] = dataset[:]
                    else:
                        self.logger.warning(
                            f"Dataset {dataset_name} not found in {group_path}"
                        )
                except Exception as e:
                    self.logger.warning(
                        f"Could not extract {dataset_name} from {group_path}: {e}"
                    )

            # self.logger.info( f"Extracted {len(dataset_dict)} entries
            #     for dataset {dataset_name}" )

        except Exception as e:
            self.logger.error(f"Failed to extract dataset {dataset_name}: {e}")

        return dataset_dict

    def _extract_filename_from_group_path(self, group_path: str) -> Optional[str]:
        """
        Extract filename from HDF5 group path.

        The HDF5 structure has filenames as the deepest level group
        names:
        sign_squared_violation/Zolotarev_several_config_varying_n/filename.txt
        """
        parts = group_path.strip("/").split("/")
        if parts:
            # The filename is the last part of the path
            return parts[-1]
        return None

    def _apply_aggregation(self, dataset_dict: Dict, method: str, config: Dict) -> Dict:
        """Apply the specified aggregation method to dataset values."""
        result_dict = {}

        for filename, dataset_values in dataset_dict.items():
            # Ensure dataset_values is a numpy array
            if not isinstance(dataset_values, np.ndarray):
                dataset_values = np.array(dataset_values)

            if method == "mean_with_error":
                condition = config.get("condition", "all_lengths_greater_than_one")

                if (
                    condition == "all_lengths_greater_than_one"
                    and len(dataset_values) > 1
                ):
                    mean_val = float(np.mean(dataset_values))
                    error_val = float(
                        np.std(dataset_values, ddof=1) / np.sqrt(len(dataset_values))
                    )
                    result_dict[filename] = (mean_val, error_val)
                elif (
                    condition == "all_lengths_equal_to_one" and len(dataset_values) == 1
                ):
                    result_dict[filename] = float(dataset_values[0])
                elif "fallback" in config:
                    # Apply fallback processing
                    fallback_config = config["fallback"]
                    fallback_method = fallback_config["aggregation_method"]
                    fallback_condition = fallback_config["condition"]

                    if (
                        fallback_condition == "all_lengths_equal_to_one"
                        and len(dataset_values) == 1
                    ):
                        result_dict[filename] = float(dataset_values[0])

            elif method == "single_value":
                if len(dataset_values) == 1:
                    result_dict[filename] = float(dataset_values[0])

            elif method == "sum_then_divide":
                total = float(np.sum(dataset_values))
                # Note: Division by unit would be handled in calling
                # code
                result_dict[filename] = total

            elif method == "sum_plus_length_then_divide":
                # For kernel applications: sum + length (for initial
                # application)
                total = int(np.sum(dataset_values)) + len(dataset_values)
                result_dict[filename] = total

            elif method == "mean":
                result_dict[filename] = float(np.mean(dataset_values))

            elif method == "unique_values_as_list":  # More precisle, a tuple
                data_type = config.get("data_type", "float")
                if data_type == "float":
                    unique_vals = tuple(float(val) for val in np.unique(dataset_values))
                else:
                    unique_vals = tuple(np.unique(dataset_values))
                result_dict[filename] = unique_vals

        return result_dict

    def _resolve_output_pattern(self, pattern: str, config: Dict) -> str:
        """Resolve output column name pattern with dynamic
        substitutions."""
        resolved_pattern = pattern

        # Handle {main_program_type} substitution
        if "{main_program_type}" in pattern:
            if "Main_program_type" in self.dataframe.columns:
                main_program_type = str(self.dataframe["Main_program_type"].iloc[0])
                # Remove "_values" suffix if present
                main_program_type = main_program_type.replace("_values", "")
                resolved_pattern = resolved_pattern.replace(
                    "{main_program_type}", main_program_type
                )

        # Handle {unit} substitution
        if "{unit}" in pattern:
            unit_mapping = config.get("unit_mapping", {})

            # Determine unit based on DataFrame contents
            if (
                "has_Number_of_spinors" in unit_mapping
                and "Number_of_spinors" in self.dataframe.columns
            ):
                unit_config = unit_mapping["has_Number_of_spinors"]
                unit = unit_config["unit"]
            elif "default" in unit_mapping:
                unit_config = unit_mapping["default"]
                unit = unit_config["unit"]
            else:
                unit = "vector"  # fallback

            resolved_pattern = resolved_pattern.replace("{unit}", unit)

        return resolved_pattern

    def _validate_vector_spinor_counts(self) -> None:
        """
        Validate Number_of_vectors and Number_of_spinors against HDF5
        dataset lengths.

        Validation logic:
            - Invert case: len(Total_number_of_CG_iterations_per_spinor)
              should equal Number_of_vectors × Number_of_spinors
            - Non-invert case: len(Calculation_result_per_vector) should
              equal Number_of_vectors

        This validates that the parsed parameter values match the actual
        data structure.
        """
        # Determine program type
        main_program_type = None
        if "Main_program_type" in self.dataframe.columns and len(self.dataframe) > 0:
            main_program_type = self.dataframe["Main_program_type"].iloc[0]

        if main_program_type == "invert":
            self._validate_invert_case()
        else:
            self._validate_forward_case()

    def _validate_invert_case(self) -> None:
        """Validate vector/spinor counts for inversion case."""
        # Check if the relevant dataset exists
        if (
            "Total_number_of_CG_iterations_per_spinor"
            not in self.hdf5_analyzer.list_of_output_quantity_names_from_hdf5
        ):
            self.logger.warning(
                "Cannot validate: Total_number_of_CG_iterations_per_spinor dataset not found in HDF5"
            )
            return

        # Get expected counts from DataFrame
        if "Number_of_spinors" not in self.dataframe.columns:
            self.logger.error("Number_of_spinors missing for invert case")
            raise ValueError("Number_of_spinors required for validation in invert case")

        if "Number_of_vectors" not in self.dataframe.columns:
            self.logger.error("Number_of_vectors missing for invert case")
            raise ValueError("Number_of_vectors required for validation in invert case")

        # Validate for each file/row
        validation_errors = []
        validated_count = 0  # Track successful validations
        skipped_files = []  # Track files with zero-length datasets

        for idx, row in self.dataframe.iterrows():
            filename = row["Filename"]
            expected_spinors = int(row["Number_of_spinors"])
            expected_vectors = int(row["Number_of_vectors"])
            expected_total = expected_spinors * expected_vectors

            # Get actual dataset length for this file
            try:
                # Access HDF5 file directly to get dataset for this
                # specific file
                actual_length = self._get_dataset_length_for_file(
                    filename, "Total_number_of_CG_iterations_per_spinor"
                )

                if actual_length is None:
                    self.logger.warning(
                        f"Could not retrieve dataset length for {filename}"
                    )
                    continue

                # Skip validation for zero-length datasets
                if actual_length == 0:
                    skipped_files.append(filename)
                    continue

                # Validate
                if actual_length != expected_total:
                    error_msg = (
                        f"Mismatch in {filename}: "
                        f"Total_number_of_CG_iterations_per_spinor has {actual_length} elements, "
                        f"but Number_of_spinors={expected_spinors} × Number_of_vectors={expected_vectors} "
                        f"= {expected_total} expected"
                    )
                    validation_errors.append(error_msg)
                    self.logger.error(error_msg)
                else:
                    validated_count += 1

            except Exception as e:
                self.logger.warning(f"Error validating {filename}: {e}")

        # Report skipped files summary
        if skipped_files:
            self.logger.info(
                f"Skipped validation for {len(skipped_files)}/{len(self.dataframe)} files with zero-length datasets"
            )

        # Report validation results
        if validation_errors:
            error_summary = "\n".join(validation_errors)
            raise ValueError(
                f"Vector/spinor count validation failed for {len(validation_errors)} files:\n"
                f"{error_summary}"
            )
        else:
            self.logger.info(
                f"✓ Validation passed: Number_of_spinors × Number_of_vectors matches "
                f"CG_total_calculation_time_per_spinor dataset lengths for all files"
            )

    def _validate_forward_case(self) -> None:
        """Validate vector counts for forward case."""
        # Check if the relevant dataset exists
        if (
            "Calculation_result_per_vector"
            not in self.hdf5_analyzer.list_of_output_quantity_names_from_hdf5
        ):
            self.logger.warning(
                "Cannot validate: Calculation_result_per_vector dataset not found in HDF5"
            )
            return

        # Get expected count from DataFrame
        if "Number_of_vectors" not in self.dataframe.columns:
            self.logger.error("Number_of_vectors missing for forward case")
            raise ValueError(
                "Number_of_vectors required for validation in forward case"
            )

        # Validate for each file/row
        validation_errors = []

        for idx, row in self.dataframe.iterrows():
            filename = row["Filename"]
            expected_vectors = int(row["Number_of_vectors"])

            # Get actual dataset length for this file
            try:
                actual_length = self._get_dataset_length_for_file(
                    filename, "Calculation_result_per_vector"
                )

                if actual_length is None:
                    self.logger.warning(
                        f"Could not retrieve dataset length for {filename}"
                    )
                    continue

                # Validate
                if actual_length != expected_vectors:
                    error_msg = (
                        f"Mismatch in {filename}: "
                        f"Calculation_result_per_vector has {actual_length} elements, "
                        f"but Number_of_vectors={expected_vectors} expected"
                    )
                    validation_errors.append(error_msg)
                    self.logger.error(error_msg)
                else:
                    self.logger.debug(
                        f"✓ {filename}: {actual_length} elements matches "
                        f"Number_of_vectors={expected_vectors}"
                    )

            except Exception as e:
                self.logger.warning(f"Error validating {filename}: {e}")

        # Report validation results
        if validation_errors:
            error_summary = "\n".join(validation_errors)
            raise ValueError(
                f"Vector count validation failed for {len(validation_errors)} files:\n"
                f"{error_summary}"
            )
        else:
            self.logger.info(
                f"✓ Validation passed: Number_of_vectors matches "
                f"Calculation_result_per_vector dataset lengths for all files"
            )

    def _get_dataset_length_for_file(
        self, filename: str, dataset_name: str
    ) -> Optional[int]:
        """
        Get the length of a specific dataset for a specific file.

        Args:
            filename: The filename to look up dataset_name: Name of the
            dataset

        Returns:
            Length of the dataset array, or None if not found
        """
        try:
            # Find the group path that contains this filename
            group_path = None
            for path in self.hdf5_analyzer.active_groups:
                if path.endswith(filename):
                    group_path = path
                    break

            if group_path is None:
                self.logger.warning(f"No group found for filename: {filename}")
                return None

            # Access the HDF5 file directly
            f = self.hdf5_analyzer._file
            if group_path in f and dataset_name in f[group_path]:
                dataset = f[group_path][dataset_name]
                return len(dataset)
            else:
                self.logger.warning(
                    f"Dataset {dataset_name} not found in group {group_path}"
                )
                return None

        except Exception as e:
            self.logger.error(f"Error accessing dataset for {filename}: {e}")
            return None


class AnalysisCaseProcessor:
    """
    Processor for handling different analysis cases (forward vs
    inversions).

    Implements strategy pattern for case-specific processing logic.
    """

    def __init__(self, dataframe: pd.DataFrame, logger=None):
        """Initialize with the DataFrame to process."""
        self.dataframe = dataframe
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def process_analysis_cases(self) -> pd.DataFrame:
        """Apply all analysis-case-specific calculations."""
        self.logger.info("Determining analysis case...")

        analysis_case = self._determine_analysis_case()
        self.logger.info(f"  Detected: {analysis_case}")

        # Apply case-specific calculations
        self._apply_mv_multiplication_calculations(analysis_case)
        self._apply_time_cost_calculations(analysis_case)

        return self.dataframe

    def _determine_analysis_case(self) -> str:
        """
        Determine whether this is forward operator applications or
        inversions.

        Returns:
            "inversions" or "forward_operator_applications"
        """
        # Simple logic: presence of Number_of_spinors indicates
        # inversion case
        if "Number_of_spinors" in self.dataframe.columns:
            self.logger.debug(
                "  Analysis case determined from Number_of_spinors column"
            )
            return "inversions"
        else:
            self.logger.debug(
                "  Analysis case determined from absence of Number_of_spinors"
            )
            return "forward_operator_applications"

    def _apply_mv_multiplication_calculations(self, analysis_case: str) -> None:
        """Apply matrix-vector multiplication calculations for the given
        case."""
        if analysis_case not in ANALYSIS_CASES:
            return

        case_config = ANALYSIS_CASES[analysis_case]
        mv_rules = case_config.get("mv_multiplication_rules", {})

        # Get overlap operator method to determine which rule to use
        if "Overlap_operator_method" in self.dataframe.columns:
            method = self.dataframe["Overlap_operator_method"].iloc[0]

            if method in mv_rules:
                # formula = mv_rules[method]["formula"]

                # Determine output column name based on analysis case
                if analysis_case == "forward_operator_applications":
                    output_col = "Average_number_of_MV_multiplications_per_vector"
                else:
                    output_col = "Average_number_of_MV_multiplications_per_spinor"

                # Evaluate formula - this is a simplified version In
                # practice, you'd need a proper expression evaluator
                if method == "Chebyshev":
                    if analysis_case == "forward_operator_applications":
                        self.dataframe[output_col] = (
                            2 * self.dataframe["Total_number_of_Lanczos_iterations"]
                            + 1
                            + 2 * self.dataframe["Number_of_Chebyshev_terms"]
                            - 1
                        )
                    else:  # inversions
                        self.dataframe[output_col] = (
                            2 * self.dataframe["Total_number_of_Lanczos_iterations"] + 1
                        ) + (
                            2
                            * self.dataframe[
                                "Average_number_of_CG_iterations_per_spinor"
                            ]
                            + 1
                        ) * (
                            2 * self.dataframe["Number_of_Chebyshev_terms"] - 1
                        )

                self.logger.info(
                    f"Applied {method} MV multiplication formula for {analysis_case}"
                )

    def _apply_time_cost_calculations(self, analysis_case: str) -> None:
        """Apply time and cost calculations for the given case."""
        if "Total_calculation_time" not in self.dataframe.columns:
            return

        case_suffix = (
            "forward_case"
            if analysis_case == "forward_operator_applications"
            else "inversion_case"
        )

        # Apply wall clock time calculations
        if "wall_clock_time" in TIME_COST_CALCULATIONS:
            config = TIME_COST_CALCULATIONS["wall_clock_time"]
            if case_suffix in config:
                self._calculate_wall_clock_time(config[case_suffix], analysis_case)

        # Apply core hours calculations
        if "core_hours" in TIME_COST_CALCULATIONS:
            config = TIME_COST_CALCULATIONS["core_hours"]
            if case_suffix in config:
                self._calculate_core_hours(config[case_suffix])

    def _calculate_wall_clock_time(self, config: Dict, analysis_case: str) -> None:
        """Calculate wall clock time per unit."""
        output_column = config["output_column"]

        if "Total_overhead_time" in self.dataframe.columns:
            # Use formula with overhead
            if analysis_case == "forward_operator_applications":
                self.dataframe[output_column] = (
                    self.dataframe["Total_calculation_time"]
                    - self.dataframe["Total_overhead_time"]
                ) / self.dataframe["Number_of_vectors"] + self.dataframe[
                    "Total_overhead_time"
                ]
            else:  # inversions
                self.dataframe[output_column] = (
                    self.dataframe["Total_calculation_time"]
                    - self.dataframe["Total_overhead_time"]
                ) / (
                    self.dataframe["Number_of_spinors"]
                    * self.dataframe["Number_of_vectors"]
                ) + self.dataframe[
                    "Total_overhead_time"
                ]
        else:
            # Use base formula (no overhead)
            if analysis_case == "forward_operator_applications":
                self.dataframe[output_column] = (
                    self.dataframe["Total_calculation_time"]
                    / self.dataframe["Number_of_vectors"]
                )
            else:  # inversions
                self.dataframe[output_column] = self.dataframe[
                    "Total_calculation_time"
                ] / (
                    self.dataframe["Number_of_spinors"]
                    * self.dataframe["Number_of_vectors"]
                )

        self.logger.info(f"Calculated {output_column}")

    def _calculate_core_hours(self, config: Dict) -> None:
        """Calculate core hours from wall clock time."""
        if "Number_of_cores" not in self.dataframe.columns:
            return

        output_column = config["output_column"]

        # Determine which wall clock time column to use
        if output_column == "Average_core_hours_per_spinor":
            input_column = "Average_wall_clock_time_per_spinor"
        else:  # Average_core_hours_per_vector
            input_column = "Average_wall_clock_time_per_vector"

        if input_column in self.dataframe.columns:
            self.dataframe[output_column] = (
                self.dataframe["Number_of_cores"] * self.dataframe[input_column] / 3600
            )
            self.logger.info(f"Calculated {output_column}")
        else:
            self.logger.warning(
                f"Cannot calculate {output_column}: {input_column} not found"
            )
