"""
Parameter Transformation Engine for QPB log file processing.

This module provides the main transformation engine that applies all
configured transformations systematically to parameter data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path

from ._param_transform_config import (
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
    Main engine for applying systematic parameter transformations.

    This class orchestrates all parameter transformations using the
    configuration-driven approach, replacing scattered hardcoded logic
    with systematic, maintainable processing.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the transformation engine.

        Args:
            dataframe: The DataFrame containing single-valued parameters
        """
        self.dataframe = dataframe.copy()
        self.original_dataframe = dataframe.copy()
        self.logger = logging.getLogger(__name__)

    def apply_all_transformations(self) -> pd.DataFrame:
        """
        Apply all configured transformations in the correct order.

        Returns:
            Transformed DataFrame
        """
        self.logger.info("Starting parameter transformation pipeline")

        # Apply transformations in dependency order
        self._apply_string_transformations()
        self._apply_extraction_rules()
        self._apply_column_operations()
        self._apply_math_transformations()

        self.logger.info("Parameter transformation pipeline completed")
        return self.dataframe.copy()

    def _apply_string_transformations(self) -> None:
        """Apply all string-based transformations."""
        self.logger.info("Applying string transformations")

        for column_name, config in STRING_TRANSFORMATIONS.items():
            if column_name not in self.dataframe.columns:
                continue

            # Apply value replacements
            if "replacements" in config:
                for old_value, new_value in config["replacements"].items():
                    self.dataframe[column_name] = self.dataframe[column_name].replace(
                        old_value, new_value
                    )
                    self.logger.info(
                        f"Replaced '{old_value}' with '{new_value}' in {column_name}"
                    )

            # Apply formatters
            if "formatter" in config:
                formatter_name = config["formatter"]
                formatter_func = FORMATTER_FUNCTIONS.get(formatter_name)

                if formatter_func is not None:
                    # Get additional formatter parameters
                    formatter_kwargs = {
                        k: v
                        for k, v in config.items()
                        if k not in ["formatter", "replacements"]
                    }

                    self.dataframe[column_name] = self.dataframe[column_name].apply(
                        lambda x: formatter_func(x, **formatter_kwargs)  # type: ignore
                    )
                    self.logger.info(
                        f"Applied {formatter_name} formatter to {column_name}"
                    )
                else:
                    self.logger.warning(
                        f"Formatter '{formatter_name}' not found for column '{column_name}'"
                    )

    def _apply_extraction_rules(self) -> None:
        """Apply compound parameter extraction rules."""
        self.logger.info("Applying parameter extraction rules")

        for source_column, config in EXTRACTION_RULES.items():
            if source_column not in self.dataframe.columns:
                continue

            parser_name = config.get("parser", "ast_literal_eval")
            parser_func = PARSER_FUNCTIONS.get(parser_name)

            if parser_func is None:
                self.logger.warning(
                    f"Parser '{parser_name}' not found for column '{source_column}' - skipping"
                )
                continue

            # Handle extraction to multiple columns
            if "extract_to" in config:
                for target_column, extract_config in config["extract_to"].items():
                    index = extract_config["index"]
                    target_type = extract_config.get("type", "str")

                    try:
                        self.dataframe[target_column] = self.dataframe[
                            source_column
                        ].apply(
                            lambda x: target_type(parser_func(x)[index])  # type: ignore
                        )
                        self.logger.info(
                            f"Extracted {target_column} from {source_column}"
                        )
                    except (IndexError, TypeError, ValueError) as e:
                        self.logger.error(
                            f"Error extracting {target_column} from {source_column}: {e}"
                        )

            # Handle in-place transformations
            elif "transform" in config:
                transform_name = config["transform"]
                transform_func = CALCULATION_FUNCTIONS.get(transform_name)

                if transform_func is not None:
                    try:
                        self.dataframe[source_column] = self.dataframe[
                            source_column
                        ].apply(
                            lambda x: transform_func(parser_func(x))  # type: ignore
                        )
                        self.logger.info(f"Applied {transform_name} to {source_column}")
                    except Exception as e:
                        self.logger.error(
                            f"Error applying transform {transform_name} to {source_column}: {e}"
                        )
                else:
                    self.logger.warning(
                        f"Transform function '{transform_name}' not found for column '{source_column}'"
                    )

            # Remove source column if specified
            if config.get("remove_source", False):
                self.dataframe.drop(columns=[source_column], inplace=True)
                self.logger.info(f"Removed source column {source_column}")

    def _apply_column_operations(self) -> None:
        """Apply column addition, removal, and default value
        operations."""
        self.logger.info("Applying column operations")

        # Apply default values for missing columns
        for column_name, default_value in COLUMN_OPERATIONS.get(
            "default_values", {}
        ).items():
            if column_name not in self.dataframe.columns:
                self.dataframe[column_name] = default_value
                self.logger.info(
                    f"Added default value {default_value} for {column_name}"
                )

        # Apply column additions
        for result_column, source_columns in COLUMN_OPERATIONS.get(
            "additions", {}
        ).items():
            available_columns = [
                col for col in source_columns if col in self.dataframe.columns
            ]

            if len(available_columns) == len(source_columns):
                # All source columns available - add them
                self.dataframe[result_column] = self.dataframe[available_columns].sum(
                    axis=1
                )
                self.logger.info(
                    f"Added columns {available_columns} to create {result_column}"
                )

                # Remove the additional columns (keep the first one as
                # the result)
                columns_to_remove = available_columns[1:]
                if columns_to_remove:
                    self.dataframe.drop(columns=columns_to_remove, inplace=True)

            elif len(available_columns) == 1:
                # Only one column available - rename it
                old_name = available_columns[0]
                if old_name != result_column:
                    self.dataframe.rename(
                        columns={old_name: result_column}, inplace=True
                    )
                    self.logger.info(f"Renamed {old_name} to {result_column}")

        # Apply derived calculations
        for result_column, config in COLUMN_OPERATIONS.get(
            "derived_calculations", {}
        ).items():
            formula_name = config["formula"]
            formula_func = CALCULATION_FUNCTIONS.get(formula_name)
            input_columns = config["inputs"]

            if formula_func is not None and all(
                col in self.dataframe.columns for col in input_columns
            ):
                try:
                    self.dataframe[result_column] = self.dataframe.apply(
                        lambda row: formula_func(*[row[col] for col in input_columns]),  # type: ignore
                        axis=1,
                    )
                    self.logger.info(f"Calculated {result_column} using {formula_name}")
                except Exception as e:
                    self.logger.error(
                        f"Error calculating {result_column} using {formula_name}: {e}"
                    )
            elif formula_func is None:
                self.logger.warning(
                    f"Formula function '{formula_name}' not found for {result_column}"
                )
            else:
                missing_cols = [
                    col for col in input_columns if col not in self.dataframe.columns
                ]
                self.logger.warning(
                    f"Missing input columns for {result_column}: {missing_cols}"
                )

        # Apply conditional column removal
        conditional_removals = {
            "Solver_epsilon": ("MSCG_epsilon", "not_has_Outer_solver_epsilon")
        }

        for column_to_remove, (
            condition_column,
            condition_type,
        ) in conditional_removals.items():
            if (
                column_to_remove in self.dataframe.columns
                and condition_column in self.dataframe.columns
            ):

                if condition_type == "not_has_Outer_solver_epsilon":
                    if "Outer_solver_epsilon" not in self.dataframe.columns:
                        self.dataframe.drop(columns=[column_to_remove], inplace=True)
                        self.logger.info(f"Conditionally removed {column_to_remove}")

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
                if target_type == "int":
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

    def __init__(self, hdf5_analyzer, dataframe: pd.DataFrame):
        """
        Initialize the HDF5 processor.

        Args:
            hdf5_analyzer: HDF5Analyzer instance for data access
            dataframe: DataFrame to merge results into
        """
        self.hdf5_analyzer = hdf5_analyzer
        self.dataframe = dataframe
        self.logger = logging.getLogger(__name__)

    def process_all_hdf5_parameters(self) -> pd.DataFrame:
        """
        Process all configured HDF5 parameters.

        Returns:
            DataFrame with HDF5-derived columns added
        """
        self.logger.info("Starting HDF5 parameter processing")

        # Handle special vector/spinor columns first
        self._handle_vector_spinor_columns()

        for dataset_name, config in HDF5_PROCESSING_RULES.items():
            self._process_hdf5_dataset(dataset_name, config)

        self.logger.info("HDF5 parameter processing completed")
        return self.dataframe

    def _handle_vector_spinor_columns(self) -> None:
        """
        Handle Number_of_vectors and Number_of_spinors columns with fallback logic.

        This method implements the business logic for these special columns
        that's too complex for configuration-driven processing.
        """
        main_program_type = (
            self.dataframe.get("Main_program_type", pd.Series()).iloc[0]
            if len(self.dataframe) > 0
            else None
        )

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

    def _process_hdf5_dataset(self, dataset_name: str, config: Dict) -> None:
        """Process a single HDF5 dataset according to its
        configuration."""
        try:
            # Extract dataset values using HDF5Analyzer
            dataset_dict = self._extract_dataset_to_dict(dataset_name)

            if not dataset_dict:
                self.logger.info(f"No data found for dataset {dataset_name}")
                return

            # Apply aggregation method
            aggregation_method = config["aggregation_method"]
            processed_dict = self._apply_aggregation(
                dataset_dict, aggregation_method, config
            )

            # Handle unit division for sum_then_divide methods
            if aggregation_method in ["sum_then_divide", "sum_plus_length_then_divide"]:
                processed_dict = self._apply_unit_division(processed_dict, config)

            # Determine output column name
            output_pattern = config["output_pattern"]
            output_column = self._resolve_output_pattern(output_pattern, config)

            # Map results to DataFrame
            self.dataframe[output_column] = self.dataframe["Filename"].map(
                processed_dict
            )

            self.logger.info(f"Processed {dataset_name} -> {output_column}")

        except Exception as e:
            self.logger.error(f"Error processing {dataset_name}: {e}")

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
            # Check if dataset exists in HDF5
            if (
                dataset_name
                not in self.hdf5_analyzer.list_of_output_quantity_names_from_hdf5
            ):
                self.logger.warning(f"Dataset {dataset_name} not found in HDF5 file")
                return dataset_dict

            # Get dataset values from all active groups
            dataset_values_list = self.hdf5_analyzer.dataset_values(
                dataset_name, return_gvar=False
            )

            # Map group paths to filenames from DataFrame
            active_groups = list(self.hdf5_analyzer.active_groups)

            if len(active_groups) != len(dataset_values_list):
                self.logger.error(
                    f"Mismatch between active groups ({len(active_groups)}) "
                    f"and dataset values ({len(dataset_values_list)})"
                )
                return dataset_dict

            # Create filename mapping - this is a simplification In
            # practice, you'd need a more sophisticated mapping strategy
            # based on how filenames in CSV correspond to HDF5 group
            # paths
            for group_path, values in zip(active_groups, dataset_values_list):
                # Extract filename from group path or use a mapping
                # strategy
                filename = self._extract_filename_from_group_path(group_path)
                if filename:
                    dataset_dict[filename] = values

            self.logger.info(
                f"Extracted {len(dataset_dict)} entries for dataset {dataset_name}"
            )

        except Exception as e:
            self.logger.error(f"Failed to extract dataset {dataset_name}: {e}")

        return dataset_dict

    def _extract_filename_from_group_path(self, group_path: str) -> Optional[str]:
        """
        Extract filename from HDF5 group path.

        This method needs to implement the logic for mapping HDF5 group
        paths back to the filenames used in the CSV. The exact
        implementation depends on how the HDF5 file was structured
        relative to the original log files.
        """
        # This is a placeholder implementation - would need to be
        # customized based on actual HDF5 structure and filename
        # conventions

        # Example: if group path is "/some/path/analysis_run_001", the
        # filename might be "run_001.log" or similar
        parts = group_path.strip("/").split("/")
        if parts:
            # Try to find a part that looks like a filename
            for part in reversed(parts):
                if any(char.isdigit() for char in part):
                    # This is a very simple heuristic - would need
                    # refinement
                    return f"{part}.log"

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

            elif method == "unique_values_as_list":
                data_type = config.get("data_type", "float")
                if data_type == "float":
                    unique_vals = [float(val) for val in np.unique(dataset_values)]
                else:
                    unique_vals = list(np.unique(dataset_values))
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


class AnalysisCaseProcessor:
    """
    Processor for handling different analysis cases (forward vs
    inversions).

    Implements strategy pattern for case-specific processing logic.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """Initialize with the DataFrame to process."""
        self.dataframe = dataframe
        self.logger = logging.getLogger(__name__)

    def process_analysis_cases(self) -> pd.DataFrame:
        """Process all analysis-case-specific calculations."""
        self.logger.info("Processing analysis case calculations")

        # Determine analysis case
        analysis_case = self._determine_analysis_case()
        self.logger.info(f"Detected analysis case: {analysis_case}")

        # Apply case-specific calculations
        self._apply_mv_multiplication_calculations(analysis_case)
        self._apply_time_cost_calculations(analysis_case)

        return self.dataframe

    def _determine_analysis_case(self) -> str:
        """Determine whether this is forward operator applications or
        inversions."""
        if "Number_of_spinors" in self.dataframe.columns:
            return "inversions"
        else:
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
                formula = mv_rules[method]["formula"]

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

        # Apply adjusted core hours calculations
        if "adjusted_core_hours" in TIME_COST_CALCULATIONS:
            config = TIME_COST_CALCULATIONS["adjusted_core_hours"]
            if case_suffix in config:
                self._calculate_adjusted_core_hours(config[case_suffix])

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
                ) / self.dataframe["Number_of_spinors"] + self.dataframe[
                    "Total_overhead_time"
                ]
        else:
            # Use base formula
            if analysis_case == "forward_operator_applications":
                self.dataframe[output_column] = (
                    self.dataframe["Total_calculation_time"]
                    / self.dataframe["Number_of_vectors"]
                )
            else:  # inversions
                self.dataframe[output_column] = (
                    self.dataframe["Total_calculation_time"]
                    / self.dataframe["Number_of_spinors"]
                )

        self.logger.info(f"Calculated {output_column}")

    def _calculate_core_hours(self, config: Dict) -> None:
        """Calculate core hours."""
        if "Number_of_cores" not in self.dataframe.columns:
            return

        output_column = config["output_column"]
        # Extract input column from formula string - simplified approach
        if "Average_wall_clock_time_per_vector" in config["formula"]:
            input_column = "Average_wall_clock_time_per_vector"
        else:
            input_column = "Average_wall_clock_time_per_spinor"

        if input_column in self.dataframe.columns:
            self.dataframe[output_column] = (
                self.dataframe["Number_of_cores"] * self.dataframe[input_column] / 3600
            )
            self.logger.info(f"Calculated {output_column}")

    def _calculate_adjusted_core_hours(self, config: Dict) -> None:
        """Calculate adjusted core hours with specific adjustment
        rules."""
        base_column = config["base_column"]
        output_column = config["output_column"]
        adjustment_rules = config["adjustment_rules"]

        if base_column not in self.dataframe.columns:
            return

        # Start with base values
        self.dataframe[output_column] = self.dataframe[base_column]

        # Apply adjustment rules
        for condition, multiplier in adjustment_rules.items():
            # Parse condition and apply - this is a simplified
            # implementation
            if "Number_of_cores ==" in condition:
                if "and" in condition:
                    # Handle compound conditions
                    parts = condition.split(" and ")
                    cores_condition = parts[0].strip()
                    kernel_condition = parts[1].strip()

                    cores_value = int(cores_condition.split("==")[1].strip())
                    kernel_value = (
                        kernel_condition.split("==")[1].strip().replace("'", "")
                    )

                    mask = (self.dataframe["Number_of_cores"] == cores_value) & (
                        self.dataframe["Kernel_operator_type"] == kernel_value
                    )
                else:
                    # Simple cores condition
                    cores_value = int(condition.split("==")[1].strip())
                    mask = self.dataframe["Number_of_cores"] == cores_value

                self.dataframe.loc[mask, output_column] = (
                    self.dataframe.loc[mask, base_column] * multiplier
                )

        self.logger.info(f"Calculated {output_column} with adjustments")
