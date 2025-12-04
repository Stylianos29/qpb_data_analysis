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
    CANONICAL_SOLVER_PARAMETER_NAMES,
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
        Resolve generic Rational_order to method-specific order
        parameter.

        Maps Rational_order (from filename _n pattern) to:
            - KL_diagonal_order (for KL method)
            - Neuberger_order (for Neuberger method)
            - Zolotarev_order (for Zolotarev method)

        If the specific order already exists (from file contents),
        performs cross-validation and logs any mismatches.
        """
        self.logger.info("Resolving Rational_order to method-specific names")

        if "Rational_order" not in self.dataframe.columns:
            self.logger.debug("No Rational_order column found - skipping resolution")
            return

        if "Overlap_operator_method" not in self.dataframe.columns:
            self.logger.warning(
                "Overlap_operator_method not found - cannot resolve Rational_order"
            )
            return

        for idx, row in self.dataframe.iterrows():
            idx = cast(int | str, idx)

            overlap_method = row.get("Overlap_operator_method")
            rational_order = row.get("Rational_order")

            if (
                not isinstance(overlap_method, str)
                or overlap_method not in RATIONAL_ORDER_RESOLUTION_MAPPING
            ):
                continue

            if pd.isna(rational_order):
                continue

            # Get the target column name
            target_column = RATIONAL_ORDER_RESOLUTION_MAPPING[overlap_method]

            # Check if target already exists (from file contents)
            if target_column in self.dataframe.columns:
                existing_value = self.dataframe.loc[idx, target_column]

                if not pd.isna(existing_value) and existing_value != rational_order:
                    self.logger.warning(
                        f"Row {idx}: Mismatch between Rational_order ({rational_order}) "
                        f"and {target_column} ({existing_value}) - using file contents value"
                    )
                    continue  # Keep file contents value

            # Copy Rational_order to specific column
            self.dataframe.loc[idx, target_column] = rational_order
            self.logger.debug(f"Row {idx}: Resolved Rational_order → {target_column}")

        self.logger.info("Rational order resolution completed")

    def _resolve_solver_parameters(self) -> None:
        """
        Resolve ambiguous solver parameters to canonical names.

        This method performs context-dependent renaming of solver
        precision parameters based on Overlap_operator_method and
        program type.

        Resolution logic:
            1. Determine program type from Main_program_type column
            2. For each row, get the overlap method
            3. Look up the resolution mapping
            4. Rename parameters according to the mapping
            5. Log any unresolved parameters

        The mapping resolves:
          - Inner_solver_* → MSCG_* (sign function inversion)
          - Outer_solver_* → CG_* (full overlap inversion)
          - Generic_solver_* → MSCG_* or CG_* (context-dependent)
        """
        self.logger.info("Resolving solver parameters to canonical names")

        # Check if we have the required context columns
        if "Overlap_operator_method" not in self.dataframe.columns:
            self.logger.warning(
                "Overlap_operator_method not found - skipping solver resolution"
            )
            return

        # Process each row (typically all rows have same method/type,
        # but handle general case)
        for idx, row in self.dataframe.iterrows():
            # Cast idx to the appropriate type for .at accessor
            idx = cast(int | str, idx)  # Adjust types based on your index

            overlap_method = row.get("Overlap_operator_method")
            main_program_type = row.get("Main_program_type", "")

            # Type validation: ensure overlap_method is a string
            if not isinstance(overlap_method, str) or not overlap_method:
                self.logger.warning(
                    f"Row {idx}: Invalid Overlap_operator_method value: {overlap_method}"
                )
                continue

            # Determine if this is an invert program
            is_invert = bool(main_program_type == "invert")

            # Get the resolution mapping for this combination
            mapping_key = (overlap_method, is_invert)

            if mapping_key not in SOLVER_PARAMETER_RESOLUTION_RULES:
                self.logger.warning(
                    f"Row {idx}: No resolution mapping for ({overlap_method}, invert={is_invert})"
                )
                continue

            resolution_mapping = SOLVER_PARAMETER_RESOLUTION_RULES[mapping_key]

            # Apply the mapping to this row
            for raw_param, canonical_param in resolution_mapping.items():
                if raw_param in self.dataframe.columns:
                    # Copy the value to the canonical parameter name
                    self.dataframe.at[idx, canonical_param] = self.dataframe.at[
                        idx, raw_param
                    ]

                    self.logger.debug(
                        f"Row {idx}: Resolved {raw_param} → {canonical_param}"
                    )

        # Log summary of resolution
        resolved_params = [
            col
            for col in CANONICAL_SOLVER_PARAMETER_NAMES
            if col in self.dataframe.columns
        ]
        if resolved_params:
            self.logger.info(f"Created canonical solver parameters: {resolved_params}")

        # Check for any raw parameters that weren't resolved
        unresolved_params = [
            col for col in RAW_SOLVER_PARAMETER_NAMES if col in self.dataframe.columns
        ]
        if unresolved_params:
            self.logger.info(
                f"Raw solver parameters present (will be removed): {unresolved_params}"
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

    def _process_hdf5_dataset(self, dataset_name: str, config: Dict) -> None:
        """Process a single HDF5 dataset according to its
        configuration."""
        try:
            # Extract dataset values using HDF5Analyzer
            dataset_dict = self._extract_dataset_to_dict(dataset_name)

            if not dataset_dict:
                self.logger.info(f"No data found for dataset {dataset_name}")
                return

            # Skip if all arrays are empty
            if all(len(arr) == 0 for arr in dataset_dict.values()):
                self.logger.info(f"Skipping {dataset_name}: all arrays are empty")
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

            self.logger.info(
                f"Extracted {len(dataset_dict)} entries for dataset {dataset_name}"
            )

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
