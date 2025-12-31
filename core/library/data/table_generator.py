import os
from typing import Callable, Union, Optional

import pandas as pd
import numpy as np

from .analyzer import DataFrameAnalyzer


class TableGenerator(DataFrameAnalyzer):
    """
    A specialized class for generating formatted analysis tables from
    DataFrame results.

    Extends DataFrameAnalyzer to create well-structured tables that
    summarize data across different groupings of tunable parameters.
    Supports multiple output formats with a focus on Markdown, and
    provides methods for generating comparison tables, pivot tables, and
    grouped summaries. Particularly useful for analyzing how output
    quantities vary across different parameter combinations.
    """

    def __init__(self, dataframe: pd.DataFrame, output_directory: Optional[str] = None):
        """
        Initialize the TableGenerator with a DataFrame and optional
        output directory.

        Args:
            - dataframe (pd.DataFrame): The input DataFrame to analyze
              and generate tables from.
            - output_directory (str, optional): Default directory to
              save generated table files. If None, you must provide an
              output_directory when calling export methods, otherwise a
              ValueError will be raised. Defaults to None.

        Raises:
            TypeError: If the input is not a Pandas DataFrame.
        """
        # Initialize the parent DataFrameAnalyzer class
        super().__init__(dataframe)

        # Store the default output directory
        self.output_directory = output_directory

    def _save_table_to_file(
        self,
        table_string: str,
        output_directory: Optional[str] = None,
        filename: str = "table",
        file_format: str = "md",
    ) -> None:
        """
        Save a table string to a file in the specified directory and
        format.

        Args:
            - table_string (str): The string representation of the table
              to save.
            - output_directory (str, optional): Directory path where the
              table will be saved. If None, uses the class's default
              output directory.
            - filename (str, optional): Base name of the file (without
              extension). Default is 'table'.
            - file_format (str, optional): Format of the file: 'md',
              'txt', or 'tex'. Default is 'md'.

        Raises:
            ValueError: If an unsupported file format is provided.
        """
        # Use class's default output directory if none is provided
        if output_directory is None:
            if self.output_directory is None:
                raise ValueError(
                    "No output directory was specified. "
                    "Table cannot be exported to file."
                )
            output_directory = self.output_directory

        # Normalize and create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Validate file format
        valid_formats = {"md", "txt", "tex"}
        if file_format not in valid_formats:
            raise ValueError(
                f"Unsupported format '{file_format}'. "
                f"Must be one of {valid_formats}."
            )

        # Compose full path
        file_path = os.path.join(output_directory, f"{filename}.{file_format}")

        # Write the table string to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(table_string)

    def generate_column_uniqueness_report(
        self,
        max_width=80,
        separate_by_type=True,
        export_to_file=False,
        output_directory=None,
        filename="column_uniqueness_report",
        file_format="md",
    ) -> str:
        """
        Generate a table reporting uniqueness characteristics of
        DataFrame columns, showing single-valued and multi-valued
        fields, optionally organized by parameter type (tunable
        parameters vs output quantities).

        This method creates a formatted table with two columns: 1.
        Single-valued fields with their unique values 2. Multi-valued
        fields with the count of their unique values

        When separate_by_type is True, each column is further divided
        into tunable parameters and output quantities, with headers
        centered across the entire table width.

        Args:
            - max_width (int, optional): Maximum width of the table in
              characters. Default is 80.
            - separate_by_type (bool, optional): Whether to separate
              fields by their type (tunable parameters vs output
              quantities). Default is True.
            - export_to_file (bool, optional): Whether to export the
              table to a file. Default is False.
            - output_directory (str, optional): Directory to save the
              output file. If None, uses the class's default output
              directory.
            - filename (str, optional): Base name for the output file
              (without extension). Default is
              'column_uniqueness_report'.
            - file_format (str, optional): Format of the file: 'md',
              'txt', or 'tex'. Default is 'md'.

        Returns:
            str: A formatted string containing the table, optimized for
            Markdown display.
        """
        # Calculate how much space to allocate for each column
        half_width = (max_width - 3) // 2  # -3 for the separator and spacing

        # Create the header
        header_left = "Single-valued fields: unique value"
        header_right = "Multivalued fields: No of unique values"
        header = f"{header_left:<{half_width}} | {header_right}"

        # Create the separator line
        separator = "-" * max_width

        # Start building the table with header and separator
        table_lines = [header, separator]

        if separate_by_type:
            # GROUP ITEMS BY TYPE AND VALUE COUNT
            single_valued_tunable = [
                (col, self.unique_value_columns_dictionary[col])
                for col in self.list_of_single_valued_tunable_parameter_names
            ]
            single_valued_output = [
                (col, self.unique_value_columns_dictionary[col])
                for col in self.list_of_single_valued_output_quantity_names
            ]

            multi_valued_tunable = [
                (col, self.multivalued_columns_count_dictionary[col])
                for col in self.list_of_multivalued_tunable_parameter_names
            ]
            multi_valued_output = [
                (col, self.multivalued_columns_count_dictionary[col])
                for col in self.list_of_multivalued_output_quantity_names
            ]

            # Sort items alphabetically by column name within each group
            single_valued_tunable.sort(key=lambda x: x[0])
            single_valued_output.sort(key=lambda x: x[0])
            multi_valued_tunable.sort(key=lambda x: x[0])
            multi_valued_output.sort(key=lambda x: x[0])

            # Index counters for left and right columns
            left_index = 0
            right_index = 0

            # Add tunable parameters header centered across the entire line width
            if single_valued_tunable or multi_valued_tunable:
                # Calculate the center position and create a centered header
                header_text = "TUNABLE PARAMETERS"
                padding = (max_width - len(header_text)) // 2
                row = " " * padding + header_text
                table_lines.append(row)

                # Determine maximum rows needed for this section
                max_rows = max(
                    len(single_valued_tunable) if single_valued_tunable else 0,
                    len(multi_valued_tunable) if multi_valued_tunable else 0,
                )

                # Output rows in parallel
                for _ in range(max_rows):
                    # Left column (single-valued tunable parameters)
                    left_col = ""
                    if left_index < len(single_valued_tunable):
                        col_name, value = single_valued_tunable[left_index]
                        left_index += 1

                        # Format the value appropriately
                        if isinstance(value, float):
                            if pd.isna(value):
                                formatted_value = "NaN"
                            elif value == int(value):
                                formatted_value = str(int(value))
                            else:
                                formatted_value = f"{value:.8g}"
                        else:
                            formatted_value = str(value)

                        left_col = f"{col_name}: {formatted_value}"

                    # Right column (multi-valued tunable parameters)
                    right_col = ""
                    if right_index < len(multi_valued_tunable):
                        col_name, count = multi_valued_tunable[right_index]
                        right_index += 1
                        right_col = f"{col_name}: {count}"

                    # Create the row
                    row = f"{left_col:<{half_width}} | {right_col}"
                    table_lines.append(row)

            # Add a blank row between sections if both sections have content
            if (single_valued_tunable or multi_valued_tunable) and (
                single_valued_output or multi_valued_output
            ):
                table_lines.append("")

            # Reset index counters for output quantities
            left_index = 0
            right_index = 0

            # Add output quantities header centered across the entire line width
            if single_valued_output or multi_valued_output:
                # Calculate the center position and create a centered header
                header_text = "OUTPUT QUANTITIES"
                padding = (max_width - len(header_text)) // 2
                row = " " * padding + header_text
                table_lines.append(row)

                # Determine maximum rows needed for this section
                max_rows = max(
                    len(single_valued_output) if single_valued_output else 0,
                    len(multi_valued_output) if multi_valued_output else 0,
                )

                # Output rows in parallel
                for _ in range(max_rows):
                    # Left column (single-valued output quantities)
                    left_col = ""
                    if left_index < len(single_valued_output):
                        col_name, value = single_valued_output[left_index]
                        left_index += 1

                        # Format the value appropriately
                        if isinstance(value, float):
                            if pd.isna(value):
                                formatted_value = "NaN"
                            elif value == int(value):
                                formatted_value = str(int(value))
                            else:
                                formatted_value = f"{value:.8g}"
                        else:
                            formatted_value = str(value)

                        left_col = f"{col_name}: {formatted_value}"

                    # Right column (multi-valued output quantities)
                    right_col = ""
                    if right_index < len(multi_valued_output):
                        col_name, count = multi_valued_output[right_index]
                        right_index += 1
                        right_col = f"{col_name}: {count}"

                    # Create the row
                    row = f"{left_col:<{half_width}} | {right_col}"
                    table_lines.append(row)
        else:
            # SIMPLER VERSION - NO SEPARATION BY TYPE
            # Get all single-valued and multi-valued fields
            single_valued_fields = [
                (col, self.unique_value_columns_dictionary[col])
                for col in self.list_of_single_valued_column_names
            ]

            multi_valued_fields = [
                (col, self.multivalued_columns_count_dictionary[col])
                for col in self.list_of_multivalued_column_names
            ]

            # Sort alphabetically by column name
            single_valued_fields.sort(key=lambda x: x[0])
            multi_valued_fields.sort(key=lambda x: x[0])

            # Determine the maximum number of rows needed
            max_rows = max(len(single_valued_fields), len(multi_valued_fields))

            # Build each row of the table
            for i in range(max_rows):
                # Prepare left column content (single-valued fields)
                left_col = ""
                if i < len(single_valued_fields):
                    col_name, value = single_valued_fields[i]

                    # Format the value appropriately
                    if isinstance(value, float):
                        if pd.isna(value):
                            formatted_value = "NaN"
                        elif value == int(value):
                            formatted_value = str(int(value))
                        else:
                            formatted_value = (
                                f"{value:.8g}"  # Use general format with precision
                            )
                    else:
                        formatted_value = str(value)

                    left_col = f"{col_name}: {formatted_value}"

                # Prepare right column content (multi-valued fields)
                right_col = ""
                if i < len(multi_valued_fields):
                    col_name, count = multi_valued_fields[i]
                    right_col = f"{col_name}: {count}"

                # Create the row with perfectly aligned separator
                row = left_col.ljust(half_width)
                row += " | "
                row += right_col

                table_lines.append(row)

        # Join all lines into a single string
        table_string = "\n".join(table_lines)

        # Export to file if requested
        if export_to_file:
            self._save_table_to_file(
                table_string=table_string,
                output_directory=output_directory,
                filename=filename,
                file_format=file_format,
            )

        return table_string

    def generate_grouped_summary_tables(
        self,
        value_variable: str,
        row_variable: Optional[str] = None,
        column_variable: Optional[str] = None,
        aggregation: Union[str, Callable] = "count",
        format_value: Optional[Callable] = None,
        exclude_from_grouping: Optional[list] = None,
        export_to_file: bool = False,
        output_directory: Optional[str] = None,
        filename: str = "summary_tables",
        file_format: str = "md",
        verbose: bool = False,
    ) -> str:
        """
        Generate grouped summary tables with optional row/column pivots
        and aggregations.

        Args:
            - value_variable (str): The variable to summarize.
            - row_variable (str, optional): Row variable in table.
            - column_variable (str, optional): Column variable in table.
            - aggregation (Union[str, Callable], optional): Either one
              of the built-in aggregation methods ('count', 'list',
              'len', 'min', 'max', 'mean'), or a custom callable
              function that takes a pandas Series as input and returns a
              value.
            - format_value (Callable, optional): A function to format
              aggregated values for display. Takes a value and returns a
              string. If None, str() is used. Default is None.
            - exclude_from_grouping (list, optional): Additional tunable
              parameters to exclude from grouping.
            - export_to_file (bool, optional): Whether to export the
              table to a file. Default is False.
            - output_directory (str, optional): Directory to save the
              output file. If None, uses the class's default output
              directory.
            - filename (str, optional): Filename (without extension).
              Default: "summary_tables".
            - file_format (str, optional): Format to save in ('md',
              'txt', etc.). Default: 'md'.

        Returns:
            str: A single formatted string containing all tables.
        """
        # If aggregation is a string, check if it's a supported built-in
        built_in_aggregations = {"count", "list", "len", "min", "max", "mean", "std"}
        if isinstance(aggregation, str) and aggregation not in built_in_aggregations:
            raise ValueError(
                f"Unsupported aggregation: '{aggregation}'. "
                f"Must be one of {built_in_aggregations} or a custom callable."
            )

        # If aggregation is not a string, it should be callable
        if not isinstance(aggregation, str) and not callable(aggregation):
            raise ValueError(
                f"aggregation must be either a string from {built_in_aggregations} "
                f"or a custom callable function."
            )

        for var in [value_variable, row_variable, column_variable]:
            if var is not None and var not in self.dataframe.columns:
                raise ValueError(f"'{var}' is not a column in the DataFrame.")

        # Determine which tunable parameters to exclude from grouping
        filter_out_vars = set()
        for var in [value_variable, row_variable, column_variable]:
            if (
                var is not None
                and var in self.list_of_multivalued_tunable_parameter_names
            ):
                filter_out_vars.add(var)

        if exclude_from_grouping:
            for var in exclude_from_grouping:
                if var not in self.dataframe.columns:
                    raise ValueError(
                        f"'{var}' in exclude_from_grouping is not a valid column."
                    )
                filter_out_vars.add(var)

        groupby_obj = self.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=list(filter_out_vars),
            verbose=verbose,
        )

        all_tables = []

        # Define custom aggregation wrapper
        def apply_aggregation(series, agg):
            """Apply either built-in or custom aggregation to a series."""
            if callable(agg):
                return agg(series)
            elif agg == "count":
                return series.nunique()
            elif agg == "list":
                return ", ".join(str(v) for v in sorted(series.unique()))
            elif agg == "len":
                return len(series)
            elif agg == "min":
                return series.min()
            elif agg == "max":
                return series.max()
            elif agg == "mean":
                return series.mean()
            elif agg == "std":
                return series.std() / np.sqrt(series.nunique())

        # Define formatter wrapper
        def format_result(value):
            """Apply formatting function if provided, otherwise use str()."""
            return format_value(value) if format_value is not None else str(value)

        for idx, (group_keys, group_df) in enumerate(groupby_obj):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            group_dict = dict(
                zip(self.reduced_multivalued_tunable_parameter_names_list, group_keys)
            )

            # Convert NumPy scalar types to native Python types
            clean_group_dict = {
                k: v.item() if isinstance(v, np.generic) else v
                for k, v in group_dict.items()
            }

            # Determine aggregation label for the table
            agg_label = aggregation.__name__ if callable(aggregation) else aggregation

            table_lines = [f"#### Group {idx+1}:\n{clean_group_dict}", ""]

            # 0D case (no row or column variables)
            if row_variable is None and column_variable is None:
                values = group_df[value_variable]
                result = apply_aggregation(values, aggregation)
                table_lines.append(
                    f"{value_variable} ({agg_label}): {format_result(result)}"
                )

            # 1D case (only row or column variable)
            elif (row_variable is not None) ^ (column_variable is not None):
                grouping_var = (
                    row_variable if row_variable is not None else column_variable
                )
                group = group_df.groupby(grouping_var, observed=True)[value_variable]

                result_dict = {}
                for key, subset in group:
                    result = apply_aggregation(subset, aggregation)
                    result_dict[key] = result

                table_lines.append(f"{grouping_var} | {agg_label} of {value_variable}")
                table_lines.append("-- | --")
                for k in sorted(result_dict):
                    table_lines.append(f"{k} | {format_result(result_dict[k])}")

            # 2D case (both row and column variables)
            elif row_variable is not None and column_variable is not None:
                pivot_df = group_df.groupby(
                    [row_variable, column_variable], observed=True
                )[value_variable]

                # Apply the aggregation to create the pivot table
                table_df = pivot_df.apply(
                    lambda x: apply_aggregation(x, aggregation)
                ).unstack(fill_value="" if isinstance(aggregation, str) else None)

                # Format the table
                headers = [f"{row_variable} \\ {column_variable}"] + [
                    str(col) for col in table_df.columns
                ]
                table_lines.append(" | ".join(headers))
                table_lines.append(" | ".join([":" + "-" * len(h) for h in headers]))

                for row_index, row in table_df.iterrows():
                    row_str = f"{row_index} | " + " | ".join(
                        format_result(row[col]) if pd.notnull(row[col]) else ""
                        for col in table_df.columns
                    )
                    table_lines.append(row_str)

            # Append final table
            all_tables.append("\n".join(table_lines))

        # Join all tables into one string with separators
        full_output = "\n\n---\n\n".join(all_tables)

        # Save if requested
        if export_to_file:
            self._save_table_to_file(
                table_string=full_output,
                output_directory=output_directory,
                filename=filename,
                file_format=file_format,
            )

        return full_output

    def generate_comparison_table_by_pivot(
        self,
        value_variable: str,
        pivot_variable: str,
        id_variable: str,
        comparison: str = "ratio",
        exclude_from_grouping: Optional[list] = None,
        export_to_file: bool = False,
        output_directory: Optional[str] = None,
        filename: str = "comparison_table",
        file_format: str = "md",
        verbose: bool = False,
    ) -> str:
        """
        Generate comparison tables (e.g., ratio or difference) of a
        value variable across two categories of a pivot variable,
        grouped by the rest of the multivalued tunable parameters.

        Args:
            - value_variable (str): The numeric variable to compare
              (e.g., 'Condition_number').
            - pivot_variable (str): The categorical variable to compare
              across (e.g., 'Kernel_operator_type'). Must have exactly
              two unique values per group.
            - id_variable (str): The ID variable to match entries
              between pivot values (e.g., 'Configuration_label').
            - comparison (str): 'ratio' or 'difference'. Default is
              'ratio'.
            - exclude_from_grouping (list, optional): Extra parameters
              to exclude from grouping.
            - export_to_file (bool, optional): Whether to export the
              table to a file. Default is False.
            - output_directory (str, optional): Directory to save output
              file. If None, uses the class's default output directory.
            - filename (str, optional): Name of the file to save (no
              extension). Default is 'comparison_table'.
            - file_format (str, optional): Output format: 'md', 'txt',
              etc. Default is 'md'.

        Returns:
            str: A Markdown-formatted string containing all group
            tables.
        """
        if comparison not in {"ratio", "difference"}:
            raise ValueError("comparison must be either 'ratio' or 'difference'")

        for var in [value_variable, pivot_variable, id_variable]:
            if var not in self.dataframe.columns:
                raise ValueError(f"'{var}' is not a valid column in the DataFrame.")

        # Determine which parameters to exclude from grouping
        filter_out_vars = {pivot_variable, id_variable}
        if exclude_from_grouping:
            filter_out_vars.update(exclude_from_grouping)

        filter_out_vars = list(filter_out_vars)

        groupby_obj = self.group_by_multivalued_tunable_parameters(
            filter_out_parameters_list=filter_out_vars,
            verbose=verbose,
        )

        all_tables = []

        for idx, (group_keys, group_df) in enumerate(groupby_obj):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            group_dict = dict(
                zip(self.reduced_multivalued_tunable_parameter_names_list, group_keys)
            )

            # Pivot to wide format: one row per id_variable, columns are pivot values
            pivoted = group_df.pivot_table(
                index=id_variable,
                columns=pivot_variable,
                values=value_variable,
                aggfunc="mean",  # Can be changed to other strategies if needed
                observed=True,
            )

            # Ensure exactly two pivot values exist per group
            pivot_columns = pivoted.columns.tolist()
            if len(pivot_columns) != 2:
                continue  # skip incomplete comparisons

            a, b = pivot_columns
            if comparison == "ratio":
                comparison_series = pivoted[a] / pivoted[b]
                comparison_label = f"{a} / {b}"
            elif comparison == "difference":
                comparison_series = pivoted[a] - pivoted[b]
                comparison_label = f"{a} - {b}"

            # Combine all into final DataFrame
            result_df = pivoted.copy()
            result_df[comparison_label] = comparison_series

            # Format table header
            table_lines = [f"#### Group {idx+1}:\n{group_dict}", ""]
            headers = (
                [id_variable] + [str(col) for col in pivot_columns] + [comparison_label]
            )
            table_lines.append(" | ".join(headers))
            table_lines.append(" | ".join([":" + "-" * (len(h) - 1) for h in headers]))

            for row_idx, row in result_df.iterrows():
                formatted_row = [str(row_idx)] + [
                    f"{row[col]:.6g}" if pd.notnull(row[col]) else "NaN"
                    for col in pivot_columns + [comparison_label]
                ]
                table_lines.append(" | ".join(formatted_row))

            all_tables.append("\n".join(table_lines))

        # Join all tables
        full_output = "\n\n---\n\n".join(all_tables)

        # Save if requested
        if export_to_file:
            self._save_table_to_file(
                table_string=full_output,
                output_directory=output_directory,
                filename=filename,
                file_format=file_format,
            )

        return full_output
