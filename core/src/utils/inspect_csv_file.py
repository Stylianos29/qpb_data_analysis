"""
CSV File Inspector
=================

A command-line utility for comprehensive inspection and analysis of CSV files.

This script provides a variety of analysis tools for examining CSV data,
including:
- Basic information about the file structure and data integrity
- Column uniqueness analysis to identify categorical vs continuous variables
- Statistical summaries for numeric columns
- Sampling capabilities to view representative data
- Detection of unusual data patterns or mixed data types
- Customizable output formats (text, markdown, LaTeX)
- Generation of grouped summary tables for cross-tabulation analysis

The tool is designed to help data scientists and analysts quickly understand the
structure and content of CSV datasets before deeper analysis, identifying
potential data quality issues and providing useful summaries.

Usage:
------
    python inspect_csv_file.py -csv PATH_TO_CSV [options]

Required Arguments:
------------------
    -csv, --csv_file_path PATH    Path to the CSV file to be inspected

Optional Arguments:
-----------------
    -out, --output_directory DIR   Directory where output files will be saved
    --output_format FORMAT         Format of output file (txt, md, or tex)
    --output_filename NAME         Custom filename for the output file -v,
    --verbose                  Print output to console even when writing to file
    --uniqueness_report            Generate a column uniqueness report
    --list_fields                  List all fields (columns) in the CSV file
    --sample_rows N                Show N sample rows from the CSV file
    --field_statistics             Show basic statistics for numeric fields
    --separate_by_type             Separate columns by type in the uniqueness
    report --show_unique_values COL       Show all unique values for specified
    column --add_grouped_summary          Generate grouped summary tables
    --value_variable VAR           Variable to summarize in grouped tables
    --row_variable VAR             Row variable for grouped tables
    --column_variable VAR          Column variable for grouped tables
    --aggregation METHOD           Aggregation method for grouped tables
                                   (count, list, len, min, max, mean)

Examples:
--------
    # Basic inspection with console output python inspect_csv_file.py -csv
    data.csv --sample_rows 5 --list_fields
    
    # Generate a comprehensive report in markdown format python
    inspect_csv_file.py -csv data.csv -out reports/ --output_format md \
                              --uniqueness_report --field_statistics
    
    # Create a cross-tabulation summary python inspect_csv_file.py -csv
    sales.csv --add_grouped_summary \
                              --value_variable sales --row_variable region \
                              --column_variable quarter --aggregation mean

Dependencies:
------------
    - pandas: For data manipulation and analysis
    - click: For command line interface
    - Custom library modules for analysis and validation
"""

import os
import sys

import click
import pandas as pd

# Import library components
from library.validation.click_validators import (
    csv_file,
    directory,
)
from library import (
    load_csv,
    DataFrameAnalyzer,
    TableGenerator,
)


@click.command()
@click.option(
    "-csv",
    "--csv_file_path",
    "csv_file_path",
    required=True,
    callback=csv_file.input,
    help="Path to the CSV file to be inspected.",
)
@click.option(
    "-out",
    "--output_directory",
    "output_directory",
    default=None,
    callback=directory.can_create,
    help="Directory where output files will be saved.",
)
@click.option(
    "--output_format",
    "output_format",
    default="txt",
    type=click.Choice(["txt", "md", "tex"], case_sensitive=False),
    help="Format of the output file (txt, md, or tex). Default: txt",
)
@click.option(
    "--output_filename",
    default=None,
    type=str,
    help="Custom filename for the output file (without extension).",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Print output to console even when writing to file.",
)
@click.option(
    "--uniqueness_report", is_flag=True, help="Generate a column uniqueness report."
)
@click.option(
    "--list_fields", is_flag=True, help="List all fields (columns) in the CSV file."
)
@click.option(
    "--sample_rows",
    "num_rows",
    default=0,
    type=int,
    help="Show a sample of rows from the CSV file.",
)
@click.option(
    "--field_statistics", is_flag=True, help="Show basic statistics for numeric fields."
)
@click.option(
    "--separate_by_type",
    is_flag=True,
    help=(
        "Separate columns by tunable parameters and output quantities in "
        "the uniqueness report."
    ),
)
@click.option(
    "--show_unique_values",
    "show_unique_values_for",
    default=None,
    type=str,
    help="Show all unique values for a specified column.",
)
@click.option(
    "--add_grouped_summary",
    is_flag=True,
    help="Generate grouped summary tables for specified variables.",
)
@click.option(
    "--value_variable",
    default=None,
    type=str,
    help="Variable to summarize in grouped summary tables.",
)
@click.option(
    "--row_variable",
    default=None,
    type=str,
    help="Row variable for grouped summary tables.",
)
@click.option(
    "--column_variable",
    default=None,
    type=str,
    help="Column variable for grouped summary tables.",
)
@click.option(
    "--aggregation",
    default="count",
    type=click.Choice(
        ["count", "list", "len", "min", "max", "mean"], case_sensitive=False
    ),
    help="Aggregation method for grouped summary tables.",
)
def main(
    csv_file_path,
    output_directory,
    output_format,
    output_filename,
    verbose,
    uniqueness_report,
    list_fields,
    num_rows,
    field_statistics,
    separate_by_type,
    show_unique_values_for,
    add_grouped_summary,
    value_variable,
    row_variable,
    column_variable,
    aggregation,
):
    # Setup output file handler based on format
    output_file = None
    if output_directory is not None and output_format:
        # Determine the output filename
        if output_filename:
            base_filename = output_filename
        else:
            # Generate base filename without extension from input CSV
            base_filename = (
                os.path.splitext(os.path.basename(csv_file_path))[0] + "_summary"
            )

        output_file_path = os.path.join(
            output_directory, f"{base_filename}.{output_format}"
        )
        output_file = open(output_file_path, "w", encoding="utf-8")

    def write_section(title, content=""):
        """Write a section with a title and content to output file or console."""
        if output_file:
            if output_format == "md":
                output_file.write(f"## {title}\n\n")
            elif output_format == "tex":
                output_file.write(f"\\section{{{title}}}\n\n")
            else:  # txt format
                output_file.write(f"\n{title}\n")
                output_file.write("=" * len(title) + "\n")

            if content:
                output_file.write(f"{content}\n\n")

        # Print to console if no output file or verbose mode is on
        if not output_file or verbose:
            click.echo(f"\n{title}")
            click.echo("=" * len(title))
            if content:
                click.echo(f"{content}\n")

    def write_content(content, print_to_console=True):
        """
        Write content to output file and optionally to console.

        Args:
            content (str): The content to write
            print_to_console (bool): Whether to also print to console when writing to file
        """
        if output_file:
            output_file.write(f"{content}\n\n")

            # Print to console only if verbose mode is on or print_to_console flag is True
            if verbose and print_to_console:
                click.echo(f"{content}")
        else:
            click.echo(f"{content}")

    # Load the CSV file
    try:
        # Get file name and directory separately
        file_name = os.path.basename(csv_file_path)
        parent_dir = os.path.dirname(csv_file_path)

        # Load the file
        dataframe = load_csv(csv_file_path)

        # Always write the basic info to file and conditionally to console
        loading_message = (
            f"Successfully loaded CSV file: '{file_name}' "
            f"\nfrom directory: '{parent_dir}'."
        )

        if output_file:
            write_section("File Details")
            write_content(
                loading_message, print_to_console=True
            )  # Always show loading message
        else:
            click.echo(loading_message)

        # Create analyzer instances
        analyzer = DataFrameAnalyzer(dataframe)
        table_generator = TableGenerator(dataframe, output_directory or ".")

    except Exception as e:
        error_message = f"ERROR: Failed to load CSV file: {str(e)}"
        if output_file:
            write_content(error_message)
        click.echo(error_message)  # Always print errors to console
        if output_file:
            output_file.close()
        sys.exit(1)

    # Always show basic DataFrame information section
    if output_file:
        write_section("Basic Information")

    # DataFrame shape
    rows, cols = dataframe.shape
    shape_message = f"DataFrame Shape: {rows} rows × {cols} columns"
    write_content(shape_message)

    # Check for empty values
    empty_counts = dataframe.isna().sum()
    empty_columns = empty_counts[empty_counts > 0]

    if len(empty_columns) > 0:
        write_content("Columns with empty values:")
        for col, count in empty_columns.items():
            empty_col_message = f"  {col}: {count} rows with empty values"
            write_content(empty_col_message)
    else:
        empty_message = "No empty values found in the CSV file."
        write_content(empty_message)

    # Check for unusual data types
    unusual_data_detected = False

    if output_file:
        write_content("Checking for unusual data types:")

    # Check for mixed data types in columns
    for column in dataframe.columns:
        try:
            # Get the column's data
            col_data = dataframe[column].dropna()

            # Skip empty columns
            if len(col_data) == 0:
                continue

            # Check for mixed numeric and non-numeric values
            if pd.api.types.is_numeric_dtype(col_data):
                # For numeric columns, check if there are any strings mixed in
                try:
                    pd.to_numeric(col_data)
                except:
                    unusual_data_detected = True
                    unusual_message = (
                        f"  Column '{column}' has mixed numeric and non-numeric values"
                    )
                    write_content(unusual_message)

            # Check for date-like strings mixed with other types
            if col_data.dtype == object:
                date_count = 0
                total_count = len(col_data)

                # Check a sample for date-like patterns
                sample = col_data.sample(min(100, total_count))
                for val in sample:
                    if isinstance(val, str):
                        # Simple pattern matching for dates
                        if any(pattern in val for pattern in ["/", "-"]) and any(
                            c.isdigit() for c in val
                        ):
                            date_count += 1

                # If some but not all values look like dates
                if 0 < date_count < len(sample) * 0.9:
                    unusual_data_detected = True
                    date_message = (
                        f"  Column '{column}' may have "
                        "inconsistent date formats or mixed data types"
                    )
                    write_content(date_message)

        except Exception as e:
            # If we can't analyze a column, it might have unusual data
            error_analysis_message = f"  Could not analyze column '{column}': {str(e)}"
            write_content(error_analysis_message)
            unusual_data_detected = True

    if not unusual_data_detected:
        no_unusual_message = "  No unusual data types detected in the CSV file."
        write_content(no_unusual_message)

    # If additional analysis options are specified, show these in formatted sections
    if (
        list_fields
        or num_rows > 0
        or field_statistics
        or uniqueness_report
        or show_unique_values_for
        or add_grouped_summary
    ):
        # List fields if requested
        if list_fields:
            write_section("Columns")
            cols_info = []

            # Group columns by type (tunable parameters vs output quantities)
            tunable_params = analyzer.list_of_tunable_parameter_names_from_dataframe
            output_quantities = analyzer.list_of_output_quantity_names_from_dataframe

            if tunable_params:
                cols_info.append("Tunable Parameters:")
                cols_info.append("  " + "\n  ".join(sorted(tunable_params)))

            if output_quantities:
                cols_info.append("Output Quantities:")
                cols_info.append("  " + "\n  ".join(sorted(output_quantities)))

            write_content("\n".join(cols_info))

        # Sample rows if requested
        if num_rows > 0:
            write_section(f"Sample Data ({num_rows} rows)")
            try:
                if output_format == "md":
                    # Convert DataFrame to markdown table
                    sample_data = dataframe.head(num_rows).to_markdown(index=False)
                elif output_format == "tex":
                    # Convert DataFrame to latex table
                    sample_data = dataframe.head(num_rows).to_latex(index=False)
                else:
                    # Plain text format
                    sample_data = dataframe.head(num_rows).to_string(index=False)
                write_content(
                    sample_data, print_to_console=False
                )  # Don't duplicate in console
            except Exception as e:
                write_content(f"Could not display sample rows: {str(e)}")

        # Show uniqueness report if requested
        if uniqueness_report:
            write_section("Column Uniqueness Report")
            try:
                report = table_generator.generate_column_uniqueness_report(
                    separate_by_type=separate_by_type, export_to_file=False
                )
                write_content(
                    report, print_to_console=False
                )  # Don't duplicate in console
            except Exception as e:
                write_content(f"Could not generate uniqueness report: {str(e)}")

        # Show unique values for a specific column if requested
        if show_unique_values_for:
            write_section(f"Unique Values for '{show_unique_values_for}'")
            try:
                # Capture output from unique_values method
                import io
                from contextlib import redirect_stdout

                output = io.StringIO()
                with redirect_stdout(output):
                    print(analyzer.unique_values(show_unique_values_for))

                write_content(output.getvalue())
            except Exception as e:
                write_content(f"Could not show unique values: {str(e)}")

        # Field statistics if requested
        if field_statistics:
            write_section("Statistical Summary")
            try:
                # Filter to numeric columns only
                numeric_df = dataframe.select_dtypes(include=["number"])
                if not numeric_df.empty:
                    if output_format == "md":
                        stats = numeric_df.describe().transpose().to_markdown()
                    elif output_format == "tex":
                        stats = numeric_df.describe().transpose().to_latex()
                    else:
                        stats = numeric_df.describe().transpose().to_string()
                    write_content(
                        stats, print_to_console=False
                    )  # Don't duplicate in console
                else:
                    write_content("No numeric columns found for statistical analysis.")
            except Exception as e:
                write_content(f"Could not generate statistics: {str(e)}")

        # Generate grouped summary tables if requested
        if add_grouped_summary and value_variable:
            write_section("Grouped Summary Tables")
            try:
                summary_tables = table_generator.generate_grouped_summary_tables(
                    value_variable=value_variable,
                    row_variable=row_variable,
                    column_variable=column_variable,
                    aggregation=aggregation,
                    export_to_file=False,
                )
                write_content(
                    summary_tables, print_to_console=False
                )  # Don't duplicate in console
            except Exception as e:
                write_content(f"Could not generate grouped summary tables: {str(e)}")

    # Close the output file if it was opened
    if output_file:
        output_file.close()

    click.echo(
        f"   -- Summary of the '{os.path.basename(csv_file_path)}' "
        "CSV file generated."
    )


if __name__ == "__main__":
    main()
