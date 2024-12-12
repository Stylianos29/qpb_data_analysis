import sys

import click
import pandas as pd

from library import filesystem_utilities

@click.command()
@click.option("--csv_file_path", "csv_file_path",
              "-in_csv_path", required=True,
              help="Path to the CSV file to be inspected.")
@click.option("--list_fields", is_flag=True, 
              help="List all fields (columns) in the CSV file.")
@click.option("--sample_rows", "num_rows", default=5, type=int,
              help="Show a sample of rows from the CSV file (default: 5).")
@click.option("--field_statistics", is_flag=True,
              help="Show basic statistics for numeric fields.")
@click.option("--output_file", "output_file", type=str,
              help="Path to the output text file to write the results.")

def main(csv_file_path, list_fields, num_rows, field_statistics, output_file):

    def write_or_print(content):
        """Write content to the output file or print it to the console."""
        if output_file:
            with open(output_file, "a") as file:
                file.write(content + "\n")
        else:
            print(content)

    # Check provided path to .csv file
    if not filesystem_utilities.is_valid_file(csv_file_path):
        error_message = "ERROR: Provided path to .csv file is invalid.\nExiting..."
        write_or_print(error_message)
        sys.exit(1)

    # Clear the output file if it exists
    if output_file:
        open(output_file, "w").close()

    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Default behavior: Field analysis
    write_or_print("\nField Analysis:")
    single_unique_fields = []
    multiple_unique_fields = {}

    for col in df.columns:
        unique_values = df[col].nunique()
        if unique_values == 1:
            single_unique_fields.append((col, df[col].iloc[0]))
        else:
            multiple_unique_fields[col] = unique_values

    # Display fields with single unique value
    write_or_print("\nFields with a single unique value:")
    if single_unique_fields:
        for field, value in single_unique_fields:
            write_or_print(f"  {field}: {value}")
    else:
        write_or_print("  None")

    # Display fields with multiple unique values
    write_or_print("\nFields with multiple unique values (count):")
    if multiple_unique_fields:
        for field, count in multiple_unique_fields.items():
            write_or_print(f"  {field}: {count} unique values")
    else:
        write_or_print("  None")

    # Optional flag: List fields
    if list_fields:
        write_or_print("\nFields (columns) in the CSV file:")
        for col in df.columns:
            write_or_print(f"- {col}")

    # Optional flag: Sample rows
    if num_rows > 0:
        write_or_print(f"\nSample of {num_rows} rows from the CSV file:")
        write_or_print(df.head(num_rows).to_string(index=False))

    # Optional flag: Field statistics
    if field_statistics:
        write_or_print("\nBasic statistics for numeric fields:")
        write_or_print(df.describe().transpose().to_string())

if __name__ == "__main__":
    main()
