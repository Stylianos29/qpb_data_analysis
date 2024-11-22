"""
inspect_csv_file.py

A utility script to inspect and analyze a CSV file.

Purpose:
    - Identify fields with a single unique value.
    - Count the number of unique values for fields with multiple unique values.
    - Provide additional functionalities such as listing fields, sampling rows,
      and displaying basic statistics for numeric fields.

Functionality:
    - Default behavior: Split fields into two categories:
        1. Fields with a single unique value.
        2. Fields with multiple unique values (along with the count of unique
           values).
    - Optional flags to explore specific aspects of the file.

Input:
    - Path to the CSV file.

Output:
    - Field analysis and optional exploration results.

Usage Example:
    python inspect_csv_file.py --csv_file_path path/to/file.csv python
    inspect_csv_file.py --csv_file_path path/to/file.csv --list_fields python
    inspect_csv_file.py --csv_file_path path/to/file.csv --sample_rows 10 python
    inspect_csv_file.py --csv_file_path path/to/file.csv --field_statistics
"""

import sys

import click  # type: ignore
import pandas as pd

sys.path.append('../../')
from library import filesystem_utilities


@click.command()
@click.option("--csv_file_path", "csv_file_path",
              "-csv", required=True,
              help="Path to the CSV file to be inspected.")
@click.option("--list_fields", is_flag=True, 
              help="List all fields (columns) in the CSV file.")
@click.option("--sample_rows", "num_rows", default=5, type=int,
              help="Show a sample of rows from the CSV file (default: 5).")
@click.option("--field_statistics", is_flag=True,
              help="Show basic statistics for numeric fields.")

def main(csv_file_path, list_fields, num_rows, field_statistics):

    # Check provided path to .csv file
    if not filesystem_utilities.is_valid_file(csv_file_path):
        print("ERROR: Provided path to .csv file is invalid.")
        print("Exiting...")
        sys.exit(1)

    # Load the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Default behavior: Field analysis
    print("\nField Analysis:")
    single_unique_fields = []
    multiple_unique_fields = {}

    for col in df.columns:
        unique_values = df[col].nunique()
        if unique_values == 1:
            single_unique_fields.append((col, df[col].iloc[0]))
        else:
            multiple_unique_fields[col] = unique_values
    
    # Display fields with single unique value
    print("\nFields with a single unique value:")
    if single_unique_fields:
        for field, value in single_unique_fields:
            print(f"  {field}: {value}")
    else:
        print("  None")

    # Display fields with multiple unique values
    print("\nFields with multiple unique values (count):")
    if multiple_unique_fields:
        for field, count in multiple_unique_fields.items():
            print(f"  {field}: {count} unique values")
    else:
        print("  None")
    
    # Optional flag: List fields
    if list_fields:
        print("\nFields (columns) in the CSV file:")
        for col in df.columns:
            print(f"- {col}")
    
    # Optional flag: Sample rows
    if num_rows > 0:
        print(f"\nSample of {num_rows} rows from the CSV file:")
        print(df.head(num_rows))
    
    # Optional flag: Field statistics
    if field_statistics:
        print("\nBasic statistics for numeric fields:")
        print(df.describe().transpose())


if __name__ == "__main__":
    main()
