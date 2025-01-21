import sys
import os

import click

from library import filesystem_utilities, data_processing


@click.command()
@click.option(
    "--csv_file_path",
    "csv_file_path",
    "-in_csv_path",
    required=True,
    help="Path to the CSV file to be inspected.",
)
@click.option("--generate_output_file", is_flag=True, help="Write output to a .txt file.")
@click.option(
    "--list_fields", is_flag=True, help="List all fields (columns) in the CSV file."
)
@click.option(
    "--sample_rows",
    "num_rows",
    default=0,
    type=int,
    help="Show a sample of rows from the CSV file (default: 5).",
)
@click.option(
    "--field_statistics", is_flag=True, help="Show basic statistics for numeric fields."
)
@click.option(
    "--output_files_directory",
    "output_files_directory",
    default=None,
    type=str,
    help="Path to the output text file to write the results.",
)
def main(
    csv_file_path,
    list_fields,
    num_rows,
    field_statistics,
    generate_output_file,
    output_files_directory,
):
    # VALIDATE INPUT ARGUMENTS

    if not filesystem_utilities.is_valid_file(csv_file_path):
        error_message = "Provided .csv file path is invalid."
        print("ERROR:", error_message)
        sys.exit(1)

    if output_files_directory is not None:
        if not filesystem_utilities.is_valid_directory(output_files_directory):
            error_message = "Provided output files directory is invalid."
            print("ERROR:", error_message)
            sys.exit(1)
    else:
        if generate_output_file:
            output_files_directory = os.path.dirname(csv_file_path)

    output_file_path = os.path.join(
        output_files_directory, os.path.basename(csv_file_path).replace(".csv", ".txt")
    )

    def write_or_print(content):
        """Write content to the output file or print it to the console."""
        if output_files_directory:
            with open(output_file_path, "a") as file:
                file.write(content + "\n")
        else:
            print(content)

    # Clear the output file if it exists
    if output_files_directory:
        open(output_file_path, "w").close()

    # Load the CSV file
    dataframe = data_processing.load_csv(csv_file_path)
    analyzer = data_processing.DataFrameAnalyzer(dataframe)

    # Default behavior: Field analysis
    write_or_print("\nField Analysis:")

    # Display fields with single unique value
    write_or_print("\nFields with a single unique value (unique value):")
    for field, value in analyzer.single_valued_fields_dictionary.items():
        write_or_print(f"  {field}: {value}")

    # Display fields with multiple unique values
    write_or_print("\nFields with multiple unique values (count):")
    for field, count in analyzer.multivalued_fields_dictionary.items():
        write_or_print(f"  {field}: {count} unique values")

    # Optional flag: List fields
    if list_fields:
        write_or_print("\nFields (columns) in the CSV file:")
        for col in dataframe.columns:
            write_or_print(f"- {col}")

    # Optional flag: Sample rows
    if num_rows > 0:
        write_or_print(f"\nSample of {num_rows} rows from the CSV file:")
        write_or_print(dataframe.head(num_rows).to_string(index=False))

    # Optional flag: Field statistics
    if field_statistics:
        write_or_print("\nBasic statistics for numeric fields:")
        write_or_print(dataframe.describe().transpose().to_string())


if __name__ == "__main__":
    main()
