# CSV File Inspector

A comprehensive command-line utility for inspecting and analyzing CSV files.

## Overview

CSV File Inspector is a powerful command-line tool designed to help data
scientists, analysts, and researchers quickly understand the structure, content,
and quality of CSV datasets. It provides a suite of analysis capabilities that
make it easier to identify data issues, understand variable distributions, and
generate useful summary statistics—all before deeper analysis begins.

## Features

- **Basic Information**: Analyze file structure, dimensions, and data
  completeness
- **Data Quality Checks**: Detect empty values and unusual data patterns
- **Column Analysis**: Generate uniqueness reports to identify categorical vs.
  continuous variables
- **Data Sampling**: View representative rows from the dataset
- **Statistical Summaries**: Calculate basic statistics for numeric columns
- **Grouped Analysis**: Create cross-tabulation summaries with customizable
  aggregations
- **Flexible Output**: Generate reports in plain text, Markdown, or LaTeX
  formats

## Installation

### Prerequisites

The CSV File Inspector requires:

- Python 3.6+
- pandas
- click
- Custom library modules (included in the package)

### Setup

1. Ensure all dependencies are installed:
   ```
   pip install pandas click
   ```

2. Make sure the custom library module is in your Python path or in the same
   directory as the script.

## Usage

### Basic Command Structure

```
python inspect_csv_file.py -csv PATH_TO_CSV [options]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `-csv`, `--csv_file_path` | Path to the CSV file to be inspected |

### Optional Arguments

| Argument | Description |
|----------|-------------|
| `-out`, `--output_directory` | Directory where output files will be saved |
| `--output_format` | Format of the output file (txt, md, or tex) |
| `--output_filename` | Custom filename for the output file |
| `-v`, `--verbose` | Print output to console even when writing to file |
| `--uniqueness_report` | Generate a column uniqueness report |
| `--list_fields` | List all fields (columns) in the CSV file |
| `--sample_rows` | Number of sample rows to show from the CSV file |
| `--field_statistics` | Show basic statistics for numeric fields |
| `--separate_by_type` | Separate columns by tunable parameters and output quantities |
| `--show_unique_values` | Show all unique values for a specified column |
| `--add_grouped_summary` | Generate grouped summary tables for variables |
| `--value_variable` | Variable to summarize in grouped tables |
| `--row_variable` | Row variable for grouped summary tables |
| `--column_variable` | Column variable for grouped summary tables |
| `--aggregation` | Aggregation method for grouped tables (count, list, len, min, max, mean) |

## Examples

### Basic Inspection

To perform a quick inspection of a CSV file with console output:

```bash
python inspect_csv_file.py -csv data.csv --sample_rows 5 --list_fields
```

This command:
- Loads the CSV file `data.csv`
- Shows the first 5 rows of data
- Lists all columns in the file

### Comprehensive Report

To generate a detailed report in Markdown format:

```bash
python inspect_csv_file.py -csv data.csv -out reports/ --output_format md \
                          --uniqueness_report --field_statistics
```

This command:
- Analyzes the CSV file `data.csv`
- Saves a Markdown report to the `reports` directory
- Includes a uniqueness report for all columns
- Provides statistical summaries for numeric fields

### Cross-Tabulation Analysis

To create summary tables with aggregated values:

```bash
python inspect_csv_file.py -csv sales.csv --add_grouped_summary \
                          --value_variable revenue --row_variable region \
                          --column_variable quarter --aggregation mean
```

This command:
- Analyzes the sales data in `sales.csv`
- Creates a cross-tabulation with regions as rows and quarters as columns
- Calculates the mean revenue for each region-quarter combination

## Output Details

### Basic Information Section

The basic information section always includes:
- File source information
- Number of rows and columns
- Report on empty values
- Detection of unusual data types

Example output:
```
Basic Information
================
DataFrame Shape: 1000 rows × 15 columns

Columns with empty values:
  customer_notes: 243 rows with empty values
  
Checking for unusual data types:
  No unusual data types detected in the CSV file.
```

### Column Uniqueness Report

This section analyzes each column to determine:
- Number of unique values
- Percentage of uniqueness
- Whether a column is likely categorical or continuous
- Distribution of values for categorical columns

Example output:
```
Column Uniqueness Report
=======================
Tunable Parameters:
  region (7 unique values, 0.7% uniqueness) - Categorical
    Values: North (234), South (210), East (198), West (180), ...
  
  customer_type (3 unique values, 0.3% uniqueness) - Categorical
    Values: Retail (543), Corporate (312), Government (145)

Output Quantities:
  price (987 unique values, 98.7% uniqueness) - Continuous
  quantity (37 unique values, 3.7% uniqueness) - Discrete
```

### Statistical Summary

For numeric columns, the tool generates descriptive statistics:
- Count, mean, standard deviation
- Minimum, maximum
- Percentiles (25%, 50%, 75%)

Example output in markdown format:
```
Statistical Summary
==================
|           | count | mean     | std      | min  | 25%   | 50%   | 75%    | max     |
|-----------|-------|----------|----------|------|-------|-------|--------|---------|
| price     | 1000  | 157.34   | 83.21    | 9.99 | 99.99 | 149.99| 199.99 | 499.99  |
| quantity  | 1000  | 3.45     | 2.87     | 1    | 1     | 2     | 5      | 24      |
| total     | 1000  | 472.58   | 395.48   | 9.99 | 199.98| 299.98| 599.95 | 3999.92 |
```

### Grouped Summary Tables

When grouped summaries are requested, the tool generates cross-tabulations:

Example output:
```
Grouped Summary Tables
=====================
Mean Revenue by Region and Quarter
---------------------------------
|         | Q1     | Q2     | Q3     | Q4     |
|---------|--------|--------|--------|--------|
| North   | 12,453 | 13,566 | 15,211 | 17,988 |
| South   | 10,322 | 11,877 | 12,344 | 16,422 |
| East    | 14,566 | 15,232 | 15,788 | 18,344 |
| West    | 13,211 | 13,877 | 16,322 | 19,788 |
```

## Advanced Usage

### Working with Large Files

For very large CSV files, consider:
- Using just the basic information and uniqueness report first
- Sampling a small number of rows to get a quick overview
- Running field statistics only on selected columns

### Custom Output

The `--output_filename` parameter lets you specify a custom base filename for
the generated report. By default, the script uses the input CSV filename with a
`_summary` suffix.

### Integration with Other Tools

The output from CSV File Inspector can be used as input to other data analysis
processes:
- Text output can be redirected to other command-line tools
- Markdown output can be integrated into documentation systems
- LaTeX output can be included in academic papers or reports

## Under the Hood

The script uses several key components from its custom library:

- **DataFrameAnalyzer**: Handles the core analysis of the DataFrame
- **TableGenerator**: Creates formatted tables and reports
- **Validation functions**: Ensure input files and directories are valid

## Troubleshooting

### Common Issues

- **File Loading Errors**: Ensure your CSV is properly formatted and try
  specifying the encoding if needed
- **Empty Value Detection**: The script considers both NaN and empty strings as
  empty values
- **Statistical Calculation Errors**: May occur if numeric columns contain
  non-numeric values

### Debugging Tips

- Use the `-v` (verbose) flag to see more detailed output
- Check for unusual characters or encoding issues in your CSV file
- For mixed data types, consider pre-processing your CSV before analysis

## Conclusion

CSV File Inspector provides a quick, comprehensive way to understand CSV
datasets before committing to deeper analysis. By identifying potential data
quality issues, understanding variable distributions, and generating useful
summaries, it helps streamline the data exploration process and ensures analysts
start with a solid understanding of their dataset.
