# API Reference: TableGenerator

## Overview

The `TableGenerator` class extends `DataFrameAnalyzer` to provide specialized
functionality for creating well-formatted analysis tables from DataFrame
results. It supports multiple output formats with a focus on Markdown, and
offers methods for generating comparison tables, pivot tables, and grouped
summaries. This class is particularly useful for analyzing how output quantities
vary across different parameter combinations.

## Class Definition

```python
class TableGenerator(DataFrameAnalyzer):
    def __init__(dataframe: pd.DataFrame, output_directory: str = ".")
```

## Constructor

### `TableGenerator(dataframe, output_directory)`

Creates a new TableGenerator instance with the provided DataFrame and output
directory.

**Parameters:**
- `dataframe` (pd.DataFrame): The input DataFrame to analyze and generate tables
  from.
- `output_directory` (str, optional): Default directory to save generated table
  files. Defaults to current directory (".").

**Raises:**
- `TypeError`: If the input is not a Pandas DataFrame.

**Example:**
```python
import pandas as pd
from your_library import TableGenerator

# Create a sample DataFrame
df = pd.DataFrame({
    'parameter1': [1, 2, 3],
    'parameter2': ['A', 'B', 'C'],
    'output1': [10.1, 10.2, 10.3]
})

# Initialize the generator
generator = TableGenerator(df, output_directory="./tables")
```

## Methods

### `generate_column_uniqueness_report(max_width=80, separate_by_type=True, export_to_file=False, output_directory=None, filename="column_uniqueness_report", file_format="md")`

Generate a table reporting uniqueness characteristics of DataFrame columns.

**Parameters:**
- `max_width` (int, optional): Maximum width of the table in characters. Default
  is 80.
- `separate_by_type` (bool, optional): Whether to separate fields by their type
  (tunable parameters vs output quantities). Default is True.
- `export_to_file` (bool, optional): Whether to export the table to a file.
  Default is False.
- `output_directory` (str, optional): Directory to save the output file. If
  None, uses the class's default output directory.
- `filename` (str, optional): Base name for the output file (without extension).
  Default is 'column_uniqueness_report'.
- `file_format` (str, optional): Format of the file: 'md', 'txt', or 'tex'.
  Default is 'md'.

**Returns:**
- `str`: A formatted string containing the table, optimized for Markdown
  display.

**Example:**
```python
# Generate a report showing single-valued and multi-valued columns
report = generator.generate_column_uniqueness_report(
    max_width=100,
    separate_by_type=True,
    export_to_file=True,
    filename="uniqueness_analysis"
)
print(report)
```

### `generate_grouped_summary_tables(value_variable, row_variable=None, column_variable=None, aggregation="count", exclude_from_grouping=None, export_to_file=False, output_directory=None, filename="summary_tables", file_format="md")`

Generate grouped summary tables with optional row/column pivots and
aggregations.

**Parameters:**
- `value_variable` (str): The variable to summarize.
- `row_variable` (str, optional): Row variable in table.
- `column_variable` (str, optional): Column variable in table.
- `aggregation` (str, optional): One of 'count', 'list', 'len', 'min', 'max',
  'mean'.
- `exclude_from_grouping` (list, optional): Additional tunable parameters to
  exclude from grouping.
- `export_to_file` (bool, optional): Whether to export the table to a file.
- `output_directory` (str, optional): Directory to save the output file.
- `filename` (str, optional): Filename (without extension).
- `file_format` (str, optional): Output format ('md', 'txt', etc.).

**Returns:**
- `str`: A single formatted string containing all tables.

**Example:**
```python
# Generate summary tables of average output1 values
tables = generator.generate_grouped_summary_tables(
    value_variable='output1',
    row_variable='parameter1',
    column_variable='parameter2',
    aggregation='mean'
)
print(tables)
```

### `generate_comparison_table_by_pivot(value_variable, pivot_variable, id_variable, comparison="ratio", exclude_from_grouping=None, export_to_file=False, output_directory=None, filename="comparison_table", file_format="md")`

Generate comparison tables of a value variable across two categories.

**Parameters:**
- `value_variable` (str): The numeric variable to compare.
- `pivot_variable` (str): The categorical variable to compare across (must have
  exactly two unique values per group).
- `id_variable` (str): The ID variable to match entries between pivot values.
- `comparison` (str): 'ratio' or 'difference'. Default is 'ratio'.
- `exclude_from_grouping` (list, optional): Extra parameters to exclude from
  grouping.
- `export_to_file` (bool, optional): Whether to export the table to a file.
- `output_directory` (str, optional): Directory to save output file.
- `filename` (str, optional): Name of the file to save.
- `file_format` (str, optional): Output format ('md', 'txt', etc.).

**Returns:**
- `str`: A Markdown-formatted string containing all group tables.

**Example:**
```python
# Compare output1 values between two categories of parameter2
comparison = generator.generate_comparison_table_by_pivot(
    value_variable='output1',
    pivot_variable='parameter2',
    id_variable='parameter1',
    comparison='ratio'
)
print(comparison)
```

## Usage Examples

### Basic Table Generation

```python
import pandas as pd
from your_library import TableGenerator

# Create a DataFrame with experiment data
df = pd.DataFrame({
    'temperature': [298, 310, 323, 298, 310, 323],
    'pressure': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    'catalyst': ['A', 'A', 'A', 'A', 'A', 'A'],
    'yield': [0.45, 0.52, 0.61, 0.47, 0.55, 0.63],
    'selectivity': [0.78, 0.75, 0.72, 0.79, 0.76, 0.73]
})

# Initialize generator
generator = TableGenerator(df, output_directory="./analysis_tables")

# Generate uniqueness report
uniqueness_report = generator.generate_column_uniqueness_report(
    export_to_file=True,
    filename="parameter_uniqueness"
)

# Generate summary tables for yield grouped by temperature and pressure
summary_tables = generator.generate_grouped_summary_tables(
    value_variable='yield',
    row_variable='temperature',
    column_variable='pressure',
    aggregation='mean',
    export_to_file=True,
    filename="yield_summary"
)
```

### Comparison Analysis

```python
# Compare selectivity between different pressure levels
comparison_table = generator.generate_comparison_table_by_pivot(
    value_variable='selectivity',
    pivot_variable='pressure',
    id_variable='temperature',
    comparison='ratio',
    export_to_file=True,
    filename="pressure_comparison"
)

# Print the results
print("Selectivity Comparison Across Pressure Levels:")
print(comparison_table)
```

## Notes

- The class inherits all functionality from `DataFrameAnalyzer`, including
  DataFrame filtering and grouping capabilities.
- All table generation methods support multiple output formats (Markdown, plain
  text, LaTeX).
- Tables can be displayed directly or saved to files in the specified output
  directory.
- The class expects the same constants from the `constants` module as
  `DataFrameAnalyzer`.
