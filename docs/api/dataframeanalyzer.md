# API Reference: DataFrameAnalyzer

## Overview

The `DataFrameAnalyzer` class provides tools for analyzing and manipulating
Pandas DataFrames with a focus on distinguishing between tunable parameters and
output quantities. It helps with categorizing DataFrame columns, identifying
columns with single vs. multiple unique values, grouping data by parameters, and
filtering the DataFrame based on various conditions.

## Class Definition

```python
class DataFrameAnalyzer:
    def __init__(dataframe: pd.DataFrame)
```

## Constructor

### `DataFrameAnalyzer(dataframe)`

Creates a new DataFrameAnalyzer instance with the provided DataFrame.

**Parameters:**
- `dataframe` (pd.DataFrame): The input DataFrame to analyze.

**Raises:**
- `TypeError`: If the input is not a Pandas DataFrame.

**Example:**
```python
import pandas as pd
from your_library import DataFrameAnalyzer

# Create a sample DataFrame
df = pd.DataFrame({
    'parameter1': [1, 2, 3],
    'parameter2': ['A', 'B', 'C'],
    'output1': [10.1, 10.2, 10.3]
})

# Initialize the analyzer
analyzer = DataFrameAnalyzer(df)
```

## Key Attributes

After initialization, the following attributes are available:

- `original_dataframe` (pd.DataFrame): A copy of the original input DataFrame.
- `dataframe` (pd.DataFrame): The working DataFrame that can be filtered and
  manipulated.
- `list_of_dataframe_column_names` (list): List of all column names in the
  DataFrame.
- `list_of_tunable_parameter_names_from_dataframe` (list): Columns identified as
  tunable parameters.
- `list_of_output_quantity_names_from_dataframe` (list): Columns identified as
  output quantities.
- `unique_value_columns_dictionary` (dict): Columns with single unique values
  and their values.
- `multivalued_columns_count_dictionary` (dict): Columns with multiple unique
  values and counts.
- `list_of_single_valued_column_names` (list): Column names with only one unique
  value.
- `list_of_multivalued_column_names` (list): Column names with multiple unique
  values.
- `list_of_single_valued_tunable_parameter_names` (list): Tunable parameters
  with one unique value.
- `list_of_multivalued_tunable_parameter_names` (list): Tunable parameters with
  multiple values.
- `list_of_single_valued_output_quantity_names` (list): Output quantities with
  one unique value.
- `list_of_multivalued_output_quantity_names` (list): Output quantities with
  multiple values.

## Methods

### `group_by_multivalued_tunable_parameters(filter_out_parameters_list=None)`

Groups the DataFrame by multivalued tunable parameters, optionally excluding
some parameters.

**Parameters:**
- `filter_out_parameters_list` (list, optional): List of parameter names to
  exclude from the grouping operation. Default is None (include all multivalued
  parameters).

**Returns:**
- `pandas.core.groupby.DataFrameGroupBy`: A GroupBy object created by grouping
  on the specified multivalued tunable parameters.

**Raises:**
- `TypeError`: If filter_out_parameters_list is not a list.
- `ValueError`: If filter_out_parameters_list contains invalid parameter names.

**Example:**
```python
# Group by all multivalued tunable parameters
groups = analyzer.group_by_multivalued_tunable_parameters()

# Process each group
for name, group in groups:
    print(f"Group: {name}")
    print(group)
    
# Group by all except 'parameter1'
groups = analyzer.group_by_multivalued_tunable_parameters(
    filter_out_parameters_list=['parameter1']
)
```

---

### `restrict_dataframe(condition=None, filter_func=None)`

Restricts the DataFrame to rows that satisfy given conditions.

**Parameters:**
- `condition` (str, optional): A string condition to filter rows using Pandas
  query syntax.
- `filter_func` (callable, optional): A function that takes a DataFrame and
  returns a boolean Series for more complex filtering needs.

**Returns:** None (modifies the DataFrameAnalyzer's DataFrame in-place)

**Raises:**
- `ValueError`: If neither condition nor filter_func is provided, or if filtering fails.

**Example:**
```python
# Example 1: Using a query string for simpler conditions
analyzer.restrict_dataframe("parameter1 > 1 and parameter2 == 'B'")

# Example 2: Using a filter function for more complex conditions
def complex_filter(df):
    return (
        ((df["parameter1"] >= 2) & (df["parameter2"] == "B")) |
        ((df["parameter1"] == 1) & (df["parameter2"] == "A"))
    )
analyzer.restrict_dataframe(filter_func=complex_filter)
```

---

### `restore_original_dataframe()`

Resets the working DataFrame to the original, unfiltered state.

**Returns:** None (modifies the DataFrameAnalyzer's DataFrame in-place)

**Example:**
```python
# After applying filters, reset to original
analyzer.restrict_dataframe("parameter1 > 1")
# Now we have a filtered dataframe
print(len(analyzer.dataframe))  # Shows reduced number of rows

# Reset to original
analyzer.restore_original_dataframe()
print(len(analyzer.dataframe))  # Shows original number of rows
```

## Usage Examples

### Basic Analysis Workflow

```python
import pandas as pd
from your_library import DataFrameAnalyzer

# Create a DataFrame with experiment data
df = pd.DataFrame({
    'temperature': [298, 310, 323, 298, 310, 323],
    'pressure': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    'catalyst': ['A', 'A', 'A', 'A', 'A', 'A'],
    'yield': [0.45, 0.52, 0.61, 0.47, 0.55, 0.63],
    'selectivity': [0.78, 0.75, 0.72, 0.79, 0.76, 0.73]
})

# Initialize analyzer
analyzer = DataFrameAnalyzer(df)

# Print summary of parameter types
print(f"Tunable parameters with multiple values: {analyzer.list_of_multivalued_tunable_parameter_names}")
print(f"Tunable parameters with single value: {analyzer.list_of_single_valued_tunable_parameter_names}")
print(f"Output quantities: {analyzer.list_of_output_quantity_names_from_dataframe}")

# Group by temperature only (filtering out pressure)
temp_groups = analyzer.group_by_multivalued_tunable_parameters(
    filter_out_parameters_list=['pressure']
)

# Calculate average yield for each temperature
for name, group in temp_groups:
    avg_yield = group['yield'].mean()
    print(f"Temperature {name}: Average yield = {avg_yield:.2f}")

# Filter to high-performance conditions
analyzer.restrict_dataframe("yield > 0.5 and selectivity > 0.7")
print("High performing conditions:")
print(analyzer.dataframe)

# Reset for further analysis
analyzer.restore_original_dataframe()
```

### Complex Filtering Example

```python
# Filter for optimal conditions based on multiple criteria
def optimal_conditions(df):
    high_yield = df['yield'] > 0.55
    good_selectivity = df['selectivity'] > 0.72
    energy_efficient = df['temperature'] <= 310
    
    return high_yield & good_selectivity & energy_efficient

analyzer.restrict_dataframe(filter_func=optimal_conditions)
print("Optimal processing conditions:")
print(analyzer.dataframe)
```

## Notes

- The class expects two predefined constants from a `constants` module:
  - `constants.TUNABLE_PARAMETER_NAMES_LIST`: List of column names that are
    considered tunable parameters
  - `constants.OUTPUT_QUANTITY_NAMES_LIST`: List of column names that are
    considered output quantities
