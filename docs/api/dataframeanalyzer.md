# API Reference: DataFrameAnalyzer

## Overview

The `DataFrameAnalyzer` class provides comprehensive tools for analyzing and
manipulating Pandas DataFrames with a focus on distinguishing between tunable
parameters and output quantities. It offers both read-only inspection
capabilities and data manipulation features, including filtering, grouping,
column derivation, and state management through context managers.

## Class Definition

```python
from library.data import DataFrameAnalyzer

analyzer = DataFrameAnalyzer(dataframe: pd.DataFrame)
```

## Constructor

### `DataFrameAnalyzer(dataframe)`

Creates a new DataFrameAnalyzer instance with the provided DataFrame. The
analyzer maintains both an original copy (for restoration) and a working copy
(for modifications) of the input DataFrame.

**Parameters:**
- `dataframe` (pd.DataFrame): The input DataFrame to analyze.

**Raises:**
- `TypeError`: If the input is not a Pandas DataFrame.

**Example:**
```python
import pandas as pd
from library.data import DataFrameAnalyzer

# Create a sample DataFrame
df = pd.DataFrame({
    'Kernel_operator_type': ['Wilson', 'Brillouin'],
    'MSCG_epsilon': [1e-6, 1e-5],
    'Total_calculation_time': [10.5, 12.3]
})

# Initialize the analyzer
analyzer = DataFrameAnalyzer(df)
```

## Key Attributes

After initialization, the analyzer automatically categorizes columns and
provides the following attributes:

### DataFrame References
- `original_dataframe` (pd.DataFrame): Immutable copy of the original input
  DataFrame
- `dataframe` (pd.DataFrame): Working DataFrame that can be filtered and
  manipulated

### Column Lists
- `list_of_dataframe_column_names` (list): All column names
- `list_of_tunable_parameter_names_from_dataframe` (list): Columns identified as
  tunable parameters
- `list_of_output_quantity_names_from_dataframe` (list): Columns identified as
  output quantities

### Value Analysis Dictionaries
- `unique_value_columns_dictionary` (dict): Maps single-valued columns to their
  unique values
- `multivalued_columns_count_dictionary` (dict): Maps multi-valued columns to
  their unique value counts

### Categorized Column Lists
- `list_of_single_valued_column_names` (list): Columns with only one unique
  value
- `list_of_multivalued_column_names` (list): Columns with multiple unique values
- `list_of_single_valued_tunable_parameter_names` (list): Tunable parameters
  with one unique value
- `list_of_multivalued_tunable_parameter_names` (list): Tunable parameters with
  multiple values
- `list_of_single_valued_output_quantity_names` (list): Output quantities with
  one unique value
- `list_of_multivalued_output_quantity_names` (list): Output quantities with
  multiple values

### Computed Properties
- `reduced_multivalued_tunable_parameter_names_list` (list): Multivalued
  parameters after filtering (computed property)

## Methods

### Data Inspection Methods

#### `column_unique_values(column_name)`

Returns a sorted list of unique values for the specified column.

**Parameters:**
- `column_name` (str): The name of the column to analyze

**Returns:**
- `list`: Sorted list of unique values (converted to Python native types)

**Raises:**
- `ValueError`: If the column doesn't exist in the DataFrame

**Example:**
```python
# Get unique values for a column
values = analyzer.column_unique_values('Kernel_operator_type')
print(values)  # ['Brillouin', 'Wilson']
```

---

### Data Manipulation Methods

#### `restrict_dataframe(condition=None, filter_func=None)`

Restricts the DataFrame to rows that satisfy given conditions. Supports method
chaining.

**Parameters:**
- `condition` (str, optional): String condition using Pandas query syntax
- `filter_func` (callable, optional): Function that takes a DataFrame and
  returns a boolean Series

**Returns:**
- `self`: The analyzer instance (for method chaining)

**Raises:**
- `ValueError`: If neither condition nor filter_func is provided
- `TypeError`: If types are incorrect

**Example:**
```python
# Using query syntax
analyzer.restrict_dataframe(
  "MSCG_epsilon >= 1e-6 and Kernel_operator_type == 'Wilson'"
  )

# Using a filter function
def custom_filter(df):
    return df['Total_calculation_time'] > 11.0
analyzer.restrict_dataframe(filter_func=custom_filter)

# Method chaining
analyzer.restrict_dataframe("param > 5").add_derived_column(
  "new_col", expression="param * 2"
  )
```

---

#### `add_derived_column(new_column_name, derivation_function=None, expression=None)`

Adds a new column derived from existing columns. Supports method chaining.

**Parameters:**
- `new_column_name` (str): Name for the new column
- `derivation_function` (callable, optional): Function that takes the DataFrame
  and returns a Series
- `expression` (str, optional): String expression to evaluate

**Returns:**
- `self`: The analyzer instance (for method chaining)

**Raises:**
- `ValueError`: If column exists, no derivation method provided, or operation
  fails

**Example:**
```python
# Using a function
analyzer.add_derived_column(
    'efficiency',
    derivation_function=lambda df: 100 / df['Total_calculation_time']
)

# Using an expression
analyzer.add_derived_column(
  'double_time', expression='Total_calculation_time * 2'
  )
```

---

#### `restore_original_dataframe()`

Resets the working DataFrame to the original state. Supports method chaining.

**Returns:**
- `self`: The analyzer instance (for method chaining)

**Example:**
```python
# Make changes and then restore
analyzer.restrict_dataframe("param > 5")
analyzer.add_derived_column("new_col", expression="param * 2")
analyzer.restore_original_dataframe()  # Back to original
```

---

### Grouping Methods

#### `group_by_multivalued_tunable_parameters(filter_out_parameters_list=None, verbose=False)`

Groups the DataFrame by multivalued tunable parameters, optionally excluding
some.

**Parameters:**
- `filter_out_parameters_list` (list, optional): Parameters to exclude from
  grouping
- `verbose` (bool, optional): If True, prints the grouping columns

**Returns:**
- `pandas.core.groupby.DataFrameGroupBy`: GroupBy object

**Raises:**
- `TypeError`: If filter_out_parameters_list is not a list
- `ValueError`: If invalid parameters are specified

**Example:**
```python
# Group by all multivalued parameters
groups = analyzer.group_by_multivalued_tunable_parameters()

# Exclude specific parameters
groups = analyzer.group_by_multivalued_tunable_parameters(
    filter_out_parameters_list=['MSCG_epsilon']
)

# Process groups
for name, group in groups:
    print(f"Group {name}: {len(group)} rows")
```

---

### Context Manager Support

The analyzer can be used as a context manager for temporary modifications:

```python
# Single context
with analyzer:
    analyzer.restrict_dataframe("column > 5")
    # DataFrame is modified here
# DataFrame is automatically restored here

# Nested contexts
with analyzer:
    analyzer.restrict_dataframe("param1 > 5")
    with analyzer:
        analyzer.restrict_dataframe("param2 == 'A'")
        # Both filters applied
    # Only first filter applied
# Original state restored
```

## Complete Example

```python
import pandas as pd
from library.data import DataFrameAnalyzer

# Create experimental data
df = pd.DataFrame({
    'Kernel_operator_type': ['Wilson', 'Wilson', 'Brillouin', 'Brillouin'],
    'MSCG_epsilon': [1e-6, 1e-5, 1e-6, 1e-5],
    'Configuration_label': ['001', '001', '002', '002'],
    'Total_calculation_time': [10.5, 12.3, 11.8, 13.2],
    'Efficiency': [0.85, 0.82, 0.87, 0.83]
})

# Initialize analyzer
analyzer = DataFrameAnalyzer(df)

# Inspect the data structure
print("Multivalued parameters:", analyzer.list_of_multivalued_tunable_parameter_names)
print("Output quantities:", analyzer.list_of_output_quantity_names_from_dataframe)

# Check unique values
kernel_types = analyzer.column_unique_values('Kernel_operator_type')
print(f"Kernel types: {kernel_types}")

# Add derived column
analyzer.add_derived_column(
    'time_per_epsilon',
    derivation_function=lambda df: df['Total_calculation_time'] / df['MSCG_epsilon']
)

# Group and analyze
groups = analyzer.group_by_multivalued_tunable_parameters(
    filter_out_parameters_list=['Configuration_label']
)

for (kernel, epsilon), group in groups:
    avg_time = group['Total_calculation_time'].mean()
    print(f"Kernel={kernel}, Epsilon={epsilon}: Avg time = {avg_time:.2f}")

# Use context manager for temporary analysis
with analyzer:
    analyzer.restrict_dataframe("Efficiency > 0.84")
    high_efficiency_count = len(analyzer.dataframe)
    print(f"High efficiency runs: {high_efficiency_count}")

# DataFrame is automatically restored after the context
print(f"Total runs: {len(analyzer.dataframe)}")
```

## Notes

- Column categorization as tunable parameters or output quantities is determined
  by the `TUNABLE_PARAMETER_NAMES_LIST` constant in the library
- All manipulation methods support method chaining by returning `self`
- The analyzer maintains the original DataFrame for restoration at any time
- Context manager support allows for safe temporary modifications