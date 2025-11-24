# API Reference: load_csv

## Overview

The `load_csv` function provides robust CSV file loading with automatic data
type handling, categorical conversion, missing value detection, and
comprehensive error handling. It serves as an enhanced wrapper around Pandas'
`read_csv` function, specifically designed for scientific data analysis
workflows where data consistency and quality validation are critical.

## Function Definition

```python
from library.data import load_csv

dataframe = load_csv(
    input_csv_file_path,
    dtype_mapping=None,
    converters_mapping=None,
    categorical_columns=None,
    validate_required_columns=None,
    encoding='utf-8',
)
```

## Function Signature

### `load_csv(input_csv_file_path, **kwargs)`

Loads a CSV file into a Pandas DataFrame with robust error handling and optional
filtering of dtypes and converters based on the CSV file's header.

**Parameters:**

- **`input_csv_file_path`** (str or Path): Path to the input CSV file
- **`dtype_mapping`** (dict, optional): Dictionary mapping column names to data
  types. Defaults to `DTYPE_MAPPING` from constants
- **`converters_mapping`** (dict, optional): Dictionary mapping column names to
  converter functions. Defaults to `CONVERTERS_MAPPING` from constants
- **`categorical_columns`** (dict, optional): Dictionary mapping column names to
  categorical configuration. Format: `{column_name: {'categories': [list],
  'ordered': bool}}`. Defaults to Wilson/Brillouin kernel types
- **`validate_required_columns`** (set, optional): Set of column names that must
  be present in the CSV
- **`encoding`** (str, optional): File encoding. Defaults to 'utf-8'

**Returns:**
- **`pd.DataFrame`**: The loaded DataFrame with specified dtypes, converters,
  and categorical types applied

**Raises:**
- **`FileNotFoundError`**: If the CSV file doesn't exist
- **`ValueError`**: If required columns are missing or if the file is not a
  valid CSV
- **`pd.errors.EmptyDataError`**: If the CSV file is empty
- **`UnicodeDecodeError`**: If there are encoding issues

## Core Features

### 1. Intelligent Header-Based Filtering

The function automatically reads the CSV header and filters the provided
mappings to only apply configurations for columns that actually exist in the
file.

```python
# Only applies mappings for columns present in the CSV
df = load_csv('data.csv', dtype_mapping={'Column1': int, 'NonExistentCol': str})
# 'NonExistentCol' mapping is ignored without error
```

### 2. Automatic Data Type Application

Applies appropriate data types based on predefined or custom mappings:

```python
# Using default mappings from constants
df = load_csv('experiment_data.csv')

# Using custom data types
custom_types = {
    'Configuration_label': str,
    'QCD_beta_value': float,
    'Clover_coefficient': int
}
df = load_csv('data.csv', dtype_mapping=custom_types)
```

### 3. Categorical Data Handling

Automatically converts specified columns to categorical data types with
validation:

```python
# Default categorical handling (Wilson/Brillouin kernel types)
df = load_csv('qpb_data.csv')  # Kernel_operator_type becomes categorical

# Custom categorical configuration
categorical_config = {
    'Status': {
        'categories': ['Active', 'Inactive', 'Pending'],
        'ordered': False
    },
    'Priority': {
        'categories': ['Low', 'Medium', 'High'],
        'ordered': True
    }
}
df = load_csv('data.csv', categorical_columns=categorical_config)
```

### 4. Missing Value Detection and Reporting

Comprehensively detects and reports various types of missing values:

- Standard missing values (NaN, None, pd.NA)
- Empty strings
- Common placeholders ('N/A', 'NULL', 'none', etc.)

```python
# Automatic missing value detection with warnings
df = load_csv('data_with_gaps.csv')
# Outputs warnings about missing values with counts and percentages
```

### 5. Robust Error Handling

Provides detailed error messages for common issues:

```python
# File validation
try:
    df = load_csv('nonexistent.csv')
except FileNotFoundError as e:
    print(f"File error: {e}")

# Column validation
try:
    df = load_csv(
        'data.csv', validate_required_columns={'RequiredCol1', 'RequiredCol2'}
        )
except ValueError as e:
    print(f"Missing columns: {e}")
```

## Usage Examples

### Basic Usage

```python
from library.data import load_csv

# Simple loading with defaults
df = load_csv('experimental_results.csv')
print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
```

### Advanced Configuration

```python
# Complete configuration example
df = load_csv(
    input_csv_file_path='complex_dataset.csv',
    dtype_mapping={
        'sample_id': str,
        'measurement': float,
        'replicate_count': int
    },
    converters_mapping={
        'beta_value': lambda x: f"{float(x):.3f}" if x.strip() else x,
        'timestamp': pd.to_datetime
    },
    categorical_columns={
        'method': {
            'categories': ['Method_A', 'Method_B', 'Method_C'],
            'ordered': False
        },
        'quality': {
            'categories': ['Poor', 'Good', 'Excellent'],
            'ordered': True
        }
    },
    validate_required_columns={'sample_id', 'measurement'},
    encoding='utf-8',
)
```

### Disabling Categorical Conversion

```python
# Load without categorical conversion for raw data analysis
df = load_csv('raw_data.csv')
# All string columns remain as 'object' dtype
```

### Handling Different File Encodings

```python
# For files with special characters or non-UTF-8 encoding
df = load_csv('european_data.csv', encoding='latin-1')
```

### Column Validation

```python
# Ensure critical columns are present
required_cols = {'experiment_id', 'timestamp', 'result_value'}

try:
    df = load_csv('results.csv', validate_required_columns=required_cols)
    print("All required columns present")
except ValueError as e:
    print(f"Missing required columns: {e}")
```

## Integration with qpb Analysis Workflow

The function is specifically designed for qpb-generated data:

```python
# Typical qpb data loading
qpb_data = load_csv('lattice_results.csv')

# Automatic categorization of Wilson/Brillouin kernel types
print("Kernel types:", qpb_data['Kernel_operator_type'].cat.categories.tolist())

# Default converters handle scientific notation and special formatting
print("Beta values:", qpb_data['QCD_beta_value'].head())
```

## Error Handling Patterns

### File System Issues

```python
from pathlib import Path

def safe_load_csv(file_path):
    try:
        return load_csv(file_path)
    except FileNotFoundError:
        print(f"CSV file not found: {file_path}")
        return None
    except ValueError as e:
        if "Required columns missing" in str(e):
            print(f"Data validation failed: {e}")
        else:
            print(f"CSV parsing error: {e}")
        return None
    except UnicodeDecodeError:
        print("Encoding issue - try specifying encoding parameter")
        return None
```

### Converter Error Handling

```python
# Safe converter that handles missing values
def safe_float_converter(x):
    """Convert to float, handling empty strings gracefully."""
    if not x or not x.strip():
        return None
    try:
        return float(x)
    except ValueError:
        return None

# Use with the function
df = load_csv(
    'data.csv',
    converters_mapping={'numeric_col': safe_float_converter}
)
```

## Best Practices

### 1. Always Handle Exceptions

```python
def load_experiment_data(file_path):
    """Load experimental data with proper error handling."""
    try:
        df = load_csv(
            file_path,
            validate_required_columns={'experiment_id', 'timestamp'}
        )
        print(f"✓ Loaded {len(df)} experimental records")
        return df
    except Exception as e:
        print(f"✗ Failed to load {file_path}: {e}")
        return None
```

### 2. Use Path Objects

```python
from pathlib import Path

data_dir = Path("data/experiments")
for csv_file in data_dir.glob("*.csv"):
    df = load_csv(csv_file)  # Path objects work directly
    process_data(df)
```

### 3. Validate Data After Loading

```python
def validate_loaded_data(df):
    """Validate data quality after loading."""
    issues = []
    
    # Check for completely empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        issues.append(f"Empty columns: {empty_cols}")
    
    # Check data types
    unexpected_objects = df.select_dtypes(include=['object']).columns.tolist()
    expected_objects = ['Configuration_label', 'Overlap_operator_method']
    unexpected = set(unexpected_objects) - set(expected_objects)
    if unexpected:
        issues.append(f"Unexpected object columns: {list(unexpected)}")
    
    return issues

# Usage
df = load_csv('data.csv')
issues = validate_loaded_data(df)
if issues:
    print("Data quality issues:", issues)
```

### 4. Use Context-Appropriate Settings

```python
# For exploratory analysis - disable strict validation
df_explore = load_csv('messy_data.csv')

# For production analysis - enable all validations
df_production = load_csv(
    'clean_data.csv',
    validate_required_columns={'key_column'}    
)
```

## Common Issues and Solutions

### Issue: Converter Errors with Missing Values

**Problem**: Custom converters fail on empty strings or NaN values

**Solution**: Make converters robust to missing data
```python
# Bad
bad_converter = lambda x: float(x) * 2  # Fails on empty strings

# Good
good_converter = lambda x: float(x) * 2 if x and str(x).strip() else None
```

### Issue: Unexpected Categorical Values

**Problem**: Data contains values not in the expected categorical list

**Solution**: The function automatically detects this and issues warnings,
keeping data as non-categorical

### Issue: Encoding Problems

**Problem**: CSV contains special characters that cause UnicodeDecodeError

**Solution**: Try different encodings
```python
encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

for encoding in encodings_to_try:
    try:
        df = load_csv('problematic.csv', encoding=encoding)
        print(f"Success with encoding: {encoding}")
        break
    except UnicodeDecodeError:
        continue
```

## Performance Considerations

- The function reads the CSV header twice (once for column detection, once for
  full loading) but this overhead is minimal compared to the robustness benefits
- For very large files, consider using `pd.read_csv` directly if you don't need
  the additional validation features
- Categorical conversion happens after loading, so memory usage may temporarily
  increase for large datasets

## Integration with Other Library Components

The `load_csv` function works seamlessly with other library components:

```python
from library.data import load_csv, DataFrameAnalyzer

# Load and analyze in one workflow
df = load_csv('experiment_data.csv')
analyzer = DataFrameAnalyzer(df)

# Inspect the loaded data structure
print("Tunable parameters:", analyzer.list_of_multivalued_tunable_parameter_names)
print("Output quantities:", analyzer.list_of_output_quantity_names_from_dataframe)
```

This function serves as the primary entry point for data loading in the qpb
analysis library, providing the reliability and consistency needed for
scientific data analysis workflows.
