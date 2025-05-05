# HDF5Plotter

## Overview

`HDF5Plotter` is a specialized class for visualizing datasets from HDF5 files.
It extends `EnhancedHDF5Analyzer` with DataFrame generation capabilities that
are compatible with the `DataPlotter` class, allowing for seamless visualization
of HDF5 datasets without modifying the existing plotting infrastructure.

## Class Hierarchy

- Inherits from: `EnhancedHDF5Analyzer`
- Uses composition with: `DataPlotter`

## Constructor

```python
HDF5Plotter(hdf5_file_path, default_output_directory=None)
```

**Parameters:**
- `hdf5_file_path` (str): Path to the HDF5 file to analyze and plot
- `default_output_directory` (str, optional): Default directory where plots will
  be saved. If None, a temporary directory will be used.

## Methods

### DataFrame Creation

#### `create_dataset_dataframe`

```python
create_dataset_dataframe(dataset_name, add_time_column=True, time_offset=0, 
                      filter_func=None, include_group_path=False, flatten_arrays=True)
```

Creates a pandas DataFrame from a specific dataset across all compatible groups.

**Parameters:**
- `dataset_name` (str): Name of the dataset to extract
- `add_time_column` (bool, optional): Whether to add a time/index column.
  Defaults to True.
- `time_offset` (int, optional): Offset to apply to the time/index values.
  Defaults to 0.
- `filter_func` (callable, optional): Function to filter groups, takes group
  path and returns bool
- `include_group_path` (bool, optional): Include the full HDF5 group path as a
  column. Defaults to False.
- `flatten_arrays` (bool, optional): Whether to convert each array element to a
  separate row. If False, stores whole arrays as DataFrame values. Defaults to
  True.

**Returns:**
- `pd.DataFrame`: DataFrame containing the dataset values and associated
  parameters

**Raises:**
- `ValueError`: If the dataset doesn't exist in the HDF5 file

---

#### `create_multi_dataset_dataframe`

```python
create_multi_dataset_dataframe(dataset_names, time_aligned=True, add_time_column=True, 
                            time_offset=0, filter_func=None, include_group_path=False)
```

Creates a DataFrame containing multiple datasets, with values aligned by time
index.

**Parameters:**
- `dataset_names` (list): List of dataset names to include
- `time_aligned` (bool, optional): Whether datasets should align by time index.
  If True, assumes all datasets have same length. Defaults to True.
- `add_time_column` (bool, optional): Whether to add a time/index column.
  Defaults to True.
- `time_offset` (int, optional): Offset to apply to the time/index values.
  Defaults to 0.
- `filter_func` (callable, optional): Function to filter groups, takes group
  path and returns bool
- `include_group_path` (bool, optional): Include the full HDF5 group path as a
  column. Defaults to False.

**Returns:**
- `pd.DataFrame`: DataFrame containing all specified datasets and parameters

**Raises:**
- `ValueError`: If any dataset doesn't exist or datasets have incompatible
  shapes

---

#### `create_merged_value_error_dataframe`

```python
create_merged_value_error_dataframe(base_name, add_time_column=True, time_offset=0,
                                 filter_func=None, include_group_path=False)
```

Creates a DataFrame with value-error tuples for a given base dataset name.

**Parameters:**
- `base_name` (str): Base name of the dataset (without "_mean_values" or
  "_error_values" suffix)
- `add_time_column` (bool, optional): Whether to add a time/index column.
  Defaults to True.
- `time_offset` (int, optional): Offset to apply to the time/index values.
  Defaults to 0.
- `filter_func` (callable, optional): Function to filter groups
- `include_group_path` (bool, optional): Include the full HDF5 group path as a
  column

**Returns:**
- `pd.DataFrame`: DataFrame with merged value-error tuples and associated
  parameters

**Raises:**
- `ValueError`: If corresponding datasets can't be found or have incompatible
  shapes

---

### Plotting

#### `plot_datasets`

```python
plot_datasets(dataset_names, output_directory=None, 
           x_axis='time_index', filter_func=None, plot_kwargs=None, 
           group_by=None, exclude_from_grouping=None, merge_value_error=False)
```

Plots datasets directly from the HDF5 file using DataPlotter.

**Parameters:**
- `dataset_names` (str or list): Name(s) of dataset(s) to plot
- `output_directory` (str, optional): Directory where plots will be saved
- `x_axis` (str, optional): Column to use as x-axis. Defaults to 'time_index'.
- `filter_func` (callable, optional): Function to filter groups
- `plot_kwargs` (dict, optional): Additional keyword arguments for
  DataPlotter.plot()
- `group_by` (str, optional): Column to use for grouping in combined plots
- `exclude_from_grouping` (list, optional): Parameters to exclude from grouping
- `merge_value_error` (bool, optional): If True, try to merge value/error
  datasets. Default is False.

**Returns:**
- `DataPlotter`: The configured DataPlotter instance

**Raises:**
- `ValueError`: If datasets don't exist or are incompatible

---

### Private Methods

#### `_merge_value_error_datasets`

```python
_merge_value_error_datasets(base_name, group_path=None)
```

Private method to merge corresponding mean and error datasets into value-error
tuples.

**Parameters:**
- `base_name` (str): Base name of the dataset without the "_mean_values" or
  "_error_values" suffix
- `group_path` (str, optional): If provided, only look for datasets in this
  specific group path

**Returns:**
- `dict`: A dictionary mapping group paths to numpy arrays of tuples (value,
  error)
- `str`: The cleaned base name

**Raises:**
- `ValueError`: If corresponding mean/error datasets can't be found or have
  mismatched shapes

## Usage Examples

### Basic Dataset Plotting

```python
# Initialize the plotter
plotter = HDF5Plotter('my_data.h5', 'output_plots')

# Plot a single dataset
plotter.plot_datasets("Jackknife_average_of_g5_g5_correlator_mean_values")
```

### Plotting with Error Bars

```python
# Plot a dataset with its corresponding error values
plotter.plot_datasets(
    "Jackknife_average_of_g5_g5_correlator",
    merge_value_error=True,
    plot_kwargs={'yaxis_log_scale': True}
)
```

### Creating a Custom DataFrame

```python
# Create a DataFrame for custom analysis
df = plotter.create_dataset_dataframe("Jackknife_average_of_g5_g5_correlator_mean_values")

# Create a DataFrame with error tuples
df_with_errors = plotter.create_merged_value_error_dataframe("Jackknife_average_of_g5_g5_correlator")
```

### Grouped Plotting with Custom Options

```python
plotter.plot_datasets(
    "Jackknife_average_of_g5_g5_correlator_mean_values",
    group_by="Kernel_operator_type",
    exclude_from_grouping=["Configuration_label"],
    plot_kwargs={
        'figure_size': (10, 7),
        'yaxis_log_scale': True,
        'legend_location': 'best'
    }
)
```

### Multiple Dataset Analysis

```python
# Create a DataFrame with multiple related datasets
datasets = [
    "Jackknife_average_of_g5_g5_correlator_mean_values",
    "Jackknife_average_of_g4g5_g5_correlator_mean_values"
]
multi_df = plotter.create_multi_dataset_dataframe(datasets)

# Use with DataPlotter directly for advanced customization
from custom_library import DataPlotter
custom_plotter = DataPlotter(multi_df, 'output_plots')
custom_plotter.set_plot_variables('time_index', datasets[0])
custom_plotter.plot(grouping_variable="Kernel_operator_type")
```

## Notes

1. The class intelligently handles dataset naming patterns:
   - It automatically detects and manages dataset suffixes (`_mean_values`,
     `_error_values`)
   - For error bar plotting, it merges corresponding mean and error datasets
     into tuple format

2. All plotting methods ultimately use the `DataPlotter` class, leveraging its
   existing functionality for:
   - Grouped plots
   - Customized formatting
   - Publication-quality output

3. The created DataFrames follow a consistent structure:
   - Time/index as the first column (when applicable)
   - Tunable parameters (both single and multi-valued) as additional columns
   - Dataset values as the final columns

4. Always remember to close the HDF5 file when done:
   ```python
   plotter.close()
   ```
