# Data Concepts and HDF5 Structure

This document explains the fundamental concepts used throughout the
`qpb_data_analysis` library: tunable parameters, output quantities, and
how they relate to the HDF5 file hierarchy.

## Table of Contents

1. [Overview](#overview)
2. [Tunable Parameters](#tunable-parameters)
3. [Output Quantities](#output-quantities)
4. [HDF5 File Hierarchy](#hdf5-file-hierarchy)
5. [Data Organization Examples](#data-organization-examples)
6. [Working with the Data](#working-with-the-data)

## Overview

The `qpb_data_analysis` library is designed to analyze data from Lattice
QCD simulations, particularly lattice QCD calculations using overlap
fermions. The data analysis revolves around three key concepts:

- **Tunable Parameters**: Input variables that control the simulation or
  analysis
- **Output Quantities**: Measured or calculated results from the
  simulations
- **HDF5 Hierarchy**: The structured file format that organizes this
  data

## Tunable Parameters

Tunable parameters are the input variables that define the conditions
under which simulations are run or analyses are performed. These
parameters can be **single-valued** (constant across all data) or
**multi-valued** (varied to explore different conditions).

### Categories of Tunable Parameters

#### Physical Parameters
- `QCD_beta_value`: The QCD coupling parameter β
- `Bare_mass`: The bare fermion mass parameter
- `Kappa_value`: The hopping parameter κ
- `Clover_coefficient`: The clover coefficient c_SW

#### Algorithmic Parameters
- `Overlap_operator_method`: Method for overlap operator construction
  ("Chebyshev", "KL", "Bare")
- `Kernel_operator_type`: Type of kernel operator ("Wilson",
  "Brillouin")
- `Number_of_Chebyshev_terms`: Number of terms in Chebyshev
  approximation
- `KL_diagonal_order`: Order of KL expansion
- `Delta_Min`, `Delta_Max`: Eigenvalue bounds for approximations

#### Computational Parameters
- `Configuration_label`: Identifier for gauge configurations
- `MPI_geometry`: Parallel computing layout
- `Threads_per_process`: OpenMP thread count
- `CG_epsilon`, `MSCG_epsilon`: Solver convergence criteria

#### Analysis Parameters
- `APE_iterations`: APE smearing iterations
- `Rho_value`: APE smearing parameter ρ
- `Lanczos_epsilon`: Lanczos algorithm convergence

### Single-Valued vs Multi-Valued Parameters

**Single-valued parameters** remain constant throughout a dataset:
```python
# Example: All data uses the same QCD beta value
single_valued = {"QCD_beta_value": 6.0, "Clover_coefficient": 1.5}
```

**Multi-valued parameters** take different values, creating distinct
experimental conditions:
```python
# Example: Data explores different masses and Chebyshev terms
multi_valued = {
    "Bare_mass": [0.1, 0.2, 0.3],
    "Number_of_Chebyshev_terms": [50, 100, 150]
}
```

## Output Quantities

Output quantities are the results obtained from simulations or analyses.
These represent the physical or computational quantities of interest.

### Physics Output Quantities

#### Mass Measurements
- `PCAC_mass_estimate`: PCAC mass estimates
- `Pion_effective_mass_estimate`: Effective pion mass
- `Critical_bare_mass`: Critical bare mass values

#### Correlators and Observables
- `Jackknife_average_of_g5_g5_correlator`: Pseudoscalar correlators
- `Jackknife_average_of_g4g5_g5_correlator`: Axial-pseudoscalar
  correlators
- `Average_sign_squared_violation_values`: Sign function violations
- `Plaquette`: Gauge field plaquette values

#### Spectral Properties
- `Condition_number`: Matrix condition numbers
- `Minimum_eigenvalue_squared`: Smallest eigenvalues λ²_min
- `Maximum_eigenvalue_squared`: Largest eigenvalues λ²_max

### Computational Output Quantities

#### Performance Metrics
- `Average_calculation_time_per_spinor_per_config`: Wall-clock
  time
- `Average_core_hours_per_spinor_per_config`: Computational cost
- `Average_number_of_MV_multiplications_per_spinor`: Matrix-vector
  operations
- `Total_number_of_Lanczos_iterations`: Algorithm iterations

#### Quality Measures
- `Average_normality_values`: Operator normality violations
- `Average_ginsparg_wilson_relation_values`: Ginsparg-Wilson relation
  satisfaction

### Data Types

Output quantities can have different formats:

**Scalar values**: Single numbers
```python
energy = 2.45
```

**Value-error pairs**: Measurements with uncertainties
```python
mass_with_error = (0.125, 0.003)  # (value, uncertainty)
```

**Time series**: Arrays of measurements
```python
correlator_timeseries = [1.0, 0.8, 0.6, 0.4, ...]
```

## HDF5 File Hierarchy

The HDF5 file format provides a hierarchical structure that organizes
tunable parameters and output quantities in a logical, efficient manner.

### Hierarchical Organization

```
correlators_jackknife_analysis.h5
└── invert/                                      # Physical directory structure
    └── KL_several_config_varying_n/             # Second-to-deepest group (18 attributes)
        ├── Correlators_jackknife_analysis_0     # Deepest group (4 attributes)
        │   ├── Jackknife_average_of_PCAC_mass_correlator_mean_values  [float64: 48]
        │   ├── Jackknife_average_of_PCAC_mass_correlator_error_values [float64: 48]
        │   ├── Jackknife_average_of_g4g5_g5_correlator_mean_values    [float64: 48]
        │   ├── Average_core_hours_per_spinor_values_list              [float64: 3]
        │   ├── Configuration_label_values_list                        [UTF-8: 3]
        │   └── ... (all datasets have same names across deepest groups)
        ├── Correlators_jackknife_analysis_1     # Deepest group (4 attributes)
        │   ├── Jackknife_average_of_PCAC_mass_correlator_mean_values  [float64: 48]
        │   ├── Jackknife_average_of_PCAC_mass_correlator_error_values [float64: 48]
        │   ├── Average_core_hours_per_spinor_values_list              [float64: 25]
        │   └── ... (same dataset names, different array sizes/values)
        └── Correlators_jackknife_analysis_2     # Additional analysis instances...
```

### Parameter Storage Strategy

**Deepest-level group attributes**: **Multi-valued parameters** that are
unique to each analysis instance. These vary between the deepest groups
and define the specific experimental conditions for that particular
analysis.

**Second-to-deepest group attributes**: **Single-valued parameters**
that are constant across all analysis instances within that branch of
the hierarchy.

**Directory structure**: The upper levels of the hierarchy mirror the
physical directory structure where the HDF5 file is stored, preserving
the organizational context of the original data.

### Key Design Principles

1. **Systematic dataset naming**: All deepest-level groups contain
   datasets with identical names, enabling consistent access patterns
   across different analysis instances.

2. **Attribute organization by scope**:
   - Parameters that vary between analyses → deepest-level group
     attributes
   - Parameters that are constant for the study → second-to-deepest
     group attributes

3. **Directory preservation**: The HDF5 structure preserves the original
   directory organization, making it easy to trace data back to its
   source.

Example attribute organization:
```python
# Second-to-deepest group attributes (constant across all analyses)
single_valued = {
    "QCD_beta_value": 6.0,
    "Kernel_operator_type": "Wilson", 
    "Overlap_operator_method": "KL"
}

# Deepest group attributes (vary between analysis instances)
# For Correlators_jackknife_analysis_0:
multi_valued_0 = {"KL_diagonal_order": 3, "Bare_mass": 0.1}
# For Correlators_jackknife_analysis_1:
multi_valued_1 = {"KL_diagonal_order": 5, "Bare_mass": 0.1}
```

## Data Organization Examples

### Example 1: Parameter Scan

A study varying bare mass and Chebyshev terms:

```
study.h5
├── Attributes: {QCD_beta_value: 6.0, Kernel_operator_type: "Wilson"}
├── /mass_0.10/
│   ├── /mass_0.10/chebyshev_50/
│   │   ├── PCAC_mass_estimate_mean_values: [0.095, 0.098, ...]
│   │   └── PCAC_mass_estimate_error_values: [0.002, 0.003, ...]
│   └── /mass_0.10/chebyshev_100/
└── /mass_0.20/
    ├── /mass_0.20/chebyshev_50/
    └── /mass_0.20/chebyshev_100/
```

### Example 2: Multi-Configuration Analysis

Data from multiple gauge configurations:

```
configurations.h5
├── Attributes: {Bare_mass: 0.15, Number_of_Chebyshev_terms: 75}
├── /config_1001/
│   ├── g5_g5_correlator_mean_values: [1.0, 0.85, 0.72, ...]
│   └── plaquette_values: [0.588234, 0.588156, ...]
├── /config_1002/
└── /config_1003/
```

### Example 3: Gvar (Jackknife) Data

Statistical analysis with jackknife resampling:

```
jackknife_analysis.h5
├── /analysis_results/
│   ├── PCAC_mass_estimate_mean_values: [0.1245]
│   ├── PCAC_mass_estimate_error_values: [0.0032]
│   ├── pion_effective_mass_mean_values: [0.1389, 0.1391, ...]
│   └── pion_effective_mass_error_values: [0.0041, 0.0045, ...]
```

## Working with the Data

### Using the Library Classes

The library provides specialized classes for different data access
patterns:

#### HDF5Analyzer
For exploring and understanding HDF5 file structure:
```python
from library.data import HDF5Analyzer

analyzer = HDF5Analyzer('data.h5')
print(analyzer.list_of_tunable_parameter_names_from_hdf5)
print(analyzer.list_of_output_quantity_names_from_hdf5)
```

#### HDF5Plotter
For plotting data directly from HDF5 files:
```python
from library.visualization import HDF5Plotter

plotter = HDF5Plotter('data.h5', 'plots/')
plotter.set_plot_variables('PCAC_mass_estimate')  # vs time
plotter.plot(grouping_variable='Bare_mass', use_gvar=True)
```

#### DataPlotter
For plotting data from DataFrames:
```python
from library.visualization import DataPlotter

# Convert HDF5 to DataFrame first
df = analyzer.to_dataframe(['PCAC_mass_estimate'])
plotter = DataPlotter(df, 'plots/')
plotter.set_plot_variables('time_index', 'PCAC_mass_estimate')
plotter.plot(grouping_variable='Configuration_label')
```

### Data Analysis Workflow

1. **Exploration**: Use `HDF5Analyzer` to understand file structure
2. **Extraction**: Convert relevant data to DataFrames
3. **Analysis**: Use pandas operations for data manipulation
4. **Visualization**: Create plots with `DataPlotter` or `HDF5Plotter`
5. **Results**: Export findings in desired formats

### Best Practices

- **Parameter Naming**: Use descriptive, consistent parameter names
- **Units**: Document units in parameter and output quantity names
- **Metadata**: Store important metadata as attributes
- **Versioning**: Include analysis version information
- **Documentation**: Maintain clear records of parameter meanings

## Conclusion

Understanding the relationship between tunable parameters, output
quantities, and the HDF5 hierarchy is crucial for effective use of the
`qpb_data_analysis` library. This organization enables:

- **Scalable analysis** of large parameter spaces
- **Efficient data access** through hierarchical organization
- **Flexible visualization** of multi-dimensional datasets
- **Reproducible research** through comprehensive metadata storage

The library's design abstracts away much of the complexity while
providing powerful tools for exploring and analyzing Lattice QCD
simulation data.