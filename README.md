# qpb_data_analysis

A Python-based toolkit for analyzing log files and data files generated
by the 'qpb' project. This repository provides scripts for processing
and visualizing data, with outputs in CSV, HDF5, and image formats. It
includes modular libraries, usage examples, and unit tests to ensure
reliability and ease of use.

## Installation

1. Clone the repository:  
```bash
git clone https://github.com/Stylianos29/qpb_data_analysis.git cd
qpb_data_analysis
```

2. Install the project as a Python package with dependencies:  
```bash
pip install -e .
```

## Project Structure

```
qpb_data_analysis/
├── core/                   # Main Python package
│   ├── library/           # Generic functions and classes (reusable utilities)
│   ├── src/               # Analysis scripts and domain-specific modules
│   └── tests/             # Unit tests and integration tests
├── bash_scripts/          # Automation scripts for data processing pipeline
├── data_files/            # Input and output data storage
│   ├── raw/              # Unprocessed data files (.txt, .dat, .err)
│   └── processed/        # Processed data (.csv, .h5)
├── output/                # Analysis results
│   ├── plots/            # Generated visualizations
│   └── tables/           # Formatted output tables
├── notebooks/             # Jupyter notebooks for interactive analysis
├── examples/              # Usage examples and demonstrations
└── docs/                  # Comprehensive documentation
```

For detailed information about each directory and the project
architecture, see
[docs/project_structure.md](docs/project_structure.md).

## Quick Start

### Main Data Analysis Pipeline

1. **Store raw data**: Place your data files in
   `data_files/raw/<experiment_name>/`
   - Supported formats: `.txt`, `.dat`, `.err`
   - Use descriptive names like `Chebyshev_several_config_varying_N`

2. **Run the pipeline**: Use BASH scripts in `bash_scripts/` to:
   - Pre-process raw data
   - Process and analyze using scripts from `core/src/`
   - Generate outputs in `data_files/processed/` and `output/`

3. **Interactive analysis**: Use Jupyter notebooks in `notebooks/` to:
   - Explore processed `.csv` and `.h5` files
   - Create custom visualizations
   - Perform additional analyses

### Example Workflow

```bash
# Process a specific data set
cd bash_scripts
./process_raw_data_files_set.sh --set_dir ../data_files/raw/my_experiment

# Analyze all processed data
./analyze_all_processed_data.sh

# Or use Jupyter notebooks for interactive exploration
jupyter notebook ../notebooks/
```

## Documentation

- **[project_structure.md](docs/project_structure.md)** - Detailed
  directory layout and purposes
- **[install.md](docs/install.md)** - Installation and setup guide
- **[usage.md](docs/usage.md)** - Comprehensive usage instructions
- **[api.md](docs/api.md)** - API documentation for modules and
  functions
