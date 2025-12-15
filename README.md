# qpb_data_analysis

A Python-based toolkit for analyzing log files and data files generated
by the 'qpb' project. This repository provides scripts for processing
and visualizing data, with outputs in CSV, HDF5, and image formats. It
includes modular libraries, usage examples, and unit tests to ensure
reliability and ease of use.

## System Requirements

This project requires the following system tools to be installed:

- **Python** 3.8 or higher
- **jq** - Command-line JSON processor
  - Ubuntu/Debian: `sudo apt-get install jq`
  - macOS: `brew install jq`
  - Windows: Download from https://jqlang.github.io/jq/download/

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Stylianos29/qpb_data_analysis.git
cd qpb_data_analysis
```

### 2. Set up your Python environment

#### Option A: Using conda (recommended)

```bash
# Create a new conda environment
conda create -n qpb_analysis python=3.11
conda activate qpb_analysis
```

#### Option B: Using venv

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the package

Choose the installation option that fits your needs:

#### Basic installation (core functionality only)

```bash
pip install -e .
```

#### With Jupyter notebook support (for interactive analysis)

```bash
pip install -e ".[notebooks]"
```

#### Development installation (includes testing tools)

```bash
pip install -e ".[dev]"
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

## Running Tests

If you installed with the `dev` option:

```bash
pytest core/tests/
```

## Documentation

- **[project_structure.md](docs/project_structure.md)** - Detailed
  directory layout and purposes
- **[install.md](docs/install.md)** - Installation and setup guide
- **[usage.md](docs/usage.md)** - Comprehensive usage instructions
- **[api.md](docs/api.md)** - API documentation for modules and
  functions

## Contributing

This project is maintained by Stylianos Gregoriou and team. For
questions or contributions, please contact s.gregoriou@cyi.ac.cy.

## License

[Add your license information here]
