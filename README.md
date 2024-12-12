# qpb_data_analysis

A Python-based toolkit for analyzing log files and data files generated by the
'qpb' project. This repository provides scripts for processing and visualizing
data, with outputs in CSV, HDF5, and image formats. It includes modular
libraries, usage examples, and unit tests to ensure reliability and ease of use.

## Installation

1. Clone the repository:  
```
git clone https://github.com/Stylianos29/qpb_data_analysis.git cd
qpb_data_analysis
```

2. Standard Python packaging script. Use `setup.py` to install the project as a
 Python package and install Python dependencies:  
```
pip install -e .
```

## Project Structure
<!-- TODO: List the project structure -->

### Key Directories

#### `qpb_data_analysis/`
This is the main Python package and contains:
- **`library/`**: Reusable modules for tasks like file handling, calculations,
  and more.
- **`src/`**: Scripts for data processing, analysis, and plotting.
- **`unit_tests/`**: Test suite for validating functionality. Includes a
  `mock_data/` subdirectory for sample data.

#### `docs/`
Documentation for users, including setup instructions, data file organization
guidelines, and more.

#### `data_files/`
Stores input and output data:
- `raw/`: Unprocessed data.
- `processed/`: Cleaned or transformed data.

### Usage

* Data Processing: Scripts in `src/` handle raw-to-processed data transformations.
* Data Analysis: Analyze simulation outputs using scripts in `src/`.
* Plotting: Generate visualizations with plotting scripts in `src/`.
