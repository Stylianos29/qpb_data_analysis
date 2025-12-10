"""
QPB Data Analysis Library
========================

A comprehensive toolkit for processing, analyzing, and visualizing
Lattice QCD data from the QPB project.

This library provides modular components for:

Data Processing
--------------
    - **DataFrameAnalyzer**: Advanced pandas DataFrame analysis with
      automatic parameter categorization
    - **HDF5Analyzer**: Specialized HDF5 file handling with gvar support
    - **TableGenerator**: Formatted table creation for reports
    - **load_csv**: Robust CSV loading with validation and type
      conversion

Visualization
------------
    - **DataPlotter**: Comprehensive plotting interface for grouped data
    - **HDF5Plotter**: Direct plotting from HDF5 files

Validation & I/O
----------------
    - File and directory validation utilities
    - Click-based CLI validators for robust command-line interfaces
    - Filesystem checking and path validation

Utilities
---------
    - **QPBLogger**: Specialized logging system for QPB workflows
    - **LoggingWrapper**: Flexible logging configuration

Constants
--------
    - Project-wide constants for paths, patterns, and configurations
    - Domain-specific parameter definitions

Examples
--------
Basic data analysis workflow:

    >>> from library import load_csv, DataFrameAnalyzer, DataPlotter
    >>> 
    >>> # Load and analyze data
    >>> df = load_csv('experiment_data.csv')
    >>> analyzer = DataFrameAnalyzer(df)
    >>> 
    >>> # Visualize results
    >>> plotter = DataPlotter(df, output_dir='plots/')
    >>> plotter.plot_grouped_data('parameter', 'measurement')

HDF5 workflow:

    >>> from library import HDF5Analyzer, HDF5Plotter
    >>> 
    >>> # Analyze HDF5 structure
    >>> h5_analyzer = HDF5Analyzer('data.h5')
    >>> print(h5_analyzer.list_of_output_quantity_names_from_hdf5)
    >>> 
    >>> # Plot directly from HDF5
    >>> h5_plotter = HDF5Plotter('data.h5', output_dir='plots/')

For detailed documentation, see: docs/api/
"""

# ============================================================================
# IMPORTS - Organized by category
# ============================================================================

# Data Processing Components
from .data import (
    DataFrameAnalyzer,
    HDF5Analyzer,
    TableGenerator,
    load_csv,
)

# Visualization Components
from .visualization import (
    DataPlotter,
    HDF5Plotter,
)

# Core Constants
from .constants import (
    ROOT,
    RAW_DATA_FILES_DIRECTORY,
    PROCESSED_DATA_FILES_DIRECTORY,
)

# Validation Utilities
from .validation import (
    # Click validators for CLI
    validate_input_directory,
    validate_output_directory,
    validate_input_file,
    validate_output_file,
)

# Filesystem validation utilities
from .validation.filesystem import (
    is_valid_directory,
    is_valid_file,
    validate_path,
)

# Logging Utilities
from .utils import (
    LoggingWrapper,
    QPBLogger,
)


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Data components
    "DataFrameAnalyzer",
    "HDF5Analyzer",
    "TableGenerator",
    "load_csv",
    # Visualization components
    "DataPlotter",
    "HDF5Plotter",
    # Core constants
    "ROOT",
    "RAW_DATA_FILES_DIRECTORY",
    "PROCESSED_DATA_FILES_DIRECTORY",
    # Validation functions
    "validate_input_directory",
    "validate_output_directory",
    "validate_input_file",
    "validate_output_file",
    "is_valid_directory",
    "is_valid_file",
    "validate_path",
    # Logging system
    "LoggingWrapper",
    "QPBLogger",
]

# ============================================================================
# VERSION INFORMATION
# ============================================================================

__version__ = "0.1.0"
__author__ = "Stylianos Gregoriou"
__email__ = "s.gregoriou@cyi.ac.cy"
