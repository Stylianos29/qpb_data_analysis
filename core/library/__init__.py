"""
QPB Data Analysis Library
========================

A comprehensive toolkit for processing, analyzing, and visualizing qpb data

Main Components:
---------------
- Data Analysis: Tools for manipulating and analyzing pandas DataFrames
- Visualization: Plotting utilities for data visualization
- I/O: Utilities for handling various file formats and filesystem operations

For more information, see the documentation at: docs/api
"""

# Import from data module
from .data import DataFrameAnalyzer, EnhancedHDF5Analyzer, TableGenerator, load_csv

# Import from visualization module
from .visualization import DataPlotter, HDF5Plotter

# Import constants 
from .constants import ROOT, RAW_DATA_FILES_DIRECTORY, PROCESSED_DATA_FILES_DIRECTORY

# Define public API
__all__ = [
    # Data components
    "DataFrameAnalyzer",
    "EnhancedHDF5Analyzer", 
    "TableGenerator",
    "load_csv",
    
    # Visualization components
    "DataPlotter",
    "HDF5Plotter",
    
    # Important constants
    "ROOT",
    "RAW_DATA_FILES_DIRECTORY",
    "PROCESSED_DATA_FILES_DIRECTORY",
]