"""
Data Processing Module
=====================

Core data manipulation and analysis components for the QPB library.

This module provides:
    - DataFrame analysis and manipulation
    - HDF5 file handling
    - Table generation for reports
    - CSV loading with validation

Components
----------
    - **DataFrameAnalyzer**: Automatic parameter categorization and
      analysis
    - **HDF5Analyzer**: HDF5 file inspection and data extraction
    - **TableGenerator**: Report table creation
    - **load_csv**: Robust CSV file loading
"""

from .analyzer import DataFrameAnalyzer
from .hdf5_analyzer import HDF5Analyzer
from .table_generator import TableGenerator
from .processing import load_csv

__all__ = [
    "DataFrameAnalyzer",
    "HDF5Analyzer",
    "TableGenerator",
    "load_csv",
]
