"""
Visualization Module
===================

Comprehensive plotting and visualization components for QPB data
analysis.

This module provides flexible, publication-quality plotting capabilities
with support for:
    - Grouped data visualization
    - Direct HDF5 plotting
    - Customizable plot styling
    - Automated file management
    - Curve fitting and annotations

Components
----------
    - **DataPlotter**: Main plotting interface for DataFrame data
    - **HDF5Plotter**: Direct plotting from HDF5 files

The module uses a modular architecture with specialized managers:
    - FileManager: Output file handling
    - LayoutManager: Figure and axes layout
    - StyleManager: Visual styling and theming
    - DataProcessor: Data validation and transformation
    - AnnotationManager: Text and label management
"""

from .plotters import DataPlotter, HDF5Plotter

# Consider exposing managers for advanced users
from .managers import (
    PlotFileManager,
    PlotLayoutManager,
    PlotStyleManager,
    PlotDataProcessor,
    PlotAnnotationManager,
)

from .specialized import (
    CurveFitter,
    PlotInsetManager,
)

__all__ = [
    # Main plotters
    "DataPlotter",
    "HDF5Plotter",
    # Advanced components (optional)
    "PlotFileManager",
    "PlotLayoutManager",
    "PlotStyleManager",
    "PlotDataProcessor",
    "PlotAnnotationManager",
    "CurveFitter",
    "PlotInsetManager",
]
