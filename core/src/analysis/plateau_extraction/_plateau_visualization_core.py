#!/usr/bin/env python3
"""
Core plotting functions for plateau extraction visualization.

This module provides plotting functions for visualizing plateau extraction
results, including multi-panel plots showing individual samples with their
detected plateau regions.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from library.visualization.builders.title_builder import PlotTitleBuilder
from library import constants

from src.analysis.plateau_extraction._plateau_visualization_config import (
    DEFAULT_FONT_SIZE,
    PLOT_STYLING,
    get_plot_styling,
)


def _plot_single_panel(
    ax: Axes,
    time_index: np.ndarray,
    sample_data: np.ndarray,
    plateau_bounds: Tuple[int, int],
    plateau_value: float,
    plateau_error: float,
    config_label: str,
    analysis_config: Dict,
) -> None:
    """
    Plot a single panel showing time series data with plateau fit.
    
    Args:
        ax: Matplotlib axes object
        time_index: Array of time values
        sample_data: Time series data for this sample
        plateau_bounds: (start, end) indices of plateau region
        plateau_value: Extracted plateau mean value
        plateau_error: Plateau uncertainty
        config_label: Configuration label for legend
        analysis_config: Analysis-specific configuration
    """
    plot_style = get_plot_styling()
    
    # Plot time series data
    ax.errorbar(
        time_index,
        sample_data,
        yerr=None,  # Individual samples don't have error bars
        label=plot_style["legend"]["template"].format(config_label=config_label),
        marker=plot_style["time_series"]["marker"],
        markersize=plot_style["time_series"]["markersize"],
        linestyle=plot_style["time_series"]["linestyle"],
        linewidth=plot_style["time_series"]["linewidth"],
        alpha=plot_style["time_series"]["alpha"],
        color=plot_style["time_series"]["color"],
    )
    
    # Plot plateau fit line and uncertainty band
    if plateau_bounds is not None:
        plateau_start, plateau_end = plateau_bounds
        # Add time offset to plateau bounds
        plateau_start_time = plateau_start + analysis_config["time_offset"]
        plateau_end_time = plateau_end + analysis_config["time_offset"]
        
        # Create plateau x-range
        plateau_x = np.array([plateau_start_time, plateau_end_time])
        plateau_y = np.array([plateau_value, plateau_value])
        
        # Plot plateau line
        ax.plot(
            plateau_x,
            plateau_y,
            color=plot_style["plateau_fit"]["color"],
            linestyle=plot_style["plateau_fit"]["linestyle"],
            linewidth=plot_style["plateau_fit"]["linewidth"],
            alpha=plot_style["plateau_fit"]["alpha"],
        )
        
        # Add uncertainty band
        ax.fill_between(
            plateau_x,
            plateau_y - plateau_error,
            plateau_y + plateau_error,
            color=plot_style["plateau_fit"]["color"],
            alpha=plot_style["plateau_fit"]["fill_alpha"],
        )
        
        # Add text annotation with plateau value
        mid_point = (plateau_start_time + plateau_end_time) / 2
        ax.text(
            mid_point,
            plateau_value,
            f"{plateau_value:.4f}±{plateau_error:.4f}",
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=DEFAULT_FONT_SIZE - 2,
        )
    
    # Configure axes
    ax.set_xlabel(analysis_config["x_label"], fontsize=DEFAULT_FONT_SIZE)
    ax.set_ylabel(analysis_config["y_label"], fontsize=DEFAULT_FONT_SIZE)
    
    # Add grid
    if plot_style["axes"]["grid"]:
        ax.grid(True, alpha=plot_style["axes"]["grid_alpha"])
    
    # Add legend
    ax.legend(
        loc=plot_style["legend"]["location"],
        fontsize=DEFAULT_FONT_SIZE + plot_style["legend"]["fontsize_offset"],
    )


def create_multi_panel_figure(
    extraction_results: List[Dict],
    group_metadata: Dict,
    analysis_config: Dict,
    title_builder: PlotTitleBuilder,
) -> Figure:
    """
    Create a multi-panel figure showing plateau extractions.
    
    Args:
        extraction_results: List of extraction result dictionaries
        group_metadata: Metadata for the parameter group
        analysis_config: Analysis-specific configuration
        title_builder: PlotTitleBuilder instance
        
    Returns:
        Matplotlib figure with multi-panel plateau visualizations
    """
    plot_style = get_plot_styling()
    n_panels = len(extraction_results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(
        n_panels, 
        1,
        figsize=plot_style["figure"]["size"],
        sharex=plot_style["axes"]["share_x"],
    )
    
    if n_panels == 1:
        axes = [axes]  # Ensure axes is always a list
    
    # Adjust subplot spacing
    plt.subplots_adjust(hspace=plot_style["figure"]["subplot_spacing"])
    
    # Create main title
    main_title = title_builder.build(
        metadata_dict=group_metadata,
        tunable_params=list(group_metadata.keys()),
        leading_substring=analysis_config["title_prefix"],
        wrapping_length=plot_style["title"]["wrapping_length"],
    )
    
    fig.suptitle(
        main_title,
        fontsize=DEFAULT_FONT_SIZE + plot_style["title"]["main_fontsize_offset"],
        y=0.99,
    )
    
    # Plot each panel
    for ax, result in zip(axes, extraction_results):
        # Extract data from result dictionary
        sample_idx = result["sample_index"]
        config_label = result["config_label"]
        time_series = result["time_series"]
        plateau_bounds = result.get("plateau_bounds")
        plateau_value = result.get("plateau_value")
        plateau_error = result.get("plateau_error", 0.0)
        
        # Create time index with offset
        time_index = np.arange(len(time_series)) + analysis_config["time_offset"]
        
        # Plot the panel
        _plot_single_panel(
            ax,
            time_index,
            time_series,
            plateau_bounds,
            plateau_value,
            plateau_error,
            config_label,
            analysis_config,
        )
    
    plt.tight_layout()
    return fig


def split_extractions_into_figures(
    all_extractions: List[Dict],
    max_panels: int,
) -> List[List[Dict]]:
    """
    Split extraction results into batches for multiple figures.
    
    Args:
        all_extractions: All extraction results for a group
        max_panels: Maximum panels per figure
        
    Returns:
        List of extraction batches, each for one figure
    """
    batches = []
    for i in range(0, len(all_extractions), max_panels):
        batch = all_extractions[i:i + max_panels]
        batches.append(batch)
    return batches
