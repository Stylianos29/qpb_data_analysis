#!/usr/bin/env python3
"""
Core plotting functions for correlator analysis visualization.

This module provides the fundamental plotting functions used by
correlator analysis visualization scripts. It handles the creation of
multi-sample correlator plots showing both individual jackknife samples
and their statistical averages with error bars.

The module is designed to work with the project's visualization
infrastructure (PlotLayoutManager, PlotStyleManager, etc.) and uses
configuration-driven styling through PLOT_STYLING constants.

Key Functions:
    - _create_multi_sample_plots: Creates multiple plots for a
      correlator group, batching samples according to samples_per_plot
      configuration
    - _create_single_correlator_plot: Creates individual plots showing
      sample data overlaid with jackknife averages and error bars

Features:
    - Configuration-driven styling and layout
    - Proper matplotlib resource management with automatic figure
      cleanup
    - Integration with project's visualization managers and builders
    - Support for both PCAC mass and effective mass visualizations
    - Configurable legend titles with dynamic sample range formatting
    - Z-order controlled layering for optimal visual hierarchy

Usage:
    This is a private module imported by visualization scripts such as
    visualize_correlator_analysis.py. Functions expect preprocessed data
    and configured visualization managers.

Dependencies:
    - Project visualization infrastructure (managers, builders)
    - PLOT_STYLING configuration from _correlator_visualization_config
    - Matplotlib with Agg backend for non-interactive plotting
"""

import os
from typing import Dict, List, Tuple

from matplotlib.figure import Figure
import click
import numpy as np

# Configure matplotlib to use non-interactive backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import library components
from library.visualization.managers.file_manager import PlotFileManager
from library.visualization.managers.layout_manager import PlotLayoutManager
from library.visualization.managers.style_manager import PlotStyleManager
from library.visualization.builders.title_builder import PlotTitleBuilder
from library import constants

# Import from auxiliary files
from src.analysis.correlator_calculations._correlator_visualization_config import (
    DEFAULT_FONT_SIZE,
    PLOT_STYLING,
)


def _get_time_slice_indices(
    time_index: np.ndarray, time_range_config: Dict
) -> Tuple[int, int]:
    """Convert time range configuration to slice indices."""
    t_min = time_range_config.get("min", 0)
    t_max_config = time_range_config.get("max")

    # Calculate t_max
    if t_max_config is None:
        t_max = np.max(time_index) + 1
    elif t_max_config < 0:
        t_max = np.max(time_index) + t_max_config + 1
    else:
        t_max = t_max_config + 1

    # Convert to indices
    i_min = np.searchsorted(time_index, t_min)
    i_max = np.searchsorted(time_index, t_max)

    return int(i_min), int(i_max)


def _create_multi_sample_plots(
    samples_data: np.ndarray,
    mean_data: np.ndarray,
    error_data: np.ndarray,
    config_labels: List[str],
    group_name: str,
    base_plots_dir: str,
    analysis_config: Dict,
    group_metadata: Dict,
    file_manager: PlotFileManager,
    layout_manager: PlotLayoutManager,
    style_manager: PlotStyleManager,
    title_builder: PlotTitleBuilder,
    verbose: bool,
) -> int:
    """Create multi-sample plots for correlator data."""
    n_samples, n_time_points = samples_data.shape
    samples_per_plot = analysis_config.get("samples_per_plot", 8)
    n_plots = (n_samples + samples_per_plot - 1) // samples_per_plot

    # Create time index
    time_offset = analysis_config.get("time_offset", 0)
    time_index = np.arange(n_time_points) + time_offset

    # Create group subdirectory in base plots subdirectory
    group_plots_dir = os.path.join(base_plots_dir, group_name)
    os.makedirs(group_plots_dir, exist_ok=True)

    plots_created = 0
    if verbose:
        click.echo(f"    Creating {n_plots} plots ({samples_per_plot} samples each)")

    # Apply time range restriction if configured
    time_range_config = analysis_config.get("time_range", {})
    if time_range_config and set(time_range_config.values()) != {None}:
        i_min, i_max = _get_time_slice_indices(time_index, time_range_config)
        # Slice all data arrays
        time_index = time_index[i_min:i_max]
        mean_data = mean_data[i_min:i_max]
        error_data = error_data[i_min:i_max]
        samples_data = samples_data[:, i_min:i_max]

    for plot_idx in range(n_plots):
        start_idx = plot_idx * samples_per_plot
        end_idx = min(start_idx + samples_per_plot, n_samples)

        # Extract data for this plot
        plot_samples = samples_data[start_idx:end_idx]
        plot_labels = config_labels[start_idx:end_idx]

        # Create the plot
        fig = _create_single_correlator_plot(
            time_index,
            plot_samples,
            plot_labels,
            mean_data,
            error_data,
            (start_idx + 1, end_idx),
            analysis_config,
            group_metadata,
            layout_manager,
            style_manager,
            title_builder,
        )

        # Generate filename and save using file manager
        base_name = f"correlator_samples_{start_idx+1:03d}_{end_idx:03d}"
        plot_path = file_manager.plot_path(group_plots_dir, base_name)

        styling = PLOT_STYLING.copy()
        fig.savefig(
            plot_path,
            **styling.get("output", {"dpi": 300, "bbox_inches": "tight"}),
        )
        plt.close(fig)

        plots_created += 1
        if verbose:
            click.echo(f"      Created: {os.path.basename(plot_path)}")

    return plots_created


def _create_single_correlator_plot(
    time_index: np.ndarray,
    samples_data: np.ndarray,
    sample_labels: List[str],
    mean_data: np.ndarray,
    error_data: np.ndarray,
    sample_range: Tuple[int, int],
    analysis_config: Dict,
    group_metadata: Dict,
    layout_manager: PlotLayoutManager,
    style_manager: PlotStyleManager,
    title_builder: PlotTitleBuilder,
) -> Figure:
    """Create a single correlator plot with samples and average."""

    # Create figure using layout manager
    fig, ax = layout_manager.create_figure(
        figure_size=analysis_config.get("figure_size", (12, 8))
    )

    styling = PLOT_STYLING.copy()

    # Plot jackknife average (keep prominent but balanced)
    ax.errorbar(
        time_index,
        mean_data,
        yerr=error_data,
        label=styling["average"]["legend_label"],
        color=styling["average"]["color"],
        marker=styling["average"]["marker"],
        markersize=styling["average"]["marker_size"],
        capsize=styling["average"]["capsize"],
        capthick=styling["average"]["capthick"],
        elinewidth=styling["average"]["elinewidth"],
        alpha=styling["average"]["alpha"],
        zorder=styling["average"]["zorder"],
    )

    # Generate style mapping for sample labels
    style_map = style_manager.generate_marker_color_map(sample_labels)

    # Plot individual samples with LARGER markers
    for i, (sample_data, label) in enumerate(zip(samples_data, sample_labels)):
        marker, color = style_map.get(label, ("o", f"C{i}"))
        ax.plot(
            time_index,
            sample_data,
            label=label,
            color=color,
            marker=marker,
            markersize=styling["samples"]["marker_size"],
            alpha=styling["samples"]["alpha"],
            linestyle=styling["samples"]["linestyle"],
            zorder=styling["samples"]["zorder"],
        )

    # Configure axes using constants and analysis config
    plot_config = analysis_config["plot_config"]

    # Use AXES_LABELS_BY_COLUMN_NAME for proper labels
    x_label = constants.AXES_LABELS_BY_COLUMN_NAME.get("time_index") or plot_config.get(
        "x_label"
    )
    y_label = constants.AXES_LABELS_BY_COLUMN_NAME.get(
        analysis_config["samples_dataset"]
    ) or plot_config.get("y_label")

    ax.set_xlabel(x_label, fontsize=DEFAULT_FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=DEFAULT_FONT_SIZE)
    ax.set_yscale(plot_config["y_scale"])

    # Set x-axis limits to start from zero
    ax.set_xlim(xmin=0)

    # Create title using title builder (no fallback!)
    title_metadata = group_metadata.copy()

    title = title_builder.build(
        metadata_dict=title_metadata,
        tunable_params=list(group_metadata.keys()),
        leading_substring=styling["title"]["leading_substring"],
        wrapping_length=styling["title"]["wrapping_length"],
    )
    ax.set_title(
        title, fontsize=DEFAULT_FONT_SIZE + styling["title"]["font_size_offset"]
    )

    # Format the legend title with actual values
    legend_title = styling["legend"]["title"].format(
        sample_range=sample_range,  # Pass the tuple
        total_samples=int(title_metadata["Number_of_gauge_configurations"]),
    )

    # Add LEGEND with configuration labels
    ax.legend(
        title=legend_title,
        fontsize=DEFAULT_FONT_SIZE + styling["legend"]["font_size_offset"],
        loc=styling["legend"]["location"],
    )

    # Add configurable grid
    if styling["grid"]["enabled"]:
        ax.grid(True, alpha=styling["grid"]["alpha"])

    plt.tight_layout()

    return fig
