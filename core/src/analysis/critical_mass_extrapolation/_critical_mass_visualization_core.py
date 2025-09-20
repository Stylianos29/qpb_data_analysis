#!/usr/bin/env python3
"""
Core plotting functions for critical mass extrapolation visualization.

This module provides functions for creating linear extrapolation plots
showing plateau mass vs bare mass with critical mass determination.
"""

import os
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from library.visualization.builders.filename_builder import PlotFilenameBuilder
from library.constants.labels import FILENAME_LABELS_BY_COLUMN_NAME

from src.analysis.critical_mass_extrapolation._critical_mass_visualization_config import (
    get_plot_styling,
)


# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================


def load_and_validate_results_data(csv_path, analysis_type):
    """Load critical mass results CSV and validate columns."""
    df = pd.read_csv(csv_path)

    required_cols = [
        "critical_mass_mean",
        "critical_mass_error",
        "slope_mean",
        "intercept_mean",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Results CSV missing required columns: {missing_cols}")

    return df


def load_and_validate_plateau_data(csv_path, analysis_type):
    """Load plateau mass CSV and validate columns."""
    df = pd.read_csv(csv_path)

    # Check for required columns based on analysis type
    if analysis_type == "pcac":
        mass_col = "PCAC_plateau_mean"
        error_col = "PCAC_plateau_error"
    else:  # pion
        mass_col = "pion_plateau_mean"
        error_col = "pion_plateau_error"

    required_cols = ["Bare_mass", mass_col, error_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Plateau CSV missing required columns: {missing_cols}")

    return df


def group_data_for_visualization(results_df, plateau_df, analysis_type):
    """Group results and plateau data for visualization."""
    grouped_data = []

    # Group by the same parameters used in analysis
    grouping_params = ["KL_diagonal_order", "Kernel_operator_type"]
    available_params = [p for p in grouping_params if p in plateau_df.columns]

    if not available_params:
        # Single group case
        group_info = {
            "group_id": "all_data",
            "plateau_data": plateau_df,
            "results_data": results_df.iloc[0] if len(results_df) > 0 else None,
            "group_metadata": {},
        }
        grouped_data.append(group_info)
    else:
        # Group plateau data
        for group_values, plateau_group in plateau_df.groupby(available_params):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)

            # Find matching results row
            results_mask = pd.Series([True] * len(results_df))
            for i, param in enumerate(available_params):
                if param in results_df.columns:
                    results_mask &= results_df[param] == group_values[i]

            matching_results = results_df[results_mask]
            if len(matching_results) == 0:
                continue

            # Create group info
            group_id = "_".join(
                [f"{p}{v}" for p, v in zip(available_params, group_values)]
            )
            group_metadata = dict(zip(available_params, group_values))

            group_info = {
                "group_id": group_id,
                "plateau_data": plateau_group,
                "results_data": matching_results.iloc[0],
                "group_metadata": group_metadata,
            }
            grouped_data.append(group_info)

    return grouped_data


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def create_linear_fit_line(x_range, slope, intercept, n_points=100):
    """Create smooth linear fit line for plotting."""
    x_fit = np.linspace(x_range[0], x_range[1], n_points)
    y_fit = slope * x_fit + intercept
    return x_fit, y_fit


def calculate_plot_ranges(plateau_data, results_data, analysis_type):
    """Calculate appropriate plot ranges including extrapolation."""
    bare_mass = plateau_data["Bare_mass"].values

    # Get plateau mass column
    if analysis_type == "pcac":
        plateau_mass = plateau_data["PCAC_plateau_mean"].values
    else:
        plateau_mass = plateau_data["pion_plateau_mean"].values

    # X-range: extend beyond data to show extrapolation
    x_min = min(bare_mass.min(), results_data["critical_mass_mean"] * 1.2)
    x_max = max(bare_mass.max(), 0.0) * 1.1
    x_range = (x_min, x_max)

    # Y-range: include zero and some margin
    y_min = min(plateau_mass.min() * 1.1, -0.01)
    y_max = plateau_mass.max() * 1.1
    y_range = (y_min, y_max)

    return x_range, y_range


def annotate_critical_mass(ax, critical_mass_mean, critical_mass_error, styling):
    """Add critical mass annotation to plot."""
    # Add vertical line at critical mass
    ax.axvline(
        critical_mass_mean,
        color=styling["critical_mass_line_color"],
        linestyle=styling["critical_mass_line_style"],
        alpha=styling["critical_mass_line_alpha"],
        linewidth=styling["critical_mass_line_width"],
    )

    # Add annotation
    annotation_text = (
        f"Critical mass = {critical_mass_mean:.6f} ± {critical_mass_error:.6f}"
    )
    ax.annotate(
        annotation_text,
        xy=(critical_mass_mean, 0),
        xytext=(0.05, 0.95),
        textcoords="axes fraction",
        bbox=styling["annotation_bbox"],
        fontsize=styling["annotation_font_size"],
        ha="left",
        va="top",
    )


def create_extrapolation_plot(group_info, analysis_type, styling):
    """Create single critical mass extrapolation plot."""
    plateau_data = group_info["plateau_data"]
    results_data = group_info["results_data"]

    # Set up figure
    fig, ax = plt.subplots(figsize=styling["figure_size"])

    # Get data
    x_data = plateau_data["Bare_mass"].values

    if analysis_type == "pcac":
        y_data = plateau_data["PCAC_plateau_mean"].values
        y_errors = plateau_data["PCAC_plateau_error"].values
        y_label = "PCAC Mass"
    else:
        y_data = plateau_data["pion_plateau_mean"].values
        y_errors = plateau_data["pion_plateau_error"].values
        y_label = "Pion Effective Mass"

    # Plot data points with error bars
    ax.errorbar(
        x_data,
        y_data,
        yerr=y_errors,
        fmt=styling["data_marker_style"],
        color=styling["data_color"],
        markersize=styling["data_marker_size"],
        capsize=styling["error_bar_cap_size"],
        label="Plateau data",
    )

    # Calculate and plot fit line
    x_range, y_range = calculate_plot_ranges(plateau_data, results_data, analysis_type)
    x_fit, y_fit = create_linear_fit_line(
        x_range, results_data["slope_mean"], results_data["intercept_mean"]
    )

    ax.plot(
        x_fit,
        y_fit,
        color=styling["fit_line_color"],
        linewidth=styling["fit_line_width"],
        linestyle=styling["fit_line_style"],
        label=f"Linear fit (R² = {results_data['r_squared']:.4f})",
    )

    # Add horizontal line at y=0
    ax.axhline(
        0,
        color=styling["zero_line_color"],
        linestyle=styling["zero_line_style"],
        alpha=styling["zero_line_alpha"],
        linewidth=styling["zero_line_width"],
    )

    # Add critical mass annotation
    annotate_critical_mass(
        ax,
        results_data["critical_mass_mean"],
        results_data["critical_mass_error"],
        styling,
    )

    # Configure axes
    ax.set_xlabel("Bare Mass", fontsize=styling["axis_label_font_size"])
    ax.set_ylabel(y_label, fontsize=styling["axis_label_font_size"])
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # Add grid
    ax.grid(True, alpha=styling["grid_alpha"], linestyle=styling["grid_style"])

    # Add legend
    ax.legend(loc=styling["legend_location"], fontsize=styling["legend_font_size"])

    return fig, ax


def create_critical_mass_extrapolation_plots(
    group_info, title_builder, file_manager, plots_subdir_path, analysis_type
):
    """Create critical mass extrapolation plot for a parameter group."""
    try:
        # Get styling configuration
        styling = get_plot_styling()

        # Create the plot
        fig, ax = create_extrapolation_plot(group_info, analysis_type, styling)

        # Generate title
        title_text = title_builder.build(
            metadata_dict=group_info["group_metadata"],
            tunable_params=list(group_info["group_metadata"].keys()),
            leading_substring=f"Critical Mass Extrapolation ({analysis_type.upper()})",
            wrapping_length=styling["title_width"],
        )
        ax.set_title(
            title_text, fontsize=styling["title_font_size"], pad=styling["title_pad"]
        )

        # Generate filename
        filename_builder = PlotFilenameBuilder(FILENAME_LABELS_BY_COLUMN_NAME)
        filename = filename_builder.build(
            metadata_dict=group_info["group_metadata"],
            base_name=f"critical_mass_extrapolation_{analysis_type}",
            multivalued_params=["KL_diagonal_order", "Kernel_operator_type"],
        )

        # Save plot
        output_path = os.path.join(plots_subdir_path, filename)
        fig.savefig(
            output_path,
            dpi=styling["output_dpi"],
            bbox_inches=styling["bbox_inches"],
            facecolor=styling["facecolor"],
        )
        plt.close(fig)

        return output_path

    except Exception as e:
        if "fig" in locals():
            plt.close(fig)
        raise e
