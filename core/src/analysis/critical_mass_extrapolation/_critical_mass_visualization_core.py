#!/usr/bin/env python3
"""
Core plotting functions for critical mass extrapolation visualization.

Provides functions for creating linear extrapolation plots showing
plateau mass vs bare mass with critical mass determination. Handles data
loading, grouping, and plot generation for both PCAC and pion mass
analyses.
"""

import os
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import gvar as gv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from library.data.analyzer import DataFrameAnalyzer
from library.visualization.builders.filename_builder import PlotFilenameBuilder
from library.constants.labels import FILENAME_LABELS_BY_COLUMN_NAME

from src.analysis.critical_mass_extrapolation._critical_mass_shared_config import (
    get_grouping_parameters,
    GROUPING_EXCLUDED_PARAMETERS,
)
from src.analysis.critical_mass_extrapolation._critical_mass_visualization_config import (
    get_plot_styling,
    get_plateau_mass_power,
    get_plateau_column_mapping,
    ANALYSIS_CONFIGS,
)


# =============================================================================
# LOW-LEVEL UTILITY FUNCTIONS (Bottom of dependency chain)
# =============================================================================


def create_linear_fit_line(
    x_range: Tuple[float, float], slope: float, intercept: float, n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create smooth linear fit line for plotting extrapolation.

    Args:
        - x_range: Tuple of (min, max) x values for line extent
        - slope: Linear fit slope coefficient
        - intercept: Linear fit y-intercept
        - n_points: Number of points for smooth line (default 100)

    Returns:
        Tuple of (x_fit, y_fit) arrays for plotting linear fit
    """
    x_fit = np.linspace(x_range[0], x_range[1], n_points)
    y_fit = slope * x_fit + intercept
    return x_fit, y_fit


def create_quadratic_fit_line(
    x_range: Tuple[float, float], a: float, b: float, c: float, n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create smooth quadratic fit line for plotting.

    Args:
        - x_range: Tuple of (min, max) x values for line extent
        - a: Quadratic coefficient (x² term)
        - b: Linear coefficient (x term)
        - c: Constant term
        - n_points: Number of points for smooth line (default 100)

    Returns:
        Tuple of (x_fit, y_fit) arrays for plotting quadratic fit
    """
    x_fit = np.linspace(x_range[0], x_range[1], n_points)
    y_fit = a * x_fit**2 + b * x_fit + c
    return x_fit, y_fit


def calculate_plot_ranges(
    plateau_data: pd.DataFrame, results_data: pd.Series, analysis_type: str
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calculate appropriate plot ranges including extrapolation.

    Args:
        - plateau_data: DataFrame containing bare mass and plateau
          values
        - results_data: Series containing critical mass results
        - analysis_type: Type of analysis ("pcac" or "pion")

    Returns:
        Tuple of (x_range, y_range) where each range is (min, max)
    """
    bare_mass = plateau_data["Bare_mass"].to_numpy(dtype=float)

    plateau_mass_power = get_plateau_mass_power(analysis_type)

    # Get plateau mass column (still hard-coded for functions using
    # analysis_type)
    if analysis_type == "pcac":
        plateau_mass = plateau_data["PCAC_plateau_mean"].to_numpy(dtype=float)
    else:
        plateau_mass = plateau_data["pion_plateau_mean"].to_numpy(dtype=float)

    # Apply power transformation to match calculation
    plateau_mass_transformed = plateau_mass**plateau_mass_power

    # X-range: extend beyond data to show extrapolation
    x_min = min(bare_mass.min(), results_data["critical_mass_mean"] * 1.2)
    x_max = max(bare_mass.max(), 0.0) * 1.1
    x_range = (x_min, x_max)

    # Y-range: include zero and some margin
    y_min = min(plateau_mass_transformed.min() * 1.1, -0.01)
    y_max = plateau_mass_transformed.max() * 1.1
    y_range = (y_min, y_max)

    return x_range, y_range


# =============================================================================
# DATA GROUPING HELPER FUNCTIONS
# =============================================================================


def _extract_metadata_from_group_id(
    group_id: str, plateau_group: pd.DataFrame
) -> Dict[str, Any]:
    """
    Extract lattice parameter metadata from plateau data group.

    Args:
        - group_id: String identifier for parameter group
        - plateau_group: DataFrame containing plateau data for this
          group

    Returns:
        Dictionary of single-valued lattice parameters for this group
    """
    # Get the first row to extract parameter values
    first_row = plateau_group.iloc[0]

    grouping_params = get_grouping_parameters()

    # Extract metadata for parameters that exist and are single-valued
    # in this group
    group_metadata = {}
    for param in grouping_params:
        if param in plateau_group.columns:
            unique_values = plateau_group[param].unique()
            if len(unique_values) == 1:  # Single-valued in this group
                group_metadata[param] = unique_values[0]

    return group_metadata


def _find_matching_results(
    results_df: pd.DataFrame, group_metadata: Dict[str, Any]
) -> Optional[pd.Series]:
    """
    Find results row matching the plateau data group parameters.

    Args:
        - results_df: DataFrame containing critical mass calculation
          results
        - group_metadata: Dictionary of group parameter values to match

    Returns:
        Matching results row as Series, or None if no match found
    """
    if len(results_df) == 0:
        return None

    # Create mask to find matching results
    results_mask = pd.Series([True] * len(results_df))

    for param, value in group_metadata.items():
        if param in results_df.columns:
            results_mask &= results_df[param] == value

    matching_results = results_df[results_mask]

    if len(matching_results) == 0:
        return None
    elif len(matching_results) == 1:
        return matching_results.iloc[0]  # Return as Series
    else:
        # Multiple matches - take the first one
        return matching_results.iloc[0]


# =============================================================================
# DATA LOADING AND VALIDATION FUNCTIONS
# =============================================================================


def load_and_validate_results_data(
    csv_path: str, output_column_names: Dict[str, str], logger
) -> pd.DataFrame:
    """
    Load critical mass results CSV and validate required columns.

    Args:
        - csv_path: Path to results CSV file
        - output_column_names: Dictionary from shared config
          OUTPUT_COLUMN_NAMES
        - logger: Logger instance from custom logging system

    Returns:
        DataFrame containing validated results data
    """
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        raise ValueError("Results CSV contains no data rows")

    # Separate linear and quadratic columns using "quadratic" string in
    # keys
    linear_cols = []
    quadratic_cols = []

    for key, csv_col_name in output_column_names.items():
        if "quadratic" in key:
            quadratic_cols.append(csv_col_name)
        else:
            linear_cols.append(csv_col_name)

    # Check linear columns (required) - raise error if missing
    missing_linear_cols = [col for col in linear_cols if col not in df.columns]
    if missing_linear_cols:
        raise ValueError(
            f"Results CSV missing required linear fit columns: {missing_linear_cols}"
        )

    # Check quadratic columns (optional) - only log warning if missing
    missing_quadratic_cols = [col for col in quadratic_cols if col not in df.columns]
    if missing_quadratic_cols:
        # Only log if some (but not all) quadratic columns are missing
        present_quadratic_cols = [col for col in quadratic_cols if col in df.columns]
        if present_quadratic_cols:  # Some quadratic cols present, some missing
            logger.warning(f"Quadratic fit columns missing: {missing_quadratic_cols}")
        # If no quadratic columns present, that's normal (quadratic
        # fitting disabled)

    return df


def load_and_validate_plateau_data(
    csv_path: str, plateau_column_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Load plateau mass estimates CSV and validate required columns.

    Args:
        - csv_path: Path to plateau data CSV file
        - plateau_column_mapping: Dictionary mapping standard names to
          CSV columns

    Returns:
        DataFrame containing validated plateau data
    """
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        raise ValueError("Plateau CSV contains no data rows")

    # Get column names from mapping
    bare_mass_col = plateau_column_mapping["bare_mass"]
    plateau_mean_col = plateau_column_mapping["plateau_mean"]
    plateau_error_col = plateau_column_mapping["plateau_error"]

    required_cols = [bare_mass_col, plateau_mean_col, plateau_error_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Plateau CSV missing required columns: {missing_cols}")

    return df


def group_data_for_visualization(
    results_df: pd.DataFrame, plateau_df: pd.DataFrame, analysis_type: str
) -> List[Dict[str, Any]]:
    """
    Group plateau and results data by lattice parameters for plotting.

    Uses DataFrameAnalyzer to intelligently group plateau data, then
    matches each group with corresponding critical mass calculation
    results.

    Args:
        - results_df: DataFrame containing critical mass calculation
          results
        - plateau_df: DataFrame containing plateau mass estimates
        - analysis_type: Type of analysis ("pcac" or "pion")

    Returns:
        List of dictionaries, each containing group metadata, plateau
        data, and matching results data for one parameter combination
    """

    # Use DataFrameAnalyzer on plateau data (more data points for better
    # analysis)
    analyzer = DataFrameAnalyzer(plateau_df)

    # Filter exclusion list to only include parameters that exist in the
    # list of multivalued parameters
    available_multivalued_params = analyzer.list_of_multivalued_tunable_parameter_names
    filtered_exclusions = [
        param
        for param in GROUPING_EXCLUDED_PARAMETERS
        if param in available_multivalued_params
    ]

    # Group plateau data using analyzer's intelligence
    grouped_plateau_data = analyzer.group_by_multivalued_tunable_parameters(
        filter_out_parameters_list=filtered_exclusions
    )

    # Build grouped data for visualization
    grouped_data = []

    for group_identifier, plateau_group in grouped_plateau_data:
        # Convert group_identifier to string if it's not already
        if isinstance(group_identifier, tuple):
            group_id = "_".join(str(val) for val in group_identifier)
        else:
            group_id = str(group_identifier)

        # Extract group parameters from plateau_group directly
        group_metadata = _extract_metadata_from_group_id(group_id, plateau_group)

        # Find matching results row using the same grouping logic
        matching_results = _find_matching_results(results_df, group_metadata)

        if matching_results is None:
            continue  # Skip if no matching results found

        # Create group info structure expected by plotting functions
        group_info = {
            "group_id": group_id,
            "plateau_data": plateau_group,
            "results_data": matching_results,
            "group_metadata": group_metadata,
        }
        grouped_data.append(group_info)

    return grouped_data


# =============================================================================
# CORE PLOTTING FUNCTIONS
# =============================================================================


def create_critical_mass_plot(
    group_info: Dict[str, Any],
    analysis_type: str,
    plateau_column_mapping: Dict[str, str],
) -> Tuple[Figure, Axes]:
    """
    Create critical mass extrapolation plot for one parameter group.

    Generates plot showing plateau mass vs bare mass data with error
    bars, linear fit line, critical mass annotation, and chiral limit
    reference.

    Args:
        - group_info: Dictionary containing plateau data and results for
          group
        - analysis_type: Type of analysis ("pcac" or "pion")
        - plateau_column_mapping: Dictionary mapping column names

    Returns:
        Tuple of (figure, axes) matplotlib objects
    """

    styling = get_plot_styling()
    plateau_data = group_info["plateau_data"]
    results_data = group_info["results_data"]

    # Set up figure
    fig, ax = plt.subplots(figsize=styling["figure_size"])

    # Get data using column mapping
    bare_mass_col = plateau_column_mapping["bare_mass"]
    plateau_mean_col = plateau_column_mapping["plateau_mean"]
    plateau_error_col = plateau_column_mapping["plateau_error"]

    x_data = plateau_data[bare_mass_col].values
    y_data_raw = plateau_data[plateau_mean_col].values
    y_errors_raw = plateau_data[plateau_error_col].values

    # Apply power transformation to match calculation
    plateau_mass_power = get_plateau_mass_power(analysis_type)
    y_data = y_data_raw**plateau_mass_power

    # Transform errors using error propagation: σ(y^n) = n * y^(n-1) *
    # σ(y)
    if plateau_mass_power == 1:
        y_errors = y_errors_raw
    else:
        y_errors = (
            plateau_mass_power * (y_data_raw ** (plateau_mass_power - 1)) * y_errors_raw
        )

    # Get y-axis label from config
    y_label = ANALYSIS_CONFIGS[analysis_type]["default_y_label"]

    # Plot data points with error bars
    ax.errorbar(
        x_data,
        y_data,
        yerr=y_errors,
        fmt=styling["data_marker_style"],
        color=styling["data_color"],
        markersize=styling["data_marker_size"],
        capsize=styling["error_bar_cap_size"],
    )

    # Calculate plot ranges
    x_range, y_range = calculate_plot_ranges(plateau_data, results_data, analysis_type)

    # Create linear fit using gvar for automatic error propagation
    x_fit = np.linspace(x_range[0], x_range[1], 100)
    slope_gv = gv.gvar(results_data["slope_mean"], results_data["slope_error"])
    intercept_gv = gv.gvar(
        results_data["intercept_mean"], results_data["intercept_error"]
    )
    y_fit_gv = slope_gv * x_fit + intercept_gv

    # Format linear fit label with equation and metrics
    if analysis_type == "pcac":
        mass_symbol = "m_{\\mathrm{PCAC}}"
    else:
        mass_symbol = "m^2_{\\pi}"

    linear_label = (
        f"Linear fit:\n"
        f"  • $a{mass_symbol} = {results_data['slope_mean']:.4f}\\,m + {results_data['intercept_mean']:.5f}$\n"
        f"  • χ²/dof = {results_data['chi2_reduced']:.3f}\n"
        f"  • R² = {results_data['r_squared']:.4f}\n"
        f"  • Q = {results_data['fit_quality']:.3f}"
    )

    # Plot linear fit line
    ax.plot(
        x_fit,
        gv.mean(y_fit_gv),
        color=styling["fit_line_color"],
        linewidth=styling["fit_line_width"],
        linestyle=styling["fit_line_style"],
        label=linear_label,
    )

    # Add linear fit uncertainty band (gvar automatically provides
    # errors)
    ax.fill_between(
        x_fit,
        gv.mean(y_fit_gv) - gv.sdev(y_fit_gv),
        gv.mean(y_fit_gv) + gv.sdev(y_fit_gv),
        color=styling["fit_line_color"],
        alpha=0.2,
    )

    # Check if quadratic fit data exists and plot if available
    has_quadratic = (
        "quadratic_a_mean" in results_data
        and "quadratic_b_mean" in results_data
        and "quadratic_c_mean" in results_data
    )

    if has_quadratic:
        # Create quadratic fit using gvar
        a_gv = gv.gvar(
            results_data["quadratic_a_mean"], results_data["quadratic_a_error"]
        )
        b_gv = gv.gvar(
            results_data["quadratic_b_mean"], results_data["quadratic_b_error"]
        )
        c_gv = gv.gvar(
            results_data["quadratic_c_mean"], results_data["quadratic_c_error"]
        )
        y_quad_fit_gv = a_gv * x_fit**2 + b_gv * x_fit + c_gv

        # Format quadratic fit label
        quadratic_label = (
            f"Quadratic fit:\n"
            f"  • $a{mass_symbol} = {results_data['quadratic_a_mean']:.4f}\\,m^2 + {results_data['quadratic_b_mean']:.4f}\\,m + {results_data['quadratic_c_mean']:.5f}$\n"
            f"  • χ²/dof = {results_data['quadratic_chi2_reduced']:.3f}\n"
            f"  • R² = {results_data['quadratic_r_squared']:.4f}\n"
            f"  • Q = {results_data['quadratic_fit_quality']:.3f}"
        )

        # Plot quadratic fit line (green)
        ax.plot(
            x_fit,
            gv.mean(y_quad_fit_gv),
            color="#28B463",  # Green color
            linewidth=styling["fit_line_width"],
            linestyle=styling["fit_line_style"],
            label=quadratic_label,
        )

        # Add quadratic fit uncertainty band (green)
        ax.fill_between(
            x_fit,
            gv.mean(y_quad_fit_gv) - gv.sdev(y_quad_fit_gv),
            gv.mean(y_quad_fit_gv) + gv.sdev(y_quad_fit_gv),
            color="#28B463",  # Green color
            alpha=0.2,
        )

    # Add horizontal line at y=0
    ax.axhline(
        0,
        color=styling["zero_line_color"],
        linestyle=styling["zero_line_style"],
        alpha=styling["zero_line_alpha"],
        linewidth=styling["zero_line_width"],
    )

    # Add vertical line at x=0
    ax.axvline(
        0,
        color=styling["zero_line_color"],
        linestyle=styling["zero_line_style"],
        alpha=styling["zero_line_alpha"],
        linewidth=styling["zero_line_width"],
    )

    # Add critical mass vertical line WITH label (no annotation box)
    critical_mass_label = f"$m_{{\\mathrm{{crit}}}} = {results_data['critical_mass_mean']:.6f} \\pm {results_data['critical_mass_error']:.6f}$"
    ax.axvline(
        results_data["critical_mass_mean"],
        color=styling["critical_mass_line_color"],
        linestyle=styling["critical_mass_line_style"],
        alpha=styling["critical_mass_line_alpha"],
        linewidth=styling["critical_mass_line_width"],
        label=critical_mass_label,
    )

    # Configure axes with new x-axis label
    ax.set_xlabel("$m$", fontsize=styling["axis_label_font_size"])
    ax.set_ylabel(y_label, fontsize=styling["axis_label_font_size"])
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # Add grid
    ax.grid(True, alpha=styling["grid_alpha"], linestyle=styling["grid_style"])

    # Add legend
    ax.legend(loc=styling["legend_location"], fontsize=styling["legend_font_size"])

    return fig, ax


# =============================================================================
# TOP-LEVEL INTERFACE FUNCTIONS (Top of dependency chain)
# =============================================================================


def create_critical_mass_extrapolation_plots(
    group_info: Dict[str, Any],
    title_builder: Any,
    file_manager: Any,
    plots_subdir_path: str,
    analysis_type: str,
) -> Optional[str]:
    """
    Create and save critical mass extrapolation plot for parameter
    group.

    Top-level function that creates plot, generates title and filename,
    and saves to disk with proper error handling.

    Args:
        - group_info: Dictionary containing group data and metadata
        - title_builder: PlotTitleBuilder instance for plot titles
        - file_manager: PlotFileManager instance (unused but kept for
          interface)
        - plots_subdir_path: Directory path for saving plots
        - analysis_type: Type of analysis ("pcac" or "pion")

    Returns:
        Output file path if successful, None if failed
    """
    try:
        # Get styling configuration
        styling = get_plot_styling()

        plateau_column_mapping = get_plateau_column_mapping(analysis_type)

        # Create the plot (pass column mapping, not styling)
        fig, ax = create_critical_mass_plot(
            group_info, analysis_type, plateau_column_mapping
        )

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
