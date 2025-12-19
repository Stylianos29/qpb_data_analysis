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
from library.data import load_csv

from src.analysis.critical_mass_extrapolation._critical_mass_shared_config import (
    get_grouping_parameters,
)
from src.analysis.critical_mass_extrapolation._critical_mass_visualization_config import (
    get_plot_styling,
    get_plateau_mass_power,
    get_plateau_column_mapping,
    get_title_excluded_parameters,
    get_filename_base_name,
    get_filename_custom_prefix,
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


def _values_are_equal(
    val1: Any, val2: Any, rtol: float = 1e-9, atol: float = 1e-9
) -> bool:
    """
    Compare two values with proper handling for floats, strings, and
    other types.

    Args:
        - val1: First value
        - val2: Second value
        - rtol: Relative tolerance for float comparison
        - atol: Absolute tolerance for float comparison

    Returns:
        True if values are equal (within tolerance for floats)
    """
    # Handle None/NaN cases
    if pd.isna(val1) and pd.isna(val2):
        return True
    if pd.isna(val1) or pd.isna(val2):
        return False

    # Try numeric comparison first
    try:
        # Convert to float if possible
        float_val1 = float(val1)
        float_val2 = float(val2)
        # Use numpy's isclose for proper floating point comparison
        return bool(np.isclose(float_val1, float_val2, rtol=rtol, atol=atol))
    except (ValueError, TypeError):
        # Not numeric - fall back to direct comparison
        return val1 == val2


# ============================================================================
# FIT RANGE FORMATTING HELPER FUNCTION
# =============================================================================


def format_fit_range_for_legend(min_val: float, max_val: float) -> str:
    """
    Format fit range [min, max] with uniform precision for legend
    display.

    Args:
        - min_val: Minimum value of fit range
        - max_val: Maximum value of fit range

    Returns:
        Formatted string like "[0.010, 0.070]" with uniform precision
    """
    # Convert to strings and find decimal places
    min_str = f"{min_val:.10f}".rstrip("0").rstrip(".")
    max_str = f"{max_val:.10f}".rstrip("0").rstrip(".")

    min_decimals = len(min_str.split(".")[-1]) if "." in min_str else 0
    max_decimals = len(max_str.split(".")[-1]) if "." in max_str else 0

    # Use max precision for both to ensure uniform formatting
    precision = max(min_decimals, max_decimals)

    return f"[{min_val:.{precision}f}, {max_val:.{precision}f}]"


# =============================================================================
# DATA GROUPING HELPER FUNCTIONS
# =============================================================================


def _find_matching_results(
    results_df: pd.DataFrame,
    group_metadata: Dict[str, Any],
    grouping_params: List[str],
) -> Optional[pd.Series]:
    """
    Find results row matching the plateau group parameters.

    Uses explicit grouping parameters for reliable matching.

    Args:
        - results_df: DataFrame containing critical mass calculation
          results
        - group_metadata: Dictionary of group parameter values
        - grouping_params: List of parameters that define the groups

    Returns:
        Matching results row as Series, or None if no match found
    """
    if len(results_df) == 0:
        return None

    # Create mask using only the grouping parameters
    results_mask = pd.Series([True] * len(results_df))

    for param in grouping_params:
        if param in group_metadata and param in results_df.columns:
            results_mask &= results_df[param] == group_metadata[param]
        elif param in group_metadata:
            # Parameter is in group_metadata but not in results - this
            # shouldn't happen after validation, but handle gracefully
            return None

    matching_results = results_df[results_mask]

    if len(matching_results) == 0:
        return None
    elif len(matching_results) == 1:
        return matching_results.iloc[0]  # Return as Series
    else:
        # Multiple matches - this indicates an issue with grouping
        print(f"WARNING: Multiple results rows match group metadata {group_metadata}")
        print(f"         Using first match of {len(matching_results)} matches")
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
    df = load_csv(csv_path)

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
    results_df: pd.DataFrame,
    plateau_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Group plateau data and match with results for visualization.

    ROBUST IMPLEMENTATION:
        1. Validates parameter consistency between results and plateau
           CSVs
        2. Checks single-valued tunable parameters match
        3. Uses actual grouping parameters to match results rows to
           plateau groups
        4. Ensures each results row maps to exactly one plateau group

    Args:
        - results_df: DataFrame containing critical mass calculation
          results
        - plateau_df: DataFrame containing plateau mass estimates

    Returns:
        List of dictionaries, each containing group metadata, plateau
        data, and matching results data for one parameter combination

    Raises:
        - ValueError: If parameter consistency checks fail or no matches
          found
    """

    # Create analyzers for both DataFrames
    results_analyzer = DataFrameAnalyzer(results_df)
    plateau_analyzer = DataFrameAnalyzer(plateau_df)

    # === STEP 1: VALIDATE PARAMETER CONSISTENCY ===
    results_tunables = set(
        results_analyzer.list_of_tunable_parameter_names_from_dataframe
    )
    plateau_tunables = set(
        plateau_analyzer.list_of_tunable_parameter_names_from_dataframe
    )

    # Results should NOT have Bare_mass, but plateau SHOULD have it
    if "Bare_mass" in results_tunables:
        raise ValueError(
            "Results CSV should not contain 'Bare_mass' column - "
            "this is the independent variable used for fitting"
        )

    if "Bare_mass" not in plateau_tunables:
        raise ValueError("Plateau CSV must contain 'Bare_mass' column")

    # Check that all other tunable parameters match (excluding
    # Bare_mass)
    plateau_tunables_without_bare_mass = plateau_tunables - {"Bare_mass"}
    param_diff = results_tunables.symmetric_difference(
        plateau_tunables_without_bare_mass
    )

    if param_diff:
        raise ValueError(
            f"Tunable parameter mismatch between results and plateau CSVs. "
            f"Parameters only in results: {param_diff & results_tunables}. "
            f"Parameters only in plateau: {param_diff & plateau_tunables_without_bare_mass}"
        )

    # === STEP 2: VALIDATE SINGLE-VALUED PARAMETERS ===
    results_single_valued = results_analyzer.unique_value_columns_dictionary
    plateau_single_valued = plateau_analyzer.unique_value_columns_dictionary

    # Compare single-valued parameters with proper float handling
    for param in results_single_valued:
        if param in plateau_single_valued:
            if not _values_are_equal(
                results_single_valued[param], plateau_single_valued[param]
            ):
                raise ValueError(
                    f"Single-valued parameter '{param}' has different values: "
                    f"results={results_single_valued[param]} (type: {type(results_single_valued[param])}), "
                    f"plateau={plateau_single_valued[param]} (type: {type(plateau_single_valued[param])})"
                )

    # === STEP 3: GROUP PLATEAU DATA ===
    grouping_excluded_params = (
        get_grouping_parameters()
    )  # ["Bare_mass", "MPI_geometry"]

    # Filter to only parameters that are actually multivalued in plateau
    # data
    available_multivalued = plateau_analyzer.list_of_multivalued_tunable_parameter_names
    filtered_exclusions = [
        param for param in grouping_excluded_params if param in available_multivalued
    ]

    # Group plateau data
    grouped_plateau_data = plateau_analyzer.group_by_multivalued_tunable_parameters(
        filter_out_parameters_list=filtered_exclusions
    )

    # Get the actual parameters used for grouping
    actual_grouping_params = (
        plateau_analyzer.reduced_multivalued_tunable_parameter_names_list
    )

    print(f"\nGrouping plateau data:")
    print(f"  Total plateau rows: {len(plateau_df)}")
    print(f"  Multivalued parameters: {available_multivalued}")
    print(f"  Excluding from grouping: {filtered_exclusions}")
    print(f"  Actual grouping parameters: {actual_grouping_params}")
    print(f"  Number of groups created: {grouped_plateau_data.ngroups}")

    # === STEP 4: MATCH RESULTS ROWS TO PLATEAU GROUPS ===
    grouped_data = []
    unmatched_plateau_groups = 0

    for group_identifier, plateau_group in grouped_plateau_data:
        # Ensure group_identifier is a tuple
        if not isinstance(group_identifier, tuple):
            group_identifier = (group_identifier,)

        # Build group_metadata from actual grouping parameters
        group_metadata = dict(zip(actual_grouping_params, group_identifier))

        # Add single-valued parameters
        group_metadata.update(plateau_single_valued)

        # Create group_id string for logging/filenames
        group_id = "_".join(str(val) for val in group_identifier)

        # Find matching results row
        matching_results = _find_matching_results(
            results_df, group_metadata, actual_grouping_params
        )

        if matching_results is None:
            # No matching results for this plateau group
            unmatched_plateau_groups += 1
            continue

        # Create group info structure
        group_info = {
            "group_id": group_id,
            "plateau_data": plateau_group,
            "results_data": matching_results,
            "group_metadata": group_metadata,
            "grouping_params": actual_grouping_params,
        }
        grouped_data.append(group_info)

    # === STEP 5: VALIDATE MATCHING ===
    print(f"\nMatching results:")
    print(f"  Total results rows: {len(results_df)}")
    print(f"  Successful matches: {len(grouped_data)}")
    print(f"  Unmatched plateau groups: {unmatched_plateau_groups}")

    if len(grouped_data) == 0:
        raise ValueError(
            f"No matches found between results ({len(results_df)} rows) "
            f"and plateau groups ({grouped_plateau_data.ngroups} groups). "
            f"Grouping parameters used: {actual_grouping_params}"
        )

    # Check if we matched all results rows
    if len(grouped_data) < len(results_df):
        unmatched_results = len(results_df) - len(grouped_data)
        print(
            f"  WARNING: {unmatched_results} results rows did not match any plateau groups"
        )

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

    # DATA POINTS
    #############

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
    x_range, _ = calculate_plot_ranges(plateau_data, results_data, analysis_type)

    # LINEAR FIT
    #############

    # Get fit ranges
    fit_range_min = results_data.get("fit_range_min", x_data.min())
    fit_range_max = results_data.get("fit_range_max", x_data.max())

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

    # Format fit range for legend
    fit_range_str = format_fit_range_for_legend(fit_range_min, fit_range_max)

    linear_label = (
        f"$\\mathbf{{Linear\\ fit:}}$\n"
        f"  • Fitting range: $m \\in {fit_range_str}$\n"
        f"  • $a{mass_symbol} = {results_data['slope_mean']:.4f}\\,m "
        f"+ {results_data['intercept_mean']:.5f}$\n"
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

    # Draw vertical lines for linear fit range (light gray, dotted)
    ax.axvline(
        fit_range_min,
        ls=":",
        lw=1.5,
        color="gray",
        alpha=0.8,
        zorder=1,  # Behind other elements
    )
    ax.axvline(
        fit_range_max,
        ls=":",
        lw=1.5,
        color="gray",
        alpha=0.8,
        zorder=1,
        label="Linear fit range",
    )

    # QUADRATIC FIT (if available)
    #############

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

        # Get quadratic fit ranges
        quad_fit_range_min = results_data.get("quadratic_fit_range_min", x_data.min())
        quad_fit_range_max = results_data.get("quadratic_fit_range_max", x_data.max())

        # Format quadratic fit range for legend
        quad_fit_range_str = format_fit_range_for_legend(
            quad_fit_range_min, quad_fit_range_max
        )

        # Format quadratic fit label
        quadratic_label = (
            f"$\\mathbf{{Quadratic\\ fit:}}$\n"
            f"  • Fitting range: $m \\in {quad_fit_range_str}$\n"
            f"  • $a{mass_symbol} = {results_data['quadratic_a_mean']:.4f}\\,m^2 "
            f"+ {results_data['quadratic_b_mean']:.4f}\\,m "
            f"+ {results_data['quadratic_c_mean']:.5f}$\n"
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

    # REFERENCE LINES
    #############

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

    # CRITICAL MASS LINE
    #############

    # Add critical mass vertical line with error band
    critical_mass_mean = results_data["critical_mass_mean"]
    critical_mass_error = results_data["critical_mass_error"]
    critical_mass_label = (
        f"a$m^{{\\mathrm{{critical}}}}_{{\\mathrm{{bare}}}} = "
        f"{critical_mass_mean:.6f} "
        f"\\pm {critical_mass_error:.6f}$"
    )
    # Add error band first (so it appears behind the line)
    ax.axvspan(
        critical_mass_mean - critical_mass_error,
        critical_mass_mean + critical_mass_error,
        alpha=styling["critical_mass_line_alpha"],
        color=styling["critical_mass_line_color"],
        zorder=1,  # Behind other elements
    )
    ax.axvline(
        critical_mass_mean,
        color=styling["critical_mass_line_color"],
        linestyle=styling["critical_mass_line_style"],
        alpha=styling["critical_mass_line_alpha"],
        linewidth=styling["critical_mass_line_width"],
        label=critical_mass_label,
        zorder=2,  # In front of band
    )

    # SAMPLE COUNT ANNOTATIONS
    #############

    # Get sample count column name based on analysis type
    if analysis_type == "pcac":
        sample_count_col = "PCAC_n_total_samples"
    else:
        sample_count_col = "pion_n_total_samples"

    # Check if sample count column exists
    if sample_count_col in plateau_data.columns:
        sample_counts = plateau_data[sample_count_col].values

        # Annotate each data point with sample count
        for i, (x, y, count) in enumerate(zip(x_data, y_data, sample_counts)):
            ax.annotate(
                f"{int(count)}",
                xy=(x, y),
                xytext=(+15, -15),
                textcoords="offset points",
                fontsize=10,
                ha="center",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="gray",
                    linewidth=0.5,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    linewidth=0.5,
                ),
            )

        # Add dummy entry to legend with box-shaped marker
        ax.plot(
            [],
            [],
            "s",  # Square marker
            color="white",
            markeredgecolor="gray",
            markeredgewidth=0.5,
            markersize=8,
            label="Number of gauge configurations",
        )

    # Configure axes with new x-axis label
    ax.set_xlabel("$m$", fontsize=styling["axis_label_font_size"])
    ax.set_ylabel(y_label, fontsize=styling["axis_label_font_size"])

    # Add grid
    ax.grid(True, alpha=styling["grid_alpha"], linestyle=styling["grid_style"])

    # Add legend
    ax.legend(
        loc=styling["legend_location"],
        fontsize=styling["legend_font_size"],
        framealpha=styling["legend_frame_alpha"],
    )

    return fig, ax


# =============================================================================
# TOP-LEVEL INTERFACE FUNCTIONS (Top of dependency chain)
# =============================================================================


def create_critical_mass_extrapolation_plots(
    group_info: Dict[str, Any],
    title_builder: Any,
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
        title_excluded = get_title_excluded_parameters()
        title_text = title_builder.build(
            metadata_dict=group_info["group_metadata"],
            tunable_params=list(group_info["group_metadata"].keys()),
            excluded=set(title_excluded),
            leading_substring=f"Critical Mass Extrapolation:",
            wrapping_length=styling["title_width"],
        )
        ax.set_title(
            title_text, fontsize=styling["title_font_size"], pad=styling["title_pad"]
        )

        # Generate filename
        filename_builder = PlotFilenameBuilder(FILENAME_LABELS_BY_COLUMN_NAME)

        # Get the actual grouping parameters used
        grouping_params = group_info.get("grouping_params", [])
        # Get configurable base name and custom prefix
        base_name = get_filename_base_name(analysis_type)
        custom_prefix = get_filename_custom_prefix(analysis_type)

        filename = filename_builder.build(
            metadata_dict=group_info["group_metadata"],
            base_name=base_name,
            multivalued_params=grouping_params,
            custom_prefix=custom_prefix,
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
