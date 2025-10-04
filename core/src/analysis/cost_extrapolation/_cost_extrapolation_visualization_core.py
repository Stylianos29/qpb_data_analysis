"""
Core visualization functions for computational cost extrapolation.

Provides plotting functions for visualizing cost extrapolation results,
including mass vs bare mass fits and cost vs bare mass fits with
extrapolation points.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import gvar as gv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from library import load_csv

from src.analysis.cost_extrapolation._cost_extrapolation_visualization_config import (
    get_figure_config,
    get_data_point_style,
    get_fit_line_style,
    get_extrapolation_marker_style,
    get_extrapolation_lines_style,
    get_legend_config,
    get_axis_labels,
    get_analysis_config,
    get_results_column_mapping,
    get_mass_data_column_mapping,
    get_cost_data_column_mapping,
    get_plot_type_subdirectories,
)


# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================


def load_and_validate_results_data(
    results_csv_path: str,
    logger,
) -> pd.DataFrame:
    """
    Load and validate cost extrapolation results CSV.

    Args:
        results_csv_path: Path to results CSV logger: Logger instance

    Returns:
        Validated DataFrame
    """
    logger.info(f"Loading results from {results_csv_path}")

    df = load_csv(results_csv_path, apply_categorical=True)

    if df.empty:
        raise ValueError("Results CSV is empty")

    logger.info(f"Loaded {len(df)} result rows")

    # Validate required columns exist
    col_mapping = get_results_column_mapping()

    required_cols = [
        col_mapping["derived_bare_mass_mean"],
        col_mapping["derived_bare_mass_error"],
        col_mapping["extrapolated_cost_mean"],
        col_mapping["extrapolated_cost_error"],
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results: {missing}")

    return df


def load_and_validate_mass_data(
    mass_csv_path: str,
    analysis_type: str,
    logger,
) -> pd.DataFrame:
    """
    Load and validate mass plateau data.

    Args:
        mass_csv_path: Path to mass data CSV analysis_type: "pcac" or
        "pion" logger: Logger instance

    Returns:
        Validated DataFrame
    """
    logger.info(f"Loading mass data from {mass_csv_path}")

    df = load_csv(mass_csv_path, apply_categorical=True)

    if df.empty:
        raise ValueError("Mass data CSV is empty")

    logger.info(f"Loaded {len(df)} mass data rows")

    # Validate required columns
    col_mapping = get_mass_data_column_mapping(analysis_type)
    required_cols = [
        col_mapping["bare_mass"],
        col_mapping["mass_mean"],
        col_mapping["mass_error"],
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in mass data: {missing}")

    return df


def load_and_validate_cost_data(
    cost_csv_path: str,
    logger,
) -> pd.DataFrame:
    """
    Load and validate cost data.

    Args:
        cost_csv_path: Path to cost data CSV logger: Logger instance

    Returns:
        Validated DataFrame
    """
    logger.info(f"Loading cost data from {cost_csv_path}")

    df = load_csv(cost_csv_path, apply_categorical=True)

    if df.empty:
        raise ValueError("Cost data CSV is empty")

    logger.info(f"Loaded {len(df)} cost data rows")

    # Validate required columns
    col_mapping = get_cost_data_column_mapping()
    required_cols = [col_mapping["bare_mass"], col_mapping["cost_mean"]]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in cost data: {missing}")

    return df


# =============================================================================
# DATA GROUPING
# =============================================================================


def group_data_for_visualization(
    results_df: pd.DataFrame,
    mass_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    analysis_type: str,
    logger,
) -> List[Dict[str, Any]]:
    """
    Group data by parameter groups for visualization.

    Args:
        results_df: Results DataFrame mass_df: Mass data DataFrame
        cost_df: Cost data DataFrame analysis_type: "pcac" or "pion"
        logger: Logger instance

    Returns:
        List of group dictionaries with all data needed for plotting
    """
    col_mapping = get_results_column_mapping()
    grouping_params = col_mapping["grouping_params"]

    # Filter to only grouping params that exist in results
    available_grouping = [p for p in grouping_params if p in results_df.columns]

    if not available_grouping:
        logger.warning("No grouping parameters found, treating as single group")
        available_grouping = []

    logger.info(
        f"Grouping by: {available_grouping if available_grouping else 'single group'}"
    )

    grouped_data = []

    if available_grouping:
        # Group by available parameters
        for group_keys, group_df in results_df.groupby(
            available_grouping, observed=False
        ):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)

            group_info = _build_group_info(
                group_keys,
                available_grouping,
                group_df.iloc[0],
                mass_df,
                cost_df,
                analysis_type,
            )

            if group_info:
                grouped_data.append(group_info)
    else:
        # Single group
        group_info = _build_group_info(
            (),
            [],
            results_df.iloc[0],
            mass_df,
            cost_df,
            analysis_type,
        )
        if group_info:
            grouped_data.append(group_info)

    logger.info(f"Created {len(grouped_data)} visualization groups")

    return grouped_data


def _build_group_info(
    group_keys: Tuple,
    grouping_params: List[str],
    results_row: pd.Series,
    mass_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    analysis_type: str,
) -> Optional[Dict[str, Any]]:
    """Build group information dictionary for visualization."""

    # Match mass and cost data to this group
    mass_group_df = _match_group_data(mass_df, group_keys, grouping_params)
    cost_group_df = _match_group_data(cost_df, group_keys, grouping_params)

    if mass_group_df.empty or cost_group_df.empty:
        return None

    # Build group identifier string
    group_id = "_".join([str(k) for k in group_keys]) if group_keys else "all_data"

    group_info = {
        "group_id": group_id,
        "group_keys": group_keys,
        "grouping_params": grouping_params,
        "results_row": results_row,
        "mass_data": mass_group_df,
        "cost_data": cost_group_df,
        "analysis_type": analysis_type,
    }

    return group_info


def _match_group_data(
    df: pd.DataFrame,
    group_keys: Tuple,
    grouping_params: List[str],
) -> pd.DataFrame:
    """Match data rows to group parameters."""

    if not grouping_params:
        return df

    mask = pd.Series([True] * len(df))

    for param, value in zip(grouping_params, group_keys):
        if param in df.columns:
            mask &= df[param] == value

    return df[mask]


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def create_cost_extrapolation_plots(
    group_info: Dict[str, Any],
    plots_directory: Path,
    file_manager,
    title_builder,
    logger,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Create both mass and cost extrapolation plots for a group.

    Args:
        group_info: Group information dictionary plots_directory:
        Directory for saving plots file_manager: PlotFileManager
        instance title_builder: PlotTitleBuilder instance logger: Logger
        instance

    Returns:
        Tuple of (mass_plot_path, cost_plot_path) or (None, None) if
        failed
    """
    group_id = group_info["group_id"]
    logger.info(f"Creating plots for group: {group_id}")

    try:
        # Create mass fit plot
        mass_plot_path = create_mass_fit_plot(
            group_info,
            plots_directory,
            file_manager,
            title_builder,
            logger,
        )

        # Create cost fit plot
        cost_plot_path = create_cost_fit_plot(
            group_info,
            plots_directory,
            file_manager,
            title_builder,
            logger,
        )

        return mass_plot_path, cost_plot_path

    except Exception as e:
        logger.error(f"Failed to create plots for group {group_id}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None, None


def create_mass_fit_plot(
    group_info: Dict[str, Any],
    plots_directory: Path,
    file_manager,
    title_builder,
    logger,
) -> Optional[Path]:
    """Create mass vs bare mass fit plot."""

    analysis_type = group_info["analysis_type"]
    analysis_cfg = get_analysis_config(analysis_type)

    # Get data
    mass_df = group_info["mass_data"]
    results_row = group_info["results_row"]

    mass_col_mapping = get_mass_data_column_mapping(analysis_type)

    x_data = mass_df[mass_col_mapping["bare_mass"]].values
    y_mean = mass_df[mass_col_mapping["mass_mean"]].values
    y_error = mass_df[mass_col_mapping["mass_error"]].values

    # Apply mass power transformation
    mass_power = analysis_cfg["mass_power"]
    y_mean_transformed = y_mean**mass_power
    y_error_transformed = mass_power * (y_mean ** (mass_power - 1)) * y_error

    # Create figure
    fig_cfg = get_figure_config()
    fig, ax = plt.subplots(figsize=fig_cfg["figure_size"], dpi=fig_cfg["dpi"])

    # Plot data points
    data_style = get_data_point_style()
    data_style["color"] = analysis_cfg["data_color"]

    ax.errorbar(
        x_data,
        y_mean_transformed,
        yerr=y_error_transformed,
        fmt=data_style["marker"],
        markersize=data_style["marker_size"],
        color=data_style["color"],
        capsize=data_style["capsize"],
        label="Data",
    )

    # Plot fit line
    col_mapping = get_results_column_mapping()
    slope = results_row[col_mapping["mass_fit"]["slope_mean"]]
    intercept = results_row[col_mapping["mass_fit"]["intercept_mean"]]
    r_squared = results_row[col_mapping["mass_fit"]["r_squared"]]

    x_fit = np.linspace(x_data.min() * 0.95, x_data.max() * 1.05, 100)
    y_fit = slope * x_fit + intercept

    fit_style = get_fit_line_style()
    fit_style["color"] = analysis_cfg["fit_color"]

    ax.plot(
        x_fit,
        y_fit,
        color=fit_style["color"],
        linewidth=fit_style["linewidth"],
        linestyle=fit_style["linestyle"],
        label=f"Linear fit (R²={r_squared:.4f})",
    )

    # Mark derived bare mass
    derived_bare_mass = results_row[col_mapping["derived_bare_mass_mean"]]
    ax.axvline(
        derived_bare_mass,
        color="green",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Derived bare mass = {derived_bare_mass:.6f}",
    )

    # Styling
    axis_labels = get_axis_labels()
    ax.set_xlabel(
        axis_labels["bare_mass"]["label"], fontsize=axis_labels["bare_mass"]["fontsize"]
    )

    y_label_key = "pion_mass_squared" if analysis_type == "pion" else "pcac_mass"
    ax.set_ylabel(
        axis_labels[y_label_key]["label"], fontsize=axis_labels[y_label_key]["fontsize"]
    )

    if fig_cfg["grid"]:
        ax.grid(True, alpha=fig_cfg["grid_alpha"], linestyle=fig_cfg["grid_linestyle"])

    legend_cfg = get_legend_config()
    ax.legend(**legend_cfg)

    title = f"{analysis_cfg['plot_title_prefix']}: Mass Fit - {group_info['group_id']}"
    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()

    plot_subdirs = get_plot_type_subdirectories()
    mass_subdir = plots_directory / plot_subdirs["mass_fit"]
    mass_subdir.mkdir(parents=True, exist_ok=True)

    filename = f"mass_fit_{group_info['group_id']}.png"
    plot_path = mass_subdir / filename
    fig.savefig(plot_path, dpi=fig_cfg["dpi"], bbox_inches="tight")
    plt.close(fig)

    logger.info(f"  Saved mass fit plot: {plot_path}")
    return plot_path


def create_cost_fit_plot(
    group_info: Dict[str, Any],
    plots_directory: Path,
    file_manager,
    title_builder,
    logger,
) -> Optional[Path]:
    """Create cost vs bare mass fit plot with extrapolation."""

    analysis_type = group_info["analysis_type"]
    analysis_cfg = get_analysis_config(analysis_type)

    # Get data
    cost_df = group_info["cost_data"]
    results_row = group_info["results_row"]

    cost_col_mapping = get_cost_data_column_mapping()

    x_data = cost_df[cost_col_mapping["bare_mass"]].values
    y_data = cost_df[cost_col_mapping["cost_mean"]].values

    # Create figure
    fig_cfg = get_figure_config()
    fig, ax = plt.subplots(figsize=fig_cfg["figure_size"], dpi=fig_cfg["dpi"])

    # Plot data points
    data_style = get_data_point_style()
    data_style["color"] = analysis_cfg["data_color"]

    ax.plot(
        x_data,
        y_data,
        marker=data_style["marker"],
        markersize=data_style["marker_size"],
        color=data_style["color"],
        linestyle="",
        label="Data",
    )

    # Plot fit line (shifted power law)
    col_mapping = get_results_column_mapping()
    a = results_row[col_mapping["cost_fit"]["param_a_mean"]]
    b = results_row[col_mapping["cost_fit"]["param_b_mean"]]
    c = results_row[col_mapping["cost_fit"]["param_c_mean"]]
    r_squared = results_row[col_mapping["cost_fit"]["r_squared"]]

    x_fit = np.linspace(x_data.min() * 0.95, x_data.max() * 1.05, 200)
    # Avoid singularity
    x_fit = x_fit[np.abs(x_fit - b) > 1e-6]
    y_fit = a / (x_fit - b) + c

    fit_style = get_fit_line_style()
    fit_style["color"] = analysis_cfg["fit_color"]

    ax.plot(
        x_fit,
        y_fit,
        color=fit_style["color"],
        linewidth=fit_style["linewidth"],
        linestyle=fit_style["linestyle"],
        label=f"Shifted power law (R²={r_squared:.4f})",
    )

    # Mark extrapolation point
    derived_bare_mass = results_row[col_mapping["derived_bare_mass_mean"]]
    extrapolated_cost = results_row[col_mapping["extrapolated_cost_mean"]]

    extrap_marker = get_extrapolation_marker_style()

    ax.plot(
        derived_bare_mass,
        extrapolated_cost,
        marker=extrap_marker["marker"],
        markersize=extrap_marker["marker_size"],
        color=extrap_marker["color"],
        markeredgecolor=extrap_marker["edge_color"],
        markeredgewidth=extrap_marker["edge_width"],
        alpha=extrap_marker["alpha"],
        zorder=extrap_marker["zorder"],
        label=f"Extrapolation: {extrapolated_cost:.1f} core-hrs",
    )

    # Add extrapolation lines
    extrap_lines = get_extrapolation_lines_style()

    ax.axvline(
        derived_bare_mass,
        color=extrap_lines["color"],
        linestyle=extrap_lines["linestyle"],
        linewidth=extrap_lines["linewidth"],
        alpha=extrap_lines["alpha"],
    )

    ax.axhline(
        extrapolated_cost,
        color=extrap_lines["color"],
        linestyle=extrap_lines["linestyle"],
        linewidth=extrap_lines["linewidth"],
        alpha=extrap_lines["alpha"],
    )

    # Styling
    axis_labels = get_axis_labels()
    ax.set_xlabel(
        axis_labels["bare_mass"]["label"], fontsize=axis_labels["bare_mass"]["fontsize"]
    )
    ax.set_ylabel(
        axis_labels["cost"]["label"], fontsize=axis_labels["cost"]["fontsize"]
    )

    if fig_cfg["grid"]:
        ax.grid(True, alpha=fig_cfg["grid_alpha"], linestyle=fig_cfg["grid_linestyle"])

    legend_cfg = get_legend_config()
    ax.legend(**legend_cfg)

    title = f"{analysis_cfg['plot_title_prefix']}: Cost Extrapolation - {group_info['group_id']}"
    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()

    plot_subdirs = get_plot_type_subdirectories()
    cost_subdir = plots_directory / plot_subdirs["cost_fit"]
    cost_subdir.mkdir(parents=True, exist_ok=True)

    filename = f"cost_fit_{group_info['group_id']}.png"
    plot_path = cost_subdir / filename
    fig.savefig(plot_path, dpi=fig_cfg["dpi"], bbox_inches="tight")
    plt.close(fig)

    logger.info(f"  Saved cost fit plot: {plot_path}")
    return plot_path
