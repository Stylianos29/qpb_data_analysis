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
    get_title_styling,
    get_analysis_config,
    get_results_column_mapping,
    get_mass_data_column_mapping,
    get_cost_data_column_mapping,
    get_plot_type_subdirectories,
)


# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================


def _average_cost_data_for_group(
    cost_df: pd.DataFrame, grouping_params: List[str]
) -> pd.DataFrame:
    """
    Average cost data across configurations for each (bare_mass, group)
    combination.

    Args:
        cost_df: Raw cost DataFrame grouping_params: List of grouping
        parameter names

    Returns:
        DataFrame with averaged costs and uncertainties
    """
    # Group by bare mass + grouping parameters, excluding
    # Configuration_label
    group_cols = ["Bare_mass"] + grouping_params

    # Filter to only existing columns
    group_cols = [col for col in group_cols if col in cost_df.columns]

    if not group_cols:
        return cost_df

    # Group and aggregate
    averaged_data: List[Dict[str, Any]] = []  # Explicit type annotation

    for group_keys, group_data in cost_df.groupby(group_cols, observed=False):
        # Extract cost values for this group
        costs = group_data["Average_core_hours_per_spinor"].to_numpy()

        # Calculate mean and standard error
        cost_mean = float(np.mean(costs))  # Explicit cast to float
        cost_error = (
            float(np.std(costs, ddof=1) / np.sqrt(len(costs)))
            if len(costs) > 1
            else 0.0
        )

        # Build result row - explicitly typed as Dict[str, Any]
        result: Dict[str, Any] = {}

        if isinstance(group_keys, tuple):
            result.update(dict(zip(group_cols, group_keys)))
        else:
            result[group_cols[0]] = group_keys

        result["cost_mean"] = cost_mean
        result["cost_error"] = cost_error
        result["n_configurations"] = int(len(costs))  # Explicit cast to int

        averaged_data.append(result)

    return pd.DataFrame(averaged_data)


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
    Group data for visualization by matching results with mass and cost
    data.

    Uses DataFrameAnalyzer to intelligently detect grouping parameters
    and extracts metadata for title generation.
    """
    from library import DataFrameAnalyzer

    # Analyze results DataFrame
    analyzer = DataFrameAnalyzer(results_df)
    grouping_params = analyzer.reduced_multivalued_tunable_parameter_names_list
    single_valued_params = analyzer.list_of_single_valued_tunable_parameter_names

    # Combine for complete title
    all_tunable_params = grouping_params + single_valued_params

    logger.info(f"Grouping parameters (multivalued): {grouping_params}")
    logger.info(f"Single-valued parameters: {single_valued_params}")
    logger.info(f"Processing {len(results_df)} result groups")

    grouped_data = []

    for idx, results_row in results_df.iterrows():
        # Extract group metadata - multivalued parameters
        group_metadata = {
            param: results_row[param]
            for param in grouping_params
            if param in results_row.index
        }

        # Add single-valued parameters to metadata
        group_metadata.update(analyzer.unique_value_columns_dictionary)

        # Build group identifier for logging/filenames (only
        # multivalued)
        group_id = "_".join([str(results_row[param]) for param in grouping_params])

        # Get group keys as tuple for filtering
        group_keys = tuple(results_row[param] for param in grouping_params)

        # Match mass and cost data for this group
        mass_data = _filter_data_for_group(mass_df, grouping_params, group_keys)
        # cost_data = _filter_data_for_group(cost_df, grouping_params,
        # group_keys)

        if mass_data.empty:
            logger.warning(f"Skipping group {group_id} - no matching mass data")
            continue

        # if cost_data.empty: logger.warning(f"Skipping group {group_id}
        #     - no matching cost data") continue

        # logger.debug( f"Group {group_id}: {len(mass_data)} mass
        #     points, " f"{len(cost_data)} cost points" )

        # grouped_data.append( { "group_id": group_id, "group_metadata":
        #     group_metadata, "grouping_params": grouping_params,
        #         "all_tunable_params": all_tunable_params,  # NEW: All
        #         params for title "analysis_type": analysis_type,
        #         "results_row": results_row, "mass_data": mass_data,
        #         "cost_data": cost_data, } )

        # Match cost data for this group
        cost_data = _filter_data_for_group(cost_df, grouping_params, group_keys)

        if cost_data.empty:
            logger.warning(f"Skipping group {group_id} - no matching cost data")
            continue

        # Average cost data across configurations
        cost_data_averaged = _average_cost_data_for_group(cost_data, grouping_params)

        if cost_data_averaged.empty:
            logger.warning(f"Skipping group {group_id} - cost averaging failed")
            continue

        logger.debug(
            f"Group {group_id}: {len(mass_data)} mass points, "
            f"{len(cost_data_averaged)} averaged cost points"
        )

        grouped_data.append(
            {
                "group_id": group_id,
                "group_metadata": group_metadata,
                "grouping_params": grouping_params,
                "all_tunable_params": all_tunable_params,
                "analysis_type": analysis_type,
                "results_row": results_row,
                "mass_data": mass_data,
                "cost_data": cost_data_averaged,  # Use averaged data
            }
        )

    logger.info(f"Successfully grouped {len(grouped_data)} parameter combinations")

    return grouped_data


def _filter_data_for_group(
    df: pd.DataFrame,
    grouping_params: List[str],
    group_keys: Tuple,
) -> pd.DataFrame:
    """
    Filter DataFrame to rows matching specific group parameter values.

    Args:
        df: DataFrame to filter grouping_params: List of parameter names
        used for grouping group_keys: Tuple of values corresponding to
        grouping_params

    Returns:
        Filtered DataFrame
    """
    if not grouping_params:
        return df

    mask = pd.Series([True] * len(df))

    for param, value in zip(grouping_params, group_keys):
        if param in df.columns:
            mask &= df[param] == value

    return df[mask]


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
    """
    Create mass vs bare mass fit plot.

    Shows linear fit of plateau mass vs bare mass with: - Data points
    with error bars - Linear fit line with detailed parameters - Derived
    bare mass vertical marker - Reference plateau mass horizontal line -
    Sample count annotations - Fitting range markers - Reference axes at
    origin
    """

    analysis_type = group_info["analysis_type"]
    analysis_cfg = get_analysis_config(analysis_type)

    # Get data
    mass_df = group_info["mass_data"]
    results_row = group_info["results_row"]

    # Get column mappings
    col_mapping = get_results_column_mapping()
    mass_col_mapping = get_mass_data_column_mapping(analysis_type=analysis_type)

    # Determine mass columns and power based on analysis type
    mass_mean_col = analysis_cfg["mass_column_mean"]
    mass_error_col = analysis_cfg["mass_column_error"]
    mass_power = analysis_cfg["mass_power"]

    # Extract data
    x_data = mass_df[mass_col_mapping["bare_mass"]].values
    y_mean = mass_df[mass_mean_col].values
    y_error = mass_df[mass_error_col].values

    # Apply power transformation
    y_data = y_mean**mass_power
    y_data_error = mass_power * (y_mean ** (mass_power - 1)) * y_error

    # Create figure
    fig_cfg = get_figure_config()
    fig, ax = plt.subplots(figsize=fig_cfg["figure_size"], dpi=fig_cfg["dpi"])

    # Plot data points with error bars
    data_style = get_data_point_style()
    data_style["color"] = analysis_cfg["data_color"]

    ax.errorbar(
        x_data,
        y_data,
        yerr=y_data_error,
        fmt=data_style["marker"],
        markersize=data_style["marker_size"],
        color=data_style["color"],
        capsize=data_style["capsize"],
        capthick=data_style["capthick"],
        elinewidth=data_style["error_linewidth"],
    )

    # Extract fit parameters
    slope = results_row[col_mapping["mass_fit"]["slope_mean"]]
    intercept = results_row[col_mapping["mass_fit"]["intercept_mean"]]
    r_squared = results_row[col_mapping["mass_fit"]["r_squared"]]
    chi2_reduced = results_row[col_mapping["mass_fit"]["chi2_reduced"]]
    q_value = results_row[col_mapping["mass_fit"]["q_value"]]

    # Create fit line extending to y=0 (and slightly below)
    from src.analysis.cost_extrapolation._cost_extrapolation_visualization_config import (
        get_fit_line_extension,
    )

    fit_extension = get_fit_line_extension()

    # Calculate x value where fit line reaches target y value
    y_min_target = fit_extension["mass_fit_y_min"]
    if slope != 0:
        x_fit_min = (y_min_target - intercept) / slope
    else:
        x_fit_min = x_data.min() * 0.95  # Fallback if slope is zero

    # Extend right side beyond data
    x_fit_max = x_data.max() * fit_extension["mass_fit_x_max_factor"]

    # Create fit line
    x_fit = np.linspace(x_fit_min, x_fit_max, 200)
    y_fit = slope * x_fit + intercept

    # Determine y-label for fit equation based on analysis type
    if analysis_type == "pion":
        y_label_fit = "a^2m^2_{\\pi}"
    else:
        y_label_fit = "am_{\\mathrm{PCAC}}"

    # Create detailed fit label
    fit_label = (
        f"$\\mathbf{{Linear\\ fit:}}$\n"
        f"  • Fitting range: $m$ ∈ [{x_data.min():.2f}, {x_data.max():.2f}]\n"
        f"  • ${y_label_fit}$ = {slope:.4f}$m$ + {intercept:.5f}\n"
        f"  • χ²/dof = {chi2_reduced:.3f}\n"
        f"  • R² = {r_squared:.4f}\n"
        f"  • Q = {q_value:.3f}"
    )

    # Plot fit line
    fit_style = get_fit_line_style()
    fit_style["color"] = analysis_cfg["fit_color"]

    ax.plot(
        x_fit,
        y_fit,
        color=fit_style["color"],
        linewidth=fit_style["linewidth"],
        linestyle=fit_style["linestyle"],
        label=fit_label,
    )

    # Mark derived bare mass with uncertainty band (like critical mass
    # does)
    derived_bare_mass_mean = results_row[col_mapping["derived_bare_mass_mean"]]
    derived_bare_mass_error = results_row[col_mapping["derived_bare_mass_error"]]

    extrap_marker = get_extrapolation_marker_style()

    # Add error band first (behind the line)
    ax.axvspan(
        derived_bare_mass_mean - derived_bare_mass_error,
        derived_bare_mass_mean + derived_bare_mass_error,
        alpha=0.2,
        color=extrap_marker["color"],
        zorder=1,
    )

    # Add vertical line on top
    ax.axvline(
        derived_bare_mass_mean,
        color=extrap_marker["color"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Derived bare mass: {derived_bare_mass_mean:.6f} ± {derived_bare_mass_error:.6f}",
        zorder=2,
    )

    # Add reference plateau mass (horizontal line)
    _add_reference_plateau_mass_line(ax, analysis_type)

    # Add sample count annotations
    _add_sample_count_annotations(ax, x_data, y_data, mass_df, analysis_type)

    # Add fitting range markers
    _add_fitting_range_markers(ax, x_data.min(), x_data.max())

    # Add reference axes at origin
    _add_reference_axes(ax)

    # Configure axes
    axis_labels = get_axis_labels()
    ax.set_xlabel(
        axis_labels["bare_mass"]["label"],
        fontsize=axis_labels["bare_mass"]["fontsize"],
    )

    y_label_key = "pion_mass_squared" if analysis_type == "pion" else "pcac_mass"
    ax.set_ylabel(
        axis_labels[y_label_key]["label"],
        fontsize=axis_labels[y_label_key]["fontsize"],
    )

    if fig_cfg["grid"]:
        ax.grid(True, alpha=fig_cfg["grid_alpha"], linestyle=fig_cfg["grid_linestyle"])

    # Add legend
    legend_cfg = get_legend_config(plot_type="mass_fit")
    ax.legend(**legend_cfg)

    # Generate title using PlotTitleBuilder
    title_style = get_title_styling()
    title = title_builder.build(
        metadata_dict=group_info["group_metadata"],
        tunable_params=group_info["all_tunable_params"],
        leading_substring="Bare Mass Extrapolation at Fixed Plateau Mass:",
        wrapping_length=80,
    )
    ax.set_title(title, fontsize=title_style["fontsize"], pad=title_style["pad"])

    plt.tight_layout()

    # Save to sub-subdirectory
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
    """
    Create cost vs bare mass fit plot with extrapolation.

    Shows shifted power law fit of computational cost vs bare mass with:
        - Data points - Shifted power law fit line with detailed
          parameters
        - Extrapolation point at derived bare mass - Extrapolation guide
          lines
        - Sample count annotations (from mass data) - Fitting range
          markers
        - Reference axes at origin
    """

    analysis_type = group_info["analysis_type"]
    analysis_cfg = get_analysis_config(analysis_type)

    # Get data
    cost_df = group_info["cost_data"]
    mass_df = group_info["mass_data"]  # For sample count annotations
    results_row = group_info["results_row"]

    # cost_col_mapping = get_cost_data_column_mapping()

    # x_data = cost_df[cost_col_mapping["bare_mass"]].values y_data =
    # cost_df[cost_col_mapping["cost_mean"]].values

    # Create figure
    fig_cfg = get_figure_config()
    fig, ax = plt.subplots(figsize=fig_cfg["figure_size"], dpi=fig_cfg["dpi"])

    # # Plot data points
    # data_style = get_data_point_style() data_style["color"] =
    # analysis_cfg["data_color"]

    # ax.plot( x_data, y_data, marker=data_style["marker"],
    #     markersize=data_style["marker_size"],
    #     color=data_style["color"], linestyle="", )

    # Extract averaged cost data
    x_data = cost_df["Bare_mass"].values
    y_data = cost_df["cost_mean"].values
    y_error = cost_df["cost_error"].values

    # Plot data points with error bars
    data_style = get_data_point_style()
    data_style["color"] = analysis_cfg["data_color"]

    ax.errorbar(
        x_data,
        y_data,
        yerr=y_error,
        fmt=data_style["marker"],
        markersize=data_style["marker_size"],
        color=data_style["color"],
        capsize=data_style["capsize"],
        capthick=data_style["capthick"],
        elinewidth=data_style["error_linewidth"],
    )

    # Extract fit parameters (shifted power law: a/(x-b) + c)
    col_mapping = get_results_column_mapping()
    a = results_row[col_mapping["cost_fit"]["param_a_mean"]]
    b = results_row[col_mapping["cost_fit"]["param_b_mean"]]
    c = results_row[col_mapping["cost_fit"]["param_c_mean"]]
    r_squared = results_row[col_mapping["cost_fit"]["r_squared"]]
    chi2_reduced = results_row[col_mapping["cost_fit"]["chi2_reduced"]]
    q_value = results_row[col_mapping["cost_fit"]["q_value"]]

    # Create fit line
    x_fit = np.linspace(x_data.min() * 0.95, x_data.max() * 1.05, 200)
    # Avoid singularity
    x_fit = x_fit[np.abs(x_fit - b) > 1e-6]
    y_fit = a / (x_fit - b) + c

    # Create detailed fit label
    fit_label = (
        f"$\\mathbf{{Shifted\\ power\\ law:}}$\n"
        f"  • Fitting range: $m$ ∈ [{x_data.min():.2f}, {x_data.max():.2f}]\n"
        f"  • Cost = {a:.2f}/($m$ - {b:.5f}) + {c:.2f}\n"
        f"  • χ²/dof = {chi2_reduced:.3f}\n"
        f"  • R² = {r_squared:.4f}\n"
        f"  • Q = {q_value:.3f}"
    )

    # Plot fit line
    fit_style = get_fit_line_style()
    fit_style["color"] = analysis_cfg["fit_color"]

    ax.plot(
        x_fit,
        y_fit,
        color=fit_style["color"],
        linewidth=fit_style["linewidth"],
        linestyle=fit_style["linestyle"],
        label=fit_label,
    )

    # Mark extrapolation point
    derived_bare_mass = results_row[col_mapping["derived_bare_mass_mean"]]
    extrapolated_cost = results_row[col_mapping["extrapolated_cost_mean"]]
    extrapolated_cost_error = results_row[col_mapping["extrapolated_cost_error"]]

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
        label=f"Extrapolation: {extrapolated_cost:.1f} ± {extrapolated_cost_error:.1f} core-hrs",
    )

    # Add extrapolation guide lines
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

    # # Add sample count annotations (using mass data) Note: We need to
    # # match cost x_data with mass data bare mass values to get correct
    # # annotations for the cost plot points
    # mass_col_mapping =
    # get_mass_data_column_mapping(analysis_type=analysis_type)
    # mass_bare_mass = mass_df[mass_col_mapping["bare_mass"]].values

    # # Create mapping from bare mass to cost for annotation positioning
    # # Only annotate points that exist in both datasets
    # for cost_x, cost_y in zip(x_data, y_data): # Find matching bare
    #     mass in mass_df (with tolerance for # floating point) mass_idx
    #     = np.where(np.abs(mass_bare_mass - cost_x) < 1e-10)[0] if
    #     len(mass_idx) > 0: # Use the first matched index for
    #     annotation _add_sample_count_annotations( ax, [cost_x],
    #         [cost_y], mass_df.iloc[mass_idx[0:1]], analysis_type )

    # Add sample count annotations (using mass data) Match by bare mass
    # values
    mass_col_mapping = get_mass_data_column_mapping(analysis_type=analysis_type)
    mass_bare_mass = mass_df[mass_col_mapping["bare_mass"]].values

    for cost_x, cost_y in zip(x_data, y_data):
        # Find matching bare mass in mass_df
        mass_matches = mass_df[
            np.abs(mass_df[mass_col_mapping["bare_mass"]] - cost_x) < 1e-10
        ]
        if not mass_matches.empty:
            _add_sample_count_annotations(
                ax, [cost_x], [cost_y], mass_matches.iloc[0:1], analysis_type
            )

    # Add fitting range markers
    _add_fitting_range_markers(ax, x_data.min(), x_data.max())

    # Add reference axes at origin
    _add_reference_axes(ax)

    # Configure axes
    axis_labels = get_axis_labels()
    ax.set_xlabel(
        axis_labels["bare_mass"]["label"],
        fontsize=axis_labels["bare_mass"]["fontsize"],
    )
    ax.set_ylabel(
        axis_labels["cost"]["label"],
        fontsize=axis_labels["cost"]["fontsize"],
    )

    if fig_cfg["grid"]:
        ax.grid(True, alpha=fig_cfg["grid_alpha"], linestyle=fig_cfg["grid_linestyle"])

    # Add legend
    legend_cfg = get_legend_config(plot_type="cost_fit")
    ax.legend(**legend_cfg)

    # Generate title using PlotTitleBuilder
    title_style = get_title_styling()
    title = title_builder.build(
        metadata_dict=group_info["group_metadata"],
        tunable_params=group_info["all_tunable_params"],
        leading_substring="Cost Extrapolation at Fixed Bare Mass:",
        wrapping_length=80,
    )
    ax.set_title(title, fontsize=title_style["fontsize"], pad=title_style["pad"])

    plt.tight_layout()

    # Save to sub-subdirectory
    plot_subdirs = get_plot_type_subdirectories()
    cost_subdir = plots_directory / plot_subdirs["cost_fit"]
    cost_subdir.mkdir(parents=True, exist_ok=True)

    filename = f"cost_fit_{group_info['group_id']}.png"
    plot_path = cost_subdir / filename
    fig.savefig(plot_path, dpi=fig_cfg["dpi"], bbox_inches="tight")
    plt.close(fig)

    logger.info(f"  Saved cost fit plot: {plot_path}")

    return plot_path


# =============================================================================
# HELPER FUNCTIONS FOR PLOT ELEMENTS
# =============================================================================


def _add_reference_axes(ax):
    """Add reference axes at x=0 and y=0."""
    from src.analysis.cost_extrapolation._cost_extrapolation_visualization_config import (
        get_reference_axes_style,
    )

    ref_style = get_reference_axes_style()
    ax.axhline(0, **ref_style)
    ax.axvline(0, **ref_style)


def _add_sample_count_annotations(ax, x_data, y_data, mass_df, analysis_type):
    """
    Add gauge configuration count annotations to data points.

    Args:
        ax: Matplotlib axes object x_data: X coordinates for annotations
        y_data: Y coordinates for annotations mass_df: DataFrame
        containing sample count column analysis_type: "pcac" or "pion"
    """
    from src.analysis.cost_extrapolation._cost_extrapolation_visualization_config import (
        get_sample_count_columns,
        get_annotation_style,
    )

    sample_cols = get_sample_count_columns()
    sample_count_col = sample_cols.get(analysis_type)

    if not sample_count_col or sample_count_col not in mass_df.columns:
        return  # Skip if column doesn't exist

    sample_counts = mass_df[sample_count_col].values
    annot_style = get_annotation_style()

    for x, y, count in zip(x_data, y_data, sample_counts):
        ax.annotate(
            f"{int(count)}",
            xy=(x, y),
            xytext=(annot_style["offset_x"], annot_style["offset_y"]),
            textcoords="offset points",
            fontsize=annot_style["fontsize"],
            ha=annot_style["ha"],
            bbox=dict(
                boxstyle=annot_style["bbox_style"],
                facecolor=annot_style["bbox_facecolor"],
                edgecolor=annot_style["bbox_edgecolor"],
                linewidth=annot_style["bbox_linewidth"],
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=annot_style["arrow_color"],
                linewidth=annot_style["arrow_linewidth"],
            ),
        )


def _add_fitting_range_markers(ax, x_min, x_max):
    """
    Add vertical lines marking the fitting range.

    Args:
        ax: Matplotlib axes object x_min: Minimum x value of fitting
        range x_max: Maximum x value of fitting range
    """
    ax.axvline(
        x_min,
        color="gray",
        linestyle=":",
        linewidth=1,
        alpha=0.6,
        label="Fitting range",
    )
    ax.axvline(x_max, color="gray", linestyle=":", linewidth=1, alpha=0.6)


def _add_reference_plateau_mass_line(ax, analysis_type):
    """
    Add horizontal line for reference plateau mass.

    Args:
        ax: Matplotlib axes object analysis_type: "pcac" or "pion"
    """
    if analysis_type == "pcac":
        from src.analysis.cost_extrapolation._pcac_cost_extrapolation_config import (
            get_reference_pcac_mass,
        )

        ref_mass = get_reference_pcac_mass()
        # Apply power transformation (PCAC uses power=1, so just the
        # value)
        ref_mass_transformed = ref_mass
        label = f"Reference PCAC mass: {ref_mass:.6f}"
    else:  # pion
        from src.analysis.cost_extrapolation._pion_cost_extrapolation_config import (
            get_reference_pion_mass,
            get_pion_mass_power,
        )

        ref_mass = get_reference_pion_mass()
        mass_power = get_pion_mass_power()
        # Apply power transformation (pion uses power=2 typically)
        ref_mass_transformed = ref_mass**mass_power
        label = f"Reference pion mass: {ref_mass:.6f}"

    ax.axhline(
        ref_mass_transformed,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=label,
    )
