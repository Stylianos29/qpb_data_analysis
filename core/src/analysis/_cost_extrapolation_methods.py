"""
Methods for computational cost extrapolation using DataPlotter
integration.

This module contains functions that leverage the DataFrameAnalyzer for
automatic parameter detection and grouping, and the DataPlotter class
for curve fitting and visualization of computational cost data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Import library components
from library import load_csv
from library.visualization.plotters.data_plotter import DataPlotter
from library.data.analyzer import DataFrameAnalyzer

# Import configuration
from src.analysis._cost_extrapolation_config import (
    CONFIG,
    get_input_column,
    get_averaging_config,
    get_cost_column_names,
    get_validation_config,
    get_plotting_config,
    get_extrapolation_config,
    get_validation_thresholds,
)


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================


def load_and_prepare_data(csv_path: str, logger) -> pd.DataFrame:
    """
    Load and prepare processed parameters data for cost extrapolation.

    This function:
        1. Loads the raw CSV data using library.load_csv()
        2. Creates a DataFrameAnalyzer for automatic parameter detection
        3. Averages across configurations using
           DataFrameAnalyzer.group_by_multivalued_tunable_parameters()
        4. Creates error-bar ready columns for DataPlotter

    Parameters
    ----------
    csv_path : str
        Path to the processed parameters CSV file
    logger : Logger
        Logger instance for reporting

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame ready for DataPlotter extrapolation
    """
    logger.info(f"Loading data from: {csv_path}")

    # Load CSV data - load_csv handles missing and empty cells
    try:
        df = load_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV")
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        raise

    # Validate required columns exist
    required_columns = [
        get_input_column("bare_mass"),
        get_input_column("configuration_label"),
        get_input_column("core_hours_per_spinor"),
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Average across configurations using DataFrameAnalyzer
    df_averaged = _average_across_configurations(df, logger)

    logger.info(
        f"Data preparation complete: {len(df_averaged)} groups ready for extrapolation"
    )
    return df_averaged


def _average_across_configurations(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Average computational costs across configurations using
    DataFrameAnalyzer.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with individual configuration data
    logger : Logger
        Logger instance

    Returns
    -------
    pd.DataFrame
        DataFrame with (mean, error) tuples for cost data
    """
    logger.info(
        "Creating per-configuration averages using automatic parameter grouping..."
    )

    # Create DataFrameAnalyzer for automatic parameter detection
    analyzer = DataFrameAnalyzer(df)

    # Get configuration
    averaging_config = get_averaging_config()
    validation_config = get_validation_config()

    # Get column names
    core_hours_col = get_input_column("core_hours_per_spinor")

    # Get minimum data points threshold from config
    min_count = validation_config["min_data_points_for_averaging"]

    # Determine which parameters to filter out (only if they exist in
    # DataFrame)
    potential_filter_params = averaging_config["filter_out_parameters"]
    actual_filter_params = [
        param for param in potential_filter_params if param in df.columns
    ]

    logger.info(f"Filtering out parameters: {actual_filter_params}")

    # Use analyzer's automatic grouping
    groups = analyzer.group_by_multivalued_tunable_parameters(
        filter_out_parameters_list=actual_filter_params,
        verbose=False,
    )

    logger.info(
        f"Grouping parameters: {analyzer.reduced_multivalued_tunable_parameter_names_list}"
    )

    # Build result rows by iterating through groups
    result_rows = []

    for group_keys, group_df in groups:
        # Extract statistics for core hours
        core_hours_series = group_df[core_hours_col]

        # Check count early - skip groups with insufficient data
        count = core_hours_series.count()
        if count < min_count:
            logger.warning(
                f"Skipping group {group_keys}: "
                f"insufficient data points ({count} < {min_count})"
            )
            continue

        # Calculate aggregation statistics
        mean_val = core_hours_series.mean()
        sem_val = core_hours_series.sem()
        error_val = sem_val if pd.notna(sem_val) else 0.0

        # Create result row with group parameters
        grouping_param_names = analyzer.reduced_multivalued_tunable_parameter_names_list
        result_row = dict(zip(grouping_param_names, group_keys))

        # Add single-valued parameters
        result_row.update(analyzer.unique_value_columns_dictionary)

        # Add the main output: (mean, error) tuple
        result_row["Average_core_hours_per_spinor_per_configuration"] = (
            mean_val,
            error_val,
        )
        result_row["Number_of_configurations"] = int(count)

        result_rows.append(result_row)

    return pd.DataFrame(result_rows)


# =============================================================================
# DATAPLOTTER INTEGRATION
# =============================================================================


def create_cost_plotter(df: pd.DataFrame, plots_directory: Path, logger) -> DataPlotter:
    """
    Create and configure DataPlotter for cost extrapolation.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared data for extrapolation
    plots_directory : Path
        Directory for saving plots
    logger : Logger
        Logger instance

    Returns
    -------
    DataPlotter
        Configured DataPlotter instance
    """
    logger.info("Creating DataPlotter for cost extrapolation...")

    # Create DataPlotter instance
    plotter = DataPlotter(df, str(plots_directory))

    # Set plot variables from configuration
    plotting_config = get_plotting_config()
    x_var = plotting_config["x_variable"]
    y_var = plotting_config["y_variable"]

    plotter.set_plot_variables(x_var, y_var)

    logger.info(f"DataPlotter configured: {x_var} vs {y_var}")
    logger.info(
        f"Multivalued parameters: {plotter.list_of_multivalued_tunable_parameter_names}"
    )
    logger.info(
        f"Single-valued parameters: {plotter.list_of_single_valued_tunable_parameter_names}"
    )

    return plotter


def perform_cost_extrapolation(plotter: DataPlotter, logger) -> Dict[str, Any]:
    """
    Perform cost extrapolation using DataPlotter with automatic grouping
    and fitting.
    """
    logger.info("Performing cost extrapolation with automatic grouping and fitting...")

    # Get configuration
    plotting_config = get_plotting_config()
    validation_config = get_validation_config()

    # Perform plotting with curve fitting (CurveFitter handles data
    # validation)
    logger.info("Generating plots with curve fitting...")
    plotter.plot(
        figure_size=plotting_config["figure_size"],
        marker_size=plotting_config["marker_size"],
        capsize=plotting_config["capsize"],
        include_legend=plotting_config["include_legend"],
        legend_location=plotting_config["legend_location"],
        fit_function=plotting_config["fit_function"],
        show_fit_parameters_on_plot=plotting_config["show_fit_parameters"],
        fit_label_location=plotting_config["fit_label_location"],
        fit_min_data_points=validation_config["min_data_points_for_fitting"],
        save_figure=True,
        verbose=False,
        include_plot_title=True,
        top_margin_adjustment=plotting_config["top_margin_adjustment"],
    )

    # Extract and validate results
    results = _extract_results_from_plotter(plotter, logger)

    if get_extrapolation_config()["validate_results"]:
        _validate_extrapolation_results(results, logger)

    return results


def _extract_results_from_plotter(plotter: DataPlotter, logger) -> Dict[str, Any]:
    """Extract results from DataPlotter after fitting."""

    logger.info("Extracting fit results from DataPlotter...")

    # Extract basic statistics from the data
    cost_cols = get_cost_column_names()
    df = plotter.dataframe

    # Group-level statistics
    group_results = []

    # If we have multivalued parameters, extrapolate by group
    if plotter.list_of_multivalued_tunable_parameter_names:
        grouping_params = plotter.list_of_multivalued_tunable_parameter_names
        for group_values, group_df in df.groupby(grouping_params):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)

            group_result = _extrapolate_group(
                group_values, group_df, grouping_params, logger
            )
            group_results.append(group_result)
    else:
        # Extrapolate entire dataset as one group
        group_result = _extrapolate_group(("all_data",), df, ["dataset"], logger)
        group_results.append(group_result)

    # Overall statistics
    overall_stats = {
        "total_data_points": len(df),
        "total_groups": len(group_results),
        "avg_cost": float(df[cost_cols["mean"]].mean()),
        "std_cost": float(df[cost_cols["mean"]].std()),
        "min_cost": float(df[cost_cols["mean"]].min()),
        "max_cost": float(df[cost_cols["mean"]].max()),
    }

    return {
        "group_results": group_results,
        "overall_statistics": overall_stats,
        "extrapolation_type": "with_fitting",
        "fit_function": get_plotting_config()["fit_function"],
    }


def _extrapolate_group(
    group_values: Tuple, group_df: pd.DataFrame, param_names: List[str], logger
) -> Dict[str, Any]:
    """Extrapolate a single parameter group."""

    cost_cols = get_cost_column_names()

    # Basic group information
    group_params = dict(zip(param_names, group_values))

    # Statistics
    result = {
        **group_params,
        "n_data_points": len(group_df),
        "avg_cost": float(group_df[cost_cols["mean"]].mean()),
        "std_cost": float(group_df[cost_cols["mean"]].std()),
        "min_cost": float(group_df[cost_cols["mean"]].min()),
        "max_cost": float(group_df[cost_cols["mean"]].max()),
        "total_configurations": (
            int(group_df[cost_cols["count"]].sum())
            if cost_cols["count"] in group_df.columns
            else len(group_df)
        ),
    }

    # Add bare mass range if available
    bare_mass_col = get_input_column("bare_mass")
    if f"{bare_mass_col}_min" in group_df.columns:
        result.update(
            {
                "bare_mass_min": float(group_df[f"{bare_mass_col}_min"].min()),
                "bare_mass_max": float(group_df[f"{bare_mass_col}_max"].max()),
                "bare_mass_avg": float(group_df[f"{bare_mass_col}_mean"].mean()),
            }
        )
    elif bare_mass_col in group_df.columns:
        result.update(
            {
                "bare_mass_min": float(group_df[bare_mass_col].min()),
                "bare_mass_max": float(group_df[bare_mass_col].max()),
                "bare_mass_avg": float(group_df[bare_mass_col].mean()),
            }
        )

    return result


def _validate_extrapolation_results(results: Dict[str, Any], logger) -> None:
    """Validate extrapolation results against quality thresholds."""
    thresholds = get_validation_thresholds()
    group_results = results.get("group_results", [])

    if not group_results:
        logger.warning("No group results to validate")
        return

    # Check for sufficient data points
    insufficient_data_groups = 0
    for group in group_results:
        n_points = group.get("n_data_points", 0)
        if n_points < thresholds["min_data_points"]:
            insufficient_data_groups += 1

    # Calculate success rate
    total_groups = len(group_results)
    success_rate = 1 - insufficient_data_groups / total_groups

    logger.info(f"Validation results:")
    logger.info(f"  • Total groups: {total_groups}")
    logger.info(f"  • Insufficient data: {insufficient_data_groups}")
    logger.info(f"  • Success rate: {success_rate:.1%}")


# =============================================================================
# RESULT EXPORT
# =============================================================================


def export_results(
    results: Dict[str, Any], output_directory: Path, csv_filename: str, logger
) -> pd.DataFrame:
    """
    Export extrapolation results to CSV file.

    Parameters
    ----------
    results : Dict[str, Any]
        Extrapolation results from perform_cost_extrapolation
    output_directory : Path
        Output directory
    csv_filename : str
        Name of output CSV file
    logger : Logger
        Logger instance

    Returns
    -------
    pd.DataFrame
        Exported results DataFrame
    """
    logger.info("Exporting extrapolation results...")

    group_results = results.get("group_results", [])
    if not group_results:
        logger.warning("No group results to export")
        return pd.DataFrame()

    # Convert to DataFrame
    results_df = pd.DataFrame(group_results)

    # Add overall statistics as additional columns
    overall_stats = results.get("overall_statistics", {})
    for key, value in overall_stats.items():
        results_df[f"overall_{key}"] = value

    # Add extrapolation metadata
    results_df["extrapolation_type"] = results.get("extrapolation_type", "unknown")
    if "fit_function" in results:
        results_df["fit_function"] = results["fit_function"]

    # Round floating point values
    float_precision = CONFIG["output"]["float_precision"]
    numeric_columns = results_df.select_dtypes(include=[np.number]).columns
    results_df[numeric_columns] = results_df[numeric_columns].round(float_precision)

    # Export to CSV
    csv_path = output_directory / csv_filename
    try:
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Exported {len(results_df)} results to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to export results: {e}")
        raise

    return results_df
