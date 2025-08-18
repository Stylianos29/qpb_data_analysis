"""
Private methods for computational cost estimation using DataPlotter integration.

This module contains simplified analysis functions that leverage the DataPlotter
class for automatic grouping, curve fitting, and visualization of computational
cost data.

Place this file as:
qpb_data_analysis/core/src/analysis/_cost_estimation_methods.py
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
from src.analysis._cost_estimation_config import (
    PROCESSED_PARAMS_CSV_COLUMNS,
    OUTPUT_CSV_CONFIG,
    REFERENCE_CONFIG,
    DATA_FILTERING_CONFIG,
    get_dataplotter_config,
    get_data_filtering_config,
    get_fit_validation_config,
    get_result_compilation_config,
)

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================


def load_and_prepare_data(processed_csv_path: str, logger) -> pd.DataFrame:
    """
    Load and prepare processed parameters data for analysis.

    This function:
    1. Loads the raw data
    2. Applies data filtering
    3. Creates a DataFrameAnalyzer for automatic parameter detection
    4. Creates derived column: Average_core_hours_per_spinor_per_configuration
    5. Returns DataFrame ready for DataPlotter analysis

    Parameters
    ----------
    processed_csv_path : str
        Path to processed parameter values CSV file
    logger : Logger
        Logger instance for progress reporting

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame with derived per-configuration averages

    Raises
    ------
    ValueError
        If required columns are missing or data validation fails
    """
    logger.info("Loading and preparing processed parameters data...")

    # Load CSV with apply_categorical=False to preserve raw values
    try:
        df = load_csv(processed_csv_path, apply_categorical=False)
        logger.info(f"Loaded {len(df)} rows from processed parameters CSV")
    except Exception as e:
        raise ValueError(f"Failed to load processed parameters CSV: {e}")

    # Validate required columns
    required_cols = set(PROCESSED_PARAMS_CSV_COLUMNS.values())
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Apply data filtering to raw data
    filtering_config = get_data_filtering_config()
    df_filtered = apply_data_filtering(df, filtering_config, logger)

    # Create DataFrameAnalyzer for automatic parameter detection and grouping
    logger.info("Creating DataFrameAnalyzer for automatic parameter grouping...")
    analyzer = DataFrameAnalyzer(df_filtered)

    # Log detected parameter structure
    logger.info(
        f"Detected multivalued tunable parameters: {analyzer.list_of_multivalued_tunable_parameter_names}"
    )
    logger.info(
        f"Detected single-valued tunable parameters: {analyzer.list_of_single_valued_tunable_parameter_names}"
    )
    logger.info(f"Total data points for grouping: {len(analyzer.dataframe)}")

    # Create derived dataset with per-configuration averages using automatic grouping
    df_derived = create_per_configuration_averages(analyzer, logger)

    logger.info(
        f"Data preparation completed. {len(df_derived)} configuration-averaged data points ready for analysis."
    )
    return df_derived


def apply_data_filtering(
    df: pd.DataFrame, filtering_config: Dict[str, Any], logger
) -> pd.DataFrame:
    """
    Apply data filtering based on configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    filtering_config : Dict[str, Any]
        Data filtering configuration
    logger : Logger
        Logger instance

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    initial_count = len(df)
    bare_mass_col = PROCESSED_PARAMS_CSV_COLUMNS["bare_mass"]
    core_hours_col = PROCESSED_PARAMS_CSV_COLUMNS["core_hours_per_spinor"]

    # Filter by bare mass range
    min_mass = filtering_config["min_bare_mass_for_analysis"]
    max_mass = filtering_config["max_bare_mass_for_analysis"]

    mass_mask = (df[bare_mass_col] >= min_mass) & (df[bare_mass_col] <= max_mass)
    df = df[mass_mask]

    logger.info(f"Bare mass filtering: {initial_count} → {len(df)} rows")

    # Remove rows with invalid core hours
    valid_hours_mask = (
        df[core_hours_col].notna()
        & (df[core_hours_col] > 0)
        & (df[core_hours_col] < filtering_config.get("max_reasonable_cost", 1000.0))
    )
    df = df[valid_hours_mask]

    logger.info(f"Valid core-hours filtering: → {len(df)} rows")

    # Outlier removal if enabled
    if filtering_config.get("remove_outliers", False):
        df = remove_outliers(
            df, core_hours_col, filtering_config["outlier_threshold"], logger
        )

    return df


def remove_outliers(
    df: pd.DataFrame, column: str, threshold: float, logger
) -> pd.DataFrame:
    """Remove outliers based on standard deviation threshold."""
    initial_count = len(df)

    mean_val = df[column].mean()
    std_val = df[column].std()

    outlier_mask = np.abs(df[column] - mean_val) <= (threshold * std_val)
    df_clean = df[outlier_mask]

    removed_count = initial_count - len(df_clean)
    logger.info(f"Outlier removal: removed {removed_count} outliers (>{threshold}σ)")

    return df_clean


def create_per_configuration_averages(
    analyzer: DataFrameAnalyzer, logger
) -> pd.DataFrame:
    """
    Create derived dataset with per-configuration averages using automatic parameter grouping.

    This function uses the DataFrameAnalyzer's automatic parameter detection and grouping
    to create configuration-averaged data without hardcoding grouping parameters.

    Parameters
    ----------
    analyzer : DataFrameAnalyzer
        Analyzer object containing the filtered raw data
    logger : Logger
        Logger instance

    Returns
    -------
    pd.DataFrame
        DataFrame with per-configuration averages and proper grouping
    """
    logger.info(
        "Creating per-configuration averages using automatic parameter grouping..."
    )

    # Get the DataFrame from analyzer
    df = analyzer.dataframe

    # Get column names
    core_hours_col = PROCESSED_PARAMS_CSV_COLUMNS["core_hours_per_spinor"]

    # Use analyzer's automatic grouping, excluding Configuration_label and MPI_geometry
    # Configuration_label: We want to average across configurations
    # MPI_geometry: Usually not a physics parameter, more of a computational detail
    groups = analyzer.group_by_multivalued_tunable_parameters(
        filter_out_parameters_list=["Configuration_label"],
        verbose=True,  # Log which parameters are being used for grouping
    )

    logger.info(
        f"Grouping parameters: {analyzer.reduced_multivalued_tunable_parameter_names_list}"
    )

    # Build result rows by iterating through groups
    result_rows = []

    for group_keys, group_df in groups:
        # Extract statistics for core hours
        core_hours_series = group_df[core_hours_col]

        # Calculate aggregation statistics
        stats = {
            "mean": core_hours_series.mean(),
            "std": core_hours_series.std(),
            "count": core_hours_series.count(),
            "sem": core_hours_series.sem(),  # Standard error of the mean
        }

        # Create result row starting with group parameters
        # Get parameter names that were used for grouping
        grouping_param_names = analyzer.reduced_multivalued_tunable_parameter_names_list
        result_row = dict(zip(grouping_param_names, group_keys))

        # Add statistical results. Add "_mean", "_std", "_error"
        # suffixes to avoid conflict with tuple column
        result_row.update(
            {
                "Average_core_hours_per_spinor_per_configuration_mean": stats["mean"],
                "Average_core_hours_per_spinor_per_configuration_std": stats["std"],
                "Number_of_configurations": int(stats["count"]),
                "Average_core_hours_per_spinor_per_configuration_error": (
                    stats["sem"] if pd.notna(stats["sem"]) else 0.0
                ),
            }
        )

        # Add any single-valued parameters that might be relevant
        for param in analyzer.list_of_single_valued_tunable_parameter_names:
            if param in group_df.columns and param not in result_row:
                result_row[param] = group_df[param].iloc[0]

        result_rows.append(result_row)

    # Create result DataFrame
    grouped = pd.DataFrame(result_rows)

    # Log statistics
    initial_rows = len(df)
    final_rows = len(grouped)
    avg_configs_per_group = grouped["Number_of_configurations"].mean()

    logger.info(f"Configuration averaging: {initial_rows} → {final_rows} data points")
    logger.info(f"Average configurations per group: {avg_configs_per_group:.1f}")

    # Log groups with single configurations
    single_config_groups = (grouped["Number_of_configurations"] == 1).sum()
    if single_config_groups > 0:
        logger.warning(
            f"{single_config_groups} groups have only 1 configuration (no averaging possible)"
        )

    # Create error-bar ready column for DataPlotter (value, error) tuples
    grouped["Average_core_hours_per_spinor_per_configuration"] = [
        (mean_val, error_val)
        for mean_val, error_val in zip(
            grouped["Average_core_hours_per_spinor_per_configuration_mean"],
            grouped["Average_core_hours_per_spinor_per_configuration_error"],
        )
    ]

    return grouped


# =============================================================================
# DATAPLOTTER INTEGRATION
# =============================================================================


def create_cost_plotter(df: pd.DataFrame, plots_directory: Path, logger) -> DataPlotter:
    """
    Create and configure DataPlotter for cost analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared data for analysis
    plots_directory : Path
        Directory for saving plots
    logger : Logger
        Logger instance

    Returns
    -------
    DataPlotter
        Configured DataPlotter instance
    """
    logger.info("Creating DataPlotter for cost analysis...")

    # Create DataPlotter instance
    plotter = DataPlotter(df, str(plots_directory))

    # Set plot variables
    dataplotter_config = get_dataplotter_config()
    x_var = dataplotter_config["x_variable"]
    y_var = dataplotter_config["y_variable"]

    plotter.set_plot_variables(x_var, y_var)

    logger.info(f"DataPlotter configured: {x_var} vs {y_var}")
    logger.info(
        f"Multivalued parameters detected: {plotter.list_of_multivalued_tunable_parameter_names}"
    )

    return plotter


def perform_cost_analysis(
    plotter: DataPlotter, logger
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Perform cost analysis using DataPlotter.

    Parameters
    ----------
    plotter : DataPlotter
        Configured DataPlotter instance
    logger : Logger
        Logger instance

    Returns
    -------
    Tuple[Dict[str, Any], List[Dict[str, Any]]]
        Analysis results and individual group results
    """
    logger.info("Performing cost analysis with automatic grouping and fitting...")

    # Get configuration
    dataplotter_config = get_dataplotter_config()
    filtering_config = get_data_filtering_config()

    # Check if we have sufficient data for fitting
    min_points = filtering_config["min_data_points_for_fitting"]
    if len(plotter.dataframe) < min_points:
        raise ValueError(
            f"Insufficient data points: {len(plotter.dataframe)} < {min_points}"
        )

    # Perform plotting with automatic grouping and fitting
    plotter.plot(
        # Grouping (automatic based on multivalued parameters)
        grouping_variable=None,  # Auto-detect
        # Figure settings
        figure_size=dataplotter_config["figure_size"],
        font_size=dataplotter_config["font_size"],
        # Plot titles
        include_plot_title=dataplotter_config["include_plot_title"],
        # custom_plot_title=dataplotter_config["custom_plot_title"],
        title_from_columns=dataplotter_config["title_from_columns"],
        title_size=dataplotter_config["title_size"],
        bold_title=dataplotter_config["bold_title"],
        title_wrapping_length=dataplotter_config["title_wrapping_length"],
        # Curve fitting
        fit_function=dataplotter_config["fit_function"],
        show_fit_parameters_on_plot=dataplotter_config["show_fit_parameters_on_plot"],
        fit_label_location=dataplotter_config["fit_label_location"],
        fit_label_format=dataplotter_config["fit_label_format"],
        fit_curve_range=dataplotter_config["fit_curve_range"],
        # Styling
        marker_size=dataplotter_config["marker_size"],
        capsize=dataplotter_config["capsize"],
        empty_markers=dataplotter_config["empty_markers"],
        include_legend=dataplotter_config["include_legend"],
        legend_location=dataplotter_config["legend_location"],
        # Axis settings
        xaxis_log_scale=dataplotter_config["xaxis_log_scale"],
        yaxis_log_scale=dataplotter_config["yaxis_log_scale"],
        # File output
        save_figure=dataplotter_config["save_figure"],
        file_format=dataplotter_config["file_format"],
        # Additional options
        verbose=True,
        top_margin_adjustment=0.88,
        left_margin_adjustment=0.12,
    )

    logger.info("DataPlotter analysis completed successfully")

    # Extract results from DataPlotter
    analysis_results = extract_analysis_results(plotter, logger)
    group_results = extract_group_results(plotter, logger)

    return analysis_results, group_results


def extract_analysis_results(plotter: DataPlotter, logger) -> Dict[str, Any]:
    """
    Extract overall analysis results from DataPlotter.

    Parameters
    ----------
    plotter : DataPlotter
        DataPlotter instance after analysis
    logger : Logger
        Logger instance

    Returns
    -------
    Dict[str, Any]
        Overall analysis results
    """
    results = {
        "total_data_points": len(plotter.dataframe),
        "multivalued_parameters": plotter.list_of_multivalued_tunable_parameter_names,
        "single_valued_parameters": plotter.list_of_single_valued_tunable_parameter_names,
        "output_quantities": plotter.list_of_output_quantity_names_from_dataframe,
        "reference_pcac_mass": REFERENCE_CONFIG["default_reference_pcac_mass"],
    }

    # Add basic statistics
    core_hours_col = (
        "Average_core_hours_per_spinor_per_configuration"  # Use derived column
    )
    bare_mass_col = PROCESSED_PARAMS_CSV_COLUMNS["bare_mass"]

    results["cost_statistics"] = {
        "mean": float(plotter.dataframe[core_hours_col].mean()),
        "std": float(plotter.dataframe[core_hours_col].std()),
        "min": float(plotter.dataframe[core_hours_col].min()),
        "max": float(plotter.dataframe[core_hours_col].max()),
        "median": float(plotter.dataframe[core_hours_col].median()),
    }

    results["bare_mass_range"] = {
        "min": float(plotter.dataframe[bare_mass_col].min()),
        "max": float(plotter.dataframe[bare_mass_col].max()),
        "mean": float(plotter.dataframe[bare_mass_col].mean()),
    }

    return results


def extract_group_results(plotter: DataPlotter, logger) -> List[Dict[str, Any]]:
    """
    Extract individual group results from DataPlotter.

    Parameters
    ----------
    plotter : DataPlotter
        DataPlotter instance after analysis
    logger : Logger
        Logger instance

    Returns
    -------
    List[Dict[str, Any]]
        List of results for each parameter group
    """
    logger.info("Extracting individual group results...")

    group_results = []

    # Get grouped data using DataPlotter's grouping
    grouped = plotter.group_by_multivalued_tunable_parameters(verbose=False)

    # Access fit results if available (this would need to be implemented in DataPlotter)
    # For now, we'll extract basic group statistics

    for group_keys, group_df in grouped:
        group_result = compile_group_result(group_keys, group_df, plotter, logger)
        group_results.append(group_result)

    # If no multivalued parameters detected, treat entire dataset as one group
    if len(group_results) == 0:
        logger.info(
            "No multivalued parameters detected - treating entire dataset as one group"
        )

        # Create a single group result using the entire dataset
        single_group_result = compile_single_group_result(
            plotter.dataframe, plotter, logger
        )
        group_results.append(single_group_result)

    logger.info(f"Extracted results for {len(group_results)} parameter groups")
    return group_results


def compile_single_group_result(
    group_df: pd.DataFrame, plotter: DataPlotter, logger
) -> Dict[str, Any]:
    """
    Compile results when there's only one parameter group (no multivalued parameters).

    Parameters
    ----------
    group_df : pd.DataFrame
        Data for the single group
    plotter : DataPlotter
        DataPlotter instance
    logger : Logger
        Logger instance

    Returns
    -------
    Dict[str, Any]
        Compiled results for this group
    """
    # Single-valued parameters
    single_valued_params = {}
    for param in plotter.list_of_single_valued_tunable_parameter_names:
        if param in group_df.columns:
            single_valued_params[param] = group_df[param].iloc[0]

    # Basic statistics
    core_hours_col = (
        "Average_core_hours_per_spinor_per_configuration"  # Use derived column
    )
    core_hours_error_col = (
        "Error_core_hours_per_spinor_per_configuration"  # Error column
    )
    bare_mass_col = PROCESSED_PARAMS_CSV_COLUMNS["bare_mass"]

    result = {
        **single_valued_params,
        "parameter_group": "single_group",
        "n_data_points": len(group_df),
        "avg_core_hours_per_configuration": float(group_df[core_hours_col].mean()),
        "std_core_hours_per_configuration": float(group_df[core_hours_col].std()),
        "min_core_hours_per_configuration": float(group_df[core_hours_col].min()),
        "max_core_hours_per_configuration": float(group_df[core_hours_col].max()),
        "avg_core_hours_error": (
            float(group_df[core_hours_error_col].mean())
            if core_hours_error_col in group_df.columns
            else 0.0
        ),
        "bare_mass_range_min": float(group_df[bare_mass_col].min()),
        "bare_mass_range_max": float(group_df[bare_mass_col].max()),
        "avg_bare_mass": float(group_df[bare_mass_col].mean()),
        "total_configurations": (
            int(group_df["Number_of_configurations"].sum())
            if "Number_of_configurations" in group_df.columns
            else len(group_df)
        ),
    }

    # Add cost predictions if configuration requests it
    compilation_config = get_result_compilation_config()
    if compilation_config["include_cost_predictions"]:
        result["cost_predictions"] = calculate_cost_predictions(
            group_df, compilation_config["cost_prediction_points"]
        )

    return result


def compile_group_result(
    group_keys: Tuple, group_df: pd.DataFrame, plotter: DataPlotter, logger
) -> Dict[str, Any]:
    """
    Compile results for a single parameter group.

    Parameters
    ----------
    group_keys : Tuple
        Parameter values defining the group
    group_df : pd.DataFrame
        Data for this group
    plotter : DataPlotter
        DataPlotter instance
    logger : Logger
        Logger instance

    Returns
    -------
    Dict[str, Any]
        Compiled results for this group
    """
    # Group parameter values
    param_names = plotter.list_of_multivalued_tunable_parameter_names
    group_params = dict(zip(param_names, group_keys))

    # Basic statistics
    core_hours_col = (
        "Average_core_hours_per_spinor_per_configuration_mean"  # Use derived column
    )
    core_hours_error_col = (
        "Average_core_hours_per_spinor_per_configuration_error"  # Error column
    )
    bare_mass_col = PROCESSED_PARAMS_CSV_COLUMNS["bare_mass"]

    result = {
        **group_params,
        "n_data_points": len(group_df),
        "avg_core_hours_per_configuration": float(group_df[core_hours_col].mean()),
        "std_core_hours_per_configuration": float(group_df[core_hours_col].std()),
        "min_core_hours_per_configuration": float(group_df[core_hours_col].min()),
        "max_core_hours_per_configuration": float(group_df[core_hours_col].max()),
        "avg_core_hours_error": (
            float(group_df[core_hours_error_col].mean())
            if core_hours_error_col in group_df.columns
            else 0.0
        ),
        "bare_mass_range_min": float(group_df[bare_mass_col].min()),
        "bare_mass_range_max": float(group_df[bare_mass_col].max()),
        "avg_bare_mass": float(group_df[bare_mass_col].mean()),
        "total_configurations": (
            int(group_df["Number_of_configurations"].sum())
            if "Number_of_configurations" in group_df.columns
            else len(group_df)
        ),
    }

    # Add cost predictions if configuration requests it
    compilation_config = get_result_compilation_config()
    if compilation_config["include_cost_predictions"]:
        result["cost_predictions"] = calculate_cost_predictions(
            group_df, compilation_config["cost_prediction_points"]
        )

    return result


def calculate_cost_predictions(
    group_df: pd.DataFrame, prediction_points: List[float]
) -> Dict[str, float]:
    """
    Calculate cost predictions at specific bare mass values.

    Parameters
    ----------
    group_df : pd.DataFrame
        Group data
    prediction_points : List[float]
        Bare mass values for prediction

    Returns
    -------
    Dict[str, float]
        Predicted costs at each point
    """
    # Simple linear interpolation for now
    # In the future, this could use the actual fit results from DataPlotter

    bare_mass_col = PROCESSED_PARAMS_CSV_COLUMNS["bare_mass"]
    core_hours_col = (
        "Average_core_hours_per_spinor_per_configuration"  # Use derived column
    )

    x_data = group_df[bare_mass_col].to_numpy()
    y_data = group_df[core_hours_col].to_numpy()

    predictions = {}
    for point in prediction_points:
        # Simple linear interpolation
        if np.min(x_data) <= point <= np.max(x_data):
            predicted_cost = np.interp(point, x_data, y_data)
            predictions[f"cost_at_bare_mass_{point:.3f}"] = float(predicted_cost)
        else:
            predictions[f"cost_at_bare_mass_{point:.3f}"] = None

    return predictions


# =============================================================================
# RESULT COMPILATION AND EXPORT
# =============================================================================


def compile_final_results(
    analysis_results: Dict[str, Any], group_results: List[Dict[str, Any]], logger
) -> pd.DataFrame:
    """
    Compile final results into a DataFrame for export.

    Parameters
    ----------
    analysis_results : Dict[str, Any]
        Overall analysis results
    group_results : List[Dict[str, Any]]
        Individual group results
    logger : Logger
        Logger instance

    Returns
    -------
    pd.DataFrame
        Final results ready for CSV export
    """
    logger.info("Compiling final results for export...")

    if not group_results:
        logger.warning("No group results to compile")
        return pd.DataFrame()

    # Convert group results to DataFrame
    results_df = pd.DataFrame(group_results)

    # Add overall statistics as additional columns
    for key, value in analysis_results["cost_statistics"].items():
        results_df[f"overall_{key}"] = value

    # Add metadata
    results_df["total_data_points_analyzed"] = analysis_results["total_data_points"]
    results_df["reference_pcac_mass"] = analysis_results["reference_pcac_mass"]

    # Round floating point values
    float_precision = OUTPUT_CSV_CONFIG["float_precision"]
    numeric_columns = results_df.select_dtypes(include=[np.number]).columns
    results_df[numeric_columns] = results_df[numeric_columns].round(float_precision)

    logger.info(f"Compiled {len(results_df)} group results for export")
    return results_df


def validate_results(results_df: pd.DataFrame, logger) -> bool:
    """
    Validate analysis results against quality thresholds.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame
    logger : Logger
        Logger instance

    Returns
    -------
    bool
        Whether results pass validation
    """
    if results_df.empty:
        logger.error("Results DataFrame is empty")
        return False

    validation_config = get_fit_validation_config()

    # Check for reasonable cost values
    max_cost = validation_config["max_reasonable_cost"]
    min_cost = validation_config["min_reasonable_cost"]

    cost_col = "avg_core_hours_per_configuration"  # Updated column name
    invalid_costs = (
        (results_df[cost_col] > max_cost) | (results_df[cost_col] < min_cost)
    ).sum()

    if invalid_costs > 0:
        logger.warning(f"Found {invalid_costs} groups with unreasonable cost values")

    # Check data point counts
    min_points = DATA_FILTERING_CONFIG["min_data_points_for_fitting"]
    insufficient_data = (results_df["n_data_points"] < min_points).sum()

    if insufficient_data > 0:
        logger.warning(
            f"Found {insufficient_data} groups with insufficient data points"
        )

    success_rate = 1 - (invalid_costs + insufficient_data) / len(results_df)
    logger.info(
        f"Result validation: {success_rate:.1%} of groups passed quality checks"
    )

    return success_rate > 0.5  # Require at least 50% success rate
