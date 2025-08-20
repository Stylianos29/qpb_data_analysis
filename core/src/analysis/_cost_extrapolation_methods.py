"""
Methods for computational cost extrapolation using DataPlotter
integration.

This module contains functions that leverage the DataFrameAnalyzer for
automatic parameter detection and grouping, and the DataPlotter class
for curve fitting and visualization of computational cost data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from pathlib import Path

# Import library components
from library import load_csv, DataFrameAnalyzer, DataPlotter

# Import configuration
from src.analysis._cost_extrapolation_config import (
    get_input_column,
    get_averaging_config,
    get_validation_config,
    get_plotting_config,
    get_output_config,
    get_extrapolation_config,
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

    # Apply fitting range constraints if specified
    extrapolation_config = get_extrapolation_config()
    min_mass = extrapolation_config.get("fit_range_min_bare_mass")
    max_mass = extrapolation_config.get("fit_range_max_bare_mass")

    if min_mass is not None:
        df = df[df["Bare_mass"] > min_mass]
    if max_mass is not None:
        df = df[df["Bare_mass"] <= max_mass]

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
        "Grouping parameters: "
        f"{analyzer.reduced_multivalued_tunable_parameter_names_list}"
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

    # Get plotting configuration
    plotting_config = get_plotting_config()

    # Create base subdirectory for cost extrapolation
    base_subdir = plotting_config.get(
        "base_subdirectory", "Computational_cost_extrapolation"
    )
    cost_plots_directory = plots_directory / base_subdir
    cost_plots_directory.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created base plots directory: {cost_plots_directory}")

    # Create DataPlotter instance with enhanced directory
    plotter = DataPlotter(df, str(cost_plots_directory))

    # Set plot variables from configuration
    x_var = plotting_config["x_variable"]
    y_var = plotting_config["y_variable"]

    plotter.set_plot_variables(x_var, y_var)

    logger.info(f"DataPlotter configured: {x_var} vs {y_var}")
    logger.info(f"Base directory: {cost_plots_directory}")
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

    # Perform plotting with curve fitting
    logger.info("Generating plots with curve fitting...")
    plotter.plot(
        # Figure settings
        figure_size=plotting_config["figure_size"],
        top_margin_adjustment=plotting_config["top_margin_adjustment"],
        left_margin_adjustment=plotting_config["left_margin_adjustment"],
        # Styling
        marker_size=plotting_config["marker_size"],
        capsize=plotting_config["capsize"],
        include_legend=plotting_config["include_legend"],
        legend_location=plotting_config["legend_location"],
        include_plot_title=True,
        # Curve fitting
        fit_function=plotting_config["fit_function"],
        show_fit_parameters_on_plot=plotting_config["show_fit_parameters"],
        fit_label_location=plotting_config["fit_label_location"],
        fit_min_data_points=validation_config["min_data_points_for_fitting"],
        # Output
        verbose=plotting_config["verbose"],
        file_format=plotting_config["file_format"],
        # Advanced
        post_plot_customization_function=add_extrapolation_lines,
    )

    # Extract results
    return plotter.get_fit_results()


def add_extrapolation_lines(ax, fit_results=None, **kwargs):
    """
    Add vertical and horizontal lines showing extrapolation with uncertainty bands.
    """
    if not fit_results:
        return

    # Get configuration
    extrapolation_config = get_extrapolation_config()
    target_bare_mass = extrapolation_config["target_bare_mass"]

    # Calculate extrapolated cost with uncertainty
    extrapolated_result = _calculate_extrapolation(fit_results, target_bare_mass)
    if extrapolated_result is None:
        return

    # Extract value and uncertainty
    if hasattr(extrapolated_result, "mean"):  # gvar object
        extrapolated_cost = float(extrapolated_result.mean)
        uncertainty = float(extrapolated_result.sdev)
    else:  # float
        extrapolated_cost = float(extrapolated_result)
        uncertainty = 0.0

    # Get styles and labels from config
    v_style = extrapolation_config["vertical_line_style"]
    h_style = extrapolation_config["horizontal_line_style"]
    band_style = extrapolation_config["uncertainty_band_style"]
    v_label = extrapolation_config["vertical_line_label"]
    h_label = extrapolation_config["horizontal_line_label"]

    # Create combined horizontal label with uncertainty
    if uncertainty > 0:
        import gvar

        h_label_text = (
            f"{h_label} = {gvar.gvar(extrapolated_cost, uncertainty)} core-hours"
        )
    else:
        h_label_text = f"{h_label} = {extrapolated_cost:.2f} core-hours"

    # Draw extrapolation lines
    ax.axvline(target_bare_mass, label=f"{v_label} = {target_bare_mass}", **v_style)
    ax.axhline(extrapolated_cost, label=h_label_text, **h_style)

    # Add uncertainty band (no label to avoid legend clutter)
    if uncertainty > 0:
        band_color = band_style.get("color") or h_style.get("color", "green")
        ax.axhspan(
            extrapolated_cost - uncertainty,
            extrapolated_cost + uncertainty,
            alpha=band_style["alpha"],
            color=band_color,
        )

    # Update legend
    ax.legend()


def _calculate_extrapolation(fit_result, x_target):
    """
    Calculate extrapolated y-value for given x_target using fit results.

    Parameters
    ----------
    fit_result : dict
        Fit result with 'parameters', 'function', and 'method' keys
    x_target : float
        X-value to extrapolate at

    Returns
    -------
    float or None
        Extrapolated y-value, or None if calculation fails
    """

    # Helper function for shifted power law
    def _safe_shifted_power_law(a, b, c, x_target):
        """Handle shifted power law with gvar-safe abs()."""
        # Extract mean value for comparison (works for both float and gvar)
        b_mean = b.mean if hasattr(b, "mean") else b

        if abs(x_target - b_mean) < 1e-10:
            return None
        return a / (x_target - b) + c

    if not fit_result:
        return None

    # Extract parameters
    params = fit_result["parameters"]

    # Function dispatch table
    function_map = {
        "linear": lambda a, b, *_: a * x_target + b,
        "exponential": lambda a, b, c, *_: a * np.exp(-b * x_target) + c,
        "power_law": lambda a, b, *_: a * (x_target**b) if x_target > 0 else None,
        "shifted_power_law": lambda a, b, c, *_: _safe_shifted_power_law(
            a, b, c, x_target
        ),
    }

    function_type = fit_result["function"]
    if function_type not in function_map:
        return None

    try:
        return function_map[function_type](*params)
    except (IndexError, ValueError, ZeroDivisionError):
        return None


# =============================================================================
# RESULT EXPORT
# =============================================================================


def export_results(
    fit_results: Dict[str, Any],
    plotter: DataPlotter,
    output_directory: Path,
    csv_filename: str,
    logger,
) -> pd.DataFrame:
    """
    Export extrapolation results to CSV file.

    Parameters
    ----------
    fit_results : Dict[str, Any]
        Fit results from DataPlotter.get_fit_results()
    plotter : DataPlotter
        DataPlotter instance for accessing parameter metadata
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

    if not fit_results:
        logger.warning("No fit results to export")
        return pd.DataFrame()

    # Get parameter names for proper column naming
    param_names = plotter.reduced_multivalued_tunable_parameter_names_list

    # Convert fit results to DataFrame
    results_data = []
    for group_key, fit_data in fit_results.items():
        if fit_data:
            # Handle both gvar and scipy results
            if fit_data.get("method") == "gvar":
                try:
                    import gvar

                    params = [float(gvar.mean(p)) for p in fit_data["parameters"]]
                except ImportError:
                    params = fit_data["parameters"]
            else:
                params = fit_data["parameters"]

            # Create row with group parameters
            row = {}

            # Add group parameters with actual parameter names
            if isinstance(group_key, tuple):
                for i, value in enumerate(group_key):
                    if i < len(param_names):
                        row[param_names[i]] = value
                    else:
                        row[f"group_param_{i}"] = value  # Fallback for extra values
            else:
                # Single parameter case
                if param_names:
                    row[param_names[0]] = group_key
                else:
                    row["group_key"] = group_key

            # Add single-valued parameters
            row.update(plotter.unique_value_columns_dictionary)

            # Calculate n_data_points from original data
            if isinstance(group_key, tuple):
                # Create mask for this specific group - initialize as all True Series
                mask = pd.Series(
                    [True] * len(plotter.dataframe), index=plotter.dataframe.index
                )
                for i, param_name in enumerate(param_names):
                    if i < len(group_key):
                        mask &= plotter.dataframe[param_name] == group_key[i]
                n_data_points = int(mask.sum())
            else:
                # Single parameter case or fallback
                if param_names:
                    mask = plotter.dataframe[param_names[0]] == group_key
                    n_data_points = int(mask.sum())
                else:
                    n_data_points = len(plotter.dataframe)

            row["n_data_points"] = n_data_points

            # Add fit parameters (a, b, c for shifted power law)
            for i, param in enumerate(params[:3]):  # Limit to first 3 parameters
                row[f"param_{chr(97+i)}"] = param  # a, b, c

            # Add extrapolation values
            target_bare_mass = get_extrapolation_config()["target_bare_mass"]
            extrapolated_result = _calculate_extrapolation(fit_data, target_bare_mass)

            row["target_bare_mass"] = target_bare_mass
            if extrapolated_result is not None:
                if hasattr(extrapolated_result, "mean"):  # gvar object
                    extrapolated_cost = (
                        float(extrapolated_result.mean),
                        float(extrapolated_result.sdev),
                    )
                else:  # float
                    extrapolated_cost = float(extrapolated_result)
                row["extrapolated_cost"] = extrapolated_cost

            results_data.append(row)

    # Create DataFrame
    results_df = pd.DataFrame(results_data)

    if results_df.empty:
        logger.warning("No valid fit results to export")
        return results_df

    # Round numeric values
    float_precision = get_output_config()["float_precision"]
    numeric_columns = results_df.select_dtypes(include=[np.number]).columns
    results_df[numeric_columns] = results_df[numeric_columns].round(float_precision)

    # Export to CSV
    csv_path = output_directory / csv_filename
    try:
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Exported {len(results_df)} fit results to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to export results: {e}")
        raise

    return results_df
