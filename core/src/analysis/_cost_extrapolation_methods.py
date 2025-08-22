"""
Enhanced methods for computational cost extrapolation using DataPlotter
integration.

This module supports both fixed bare mass and fixed PCAC mass
extrapolation methods. Uses DataFrameAnalyzer for automatic parameter
detection and grouping, and DataPlotter for curve fitting and
visualization of computational cost data.

Supported Methods:
    - fixed_bare_mass: Direct extrapolation using configured reference bare
      mass
    - fixed_pcac_mass: Convert reference PCAC mass to bare mass, then
      extrapolate cost
"""

import numpy as np
import pandas as pd
import gvar
from typing import Dict, Any, Optional, Union
from pathlib import Path

# Import library components
from library import load_csv, DataFrameAnalyzer, DataPlotter

# Import
from src.analysis._cost_extrapolation_config import (
    get_shared_config,
    get_pcac_config,
    get_cost_config,
    get_reference_pcac_mass,
    get_reference_bare_mass,
    get_base_subdirectory,
    # Backward compatibility
    get_averaging_config,
    get_validation_config,
    get_output_config,
    get_extrapolation_config,
)


# =============================================================================
# METHOD DETECTION
# =============================================================================


def detect_extrapolation_method(pcac_csv_path: Optional[str] = None) -> str:
    """
    Automatically detect extrapolation method based on input availability.

    Parameters
    ----------
    pcac_csv_path : str, optional
        Path to PCAC mass CSV file

    Returns
    -------
    str
        "fixed_pcac_mass" if PCAC data provided, "fixed_bare_mass" otherwise
    """
    return "fixed_pcac_mass" if pcac_csv_path is not None else "fixed_bare_mass"


# =============================================================================
# MAIN ENTRY POINT - METHOD DISPATCH
# =============================================================================


def extrapolate_computational_cost(
    processed_csv_path: str,
    output_directory: Path,
    plots_directory: Path,
    logger,
    pcac_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry point for computational cost extrapolation.

    Supports both fixed bare mass and fixed PCAC mass methods.
    Automatically dispatches based on configuration.

    Parameters
    ----------
    processed_csv_path : str
        Path to processed computational cost CSV file
    plots_directory : Path
        Directory for saving plots
    logger : Logger
        Logger instance
    pcac_csv_path : str, optional
        Path to PCAC mass CSV file (required for fixed_pcac_mass method)

    Returns
    -------
    Dict[str, Any]
        Fit results from cost extrapolation
    """
    method = detect_extrapolation_method(pcac_csv_path)
    logger.info(f"Using extrapolation method: {method}")

    if method == "fixed_bare_mass":
        return _extrapolate_fixed_bare_mass(
            processed_csv_path, output_directory, plots_directory, logger
        )

    elif method == "fixed_pcac_mass":
        if pcac_csv_path is None:
            raise ValueError("pcac_csv_path required for fixed_pcac_mass method")
        return _extrapolate_fixed_pcac_mass(
            processed_csv_path, pcac_csv_path, output_directory, plots_directory, logger
        )

    else:
        raise ValueError(f"Unknown extrapolation method: {method}")


# =============================================================================
# METHOD-SPECIFIC IMPLEMENTATIONS
# =============================================================================


def _extrapolate_fixed_bare_mass(
    processed_csv_path: str, output_directory: Path, plots_directory: Path, logger
) -> Dict[str, Any]:
    """
    Fixed bare mass method (original approach).

    Uses configured reference_bare_massas float value.
    """
    reference_bare_mass = get_reference_bare_mass()  # float from config
    logger.info(f"Using configured reference bare mass: {reference_bare_mass}")

    return _perform_cost_analysis(
        processed_csv_path,
        reference_bare_mass,
        output_directory,
        plots_directory,
        logger,
    )


def _extrapolate_fixed_pcac_mass(
    processed_csv_path: str,
    pcac_csv_path: str,
    output_directory: Path,
    plots_directory: Path,
    logger,
) -> Dict[str, Any]:
    """
    Fixed PCAC mass method: Convert reference PCAC mass to bare mass,
    then extrapolate cost.

    Data Flow:
        1. PCAC data → Linear fit → Invert to get reference bare mass (with
           uncertainty)
        2. reference bare mass → Cost extrapolation (reuse existing
           pipeline)
    """
    logger.info("Performing PCAC mass to bare mass conversion...")

    # Step 1: PCAC analysis - get reference bare mass with uncertainty
    target_bare_mass_gvar = _pcac_to_bare_mass_conversion(
        pcac_csv_path, plots_directory, logger
    )

    # Step 2: Cost extrapolation using derived reference (reuse existing infrastructure)
    logger.info(f"Reference bare mass from PCAC: {target_bare_mass_gvar}")
    return _perform_cost_analysis(
        processed_csv_path,
        target_bare_mass_gvar,
        output_directory,
        plots_directory,
        logger,
    )


# =============================================================================
# PCAC PREPROCESSING PIPELINE
# =============================================================================


def _pcac_to_bare_mass_conversion(
    pcac_csv_path: str, plots_directory: Path, logger
) -> gvar.GVar:
    """
    Convert reference PCAC mass to corresponding bare mass with
    uncertainty.

    Process:
        1. Load PCAC data
        2. Group by parameters (exclude Bare_mass)
        3. Linear fit: PCAC_mass = a * Bare_mass + b
        4. Invert: Bare_mass = (PCAC_ref - b) / a
        5. Return as gvar object with proper uncertainty propagation
    """
    logger.info(f"Loading PCAC data from: {pcac_csv_path}")

    # Load and prepare PCAC data
    pcac_df = _load_and_prepare_pcac_data(pcac_csv_path, logger)

    # Create DataPlotter for PCAC analysis
    pcac_plotter = _create_pcac_plotter(pcac_df, plots_directory, logger)

    # Perform PCAC fitting
    fit_results = _perform_pcac_fitting(pcac_plotter, logger)

    # Invert fit to get target bare mass
    reference_pcac_mass = get_reference_pcac_mass()
    target_bare_mass_gvar = _invert_pcac_fit(fit_results, reference_pcac_mass, logger)

    logger.info(f"Derived target bare mass: {target_bare_mass_gvar}")
    return target_bare_mass_gvar


def _load_and_prepare_pcac_data(pcac_csv_path: str, logger) -> pd.DataFrame:
    """
    Load and prepare PCAC data for analysis.

    Similar to load_and_prepare_data() but for PCAC input format.
    """
    logger.info(f"Loading PCAC data from: {pcac_csv_path}")

    # Load CSV data using existing infrastructure
    try:
        df = load_csv(pcac_csv_path)
        logger.info(f"Loaded {len(df)} PCAC measurements")
    except Exception as e:
        logger.error(f"Failed to load PCAC CSV: {e}")
        raise

    # Validate required PCAC columns
    pcac_config = get_pcac_config()
    required_columns = [
        pcac_config["input_columns"]["bare_mass"],
        pcac_config["input_columns"]["pcac_mass_mean"],
        pcac_config["input_columns"]["pcac_mass_error"],
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required PCAC columns: {missing_columns}")

    # Create error-bar ready format (mean, error) tuples
    # TODO: This needs to change in the future
    pcac_mass_col = pcac_config["plotting"]["y_variable"]  # "plateau_PCAC_mass"
    df[pcac_mass_col] = list(
        zip(
            df[pcac_config["input_columns"]["pcac_mass_mean"]],
            df[pcac_config["input_columns"]["pcac_mass_error"]],
        )
    )

    logger.info(f"PCAC data preparation complete: {len(df)} measurements ready")
    return df


def _create_pcac_plotter(
    df: pd.DataFrame, plots_directory: Path, logger
) -> DataPlotter:
    """
    Create DataPlotter configured for PCAC analysis.

    Similar to create_cost_plotter() but with PCAC configuration.
    """
    logger.info("Creating DataPlotter for PCAC analysis...")

    # Get configurations
    shared_config = get_shared_config()
    pcac_config = get_pcac_config()

    # Create PCAC subdirectory
    base_subdir = shared_config["base_subdirectory"]
    pcac_plots_directory = plots_directory / base_subdir

    # Create DataPlotter instance
    plotter = DataPlotter(df, str(pcac_plots_directory))

    # Set plot variables from PCAC config
    x_var = pcac_config["plotting"]["x_variable"]  # "Bare_mass"
    y_var = pcac_config["plotting"]["y_variable"]  # "PCAC_mass"
    plotter.set_plot_variables(x_var, y_var)

    logger.info(f"PCAC mass DataPlotter configured: {x_var} vs {y_var}")
    logger.info(
        f"Multivalued parameters: {plotter.list_of_multivalued_tunable_parameter_names}"
    )
    logger.info(
        f"Single-valued parameters: {plotter.list_of_single_valued_tunable_parameter_names}"
    )

    return plotter


def _perform_pcac_fitting(pcac_plotter: DataPlotter, logger) -> Dict[str, Any]:
    """
    Perform PCAC fitting using DataPlotter with linear curve fitting.
    """
    logger.info("Performing PCAC fitting with linear function...")

    # Get PCAC plotting configuration
    pcac_config = get_pcac_config()
    shared_config = get_shared_config()
    plotting_config = pcac_config["plotting"]
    validation_config = shared_config["data_validation"]

    # Create custom extrapolation line function for PCAC plots
    def add_pcac_extrapolation_lines(ax, fit_results=None, **kwargs):
        return _add_pcac_extrapolation_lines(ax, fit_results, **kwargs)

    # Perform plotting with curve fitting
    pcac_plotter.plot(
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
        fit_function=plotting_config["fit_function"],  # "linear"
        show_fit_parameters_on_plot=plotting_config["show_fit_parameters"],
        fit_label_location=plotting_config["fit_label_location"],
        fit_min_data_points=validation_config["min_data_points_for_fitting"],
        # Output
        save_figure=True,
        verbose=plotting_config["verbose"],
        # Advanced
        post_plot_customization_function=add_pcac_extrapolation_lines,
    )

    # Extract and return fit results
    fit_results = pcac_plotter.get_fit_results()
    logger.info(f"PCAC fitting completed for {len(fit_results)} parameter groups")
    return fit_results


def _add_pcac_extrapolation_lines(ax, fit_results=None, **kwargs):
    """Add extrapolation lines to PCAC plots."""
    # Add zero reference lines first (simple and always works)
    ax.axhline(0, color="black", linestyle="-", alpha=1.0, linewidth=1.4, zorder=1)
    ax.axvline(0, color="black", linestyle="-", alpha=1.0, linewidth=1.4, zorder=1)

    if not fit_results:
        return

    # Get configuration
    shared_config = get_shared_config()
    pcac_config = get_pcac_config()
    reference_pcac_mass = get_reference_pcac_mass()

    # Get line styles
    extrapolation_lines = shared_config["extrapolation_lines"]
    v_style = extrapolation_lines["vertical_line_style"]
    h_style = extrapolation_lines["horizontal_line_style"]
    band_style = extrapolation_lines["uncertainty_band_style"]

    # Get labels
    labels = pcac_config["extrapolation_labels"]
    v_label = labels["vertical_line_label"]
    h_label = labels["horizontal_line_label"]

    # Calculate derived bare mass
    try:
        # fit_results is now a single fit result, not a dict
        derived_bare_mass_gvar = _invert_single_pcac_fit(
            fit_results, reference_pcac_mass
        )

        # Extract values
        if hasattr(derived_bare_mass_gvar, "mean"):
            derived_bare_mass = float(derived_bare_mass_gvar.mean)
            uncertainty = float(derived_bare_mass_gvar.sdev)
        else:
            derived_bare_mass = float(derived_bare_mass_gvar)
            uncertainty = 0.0

        # Draw lines
        ax.axhline(
            reference_pcac_mass, label=f"{h_label} = {reference_pcac_mass}", **h_style
        )

        if uncertainty > 0:
            v_label_text = f"{v_label} = {gvar.gvar(derived_bare_mass, uncertainty)}"
        else:
            v_label_text = f"{v_label} = {derived_bare_mass:.6f}"

        ax.axvline(derived_bare_mass, label=v_label_text, **v_style)

        # Add uncertainty band
        if uncertainty > 0:
            band_color = band_style.get("color") or v_style.get("color", "green")
            ax.axvspan(
                derived_bare_mass - uncertainty,
                derived_bare_mass + uncertainty,
                alpha=band_style["alpha"],
                color=band_color,
            )

        ax.legend()

    except Exception as e:
        # Log the error instead of silently ignoring it
        print(f"Warning: Could not add PCAC extrapolation lines: {e}")


def _invert_single_pcac_fit(fit_data, reference_pcac_mass):
    """Invert a single PCAC fit result."""
    if not fit_data or fit_data.get("function") != "linear":
        raise ValueError("PCAC fit must be linear for inversion")

    params = fit_data["parameters"]
    method = fit_data.get("method", "scipy")

    if method == "gvar":
        a, b = params[0], params[1]
    else:
        a, b = gvar.gvar(params[0], 0), gvar.gvar(params[1], 0)

    if abs(gvar.mean(a)) < 1e-10:
        raise ValueError("PCAC fit slope too close to zero")

    return (reference_pcac_mass - b) / a


def _invert_pcac_fit(
    fit_results: Dict[str, Any], reference_pcac_mass: float, logger
) -> gvar.GVar:
    """
    Invert PCAC linear fit to get target bare mass from reference PCAC
    mass.

    For linear fit: PCAC_mass = a * bare_mass + b Inversion: bare_mass =
    (PCAC_mass - b) / a

    Uses gvar for automatic uncertainty propagation.
    """
    if logger:
        logger.info("Inverting PCAC fit to derive target bare mass...")

    # For individual plots, we expect a single fit result or use the
    # first available
    if isinstance(fit_results, dict) and len(fit_results) > 0:
        # Get the first (and likely only) fit result
        fit_data = next(iter(fit_results.values()))
    else:
        raise ValueError("No valid PCAC fit results available for inversion")

    if not fit_data or fit_data.get("function") != "linear":
        raise ValueError("PCAC fit must be linear for inversion")

    # Extract linear parameters: y = a*x + b
    params = fit_data["parameters"]
    method = fit_data.get("method", "scipy")

    if method == "gvar":
        # gvar parameters already have uncertainty
        a, b = params[0], params[1]  # slope, intercept
    else:
        # scipy parameters - no uncertainty available
        a, b = gvar.gvar(params[0], 0), gvar.gvar(params[1], 0)

    # Validate slope
    if abs(gvar.mean(a)) < 1e-10:
        raise ValueError("PCAC fit slope too close to zero for reliable inversion")

    # Invert: bare_mass = (pcac_mass - b) / a
    reference_bare_mass = (reference_pcac_mass - b) / a

    if logger:
        logger.info(
            f"Inversion successful: PCAC mass {reference_pcac_mass} "
            f"→ bare mass {reference_bare_mass}"
        )

    return reference_bare_mass


# =============================================================================
# UNIFIED COST ANALYSIS
# =============================================================================


def _perform_cost_analysis(
    processed_csv_path: str,
    reference_bare_mass: Union[float, gvar.GVar],
    output_directory: Path,
    plots_directory: Path,
    logger,
) -> Dict[str, Any]:
    """
    Unified cost analysis that works with both float and gvar targets.
    Handles uncertainty propagation automatically when target is gvar.
    """
    logger.info("Performing cost analysis with unified target handling...")

    # Load and prepare cost data (REUSE existing function)
    df = load_and_prepare_data(processed_csv_path, logger)

    # Create cost plotter (REUSE existing function)
    plotter = create_cost_plotter(df, plots_directory, logger)

    # Perform cost fitting (REUSE existing infrastructure with custom extrapolation lines)
    fit_results = _perform_cost_fitting(plotter, reference_bare_mass, logger)

    # Export results
    _export_results(
        fit_results,
        plotter,
        reference_bare_mass,
        output_directory,
        logger,
    )

    return fit_results


def _perform_cost_fitting(
    plotter: DataPlotter, reference_bare_mass: Union[float, gvar.GVar], logger
) -> Dict[str, Any]:
    """
    Perform cost fitting with custom target for extrapolation lines.
    """
    logger.info("Performing cost fitting with custom target...")

    # Get cost plotting configuration
    cost_config = get_cost_config()
    shared_config = get_shared_config()
    plotting_config = cost_config["plotting"]
    validation_config = shared_config["data_validation"]

    # Create custom extrapolation line function
    def add_cost_extrapolation_lines(ax, fit_results=None, **kwargs):
        return _add_cost_extrapolation_lines(
            ax, fit_results, reference_bare_mass, **kwargs
        )

    # Perform plotting with curve fitting
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
        fit_function=plotting_config["fit_function"],  # "shifted_power_law"
        show_fit_parameters_on_plot=plotting_config["show_fit_parameters"],
        fit_label_location=plotting_config["fit_label_location"],
        fit_min_data_points=validation_config["min_data_points_for_fitting"],
        # Output
        save_figure=True,
        verbose=plotting_config["verbose"],
        # Advanced
        post_plot_customization_function=add_cost_extrapolation_lines,
    )

    # Extract and return fit results
    fit_results = plotter.get_fit_results()
    logger.info(f"Cost fitting completed for {len(fit_results)} parameter groups")
    return fit_results


def _add_cost_extrapolation_lines(
    ax, fit_results=None, reference_bare_mass=None, **kwargs
):
    """
    Add extrapolation lines to cost plots with custom target.
    """
    if not fit_results:
        return

    # Get reference value
    if reference_bare_mass is not None:
        if hasattr(reference_bare_mass, "mean"):  # gvar object
            reference_value = float(reference_bare_mass.mean)
        else:  # float
            reference_value = float(reference_bare_mass)
    else:
        # Fallback to config (backward compatibility)
        reference_value = get_extrapolation_config()["reference_bare_mass"]

    # Calculate extrapolated cost with uncertainty
    extrapolated_result = _calculate_extrapolation(fit_results, reference_value)
    if extrapolated_result is None:
        return

    # Extract value and uncertainty
    if hasattr(extrapolated_result, "mean"):  # gvar object
        extrapolated_cost = float(extrapolated_result.mean)
        uncertainty = float(extrapolated_result.sdev)
    else:  # float
        extrapolated_cost = float(extrapolated_result)
        uncertainty = 0.0

    # Get configuration
    shared_config = get_shared_config()
    cost_config = get_cost_config()
    extrapolation_lines = shared_config["extrapolation_lines"]
    v_style = extrapolation_lines["vertical_line_style"]
    h_style = extrapolation_lines["horizontal_line_style"]
    band_style = extrapolation_lines["uncertainty_band_style"]

    # Get cost-specific labels
    labels = cost_config["extrapolation_labels"]
    v_label = labels["vertical_line_label"]  # Reference bare mass
    h_label = labels["horizontal_line_label"]  # Extrapolated cost

    # Create combined labels
    v_label_text = f"{v_label} = {reference_value}"
    if uncertainty > 0:
        h_label_text = (
            f"{h_label} = {gvar.gvar(extrapolated_cost, uncertainty)} core-hours"
        )
    else:
        h_label_text = f"{h_label} = {extrapolated_cost:.2f} core-hours"

    # Draw extrapolation lines
    ax.axvline(reference_value, label=v_label_text, **v_style)
    ax.axhline(extrapolated_cost, label=h_label_text, **h_style)

    # Add uncertainty band for horizontal line if uncertainty exists
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


def _export_results(
    fit_results: Dict[Any, Any],
    plotter: DataPlotter,
    reference_bare_mass: Union[float, gvar.GVar],
    output_directory: Path,
    logger,
) -> pd.DataFrame:
    """
    Export results with custom reference bare mass.
    """
    shared_config = get_shared_config()
    output_config = shared_config["output"]
    csv_filename = output_config["csv_filename"]

    return export_results(
        fit_results=fit_results,
        plotter=plotter,
        output_directory=output_directory,
        csv_filename=csv_filename,
        logger=logger,
        reference_bare_mass=reference_bare_mass,
    )


# =============================================================================
# EXISTING FUNCTIONS (REUSED WITH MINIMAL MODIFICATIONS)
# =============================================================================


def load_and_prepare_data(csv_path: str, logger) -> pd.DataFrame:
    """
    Load and prepare processed parameters data for cost extrapolation.

    UNCHANGED: Reused from original implementation.
    """
    logger.info(f"Loading data from: {csv_path}")

    # Load CSV data - load_csv handles missing and empty cells
    try:
        df = load_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV")
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        raise

    # Validate required columns exist (use cost analysis config)
    cost_config = get_cost_config()
    required_columns = [
        cost_config["input_columns"]["bare_mass"],
        cost_config["input_columns"]["configuration_label"],
        cost_config["input_columns"]["core_hours_per_spinor"],
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

    UNCHANGED: Reused from original implementation.
    """
    logger.info("Averaging computational costs across configurations...")

    # Create DataFrameAnalyzer for automatic parameter detection
    analyzer = DataFrameAnalyzer(df)

    # Get averaging configuration
    averaging_config = get_averaging_config()
    shared_config = get_shared_config()
    cost_config = get_cost_config()

    # Filter out specified parameters for averaging
    filter_params = averaging_config["filter_out_parameters"]

    # Get core hours column name
    core_hours_col = cost_config["input_columns"]["core_hours_per_spinor"]

    # Group by multivalued parameters (excluding filtered ones)
    grouped = analyzer.group_by_multivalued_tunable_parameters(
        filter_out_parameters_list=filter_params, verbose=False
    )

    # Process each group
    result_rows = []
    min_configs = shared_config["data_validation"]["min_data_points_for_averaging"]

    for group_keys, group_df in grouped:
        # Get core hours data for this group
        core_hours_series = group_df[core_hours_col]
        count = len(core_hours_series)

        # Skip groups with insufficient data
        if count < min_configs:
            logger.warning(
                f"Skipping group {group_keys}: only {count} configurations (minimum: {min_configs})"
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


def create_cost_plotter(df: pd.DataFrame, plots_directory: Path, logger) -> DataPlotter:
    """
    Create and configure DataPlotter for cost extrapolation.

    UNCHANGED: Reused from original implementation.
    """
    logger.info("Creating DataPlotter for cost extrapolation...")

    # Get configuration
    shared_config = get_shared_config()
    cost_config = get_cost_config()

    # Create base subdirectory for cost extrapolation
    base_subdir = shared_config["base_subdirectory"]
    cost_plots_directory = plots_directory / base_subdir
    cost_plots_directory.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created base plots directory: {cost_plots_directory}")

    # Create DataPlotter instance
    plotter = DataPlotter(df, str(cost_plots_directory))

    # Set plot variables from configuration
    x_var = cost_config["plotting"]["x_variable"]
    y_var = cost_config["plotting"]["y_variable"]
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


def _calculate_extrapolation(fit_result, x_reference):
    """
    Calculate extrapolated y-value with uncertainty propagation.

    UNCHANGED: Reused from original implementation with function
    dispatch table.
    """
    if not fit_result:
        return None

    # Extract parameters
    params = fit_result["parameters"]

    # Function dispatch table
    function_map = {
        "linear": lambda a, b, *_: a * x_reference + b,
        "exponential": lambda a, b, c, *_: a * np.exp(-b * x_reference) + c,
        "power_law": lambda a, b, *_: a * (x_reference**b) if x_reference > 0 else None,
        "shifted_power_law": lambda a, b, c, *_: _safe_shifted_power_law(
            a, b, c, x_reference
        ),
    }

    function_type = fit_result["function"]
    if function_type not in function_map:
        return None

    try:
        return function_map[function_type](*params)
    except (IndexError, ValueError, ZeroDivisionError):
        return None


def _safe_shifted_power_law(a, b, c, x_reference):
    """Handle shifted power law with gvar-safe abs()."""
    # Extract mean value for comparison (works for both float and gvar)
    b_mean = b.mean if hasattr(b, "mean") else b

    if abs(x_reference - b_mean) < 1e-10:
        return None
    return a / (x_reference - b) + c


def export_results(
    fit_results: Dict[Any, Any],
    plotter: DataPlotter,
    output_directory: Path,
    csv_filename: str,
    logger,
    reference_bare_mass: Optional[Union[float, gvar.GVar]] = None,  # NEW parameter
) -> pd.DataFrame:
    """
    Export extrapolation results to CSV file.
    """
    logger.info("Exporting extrapolation results...")

    if not fit_results:
        logger.warning("No fit results to export")
        return pd.DataFrame()

    # Get clean summary from DataPlotter
    summary_df = plotter.get_summary_dataframe(
        include_fit_results=True, include_data_counts=True
    )

    if summary_df.empty:
        logger.warning("No summary data to export")
        return summary_df

    # Remove unwanted columns
    columns_to_drop = ["fit_function", "fit_method"]
    summary_df = summary_df.drop(
        columns=[col for col in columns_to_drop if col in summary_df.columns]
    )

    # Get reference value
    if reference_bare_mass is not None:
        # Use getattr() with fallback for type checker compatibility
        reference_value = float(
            getattr(reference_bare_mass, "mean", reference_bare_mass)
        )
    else:
        reference_value = get_extrapolation_config()["reference_bare_mass"]

    # Add domain-specific extrapolation columns
    summary_df["reference_bare_mass"] = reference_value

    # Calculate extrapolated cost for each row with proper tuple handling
    extrapolated_costs = []
    for _, row in summary_df.iterrows():
        group_keys = tuple(
            row[param]
            for param in plotter.reduced_multivalued_tunable_parameter_names_list
        )
        if len(group_keys) == 1:
            group_keys = group_keys[0]

        fit_data = fit_results.get(group_keys)
        if fit_data:
            result = _calculate_extrapolation(fit_data, reference_value)
            if result is not None:
                if hasattr(result, "mean"):  # gvar object
                    extrapolated_cost = (float(result.mean), float(result.sdev))
                else:  # float
                    extrapolated_cost = (float(result), 0.0)
            else:
                extrapolated_cost = None
        else:
            extrapolated_cost = None

        extrapolated_costs.append(extrapolated_cost)

    summary_df["extrapolated_cost"] = extrapolated_costs

    # Round numeric values
    shared_config = get_shared_config()
    float_precision = shared_config["output"]["float_precision"]
    numeric_columns = summary_df.select_dtypes(include=[np.number]).columns
    summary_df[numeric_columns] = summary_df[numeric_columns].round(float_precision)

    # Export to CSV
    csv_path = output_directory / csv_filename
    try:
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Exported {len(summary_df)} results to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to export results: {e}")
        raise

    return summary_df
