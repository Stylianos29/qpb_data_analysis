#!/usr/bin/env python3
"""
Core utilities for critical mass extrapolation.

This module provides shared functions for calculating critical bare mass
values through linear extrapolation to the chiral limit where plateau
mass = 0.
"""

from typing import Dict, List, Optional, Any, Tuple, Mapping

import numpy as np
import pandas as pd
import gvar as gv
import lsqfit

from library.data.analyzer import DataFrameAnalyzer
from library.data import load_csv
from src.analysis.critical_mass_extrapolation._critical_mass_shared_config import (
    GROUPING_EXCLUDED_PARAMETERS,
    OUTPUT_COLUMN_NAMES,
    FILTERING_PARAMETERS,
)

# Get the minimum data points from config
MIN_DATA_POINTS = FILTERING_PARAMETERS["min_data_points_per_group"]

# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


class InsufficientDataError(Exception):
    """
    Raised when there is insufficient data for critical mass analysis.

    This is not a programming error but a data limitation - for example,
    when all parameter groups have fewer than the minimum required bare
    mass values for linear extrapolation.
    """

    pass


# =============================================================================
# LOW-LEVEL UTILITY FUNCTIONS
# =============================================================================


def linear_function(x: np.ndarray, p: List) -> np.ndarray:
    """Linear function for fitting: y = p[0] * x + p[1]."""
    return p[0] * x + p[1]


def quadratic_function(x: np.ndarray, p: List) -> np.ndarray:
    """Quadratic function for fitting: y = p[0] * x^2 + p[1] * x +
    p[2]."""
    return p[0] * x**2 + p[1] * x + p[2]


def safe_divide(
    numerator: np.ndarray, denominator: np.ndarray, default: float = np.nan
) -> np.ndarray:
    """Safe division with fallback for zero or invalid denominators."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, numerator / denominator, default)
    return result


# =============================================================================
# FITTING RANGE FUNCTIONS
# =============================================================================


def filter_data_by_fit_range(
    df: pd.DataFrame,
    bare_mass_col: str,
    range_min: Optional[float],
    range_max: Optional[float],
) -> Tuple[pd.DataFrame, Tuple[Optional[float], Optional[float]]]:
    """
    Filter dataframe to fitting range and return actual discrete bounds
    used.

    Args:
        df: Input dataframe bare_mass_col: Name of bare mass column
        range_min: Minimum bound (None = use data minimum) range_max:
        Maximum bound (None = use data maximum)

    Returns:
        Tuple of (filtered_df, (actual_min, actual_max)) where
        actual_min/max are the discrete data values included in the
        fitting range
    """
    filtered_df = df.copy()

    # Apply user-specified constraints
    if range_min is not None:
        filtered_df = filtered_df[filtered_df[bare_mass_col] >= range_min]
    if range_max is not None:
        filtered_df = filtered_df[filtered_df[bare_mass_col] <= range_max]

    # Extract actual discrete bounds from filtered data
    if len(filtered_df) > 0:
        actual_min = float(filtered_df[bare_mass_col].min())
        actual_max = float(filtered_df[bare_mass_col].max())
    else:
        # Will be caught by validation later
        actual_min, actual_max = None, None

    return filtered_df, (actual_min, actual_max)


# =============================================================================
# FITTING AND ANALYSIS FUNCTIONS
# =============================================================================


def perform_linear_fit(x_data: np.ndarray, y_data: np.ndarray) -> lsqfit.nonlinear_fit:
    """
    Perform linear fit using gvar/lsqfit.

    Args:
        x_data: Independent variable data y_data: Dependent variable
        data as gvar objects

    Returns:
        lsqfit fit result object
    """
    # Initial parameter guess: [slope, intercept]
    p0 = [1.0, 0.0]

    # Perform nonlinear fit using lsqfit
    fit_result = lsqfit.nonlinear_fit(data=(x_data, y_data), fcn=linear_function, p0=p0)

    return fit_result


def perform_quadratic_fit(
    x_data: np.ndarray, y_data: np.ndarray, linear_fit_result: lsqfit.nonlinear_fit
) -> lsqfit.nonlinear_fit:
    """
    Perform quadratic fit using gvar/lsqfit with data-driven initial
    guess.

    Args:
        x_data: Independent variable data y_data: Dependent variable
        data as gvar objects linear_fit_result: Result from linear fit
        for initial guess

    Returns:
        lsqfit fit result object
    """
    # Extract linear fit parameters
    linear_slope = gv.mean(linear_fit_result.p[0])
    linear_intercept = gv.mean(linear_fit_result.p[1])

    # Calculate data-driven initial guess for quadratic coefficient
    x_range = np.max(x_data) - np.min(x_data)
    a_guess = abs(linear_slope) / x_range * 0.1  # 10% of linear slope per unit x

    # Initial parameter guess: [a, b, c] = [quadratic, linear, constant]
    p0 = [a_guess, linear_slope, linear_intercept]

    # Perform nonlinear fit using lsqfit
    fit_result = lsqfit.nonlinear_fit(
        data=(x_data, y_data), fcn=quadratic_function, p0=p0
    )

    return fit_result


def calculate_critical_mass_from_fit(
    fit_result: lsqfit.nonlinear_fit,
) -> Optional[gv.GVar]:
    """
    Calculate critical mass from linear fit result.

    Args:
        fit_result: lsqfit result object from linear fit

    Returns:
        Critical mass as gvar object, or None if calculation fails
    """
    slope = fit_result.p[0]
    intercept = fit_result.p[1]

    # Critical mass where y = 0: x_crit = -intercept / slope
    if gv.mean(slope) == 0:
        return None

    critical_mass = -intercept / slope
    return critical_mass


def calculate_critical_mass_from_quadratic_fit(
    quadratic_fit_result: lsqfit.nonlinear_fit, linear_critical_mass: gv.GVar
) -> Optional[gv.GVar]:
    """
    Calculate critical mass from quadratic fit by selecting root closest
    to linear result.

    Args:
        quadratic_fit_result: lsqfit result object from quadratic fit
        linear_critical_mass: Critical mass from linear fit for root
        selection

    Returns:
        Critical mass as gvar object, or None if calculation fails
    """
    a = quadratic_fit_result.p[0]
    b = quadratic_fit_result.p[1]
    c = quadratic_fit_result.p[2]

    # Solve ax² + bx + c = 0 using quadratic formula
    discriminant = b**2 - 4 * a * c

    if gv.mean(discriminant) < 0:
        return None  # No real roots

    sqrt_discriminant = gv.sqrt(discriminant)
    root1 = (-b + sqrt_discriminant) / (2 * a)
    root2 = (-b - sqrt_discriminant) / (2 * a)

    # Select root closer to linear critical mass
    dist1 = abs(gv.mean(root1) - gv.mean(linear_critical_mass))
    dist2 = abs(gv.mean(root2) - gv.mean(linear_critical_mass))

    return root1 if dist1 < dist2 else root2


def calculate_fit_quality_metrics(
    fit_result: lsqfit.nonlinear_fit, x_data: np.ndarray, y_data: np.ndarray
) -> Dict[str, float]:
    """
    Calculate R², reduced chi², and Q-value for fit quality assessment.

    Args:
        fit_result: lsqfit result object x_data: Independent variable
        data y_data: Dependent variable data as gvar objects

    Returns:
        Dictionary with quality metrics
    """
    # Extract fit function based on number of parameters
    if len(fit_result.p) == 2:
        fcn = linear_function
    elif len(fit_result.p) == 3:
        fcn = quadratic_function
    else:
        raise ValueError(f"Unexpected number of parameters: {len(fit_result.p)}")

    # Calculate predicted values
    y_pred = fcn(x_data, fit_result.p)

    # Calculate R²
    y_mean_vals = gv.mean(y_data)
    ss_tot = np.sum((y_mean_vals - np.mean(y_mean_vals)) ** 2)
    ss_res = np.sum((y_mean_vals - gv.mean(y_pred)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Get chi² metrics from fit result
    chi2_reduced = fit_result.chi2 / fit_result.dof if fit_result.dof > 0 else np.inf
    Q = fit_result.Q

    return {
        "r_squared": float(r_squared),
        "chi2_reduced": float(chi2_reduced),
        "Q": float(Q),
    }


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def reorder_columns_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder DataFrame columns to put physics parameters first.

    Args:
        df: Input dataframe

    Returns:
        Dataframe with reordered columns
    """
    preferred_order = [
        "Overlap_operator_method",
        "Kernel_operator_type",
        "KL_diagonal_order",
        "Number_of_Chebyshev_terms",
    ]

    # Get existing preferred columns in order
    existing_preferred = [col for col in preferred_order if col in df.columns]

    # Get remaining columns
    remaining_cols = [col for col in df.columns if col not in existing_preferred]

    # Create final column order
    final_order = existing_preferred + remaining_cols

    return df[final_order]


def export_results_to_csv(results: List[Dict], output_csv_path: str) -> str:
    """
    Export critical mass results to CSV file.

    Args:
        results: List of result dictionaries output_csv_path: Path for
        output CSV file

    Returns:
        Path to exported CSV file
    """
    df = pd.DataFrame(results)

    # Reorder columns with physics parameters first
    df = reorder_columns_for_export(df)

    df.to_csv(output_csv_path, index=False)

    return output_csv_path


# =============================================================================
# GROUP PROCESSING FUNCTIONS
# =============================================================================


def calculate_critical_mass_for_group(
    group_df: pd.DataFrame,
    column_mapping: Dict[str, str],
    plateau_mass_power: int,
    enable_quadratic_fit: bool = False,
    fit_range_config: Optional[Mapping[str, Mapping[str, Optional[float]]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Calculate critical mass for a parameter group.

    Args:
        group_df: DataFrame for this parameter group column_mapping:
        Mapping of standard names to CSV columns plateau_mass_power:
        Power to raise plateau mass (1 for PCAC, 2 for pion)
        enable_quadratic_fit: Whether to perform quadratic fit
        fit_range_config: Dictionary with "linear" and "quadratic" range
        configs

    Returns:
        Dictionary with results, or None if calculation fails
    """
    # Get column names from config
    mass_col = column_mapping["bare_mass"]
    y_mean_col = column_mapping["plateau_mean"]
    y_error_col = column_mapping["plateau_error"]

    # === LINEAR FIT === Apply linear fit range filter
    if fit_range_config is not None:
        linear_range = fit_range_config["linear"]
        linear_df, (linear_min, linear_max) = filter_data_by_fit_range(
            group_df,
            mass_col,
            linear_range["bare_mass_min"],
            linear_range["bare_mass_max"],
        )
    else:
        # No filtering, use all data
        linear_df = group_df
        linear_min = float(group_df[mass_col].min())
        linear_max = float(group_df[mass_col].max())

    # Extract data for linear fitting
    x_data_linear = linear_df[mass_col].to_numpy()
    y_mean_linear = linear_df[y_mean_col].to_numpy()
    y_error_linear = linear_df[y_error_col].to_numpy()
    y_data_linear = gv.gvar(y_mean_linear, y_error_linear)
    y_means_transformed_linear = y_data_linear**plateau_mass_power

    # Perform linear fit
    fit_result = perform_linear_fit(x_data_linear, y_means_transformed_linear)
    quality_metrics = calculate_fit_quality_metrics(
        fit_result, x_data_linear, y_means_transformed_linear
    )
    critical_mass = calculate_critical_mass_from_fit(fit_result)

    if critical_mass is None:
        return None

    # Build core results dictionary
    result = {
        OUTPUT_COLUMN_NAMES["critical_mass_mean"]: gv.mean(critical_mass),
        OUTPUT_COLUMN_NAMES["critical_mass_error"]: gv.sdev(critical_mass),
        OUTPUT_COLUMN_NAMES["slope_mean"]: gv.mean(fit_result.p[0]),
        OUTPUT_COLUMN_NAMES["slope_error"]: gv.sdev(fit_result.p[0]),
        OUTPUT_COLUMN_NAMES["intercept_mean"]: gv.mean(fit_result.p[1]),
        OUTPUT_COLUMN_NAMES["intercept_error"]: gv.sdev(fit_result.p[1]),
        OUTPUT_COLUMN_NAMES["n_data_points"]: len(x_data_linear),
        OUTPUT_COLUMN_NAMES["r_squared"]: quality_metrics["r_squared"],
        OUTPUT_COLUMN_NAMES["chi2_reduced"]: quality_metrics["chi2_reduced"],
        OUTPUT_COLUMN_NAMES["fit_quality"]: quality_metrics["Q"],
        "fit_range_min": linear_min,
        "fit_range_max": linear_max,
    }

    # === QUADRATIC FIT ===
    if enable_quadratic_fit:
        try:
            # Apply quadratic fit range filter
            if fit_range_config is not None:
                quadratic_range = fit_range_config["quadratic"]
                quadratic_df, (quadratic_min, quadratic_max) = filter_data_by_fit_range(
                    group_df,
                    mass_col,
                    quadratic_range["bare_mass_min"],
                    quadratic_range["bare_mass_max"],
                )
            else:
                # No filtering, use all data
                quadratic_df = group_df
                quadratic_min = float(group_df[mass_col].min())
                quadratic_max = float(group_df[mass_col].max())

            # Extract data for quadratic fitting
            x_data_quad = quadratic_df[mass_col].to_numpy()
            y_mean_quad = quadratic_df[y_mean_col].to_numpy()
            y_error_quad = quadratic_df[y_error_col].to_numpy()
            y_data_quad = gv.gvar(y_mean_quad, y_error_quad)
            y_means_transformed_quad = y_data_quad**plateau_mass_power

            # Perform quadratic fit
            quad_fit_result = perform_quadratic_fit(
                x_data_quad, y_means_transformed_quad, fit_result
            )
            quad_quality_metrics = calculate_fit_quality_metrics(
                quad_fit_result, x_data_quad, y_means_transformed_quad
            )
            quad_critical_mass = calculate_critical_mass_from_quadratic_fit(
                quad_fit_result, critical_mass
            )

            # Add quadratic results
            result.update(
                {
                    OUTPUT_COLUMN_NAMES["quadratic_a_mean"]: gv.mean(
                        quad_fit_result.p[0]
                    ),
                    OUTPUT_COLUMN_NAMES["quadratic_a_error"]: gv.sdev(
                        quad_fit_result.p[0]
                    ),
                    OUTPUT_COLUMN_NAMES["quadratic_b_mean"]: gv.mean(
                        quad_fit_result.p[1]
                    ),
                    OUTPUT_COLUMN_NAMES["quadratic_b_error"]: gv.sdev(
                        quad_fit_result.p[1]
                    ),
                    OUTPUT_COLUMN_NAMES["quadratic_c_mean"]: gv.mean(
                        quad_fit_result.p[2]
                    ),
                    OUTPUT_COLUMN_NAMES["quadratic_c_error"]: gv.sdev(
                        quad_fit_result.p[2]
                    ),
                    OUTPUT_COLUMN_NAMES["quadratic_critical_mass_mean"]: (
                        gv.mean(quad_critical_mass) if quad_critical_mass else np.nan
                    ),
                    OUTPUT_COLUMN_NAMES["quadratic_critical_mass_error"]: (
                        gv.sdev(quad_critical_mass) if quad_critical_mass else np.nan
                    ),
                    OUTPUT_COLUMN_NAMES["quadratic_r_squared"]: quad_quality_metrics[
                        "r_squared"
                    ],
                    OUTPUT_COLUMN_NAMES["quadratic_chi2_reduced"]: quad_quality_metrics[
                        "chi2_reduced"
                    ],
                    OUTPUT_COLUMN_NAMES["quadratic_fit_quality"]: quad_quality_metrics[
                        "Q"
                    ],
                    "quadratic_fit_range_min": quadratic_min,
                    "quadratic_fit_range_max": quadratic_max,
                }
            )
        except Exception as e:
            # If quadratic fit fails, add NaN values for all quadratic
            # results
            result.update(
                {
                    OUTPUT_COLUMN_NAMES["quadratic_a_mean"]: np.nan,
                    OUTPUT_COLUMN_NAMES["quadratic_a_error"]: np.nan,
                    OUTPUT_COLUMN_NAMES["quadratic_b_mean"]: np.nan,
                    OUTPUT_COLUMN_NAMES["quadratic_b_error"]: np.nan,
                    OUTPUT_COLUMN_NAMES["quadratic_c_mean"]: np.nan,
                    OUTPUT_COLUMN_NAMES["quadratic_c_error"]: np.nan,
                    OUTPUT_COLUMN_NAMES["quadratic_critical_mass_mean"]: np.nan,
                    OUTPUT_COLUMN_NAMES["quadratic_critical_mass_error"]: np.nan,
                    OUTPUT_COLUMN_NAMES["quadratic_r_squared"]: np.nan,
                    OUTPUT_COLUMN_NAMES["quadratic_chi2_reduced"]: np.nan,
                    OUTPUT_COLUMN_NAMES["quadratic_fit_quality"]: np.nan,
                    "quadratic_fit_range_min": np.nan,
                    "quadratic_fit_range_max": np.nan,
                }
            )

    # Add all single-valued columns automatically
    excluded_cols = {mass_col, y_mean_col, y_error_col}
    for col in group_df.columns:
        if col not in excluded_cols:
            values = group_df[col].unique()
            if len(values) == 1:
                result[col] = values[0]

    return result


def validate_critical_mass_input_data(
    df: pd.DataFrame, analysis_type: str, column_mapping: Dict[str, str], logger
) -> None:
    """
    Validate plateau data for critical mass calculation.

    Args:
        - df: Input dataframe
        - analysis_type: Type of analysis ("pcac" or "pion")
        - column_mapping: Mapping of standard names to CSV columns
        - logger: Logger instance
    """
    # Check sufficient data points
    min_points = FILTERING_PARAMETERS["min_data_points_per_group"]
    if len(df) < min_points:
        raise ValueError(f"Need at least {min_points} data points for extrapolation")

    # Check required columns exist
    y_mean_col = column_mapping["plateau_mean"]
    y_error_col = column_mapping["plateau_error"]

    missing_cols = []
    if y_mean_col not in df.columns:
        missing_cols.append(y_mean_col)
    if y_error_col not in df.columns:
        missing_cols.append(y_error_col)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(
        f"Validated {len(df)} {analysis_type.upper()} plateau data points for group"
    )


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================


def process_critical_mass_analysis(
    input_csv_path: str,
    output_csv_path: str,
    analysis_type: str,
    column_mapping: Dict[str, str],
    required_columns: List[str],
    quadratic_config: Dict[str, Any],
    fit_range_config: Mapping[str, Mapping[str, Optional[float]]],
    plateau_mass_power: int,
    logger,
) -> str:
    """
    Process plateau data to calculate critical mass values.

    Args:
        input_csv_path: Path to input plateau CSV output_csv_path: Path
        for output results CSV analysis_type: Type of analysis ("pcac"
        or "pion") column_mapping: Mapping of standard names to CSV
        columns required_columns: List of required column names
        quadratic_config: Configuration for quadratic fitting
        fit_range_config: Configuration for fitting ranges
        plateau_mass_power: Power to raise plateau mass logger: Logger
        instance

    Returns:
        Path to output CSV file
    """
    # Get quadratic fit configuration from passed parameter
    enable_quadratic_fit = quadratic_config["enable_quadratic_fit"]

    # Load and validate input data using library function
    logger.info(f"Loading {analysis_type.upper()} plateau data")
    df = load_csv(input_csv_path, validate_required_columns=set(required_columns))

    analyzer = DataFrameAnalyzer(df)

    # Filter exclusion list to only include parameters that exist in the
    # list of multivalued parameters
    available_multivalued_params = analyzer.list_of_multivalued_tunable_parameter_names
    filtered_exclusions = [
        param
        for param in GROUPING_EXCLUDED_PARAMETERS
        if param in available_multivalued_params
    ]

    # Log what we're excluding
    if filtered_exclusions:
        logger.info(f"Excluding from grouping: {filtered_exclusions}")
    else:
        logger.info("No parameters to exclude from grouping")

    if enable_quadratic_fit:
        logger.info("Quadratic fitting enabled for validation")
    else:
        logger.info("Quadratic fitting disabled (linear only)")

    # Group data using analyzer's intelligence
    logger.info("Grouping data by lattice parameters")
    grouped_data = analyzer.group_by_multivalued_tunable_parameters(
        filter_out_parameters_list=filtered_exclusions
    )

    # Validate and collect valid groups in one pass
    valid_groups = []
    for group_id, group_df in grouped_data:
        try:
            validate_critical_mass_input_data(
                group_df, analysis_type, column_mapping, logger
            )
            valid_groups.append((group_id, group_df))
        except ValueError as e:
            logger.warning(f"Skipping group {group_id}: {e}")

    if not valid_groups:
        min_points = FILTERING_PARAMETERS["min_data_points_per_group"]
        raise InsufficientDataError(
            "No groups have sufficient data points for analysis. "
            f"Critical mass extrapolation requires at least {min_points} "
            "bare mass values per group."
        )

    logger.info(f"Processing {len(valid_groups)} valid parameter groups")

    # Log the fitting approach based on plateau mass power
    if plateau_mass_power == 1:
        logger.info(
            f"{analysis_type.upper()} analysis: fitting plateau_mass vs bare_mass"
        )
    elif plateau_mass_power == 2:
        logger.info(
            f"{analysis_type.upper()} analysis: fitting plateau_mass² vs bare_mass"
        )
    else:
        logger.info(
            f"{analysis_type.upper()} analysis: "
            f"fitting plateau_mass^{plateau_mass_power} vs bare_mass"
        )

    # Calculate critical mass for each valid group
    results = []
    for group_id, group_df in valid_groups:
        logger.info(f"Processing group: {group_id}")
        try:
            result = calculate_critical_mass_for_group(
                group_df,
                column_mapping,
                plateau_mass_power,
                enable_quadratic_fit,
                fit_range_config,
            )
            if result:
                results.append(result)
        except Exception as e:
            logger.warning(f"Failed to process group {group_id}: {e}")
            continue

    if not results:
        raise InsufficientDataError(
            "No valid critical mass calculations completed. "
            "All groups failed validation or fitting."
        )

    # Export results
    logger.info(f"Exporting {len(results)} results to {output_csv_path}")
    output_path = export_results_to_csv(results, output_csv_path)

    return output_path
