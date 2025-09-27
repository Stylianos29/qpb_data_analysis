#!/usr/bin/env python3
"""
Core utilities for critical mass extrapolation.

This module provides shared functions for calculating critical bare mass
values through linear extrapolation to the chiral limit where plateau
mass = 0.
"""

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import gvar as gv
import lsqfit

from library.data.analyzer import DataFrameAnalyzer
from library.data import load_csv
from src.analysis.critical_mass_extrapolation._critical_mass_shared_config import (
    GROUPING_EXCLUDED_PARAMETERS,
    OUTPUT_COLUMN_NAMES,
)


# =============================================================================
# LOW-LEVEL UTILITY FUNCTIONS
# =============================================================================


def linear_function(x, p):
    """Linear function for fitting: y = p[0] * x + p[1]."""
    return p[0] * x + p[1]


def quadratic_function(x, p):
    """Quadratic function for fitting: y = p[0] * x^2 + p[1] * x +
    p[2]."""
    return p[0] * x**2 + p[1] * x + p[2]


def safe_divide(numerator, denominator, default=np.nan):
    """Safe division with fallback for zero or invalid denominators."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, numerator / denominator, default)
    return result


# =============================================================================
# FITTING AND ANALYSIS FUNCTIONS
# =============================================================================


def perform_linear_fit(x_data, y_data):
    """Perform linear fit using gvar/lsqfit."""
    # Initial parameter guess: [slope, intercept]
    p0 = [1.0, 0.0]

    # Perform nonlinear fit using lsqfit
    fit_result = lsqfit.nonlinear_fit(data=(x_data, y_data), fcn=linear_function, p0=p0)

    return fit_result


def perform_quadratic_fit(x_data, y_data, linear_fit_result):
    """Perform quadratic fit using gvar/lsqfit with data-driven initial
    guess."""
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


def calculate_critical_mass_from_quadratic_fit(
    quadratic_fit_result, linear_critical_mass
):
    """Calculate critical mass from quadratic fit by selecting root
    closest to linear result."""
    a = quadratic_fit_result.p[0]  # quadratic coefficient
    b = quadratic_fit_result.p[1]  # linear coefficient
    c = quadratic_fit_result.p[2]  # constant term

    # Check if quadratic coefficient is effectively zero
    if abs(gv.mean(a)) < 1e-10:
        # Degenerate to linear case: bx + c = 0 -> x = -c/b
        if gv.mean(b) == 0:
            return None
        return -c / b

    # Calculate discriminant: b² - 4ac
    discriminant = b**2 - 4 * a * c

    # Check for real roots
    if gv.mean(discriminant) < 0:
        # No real roots
        return None

    # Calculate both roots: x = (-b ± √discriminant) / (2a)
    sqrt_discriminant = gv.sqrt(discriminant)
    root1 = (-b + sqrt_discriminant) / (2 * a)
    root2 = (-b - sqrt_discriminant) / (2 * a)

    # Select root closest to linear critical mass
    linear_value = (
        gv.mean(linear_critical_mass) if linear_critical_mass is not None else 0.0
    )

    distance1 = abs(gv.mean(root1) - linear_value)
    distance2 = abs(gv.mean(root2) - linear_value)

    return root1 if distance1 <= distance2 else root2


def calculate_fit_quality_metrics(fit_result, x_data, y_data, fit_function=None):
    """
    Calculate R², reduced chi-squared, and other fit quality metrics.

    Parameters:
        - fit_result: lsqfit result object
        - x_data: independent variable data
        - y_data: dependent variable data (gvar format)
        - fit_function: function used for fitting (defaults to
          linear_function)
    """
    # Use linear function as default if not specified
    if fit_function is None:
        fit_function = linear_function

    # Calculate fitted values
    y_fit = fit_function(x_data, fit_result.p)

    # Calculate residuals and R²
    ss_res = np.sum((gv.mean(y_data) - gv.mean(y_fit)) ** 2)
    ss_tot = np.sum((gv.mean(y_data) - np.mean(gv.mean(y_data))) ** 2)
    r_squared = 1 - safe_divide(ss_res, ss_tot, 0.0)

    # Reduced chi-squared
    dof = len(x_data) - len(fit_result.p)
    chi2_reduced = fit_result.chi2 / dof if dof > 0 else np.inf

    return {
        "r_squared": r_squared,
        "chi2_reduced": chi2_reduced,
        "chi2": fit_result.chi2,
        "dof": dof,
        "Q": fit_result.Q,
    }


def calculate_critical_mass_from_fit(fit_result):
    """Calculate critical mass from linear fit: x_critical =
    -intercept/slope."""
    slope = fit_result.p[0]
    intercept = fit_result.p[1]

    # Check if slope is non-zero
    if gv.mean(slope) == 0:
        return None

    # Calculate critical mass where y = 0: x_critical = -intercept/slope
    critical_mass = -intercept / slope

    return critical_mass


# =============================================================================
# DATA EXPORT FUNCTIONS
# =============================================================================


def reorder_columns_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder DataFrame columns with preferred physics parameters
    first."""
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
    """Export critical mass results to CSV file."""
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
) -> Optional[Dict[str, Any]]:
    """Calculate critical mass for a parameter group."""

    # Get column names from config
    mass_col = column_mapping["bare_mass"]
    y_mean_col = column_mapping["plateau_mean"]
    y_error_col = column_mapping["plateau_error"]

    # Extract data for fitting
    x_data = group_df[mass_col].values
    y_mean = group_df[y_mean_col].values
    y_error = group_df[y_error_col].values
    y_data = gv.gvar(y_mean, y_error)

    y_means_transformed = y_data**plateau_mass_power

    # Perform linear fit
    fit_result = perform_linear_fit(x_data, y_means_transformed)
    quality_metrics = calculate_fit_quality_metrics(
        fit_result, x_data, y_means_transformed
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
        OUTPUT_COLUMN_NAMES["n_data_points"]: len(x_data),
        OUTPUT_COLUMN_NAMES["r_squared"]: quality_metrics["r_squared"],
        OUTPUT_COLUMN_NAMES["chi2_reduced"]: quality_metrics["chi2_reduced"],
        OUTPUT_COLUMN_NAMES["fit_quality"]: quality_metrics["Q"],
    }

    # Perform quadratic fit if enabled
    if enable_quadratic_fit:
        try:
            # Perform quadratic fit using linear results for initial
            # guess
            quadratic_fit_result = perform_quadratic_fit(x_data, y_data, fit_result)

            # Calculate quadratic fit quality metrics
            quadratic_quality_metrics = calculate_fit_quality_metrics(
                quadratic_fit_result, x_data, y_data, quadratic_function
            )

            # Calculate quadratic critical mass
            quadratic_critical_mass = calculate_critical_mass_from_quadratic_fit(
                quadratic_fit_result, critical_mass
            )

            # Add quadratic fit results to output
            result.update(
                {
                    OUTPUT_COLUMN_NAMES["quadratic_a_mean"]: gv.mean(
                        quadratic_fit_result.p[0]
                    ),
                    OUTPUT_COLUMN_NAMES["quadratic_a_error"]: gv.sdev(
                        quadratic_fit_result.p[0]
                    ),
                    OUTPUT_COLUMN_NAMES["quadratic_b_mean"]: gv.mean(
                        quadratic_fit_result.p[1]
                    ),
                    OUTPUT_COLUMN_NAMES["quadratic_b_error"]: gv.sdev(
                        quadratic_fit_result.p[1]
                    ),
                    OUTPUT_COLUMN_NAMES["quadratic_c_mean"]: gv.mean(
                        quadratic_fit_result.p[2]
                    ),
                    OUTPUT_COLUMN_NAMES["quadratic_c_error"]: gv.sdev(
                        quadratic_fit_result.p[2]
                    ),
                    OUTPUT_COLUMN_NAMES[
                        "quadratic_r_squared"
                    ]: quadratic_quality_metrics["r_squared"],
                    OUTPUT_COLUMN_NAMES[
                        "quadratic_chi2_reduced"
                    ]: quadratic_quality_metrics["chi2_reduced"],
                    OUTPUT_COLUMN_NAMES[
                        "quadratic_fit_quality"
                    ]: quadratic_quality_metrics["Q"],
                }
            )

            # Add quadratic critical mass (may be NaN if no real roots)
            if quadratic_critical_mass is not None:
                result.update(
                    {
                        OUTPUT_COLUMN_NAMES["quadratic_critical_mass_mean"]: gv.mean(
                            quadratic_critical_mass
                        ),
                        OUTPUT_COLUMN_NAMES["quadratic_critical_mass_error"]: gv.sdev(
                            quadratic_critical_mass
                        ),
                    }
                )
            else:
                result.update(
                    {
                        OUTPUT_COLUMN_NAMES["quadratic_critical_mass_mean"]: np.nan,
                        OUTPUT_COLUMN_NAMES["quadratic_critical_mass_error"]: np.nan,
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
    """Validate plateau data for critical mass calculation."""

    # Check sufficient data points
    if len(df) < 3:
        raise ValueError("Need at least 3 data points for extrapolation")

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
    plateau_mass_power: int,
    logger,
) -> str:
    """Process plateau data to calculate critical mass values."""

    # Get quadratic fit configuration from passed parameter
    enable_quadratic_fit = quadratic_config["enable_quadratic_fit"]

    # Load and validate input data using library function
    logger.info(f"Loading {analysis_type.upper()} plateau data")
    df = load_csv(
        input_csv_path,
        validate_required_columns=set(required_columns),
        apply_categorical=True,
    )

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
        raise ValueError("No groups have sufficient data points for analysis")

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
                group_df, column_mapping, plateau_mass_power, enable_quadratic_fit
            )
            if result:
                results.append(result)
        except Exception as e:
            logger.warning(f"Failed to process group {group_id}: {e}")
            continue

    if not results:
        raise ValueError("No valid critical mass calculations completed")

    # Export results
    logger.info(f"Exporting {len(results)} results to {output_csv_path}")
    output_path = export_results_to_csv(results, output_csv_path)

    return output_path
