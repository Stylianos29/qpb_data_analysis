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
)


# =============================================================================
# LOW-LEVEL UTILITY FUNCTIONS
# =============================================================================


def linear_function(x, p):
    """Linear function for fitting: y = p[0] * x + p[1]."""
    return p[0] * x + p[1]


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


def calculate_fit_quality_metrics(fit_result, x_data, y_data):
    """Calculate R², reduced chi-squared, and other fit quality
    metrics."""
    y_fit = linear_function(x_data, fit_result.p)

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
    group_df: pd.DataFrame, column_mapping: Dict[str, str]
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

    # Perform linear fit
    fit_result = perform_linear_fit(x_data, y_data)
    quality_metrics = calculate_fit_quality_metrics(fit_result, x_data, y_data)
    critical_mass = calculate_critical_mass_from_fit(fit_result)

    if critical_mass is None:
        return None

    # Build core results dictionary
    result = {
        "critical_mass_mean": gv.mean(critical_mass),
        "critical_mass_error": gv.sdev(critical_mass),
        "slope_mean": gv.mean(fit_result.p[0]),
        "slope_error": gv.sdev(fit_result.p[0]),
        "intercept_mean": gv.mean(fit_result.p[1]),
        "intercept_error": gv.sdev(fit_result.p[1]),
        "n_data_points": len(x_data),
        "r_squared": quality_metrics["r_squared"],
        "chi2_reduced": quality_metrics["chi2_reduced"],
        "fit_quality": quality_metrics["Q"],
    }

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
    logger,
) -> str:
    """Process plateau data to calculate critical mass values."""

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

    # Calculate critical mass for each valid group
    results = []
    for group_id, group_df in valid_groups:
        logger.info(f"Processing group: {group_id}")
        try:
            result = calculate_critical_mass_for_group(group_df, column_mapping)
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
