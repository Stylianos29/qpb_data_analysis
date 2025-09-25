#!/usr/bin/env python3
"""
Core utilities for critical mass extrapolation.

This module provides shared functions for calculating critical bare mass
values through linear extrapolation to the chiral limit where plateau
mass = 0.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
import gvar as gv
import lsqfit

from library.data.analyzer import DataFrameAnalyzer
from library.constants import (
    PARAMETERS_WITH_EXPONENTIAL_FORMAT,
    PARAMETERS_OF_INTEGER_VALUE,
)
from src.analysis.critical_mass_extrapolation._critical_mass_shared_config import (
    GROUPING_EXCLUDED_PARAMETERS,
)
from library.data import load_csv


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


def process_critical_mass_analysis(
    input_csv_path: str,
    output_csv_path: str,
    analysis_type: str,
    required_columns: List[str],
    logger,
) -> str:
    """Process plateau data to calculate critical mass values."""
    # Get config based on analysis type
    if analysis_type == "pcac":
        from ._pcac_critical_mass_config import COLUMN_MAPPING
    else:
        from ._pion_critical_mass_config import COLUMN_MAPPING

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
                group_df, analysis_type, COLUMN_MAPPING, logger
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
            result = calculate_critical_mass_for_group(group_df, COLUMN_MAPPING)
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


# =============================================================================
# LOW-LEVEL UTILITY FUNCTIONS
# =============================================================================


def linear_function(x, p):
    """Linear function for fitting: y = p[0] * x + p[1]."""
    return p[0] * x + p[1]


def format_group_id(group_id, grouping_parameters):
    """Format group ID into a readable string."""
    if isinstance(group_id, str) and group_id == "single_group":
        return "single_group"

    # Handle tuple group IDs
    if isinstance(group_id, tuple):
        parts = []
        for i, param in enumerate(grouping_parameters):
            if i < len(group_id):
                value = group_id[i]
                # Clean up numpy types and format nicely
                if hasattr(value, "item"):  # numpy types
                    value = value.item()

                # Create readable parameter abbreviations
                if param == "KL_diagonal_order":
                    parts.append(f"KL{value}")
                elif param == "Kernel_operator_type":
                    parts.append(f"{value}")
                else:
                    parts.append(f"{param}_{value}")
        return "_".join(parts)

    # Fallback for other types
    return str(group_id)


def safe_divide(numerator, denominator, default=np.nan):
    """Safe division with fallback for zero or invalid denominators."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, numerator / denominator, default)
    return result
    """Safe division with fallback for zero or invalid denominators."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, numerator / denominator, default)
    return result


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


# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================


def load_and_validate_plateau_data(csv_path, analysis_type):
    """Load plateau data CSV and perform basic validation."""
    df = pd.read_csv(csv_path)

    # Basic validation
    if df.empty:
        raise ValueError(f"Empty CSV file: {csv_path}")

    return df


def group_data_by_parameters(df, grouping_parameters):
    """Group data by lattice parameters for separate analysis."""
    analyzer = DataFrameAnalyzer(df)

    # Check which grouping parameters exist in the data
    available_params = [param for param in grouping_parameters if param in df.columns]

    if not available_params:
        # If no grouping parameters, treat all data as one group
        return [("single_group", df)]

    # Group by available parameters
    grouped = df.groupby(available_params, observed=True)
    return [(group_id, group_df) for group_id, group_df in grouped]


# =============================================================================
# CRITICAL MASS CALCULATION
# =============================================================================


def perform_linear_fit(x_data, y_data):
    """Perform linear fit with gvar/lsqfit."""
    # Convert to gvar if needed
    if not hasattr(y_data[0], "mean"):
        y_gvar = gv.gvar(y_data, np.zeros_like(y_data))
    else:
        y_gvar = y_data

    # Initial parameter guess
    p0 = [0.0, np.mean(gv.mean(y_gvar))]

    # Perform fit
    fit_result = lsqfit.nonlinear_fit(data=(x_data, y_gvar), fcn=linear_function, p0=p0)

    return fit_result


def calculate_critical_mass_from_fit(fit_result):
    """Calculate critical mass from linear fit: x_critical =
    -intercept/slope."""
    slope = fit_result.p[0]
    intercept = fit_result.p[1]

    # Critical mass where y=0: 0 = slope * x + intercept => x =
    # -intercept/slope
    if gv.mean(slope) != 0:
        critical_mass = -intercept / slope
        return critical_mass
    else:
        return None


def validate_fit_quality(fit_result, quality_metrics, fit_config):
    """Validate fit quality against configured thresholds."""
    validation_results = {
        "quality_passed": True,
        "warnings": [],
    }

    # Check R²
    min_r_squared = fit_config.get("min_r_squared", 0.5)
    if quality_metrics["r_squared"] < min_r_squared:
        validation_results["quality_passed"] = False
        validation_results["warnings"].append(
            f"Low R²: {quality_metrics['r_squared']:.3f} < {min_r_squared}"
        )

    # Check fit probability
    min_q_value = fit_config.get("min_q_value", 0.01)
    if quality_metrics["Q"] < min_q_value:
        validation_results["quality_passed"] = False
        validation_results["warnings"].append(
            f"Low fit probability: {quality_metrics['Q']:.3f} < {min_q_value}"
        )

    return validation_results


def calculate_critical_mass_for_group(
    group_df: pd.DataFrame, column_mapping: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """Calculate critical mass for a parameter group."""

    mass_col = column_mapping["bare_mass"]
    y_mean_col = column_mapping["plateau_mean"]
    y_error_col = column_mapping["plateau_error"]

    # Extract data for fitting
    x_data = group_df[mass_col].values
    y_mean = group_df[y_mean_col].values
    y_error = group_df[y_error_col].values
    y_data = gv.gvar(y_mean, y_error)

    # Perform linear fit (no inner try - let outer try/catch handle it)
    fit_result = perform_linear_fit(x_data, y_data)
    quality_metrics = calculate_fit_quality_metrics(fit_result, x_data, y_data)
    critical_mass = calculate_critical_mass_from_fit(fit_result)

    if critical_mass is None:
        return None

    # Build core results dictionary (no column ordering here)
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

    # Add metadata columns that exist and are single-valued
    excluded_cols = {mass_col, y_mean_col, y_error_col}
    for col in group_df.columns:
        if col not in excluded_cols:  # Remove metadata_columns filter
            values = group_df[col].unique()
            if len(values) == 1:
                result[col] = values[0]

    return result


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================


def reorder_columns_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder DataFrame columns with preferred physics parameters first."""
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


def format_dataframe_for_export(df):
    """Apply proper formatting to DataFrame columns based on parameter
    types."""
    df_formatted = df.copy()

    # Apply exponential formatting
    for col in df_formatted.columns:
        if col in PARAMETERS_WITH_EXPONENTIAL_FORMAT:
            # Format as exponential notation
            df_formatted[col] = df_formatted[col].apply(
                lambda x: f"{float(x):.2e}" if pd.notna(x) else x
            )

    # Apply integer formatting
    for col in df_formatted.columns:
        if col in PARAMETERS_OF_INTEGER_VALUE:
            # Format as integers
            df_formatted[col] = df_formatted[col].apply(
                lambda x: f"{int(x)}" if pd.notna(x) else x
            )

    return df_formatted


def export_results_to_csv(results: List[Dict], output_csv_path: str) -> str:
    """Export critical mass results to CSV file."""
    df = pd.DataFrame(results)

    # Reorder columns with physics parameters first
    df = reorder_columns_for_export(df)

    df.to_csv(output_csv_path, index=False)

    return output_csv_path
