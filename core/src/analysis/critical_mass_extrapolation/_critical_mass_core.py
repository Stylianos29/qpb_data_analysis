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
    GROUPING_PARAMETERS,
)
from library.data import load_csv


def validate_critical_mass_input_data(df, analysis_type, logger):
    """Validate plateau data for critical mass calculation."""
    # load_csv() already handled column validation and empty file checks
    # Only need to check sufficient data points

    if len(df) < 3:
        raise ValueError("Need at least 3 data points for extrapolation")

    logger.info(f"Validated {len(df)} {analysis_type.upper()} plateau data points")


def process_critical_mass_analysis(
    input_csv_path, output_csv_path, analysis_type, required_columns, logger
):
    """Process plateau data to calculate critical mass values."""

    # Load and validate input data using library function
    logger.info(f"Loading {analysis_type.upper()} plateau data")
    df = load_csv(
        input_csv_path,
        validate_required_columns=set(required_columns),
        apply_categorical=True,
    )
    validate_critical_mass_input_data(df, analysis_type, logger)

    # Group data by lattice parameters
    logger.info("Grouping data by lattice parameters")
    grouped_data = group_data_by_parameters(df, GROUPING_PARAMETERS)
    logger.info(f"Processing {len(grouped_data)} parameter groups")

    # Calculate critical mass for each group
    results = []
    for group_id, group_df in grouped_data:
        logger.info(f"Processing group: {group_id}")
        try:
            result = calculate_critical_mass_for_group(
                group_id, group_df, analysis_type
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


def validate_group_for_fitting(group_df, min_points=3):
    """Validate that group has sufficient data for linear fitting."""
    if len(group_df) < min_points:
        return False, f"Insufficient data points: {len(group_df)} < {min_points}"

    # Check for valid mass values
    mass_col = "bare_mass" if "bare_mass" in group_df.columns else "Bare_mass"
    if mass_col not in group_df.columns:
        return False, "Missing bare mass column"

    valid_masses = group_df[mass_col].notna().sum()
    if valid_masses < min_points:
        return False, f"Insufficient valid mass values: {valid_masses} < {min_points}"

    return True, "Valid"


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


def calculate_critical_mass_for_group(group_id, group_df, analysis_type):
    """Calculate critical mass for a parameter group."""
    # Validate group
    is_valid, validation_msg = validate_group_for_fitting(group_df)
    if not is_valid:
        return None

    # Extract data for fitting
    mass_col = "bare_mass" if "bare_mass" in group_df.columns else "Bare_mass"
    x_data = group_df[mass_col].values

    # Get plateau mass data based on analysis type
    if analysis_type == "pcac":
        y_mean_col = "PCAC_plateau_mean"
        y_error_col = "PCAC_plateau_error"
    else:  # pion
        y_mean_col = "pion_plateau_mean"
        y_error_col = "pion_plateau_error"

    if y_mean_col not in group_df.columns or y_error_col not in group_df.columns:
        return None

    y_mean = group_df[y_mean_col].values
    y_error = group_df[y_error_col].values
    y_data = gv.gvar(y_mean, y_error)

    # Perform linear fit
    try:
        fit_result = perform_linear_fit(x_data, y_data)
        quality_metrics = calculate_fit_quality_metrics(fit_result, x_data, y_data)
        critical_mass = calculate_critical_mass_from_fit(fit_result)

        if critical_mass is None:
            return None

        # Package results with ordered columns
        result = {}

        # Add key physics parameters first
        for col in group_df.columns:
            if col == "Overlap_operator_method":
                result["Overlap_operator_method"] = group_df[col].iloc[0]
            elif col == "Kernel_operator_type":
                result["Kernel_operator_type"] = group_df[col].iloc[0]
            elif col == "KL_diagonal_order":
                result["KL_diagonal_order"] = group_df[col].iloc[0]

        # Add critical mass results
        result.update(
            {
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
        )

        # Add remaining metadata (only those specified in config)
        if analysis_type == "pcac":
            from src.analysis.critical_mass_extrapolation._pcac_critical_mass_config import (
                METADATA_COLUMNS,
            )
        else:  # pion
            from src.analysis.critical_mass_extrapolation._pion_critical_mass_config import (
                METADATA_COLUMNS,
            )

        for col in group_df.columns:
            if col in METADATA_COLUMNS and col not in [
                mass_col,
                y_mean_col,
                y_error_col,
                "Overlap_operator_method",
                "Kernel_operator_type",
                "KL_diagonal_order",
            ]:
                values = group_df[col].unique()
                if len(values) == 1:
                    result[col] = values[0]

        return result

    except Exception as e:
        return None


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================


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


def export_results_to_csv(results, output_path):
    """Export critical mass results to CSV file with proper
    formatting."""
    if not results:
        raise ValueError("No results to export")

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Apply proper formatting
    df_formatted = format_dataframe_for_export(df_results)

    # Export to CSV with proper float formatting for non-special columns
    df_formatted.to_csv(output_path, index=False, float_format="%.6f")

    return str(output_path)
