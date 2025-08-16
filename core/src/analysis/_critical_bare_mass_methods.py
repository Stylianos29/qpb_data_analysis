"""
Private methods for critical bare mass calculation from PCAC mass estimates.

This module contains the core analysis functions for calculating critical
bare mass values through linear extrapolation to the chiral limit where
PCAC mass = 0.

Place this file as:
qpb_data_analysis/core/src/analysis/_critical_bare_mass_methods.py
"""

import numpy as np
import pandas as pd
import gvar as gv
import lsqfit
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import warnings

# Import configuration
from src.analysis._critical_bare_mass_config import (
    INPUT_CSV_COLUMNS,
    GROUPING_PARAMETERS,
    METADATA_COLUMNS,
    DATA_FILTERING,
    LINEAR_FIT_CONFIG,
    FIT_QUALITY_THRESHOLDS,
    PHYSICAL_VALIDATION,
    ERROR_HANDLING,
    get_linear_fit_config,
    get_validation_config,
    get_error_handling_config,
)


# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================


def load_and_validate_pcac_data(csv_file_path: str, logger) -> pd.DataFrame:
    """
    Load PCAC mass estimates from CSV file and perform initial validation.

    Args:
        csv_file_path: Path to input CSV file
        logger: Logger instance for reporting

    Returns:
        Validated pandas DataFrame with PCAC mass data

    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    logger.info(f"Loading PCAC mass data from: {csv_file_path}")

    try:
        df = pd.read_csv(csv_file_path)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        raise ValueError(f"Failed to load CSV file: {e}")

    # Check for required columns
    required_columns = list(INPUT_CSV_COLUMNS.values())
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check for grouping parameter columns
    missing_grouping = [col for col in GROUPING_PARAMETERS if col not in df.columns]
    if missing_grouping:
        logger.warning(f"Missing grouping parameter columns: {missing_grouping}")

    # Basic data validation
    bare_mass_col = INPUT_CSV_COLUMNS["bare_mass"]
    pcac_mean_col = INPUT_CSV_COLUMNS["pcac_mass_mean"]
    pcac_error_col = INPUT_CSV_COLUMNS["pcac_mass_error"]

    # Check for non-finite values
    invalid_bare = df[bare_mass_col].isna() | ~np.isfinite(df[bare_mass_col])
    invalid_pcac_mean = df[pcac_mean_col].isna() | ~np.isfinite(df[pcac_mean_col])
    invalid_pcac_error = (
        df[pcac_error_col].isna()
        | ~np.isfinite(df[pcac_error_col])
        | (df[pcac_error_col] <= 0)
    )

    invalid_rows = invalid_bare | invalid_pcac_mean | invalid_pcac_error

    if invalid_rows.any():
        n_invalid = invalid_rows.sum()
        logger.warning(f"Found {n_invalid} rows with invalid data - will be excluded")
        df = df[~invalid_rows].copy()

    logger.info(f"Data validation completed. {len(df)} valid rows remain")
    return df


def group_data_by_parameters(df: pd.DataFrame, logger) -> Dict[str, pd.DataFrame]:
    """
    Group data by lattice parameters for separate analysis.

    Args:
        df: Input DataFrame with PCAC mass data
        logger: Logger instance

    Returns:
        Dictionary mapping group identifiers to DataFrames
    """
    logger.info("Grouping data by lattice parameters")

    # Use available grouping parameters
    available_grouping = [col for col in GROUPING_PARAMETERS if col in df.columns]

    if not available_grouping:
        logger.warning("No grouping parameters found - treating all data as one group")
        return {"all_data": df}

    logger.info(f"Grouping by parameters: {available_grouping}")

    # Create groups
    groups = {}
    for group_values, group_df in df.groupby(available_grouping):
        # Create group identifier string
        if isinstance(group_values, tuple):
            group_id = "_".join(
                [
                    f"{param}={val}"
                    for param, val in zip(available_grouping, group_values)
                ]
            )
        else:
            group_id = f"{available_grouping[0]}={group_values}"

        groups[group_id] = group_df.copy()
        logger.debug(f"Group '{group_id}': {len(group_df)} data points")

    logger.info(f"Created {len(groups)} parameter groups")
    return groups


def filter_data_for_fitting(
    df: pd.DataFrame, logger
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply filtering criteria to prepare data for linear fitting.

    Args:
        df: Input DataFrame for one parameter group
        logger: Logger instance

    Returns:
        Tuple of (filtered_df, filtering_info)
    """
    original_count = len(df)
    filtering_info = {"original_count": original_count}

    bare_mass_col = INPUT_CSV_COLUMNS["bare_mass"]
    success_col = INPUT_CSV_COLUMNS.get("extraction_success")

    # Apply extraction success filter if column exists
    if (
        success_col is not None
        and success_col in df.columns
        and DATA_FILTERING["require_extraction_success"]
    ):

        df = df[df[success_col] == True].copy()
        filtering_info["after_success_filter"] = len(df)
        logger.debug(f"After extraction success filter: {len(df)} points")

    # Apply upper bare mass cut
    upper_cut = DATA_FILTERING["upper_bare_mass_cut"]
    if upper_cut is not None:
        df = df[df[bare_mass_col] <= upper_cut].copy()
        filtering_info["after_mass_cut"] = len(df)
        logger.debug(f"After bare mass cut ({upper_cut}): {len(df)} points")

    # Apply maximum data points limit if specified
    max_points = DATA_FILTERING["max_data_points"]
    if max_points is not None and len(df) > max_points:
        # Keep points with smallest bare mass values for better extrapolation
        df = df.nsmallest(max_points, bare_mass_col).copy()
        filtering_info["after_max_points"] = len(df)
        logger.debug(f"After max points limit ({max_points}): {len(df)} points")

    filtering_info["final_count"] = len(df)
    return df, filtering_info


# =============================================================================
# LINEAR FITTING FUNCTIONS
# =============================================================================


def linear_function(
    x: Union[float, np.ndarray], p: List[gv.GVar]
) -> Union[gv.GVar, np.ndarray]:
    """
    Linear function for fitting: y = slope * x + intercept

    Args:
        x: Input x values
        p: Parameters [slope, intercept]

    Returns:
        y values calculated from linear function
    """
    slope, intercept = p
    return slope * x + intercept


def estimate_initial_parameters(
    x_data: np.ndarray, y_data: np.ndarray, method: str = "two_point"
) -> List[float]:
    """
    Estimate initial parameters for linear fit.

    Args:
        x_data: Independent variable data
        y_data: Dependent variable data (mean values)
        method: Method for estimation ('two_point', 'least_squares')

    Returns:
        Initial parameter estimates [slope, intercept]
    """
    if method == "two_point":
        # Use endpoints for initial guess
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min_idx, y_max_idx = np.argmin(x_data), np.argmax(x_data)
        y_min, y_max = y_data[y_min_idx], y_data[y_max_idx]

        slope = (y_max - y_min) / (x_max - x_min) if x_max != x_min else 0.0
        intercept = y_min - slope * x_min

    elif method == "least_squares":
        # Use numpy polyfit for initial guess
        coeffs = np.polyfit(x_data, y_data, 1)
        slope, intercept = coeffs[0], coeffs[1]
    else:
        raise ValueError(f"Unknown initial guess method: {method}")

    return [slope, intercept]


def perform_linear_fit(
    x_data: np.ndarray, y_gvar: np.ndarray, logger
) -> Dict[str, Any]:
    """
    Perform linear fit using lsqfit with proper error propagation.

    Args:
        x_data: Independent variable data (bare mass values)
        y_gvar: Dependent variable data as gvar objects (PCAC mass ± error)
        logger: Logger instance

    Returns:
        Dictionary containing fit results and diagnostics
    """
    config = get_linear_fit_config()

    # Estimate initial parameters using mean values
    y_mean = gv.mean(y_gvar)
    initial_params = estimate_initial_parameters(
        x_data, y_mean, config["initial_guess_method"]
    )

    logger.debug(
        f"Initial parameter guess: slope={initial_params[0]:.6f}, "
        f"intercept={initial_params[1]:.6f}"
    )

    try:
        # Perform fit using lsqfit
        fit_result = lsqfit.nonlinear_fit(
            data=(x_data, y_gvar),
            p0=initial_params,
            fcn=linear_function,
            debug=False,
        )

        # Extract results
        slope, intercept = fit_result.p

        fit_info = {
            "success": True,
            "slope": slope,
            "intercept": intercept,
            "chi2": fit_result.chi2,
            "dof": fit_result.dof,
            "chi2_per_dof": (
                fit_result.chi2 / fit_result.dof if fit_result.dof > 0 else np.inf
            ),
            "Q": fit_result.Q,  # P-value
            "fit_object": fit_result,
            "method": "lsqfit",
        }

        logger.debug(
            f"Fit successful: χ²/dof = {fit_info['chi2_per_dof']:.4f}, "
            f"Q = {fit_info['Q']:.4f}"
        )

    except Exception as e:
        logger.warning(f"Linear fit failed: {e}")
        fit_info = {
            "success": False,
            "error_message": str(e),
            "method": "lsqfit",
        }

    return fit_info


def calculate_critical_bare_mass(
    fit_info: Dict[str, Any]
) -> Tuple[Optional[gv.GVar], Dict[str, Any]]:
    """
    Calculate critical bare mass as x-intercept where PCAC mass = 0.

    Args:
        fit_info: Dictionary containing linear fit results

    Returns:
        Tuple of (critical_bare_mass, calculation_info)
    """
    if not fit_info["success"]:
        return None, {"success": False, "error": "Fit failed"}

    slope = fit_info["slope"]
    intercept = fit_info["intercept"]

    # Critical bare mass is where y = 0, so: 0 = slope * x + intercept
    # Therefore: x = -intercept / slope

    if gv.mean(slope) == 0:
        return None, {
            "success": False,
            "error": "Zero slope - cannot calculate intercept",
        }

    critical_bare_mass = -intercept / slope

    calc_info = {
        "success": True,
        "critical_bare_mass": critical_bare_mass,
        "slope": slope,
        "intercept": intercept,
        "slope_significance": (
            abs(gv.mean(slope) / gv.sdev(slope)) if gv.sdev(slope) > 0 else np.inf
        ),
    }

    return critical_bare_mass, calc_info


# =============================================================================
# FIT QUALITY VALIDATION
# =============================================================================


def validate_fit_quality(
    fit_info: Dict[str, Any], calc_info: Dict[str, Any], logger
) -> Dict[str, Any]:
    """
    Validate the quality of the linear fit and critical mass calculation.

    Args:
        fit_info: Linear fit results
        calc_info: Critical mass calculation results
        logger: Logger instance

    Returns:
        Dictionary containing validation results
    """
    if not fit_info["success"] or not calc_info["success"]:
        return {"quality_passed": False, "reason": "Fit or calculation failed"}

    thresholds = FIT_QUALITY_THRESHOLDS
    physical_val = get_validation_config()

    validation_results = {"quality_passed": True, "warnings": [], "checks": {}}

    # Check chi-squared per DOF
    chi2_per_dof = fit_info["chi2_per_dof"]
    max_chi2_dof = thresholds["max_chi2_per_dof"]
    validation_results["checks"]["chi2_per_dof"] = {
        "value": chi2_per_dof,
        "threshold": max_chi2_dof,
        "passed": chi2_per_dof <= max_chi2_dof,
    }

    if chi2_per_dof > max_chi2_dof:
        validation_results["quality_passed"] = False
        validation_results["warnings"].append(
            f"High χ²/dof: {chi2_per_dof:.3f} > {max_chi2_dof}"
        )

    # Check slope significance
    slope_sig = calc_info["slope_significance"]
    min_slope_sig = thresholds["min_slope_significance"]
    validation_results["checks"]["slope_significance"] = {
        "value": slope_sig,
        "threshold": min_slope_sig,
        "passed": slope_sig >= min_slope_sig,
    }

    if slope_sig < min_slope_sig:
        validation_results["quality_passed"] = False
        validation_results["warnings"].append(
            f"Low slope significance: {slope_sig:.2f} < {min_slope_sig}"
        )

    # Check critical mass error ratio
    critical_mass = calc_info["critical_bare_mass"]
    if critical_mass is not None:
        error_ratio = (
            gv.sdev(critical_mass) / abs(gv.mean(critical_mass))
            if gv.mean(critical_mass) != 0
            else np.inf
        )
        max_error_ratio = thresholds["max_critical_mass_error_ratio"]

        validation_results["checks"]["critical_mass_error_ratio"] = {
            "value": error_ratio,
            "threshold": max_error_ratio,
            "passed": error_ratio <= max_error_ratio,
        }

        if error_ratio > max_error_ratio:
            validation_results["quality_passed"] = False
            validation_results["warnings"].append(
                f"Large critical mass uncertainty: {error_ratio:.3f} > {max_error_ratio}"
            )

    # Physical reasonableness checks
    if critical_mass is not None:
        critical_value = gv.mean(critical_mass)
        min_critical = physical_val["min_critical_bare_mass"]
        max_critical = physical_val["max_critical_bare_mass"]

        validation_results["checks"]["physical_range"] = {
            "value": critical_value,
            "min_threshold": min_critical,
            "max_threshold": max_critical,
            "passed": min_critical <= critical_value <= max_critical,
        }

        if not (min_critical <= critical_value <= max_critical):
            validation_results["warnings"].append(
                f"Critical mass outside physical range: {critical_value:.6f} not in [{min_critical}, {max_critical}]"
            )

    # Check slope sign if required
    if physical_val["require_negative_slope"]:
        slope_value = gv.mean(calc_info["slope"])
        validation_results["checks"]["slope_sign"] = {
            "value": slope_value,
            "passed": slope_value < 0,
        }

        if slope_value >= 0:
            validation_results["warnings"].append(
                f"Non-negative slope: {slope_value:.6f}"
            )

    # Log validation results
    if validation_results["quality_passed"]:
        logger.debug("Fit quality validation passed")
    else:
        logger.warning(
            f"Fit quality issues: {'; '.join(validation_results['warnings'])}"
        )

    return validation_results


# =============================================================================
# GROUP PROCESSING
# =============================================================================


def process_parameter_group(
    group_id: str, group_df: pd.DataFrame, logger
) -> Dict[str, Any]:
    """
    Process a single parameter group to calculate critical bare mass.

    Args:
        group_id: Identifier for the parameter group
        group_df: DataFrame containing data for this group
        logger: Logger instance

    Returns:
        Dictionary containing processing results
    """
    logger.info(f"Processing group: {group_id}")

    # Filter data for fitting
    filtered_df, filtering_info = filter_data_for_fitting(group_df, logger)

    # Check minimum data points requirement
    min_points = DATA_FILTERING["min_data_points"]
    if len(filtered_df) < min_points:
        logger.warning(f"Insufficient data points: {len(filtered_df)} < {min_points}")
        return {
            "group_id": group_id,
            "success": False,
            "error": f"Insufficient data points: {len(filtered_df)} < {min_points}",
            "n_data_points": len(filtered_df),
            "filtering_info": filtering_info,
        }

    # Prepare data for fitting
    bare_mass_col = INPUT_CSV_COLUMNS["bare_mass"]
    pcac_mean_col = INPUT_CSV_COLUMNS["pcac_mass_mean"]
    pcac_error_col = INPUT_CSV_COLUMNS["pcac_mass_error"]

    x_data = filtered_df[bare_mass_col].to_numpy()
    pcac_means = filtered_df[pcac_mean_col].to_numpy()
    pcac_errors = filtered_df[pcac_error_col].to_numpy()

    # Create gvar objects for proper error propagation
    y_gvar = gv.gvar(pcac_means, pcac_errors)

    logger.debug(f"Fitting {len(x_data)} data points")

    # Perform linear fit
    fit_info = perform_linear_fit(x_data, y_gvar, logger)

    if not fit_info["success"]:
        logger.warning(f"Linear fit failed for group {group_id}")
        return {
            "group_id": group_id,
            "success": False,
            "error": f"Linear fit failed: {fit_info.get('error_message', 'Unknown error')}",
            "n_data_points": len(filtered_df),
            "filtering_info": filtering_info,
        }

    # Calculate critical bare mass
    critical_mass, calc_info = calculate_critical_bare_mass(fit_info)

    if not calc_info["success"]:
        logger.warning(f"Critical mass calculation failed for group {group_id}")
        return {
            "group_id": group_id,
            "success": False,
            "error": f"Critical mass calculation failed: {calc_info.get('error', 'Unknown error')}",
            "n_data_points": len(filtered_df),
            "filtering_info": filtering_info,
            "fit_info": fit_info,
        }

    # Validate fit quality
    validation_results = validate_fit_quality(fit_info, calc_info, logger)

    # Collect group metadata (take values from first row since they should be identical within group)
    group_metadata = {}
    available_grouping = [col for col in GROUPING_PARAMETERS if col in group_df.columns]
    for col in available_grouping:
        group_metadata[col] = group_df[col].iloc[0]

    # Include metadata columns if available
    for col in METADATA_COLUMNS:
        if col in group_df.columns:
            # For metadata, take representative values (first non-null value)
            values = group_df[col].dropna()
            if len(values) > 0:
                group_metadata[col] = values.iloc[0]

    # Prepare results
    results = {
        "group_id": group_id,
        "success": True,
        "critical_bare_mass": critical_mass,
        "critical_bare_mass_mean": gv.mean(critical_mass),
        "critical_bare_mass_error": gv.sdev(critical_mass),
        "slope": calc_info["slope"],
        "intercept": calc_info["intercept"],
        "slope_significance": calc_info["slope_significance"],
        "chi2": fit_info["chi2"],
        "chi2_per_dof": fit_info["chi2_per_dof"],
        "dof": fit_info["dof"],
        "Q_value": fit_info["Q"],
        "n_data_points": len(filtered_df),
        "data_range": [float(np.min(x_data)), float(np.max(x_data))],
        "filtering_info": filtering_info,
        "fit_validation": validation_results,
        "group_metadata": group_metadata,
        "fit_info": fit_info,  # Keep full fit info for plotting
    }

    logger.info(f"Group {group_id} completed: critical_mass = {critical_mass}")

    return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def generate_fit_diagnostics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive diagnostics from fit results.

    Args:
        results: Results from process_parameter_group

    Returns:
        Dictionary containing diagnostic information
    """
    if not results["success"]:
        return {"diagnostics_available": False}

    diagnostics = {
        "diagnostics_available": True,
        "fit_method": "lsqfit_linear",
        "data_points_used": results["n_data_points"],
        "data_range_bare_mass": results["data_range"],
        "chi_squared": float(results["chi2"]),
        "degrees_of_freedom": int(results["dof"]),
        "reduced_chi_squared": float(results["chi2_per_dof"]),
        "p_value": float(results["Q_value"]),
        "slope_value": float(gv.mean(results["slope"])),
        "slope_error": float(gv.sdev(results["slope"])),
        "intercept_value": float(gv.mean(results["intercept"])),
        "intercept_error": float(gv.sdev(results["intercept"])),
        "slope_significance": float(results["slope_significance"]),
        "fit_quality_passed": results["fit_validation"]["quality_passed"],
        "quality_warnings": results["fit_validation"]["warnings"],
    }

    return diagnostics


def create_summary_statistics(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create summary statistics from all parameter group results.

    Args:
        all_results: List of results from all processed groups

    Returns:
        Dictionary containing summary statistics
    """
    successful_results = [r for r in all_results if r["success"]]

    summary = {
        "total_groups": len(all_results),
        "successful_groups": len(successful_results),
        "failed_groups": len(all_results) - len(successful_results),
        "success_rate": (
            len(successful_results) / len(all_results) if all_results else 0
        ),
    }

    if successful_results:
        critical_masses = [r["critical_bare_mass_mean"] for r in successful_results]
        critical_errors = [r["critical_bare_mass_error"] for r in successful_results]
        chi2_values = [r["chi2_per_dof"] for r in successful_results]

        summary.update(
            {
                "critical_mass_range": [
                    float(np.min(critical_masses)),
                    float(np.max(critical_masses)),
                ],
                "critical_mass_mean": float(np.mean(critical_masses)),
                "critical_mass_std": float(np.std(critical_masses)),
                "median_error": float(np.median(critical_errors)),
                "median_chi2_per_dof": float(np.median(chi2_values)),
                "good_quality_fits": sum(
                    1
                    for r in successful_results
                    if r["fit_validation"]["quality_passed"]
                ),
            }
        )

    return summary
