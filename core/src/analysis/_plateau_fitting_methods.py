"""
Private module for PCAC mass plateau fitting and analysis methods.

This module contains functions for detecting plateau regions and estimating 
plateau values from PCAC mass time series using jackknife analysis.

Adapted from jackknife_analysis module with improved organization and clarity.

Place this file as: qpb_data_analysis/core/src/analysis/_plateau_fitting_methods.py
"""

import numpy as np
import gvar as gv
from typing import Tuple, Dict, List, Optional, Any
from scipy import stats

from src.analysis._plateau_extraction_config import (
    get_plateau_detection_config,
    get_plateau_estimation_config,
    get_error_handling_config,
)


# =============================================================================
# JACKKNIFE ANALYSIS FUNCTIONS
# =============================================================================


def calculate_jackknife_average(
    jackknife_replicas: np.ndarray, use_covariance: bool = True
) -> np.ndarray:
    """
    Calculate jackknife average with uncertainties from replica samples.

    Args:
        jackknife_replicas: Array of shape (N, T) where N is number of samples
                           and T is the time series length
        use_covariance: Whether to preserve correlations between time points

    Returns:
        Array of T gvar objects with jackknife means and uncertainties
    """
    N, T = jackknife_replicas.shape

    if use_covariance:
        # Calculate full covariance matrix
        covariance_matrix, jk_mean = _calculate_jackknife_covariance(jackknife_replicas)
        result = gv.gvar(jk_mean, covariance_matrix)
    else:
        # Independent calculation (faster)
        jk_mean = np.mean(jackknife_replicas, axis=0)
        jk_var = (N - 1) / N * np.sum((jackknife_replicas - jk_mean) ** 2, axis=0)
        jk_std = np.sqrt(jk_var)
        result = gv.gvar(jk_mean, jk_std)

    return result


def _calculate_jackknife_covariance(
    jackknife_replicas: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate jackknife covariance matrix and mean.

    Args:
        jackknife_replicas: Array of shape (N, T)

    Returns:
        Tuple of (covariance_matrix, mean_values)
    """
    N, T = jackknife_replicas.shape
    jk_mean = np.mean(jackknife_replicas, axis=0)

    # Calculate covariance matrix
    covariance_matrix = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            values_i = jackknife_replicas[:, i]
            values_j = jackknife_replicas[:, j]
            mean_i = jk_mean[i]
            mean_j = jk_mean[j]
            cov_ij = (N - 1) / N * np.sum((values_i - mean_i) * (values_j - mean_j))
            covariance_matrix[i, j] = cov_ij

    return covariance_matrix, jk_mean


# =============================================================================
# PLATEAU DETECTION FUNCTIONS
# =============================================================================


def detect_plateau_region(
    time_series_with_uncertainties: np.ndarray,
    sigma_threshold: float = 1.0,
    min_plateau_size: int = 5,
    test_method: str = "weighted_range",
    verbose: bool = False,
) -> Tuple[int, int, Dict]:
    """
    Automatically detect the plateau region in a time series using shrinking window approach.

    Args:
        time_series_with_uncertainties: Array of gvar objects
        sigma_threshold: Convergence criterion in units of sigma
        min_plateau_size: Minimum required plateau size
        test_method: Method for plateau consistency testing
        verbose: Print detailed information

    Returns:
        Tuple of (start_index, end_index, diagnostics)
    """
    T = len(time_series_with_uncertainties)

    if T < min_plateau_size:
        raise ValueError(
            f"Time series length {T} is smaller than minimum plateau size {min_plateau_size}"
        )

    # Extract means and standard deviations
    means = np.array([val.mean for val in time_series_with_uncertainties])
    stds = np.array([val.sdev for val in time_series_with_uncertainties])

    # Apply search constraints from config
    search_config = get_plateau_detection_config()["search_config"]
    min_start = max(0, search_config["min_start_time"])
    max_end = (
        min(T, T + search_config["max_end_time"])
        if search_config["max_end_time"] < 0
        else T
    )

    # Start with full valid range
    current_start = min_start
    current_end = max_end

    iteration = 0
    max_iterations = T

    diagnostics = {
        "iterations": 0,
        "initial_range": (current_start, current_end),
        "test_method": test_method,
        "sigma_threshold": sigma_threshold,
    }

    while iteration < max_iterations:
        current_size = current_end - current_start

        # Check if we've reached minimum size
        if current_size < min_plateau_size:
            raise ValueError(
                f"Could not find plateau of minimum size {min_plateau_size} "
                f"with sigma threshold {sigma_threshold}"
            )

        # Test current window for plateau consistency
        window_means = means[current_start:current_end]
        window_stds = stds[current_start:current_end]

        is_plateau, test_statistic, p_value = _test_plateau_consistency(
            window_means, window_stds, sigma_threshold, test_method
        )

        if verbose:
            print(
                f"Iteration {iteration}: range [{current_start}:{current_end}], "
                f"size {current_size}, is_plateau: {is_plateau}, "
                f"test_stat: {test_statistic:.3f}"
            )

        if is_plateau:
            # Found valid plateau
            diagnostics.update(
                {
                    "iterations": iteration,
                    "final_range": (current_start, current_end),
                    "test_statistic": test_statistic,
                    "p_value": p_value,
                    "success": True,
                }
            )
            return current_start, current_end, diagnostics

        # Find and remove most outlying point
        outlier_scores = []
        for i in range(len(window_means)):
            score = _calculate_outlier_score(window_means, window_stds, i, test_method)
            outlier_scores.append(score)

        max_score_idx = np.argmax(outlier_scores)
        global_idx = current_start + max_score_idx

        # Remove outlier (prefer removing from edges)
        if global_idx == current_start:
            current_start += 1
        elif global_idx == current_end - 1:
            current_end -= 1
        else:
            # Remove from the edge that's closer
            if global_idx - current_start <= current_end - global_idx - 1:
                current_start = global_idx + 1
            else:
                current_end = global_idx

        iteration += 1

    # If we get here, we failed to find a plateau
    diagnostics.update(
        {
            "iterations": iteration,
            "success": False,
            "failure_reason": "Maximum iterations reached",
        }
    )

    raise ValueError(
        f"Failed to detect plateau region with sigma threshold {sigma_threshold}"
    )


def _test_plateau_consistency(
    means: np.ndarray, stds: np.ndarray, sigma_threshold: float, test_method: str
) -> Tuple[bool, float, float]:
    """
    Test whether a window of data points forms a statistically consistent plateau.

    Args:
        means: Array of mean values
        stds: Array of standard deviations
        sigma_threshold: Threshold for consistency test
        test_method: Method for testing consistency

    Returns:
        Tuple of (is_plateau, test_statistic, p_value)
    """
    n_points = len(means)

    if test_method == "chi_squared":
        # Chi-squared test for constant value
        weighted_mean = np.sum(means / stds**2) / np.sum(1 / stds**2)
        chi_squared = np.sum(((means - weighted_mean) / stds) ** 2)

        expected_chi_squared = n_points
        threshold_chi_squared = expected_chi_squared + sigma_threshold * np.sqrt(
            2 * n_points
        )

        is_plateau = chi_squared <= threshold_chi_squared
        p_value = 1 - stats.chi2.cdf(chi_squared, n_points - 1)

        return is_plateau, chi_squared, p_value

    elif test_method == "range_based":
        # Simple range test
        avg_std = np.mean(stds)
        data_range = np.max(means) - np.min(means)
        threshold_range = sigma_threshold * avg_std

        is_plateau = data_range <= threshold_range
        test_statistic = data_range / avg_std
        p_value = (
            1.0 - test_statistic / sigma_threshold
            if test_statistic <= sigma_threshold
            else 0.0
        )

        return is_plateau, test_statistic, p_value

    elif test_method == "weighted_range":
        # Weighted range test using inverse-variance weighting
        weights = 1 / stds**2
        weighted_mean = np.sum(weights * means) / np.sum(weights)

        weighted_deviations = np.abs(means - weighted_mean) / stds
        max_weighted_deviation = np.max(weighted_deviations)

        is_plateau = max_weighted_deviation <= sigma_threshold
        test_statistic = max_weighted_deviation
        p_value = (
            1.0 - test_statistic / sigma_threshold
            if test_statistic <= sigma_threshold
            else 0.0
        )

        return is_plateau, test_statistic, p_value

    else:
        raise ValueError(f"Unknown test method: {test_method}")


def _calculate_outlier_score(
    window_means: np.ndarray,
    window_stds: np.ndarray,
    point_index: int,
    test_method: str,
) -> float:
    """
    Calculate outlier score for a specific point in the window.
    Higher scores indicate the point is more likely to be an outlier.

    Args:
        window_means: Mean values in current window
        window_stds: Standard deviations in current window
        point_index: Index of point to evaluate
        test_method: Method for calculating outlier score

    Returns:
        Outlier score (higher = more likely outlier)
    """
    n_points = len(window_means)

    if test_method == "chi_squared":
        weights = 1 / window_stds**2
        weighted_mean = np.sum(weights * window_means) / np.sum(weights)
        point_contribution = (
            (window_means[point_index] - weighted_mean) / window_stds[point_index]
        ) ** 2
        return point_contribution

    elif test_method in ["range_based", "weighted_range"]:
        # Calculate how much this point extends the range
        other_means = np.concatenate(
            [window_means[:point_index], window_means[point_index + 1 :]]
        )

        if len(other_means) == 0:
            return 0.0

        range_without = np.max(other_means) - np.min(other_means)
        range_with = np.max(window_means) - np.min(window_means)

        avg_std = np.mean(window_stds)
        range_increase = (range_with - range_without) / avg_std

        return range_increase

    else:
        # Default: distance from median in units of sigma
        median_value = np.median(window_means)
        return (
            np.abs(window_means[point_index] - median_value) / window_stds[point_index]
        )


# =============================================================================
# PLATEAU ESTIMATION FUNCTIONS
# =============================================================================


def estimate_plateau_value(
    jackknife_replicas: np.ndarray,
    plateau_start: int,
    plateau_end: int,
    method: str = "covariance_quadrature",
    use_inverse_variance: bool = False,
) -> Tuple[gv.GVar, Dict]:
    """
    Calculate plateau value and uncertainty from jackknife replicas.

    Args:
        jackknife_replicas: Array of shape (N, T) with jackknife samples
        plateau_start: Start index of plateau region (inclusive)
        plateau_end: End index of plateau region (exclusive)
        method: Estimation method ('simple', 'median', 'covariance_quadrature')
        use_inverse_variance: Whether to use inverse-variance weighting

    Returns:
        Tuple of (plateau_value, diagnostics)
    """
    N, T = jackknife_replicas.shape

    if plateau_end == -1:
        plateau_end = T

    plateau_region = jackknife_replicas[:, plateau_start:plateau_end]
    n_plateau_points = plateau_end - plateau_start

    if n_plateau_points < 1:
        raise ValueError(f"Invalid plateau region: [{plateau_start}:{plateau_end})")

    # Calculate individual point statistics
    individual_means = np.mean(plateau_region, axis=0)
    individual_vars = (
        (N - 1) / N * np.sum((plateau_region - individual_means) ** 2, axis=0)
    )
    individual_stds = np.sqrt(individual_vars)

    # Determine weights
    if use_inverse_variance:
        # Avoid division by zero
        weights = 1.0 / (individual_stds**2 + 1e-12)
        weights = weights / np.sum(weights)  # normalize
    else:
        weights = np.ones(n_plateau_points) / n_plateau_points

    # Calculate weighted plateau mean
    plateau_mean = np.sum(weights * individual_means)

    # Calculate uncertainty based on method
    if method == "simple":
        replica_means = np.mean(plateau_region, axis=1)
        plateau_var = (
            (N - 1) / N * np.sum((replica_means - np.mean(replica_means)) ** 2)
        )
        plateau_std = np.sqrt(plateau_var)
        covariance_matrix = None
        avg_correlation = None

    elif method == "median":
        replica_medians = np.median(plateau_region, axis=1)
        plateau_mean = np.mean(replica_medians)  # override weighted mean
        plateau_var = (N - 1) / N * np.sum((replica_medians - plateau_mean) ** 2)
        plateau_std = np.sqrt(plateau_var)
        covariance_matrix = None
        avg_correlation = None

    elif method == "covariance_quadrature":
        # Full covariance approach
        covariance_matrix = np.zeros((n_plateau_points, n_plateau_points))

        for i in range(n_plateau_points):
            for j in range(n_plateau_points):
                values_i = plateau_region[:, i]
                values_j = plateau_region[:, j]
                mean_i = individual_means[i]
                mean_j = individual_means[j]
                cov_ij = (N - 1) / N * np.sum((values_i - mean_i) * (values_j - mean_j))
                covariance_matrix[i, j] = cov_ij

        # Averaging uncertainty with correlations
        averaging_variance = np.dot(weights, np.dot(covariance_matrix, weights))

        # RMS of individual uncertainties
        rms_individual = np.sqrt(np.mean(individual_stds**2))

        # Combine in quadrature
        plateau_std = np.sqrt(averaging_variance + rms_individual**2)

        # Calculate average correlation
        correlation_matrix = covariance_matrix / np.outer(
            individual_stds, individual_stds
        )
        mask = ~np.eye(n_plateau_points, dtype=bool)
        avg_correlation = (
            np.mean(correlation_matrix[mask]) if n_plateau_points > 1 else 0.0
        )

    else:
        raise ValueError(f"Unknown estimation method: {method}")

    # Create result
    plateau_value = gv.gvar(plateau_mean, plateau_std)

    # Prepare diagnostics
    diagnostics = {
        "n_points": n_plateau_points,
        "plateau_range": (plateau_start, plateau_end),
        "individual_means": individual_means,
        "individual_stds": individual_stds,
        "weights": weights,
        "method": method,
        "use_inverse_variance": use_inverse_variance,
        "covariance_matrix": covariance_matrix,
        "avg_correlation": avg_correlation,
    }

    return plateau_value, diagnostics


# =============================================================================
# HIGH-LEVEL ORCHESTRATION FUNCTIONS
# =============================================================================


def extract_plateau_from_single_sample(
    sample_time_series: np.ndarray, config_label: str, logger
) -> Tuple[Optional[gv.GVar], Optional[Tuple[int, int]], Optional[Dict], Optional[str]]:
    """
    Extract plateau value from a single PCAC mass time series sample.

    Args:
        sample_time_series: 1D array of PCAC mass values
        config_label: Configuration label for this sample
        logger: Logger instance

    Returns:
        Tuple of (plateau_value, plateau_bounds, diagnostics, error_message)
        Returns (None, None, None, error_msg) if extraction fails
    """
    detection_config = get_plateau_detection_config()
    estimation_config = get_plateau_estimation_config()

    try:
        # Create gvar array with minimal uncertainty for detection
        # We need uncertainties for the detection algorithm
        data_std = np.std(sample_time_series)
        data_mean = np.abs(np.mean(sample_time_series))

        # Calculate minimum uncertainty with proper floor
        min_uncertainty = max(
            data_std * 0.01,  # 1% of data spread
            data_mean * 1e-6,  # 0.0001% of mean value
            1e-8,  # Absolute minimum floor
        )

        # Create uncertainty array (same shape as data)
        uncertainties = np.full_like(sample_time_series, min_uncertainty)
        gvar_series = gv.gvar(sample_time_series, uncertainties)

        # Try different sigma thresholds
        for sigma_threshold in detection_config["sigma_thresholds"]:
            try:
                plateau_start, plateau_end, detection_diagnostics = (
                    detect_plateau_region(
                        gvar_series,
                        sigma_threshold=sigma_threshold,
                        min_plateau_size=detection_config["min_plateau_size"],
                        test_method=detection_config["test_method"],
                        verbose=False,
                    )
                )

                # For single sample, we estimate value directly from plateau region
                plateau_values = sample_time_series[plateau_start:plateau_end]
                plateau_mean = np.mean(plateau_values)
                plateau_std = np.std(plateau_values) / np.sqrt(
                    len(plateau_values)
                )  # Standard error

                plateau_value = gv.gvar(plateau_mean, plateau_std)

                diagnostics = {
                    "detection": detection_diagnostics,
                    "sigma_threshold_used": sigma_threshold,
                    "plateau_mean": plateau_mean,
                    "plateau_std": plateau_std,
                    "n_plateau_points": plateau_end - plateau_start,
                }

                logger.debug(
                    f"Sample {config_label}: Plateau extracted with σ={sigma_threshold}"
                )
                return plateau_value, (plateau_start, plateau_end), diagnostics, None

            except ValueError:
                continue  # Try next sigma threshold

        # If we get here, all sigma thresholds failed
        error_msg = f"Failed to detect plateau with any sigma threshold (max={max(detection_config['sigma_thresholds'])})"
        logger.warning(f"Sample {config_label}: {error_msg}")
        return None, None, None, error_msg

    except Exception as e:
        error_msg = f"Unexpected error during plateau extraction: {e}"
        logger.error(f"Sample {config_label}: {error_msg}")
        return None, None, None, error_msg


def extract_plateau_from_group(
    jackknife_samples: np.ndarray, config_labels: List[str], group_name: str, logger
) -> Dict[str, Any]:
    """
    Extract plateau PCAC mass from a group of jackknife samples.

    Args:
        jackknife_samples: 2D array of shape (n_samples, n_time_points)
        config_labels: List of configuration labels for each sample
        group_name: Name of the group being processed
        logger: Logger instance

    Returns:
        Dictionary containing extraction results and diagnostics
    """
    error_config = get_error_handling_config()
    estimation_config = get_plateau_estimation_config()

    n_samples, n_time_points = jackknife_samples.shape

    # Track successful extractions
    successful_extractions = []
    failed_extractions = []
    plateau_bounds_list = []

    # Extract plateau from each sample
    for i, config_label in enumerate(config_labels):
        sample_series = jackknife_samples[i, :]

        plateau_value, plateau_bounds, diagnostics, error_msg = (
            extract_plateau_from_single_sample(sample_series, config_label, logger)
        )

        if plateau_value is not None:
            successful_extractions.append(
                {
                    "sample_index": i,
                    "config_label": config_label,
                    "plateau_value": plateau_value,
                    "plateau_bounds": plateau_bounds,
                    "diagnostics": diagnostics,
                }
            )
            plateau_bounds_list.append(plateau_bounds)
        else:
            failed_extractions.append(
                {
                    "sample_index": i,
                    "config_label": config_label,
                    "error_message": error_msg,
                }
            )

    n_successful = len(successful_extractions)
    n_failed = len(failed_extractions)

    logger.info(f"Group {group_name}: {n_successful}/{n_samples} samples successful")

    # Check if we have enough successful samples
    if n_successful < error_config["min_sample_size"]:
        return {
            "success": False,
            "group_name": group_name,
            "n_total_samples": n_samples,
            "n_successful": n_successful,
            "n_failed": n_failed,
            "failed_extractions": failed_extractions,
            "error_message": f"Insufficient successful samples ({n_successful} < {error_config['min_sample_size']})",
        }

    # Check failed sample fraction
    failed_fraction = n_failed / n_samples
    if failed_fraction > error_config["max_failed_fraction"]:
        return {
            "success": False,
            "group_name": group_name,
            "n_total_samples": n_samples,
            "n_successful": n_successful,
            "n_failed": n_failed,
            "failed_fraction": failed_fraction,
            "failed_extractions": failed_extractions,
            "error_message": f"Too many failed samples ({failed_fraction:.1%} > {error_config['max_failed_fraction']:.1%})",
        }

    # Determine common plateau range (use most common bounds)
    if plateau_bounds_list:
        # Find most common plateau bounds
        bounds_counter = {}
        for bounds in plateau_bounds_list:
            bounds_counter[bounds] = bounds_counter.get(bounds, 0) + 1

        common_bounds = max(bounds_counter.keys(), key=bounds_counter.get)
        common_start, common_end = common_bounds

        logger.debug(
            f"Group {group_name}: Using common plateau range [{common_start}:{common_end}]"
        )
    else:
        return {
            "success": False,
            "group_name": group_name,
            "n_total_samples": n_samples,  # Add this
            "n_successful": n_successful,  # Add this
            "n_failed": n_failed,  # Add this
            "error_message": "No valid plateau bounds found",
        }

    # Extract successful samples and re-estimate with common bounds
    successful_samples = np.array(
        [
            jackknife_samples[extraction["sample_index"], :]
            for extraction in successful_extractions
        ]
    )

    # Calculate final plateau estimate using jackknife with common bounds
    try:
        final_plateau_value, final_diagnostics = estimate_plateau_value(
            successful_samples,
            common_start,
            common_end,
            method=estimation_config["method"],
            use_inverse_variance=estimation_config["use_inverse_variance"],
        )
    except Exception as e:
        # Try fallback methods
        for fallback in estimation_config["fallback_methods"]:
            try:
                final_plateau_value, final_diagnostics = estimate_plateau_value(
                    successful_samples,
                    common_start,
                    common_end,
                    method=fallback["method"],
                    use_inverse_variance=fallback["use_inverse_variance"],
                )
                logger.info(
                    f"Group {group_name}: Used fallback method {fallback['method']}"
                )
                break
            except Exception:
                continue
        else:
            return {
                "success": False,
                "group_name": group_name,
                "n_total_samples": n_samples,  # Add this
                "n_successful": n_successful,  # Add this
                "n_failed": n_failed,  # Add this
                "error_message": f"All estimation methods failed: {e}",
            }

    # Return successful result
    return {
        "success": True,
        "group_name": group_name,
        "plateau_value": final_plateau_value,
        "plateau_bounds": common_bounds,
        "n_total_samples": n_samples,
        "n_successful": n_successful,
        "n_failed": n_failed,
        "sample_size": n_successful,
        "failed_extractions": failed_extractions,
        "successful_extractions": successful_extractions,
        "final_diagnostics": final_diagnostics,
        "common_bounds_usage": bounds_counter,
    }
