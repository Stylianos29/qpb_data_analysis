#!/usr/bin/env python3
"""
Core utilities for plateau extraction.

This module provides shared functions for detecting and extracting plateau
values from time series data using jackknife analysis.
"""

import numpy as np
import gvar as gv
from typing import Tuple, Dict, List, Optional, Any


def symmetrize_time_series(data: np.ndarray) -> np.ndarray:
    """
    Symmetrize time series data: C_sym(t) = 0.5 * (C(t) + C(T-t)).
    
    Args:
        data: 1D or 2D array (samples × time for 2D)
        
    Returns:
        Symmetrized array of same shape
    """
    if data.ndim == 1:
        reverse = data[::-1]
        return 0.5 * (data + np.roll(reverse, shift=1))
    else:
        # Handle 2D case (multiple samples)
        reverse = data[:, ::-1]
        return 0.5 * (data + np.roll(reverse, shift=1, axis=1))


def calculate_jackknife_statistics(
    jackknife_samples: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and error from jackknife samples.
    
    Args:
        jackknife_samples: Array of shape (N_samples, N_time)
        
    Returns:
        Tuple of (mean_values, error_values)
    """
    n_samples = jackknife_samples.shape[0]
    mean_values = np.mean(jackknife_samples, axis=0)
    
    # Jackknife error formula
    deviations = jackknife_samples - mean_values
    variance = np.mean(deviations**2, axis=0)
    error_values = np.sqrt((n_samples - 1) * variance)
    
    return mean_values, error_values


def detect_plateau_region(
    time_series: np.ndarray,
    errors: np.ndarray,
    sigma_threshold: float,
    min_plateau_size: int,
    search_range: Dict[str, Any],
) -> Optional[Tuple[int, int]]:
    """
    Detect plateau region in time series using weighted range test.
    
    Args:
        time_series: 1D array of values
        errors: 1D array of uncertainties
        sigma_threshold: Number of sigma for plateau detection
        min_plateau_size: Minimum number of consecutive points
        search_range: Dictionary with search constraints
        
    Returns:
        Tuple of (start_index, end_index) or None if no plateau found
    """
    n_points = len(time_series)
    min_start = search_range.get("min_start", 0)
    max_end = search_range.get("max_end", n_points)
    
    # Handle negative indexing for max_end
    if max_end < 0:
        max_end = n_points + max_end
    
    best_plateau = None
    best_score = float('inf')
    
    # Search for all possible plateau regions
    for start in range(min_start, max_end - min_plateau_size + 1):
        for end in range(start + min_plateau_size, min(max_end + 1, n_points + 1)):
            # Extract plateau region
            plateau_data = time_series[start:end]
            plateau_errors = errors[start:end]
            
            # Calculate weighted mean
            weights = 1.0 / plateau_errors**2
            weighted_mean = np.sum(weights * plateau_data) / np.sum(weights)
            
            # Check if all points are within sigma_threshold
            deviations = np.abs(plateau_data - weighted_mean) / plateau_errors
            if np.all(deviations <= sigma_threshold):
                # Calculate quality score (prefer longer, more central plateaus)
                length_score = 1.0 / (end - start)  # Prefer longer
                center = (start + end) / 2
                center_score = abs(center - n_points/2) / n_points if search_range.get("prefer_central") else 0
                score = length_score + 0.1 * center_score
                
                if score < best_score:
                    best_score = score
                    best_plateau = (start, end)
    
    return best_plateau


def extract_plateau_value_from_samples(
    jackknife_samples: np.ndarray,
    plateau_bounds: Tuple[int, int],
    method: str = "inverse_variance_weighted",
) -> Tuple[gv.GVar, Dict[str, Any]]:
    """
    Extract plateau value from jackknife samples in plateau region.
    
    Args:
        jackknife_samples: 2D array (samples × time)
        plateau_bounds: (start, end) indices of plateau
        method: Estimation method
        
    Returns:
        Tuple of (plateau_value, diagnostics)
    """
    start, end = plateau_bounds
    plateau_data = jackknife_samples[:, start:end]
    
    # Calculate statistics for each jackknife sample
    sample_means = np.mean(plateau_data, axis=1)
    
    # Calculate jackknife mean and error
    n_samples = len(sample_means)
    jk_mean = np.mean(sample_means)
    jk_variance = (n_samples - 1) / n_samples * np.sum((sample_means - jk_mean)**2)
    jk_error = np.sqrt(jk_variance)
    
    plateau_value = gv.gvar(jk_mean, jk_error)
    
    diagnostics = {
        "method": method,
        "n_points": end - start,
        "plateau_range": plateau_bounds,
        "individual_means": sample_means,
    }
    
    return plateau_value, diagnostics


def process_single_group(
    jackknife_samples: np.ndarray,
    mean_values: np.ndarray,
    error_values: np.ndarray,
    config_labels: List[str],
    sigma_thresholds: List[float],
    min_plateau_size: int,
    search_range: Dict[str, Any],
    logger,
) -> Dict[str, Any]:
    """
    Process a single parameter group to extract plateau value.
    
    Args:
        jackknife_samples: 2D array of jackknife samples
        mean_values: 1D array of mean values
        error_values: 1D array of error values
        config_labels: List of configuration labels
        sigma_thresholds: List of sigma thresholds to try
        min_plateau_size: Minimum plateau size
        search_range: Search range configuration
        logger: Logger instance
        
    Returns:
        Dictionary with extraction results
    """
    n_samples = jackknife_samples.shape[0]
    
    # Try different sigma thresholds
    for sigma in sigma_thresholds:
        plateau_bounds = detect_plateau_region(
            mean_values,
            error_values,
            sigma,
            min_plateau_size,
            search_range,
        )
        
        if plateau_bounds is not None:
            # Extract plateau value from jackknife samples
            plateau_value, diagnostics = extract_plateau_value_from_samples(
                jackknife_samples,
                plateau_bounds,
            )
            
            logger.info(
                f"Plateau found with σ={sigma}: "
                f"t=[{plateau_bounds[0]}, {plateau_bounds[1]}], "
                f"value={plateau_value}"
            )
            
            return {
                "success": True,
                "plateau_value": plateau_value,
                "plateau_bounds": plateau_bounds,
                "sigma_threshold": sigma,
                "n_samples": n_samples,
                "diagnostics": diagnostics,
            }
    
    # No plateau found with any threshold
    logger.warning("No plateau found with any sigma threshold")
    return {
        "success": False,
        "n_samples": n_samples,
        "error_message": "No plateau detected",
    }
