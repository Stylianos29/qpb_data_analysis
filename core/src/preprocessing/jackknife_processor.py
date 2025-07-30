"""
Core jackknife processing functions for correlator analysis.

This module provides functions for:
    - Jackknife resampling of correlator data
    - Finite difference derivative calculations  
    - Statistical analysis of jackknife samples
    - Data validation and error handling
"""

import numpy as np
import gvar as gv
from typing import Dict, List, Tuple, Optional, Any
import logging

from .jackknife_config import (
    DerivativeMethod,
    get_finite_difference_config,
    MIN_GAUGE_CONFIGURATIONS,
    LOG_MESSAGES,
)


class JackknifeProcessor:
    """
    Main processor class for jackknife analysis of correlator data.

    This class handles the complete workflow of jackknife analysis:
        1. Data validation and preprocessing
        2. Jackknife sample generation
        3. Derivative calculation using finite differences
        4. Statistical analysis and error estimation
    """

    def __init__(
        self,
        derivative_method: DerivativeMethod = DerivativeMethod.FOURTH_ORDER,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the jackknife processor.

        Args:
            - derivative_method: Method for finite difference
              calculation
            - logger: Optional logger instance
        """
        self.derivative_method = derivative_method
        self.finite_diff_config = get_finite_difference_config(derivative_method)
        self.logger = logger or logging.getLogger(__name__)

    def validate_input_data(
        self,
        g5g5_data: np.ndarray,
        g4g5g5_data: np.ndarray,
        min_configurations: int = MIN_GAUGE_CONFIGURATIONS,
    ) -> Tuple[bool, str]:
        """
        Validate input correlator data for jackknife analysis.

        Args:
            - g5g5_data: 2D array of g5-g5 correlator values (configs ×
              time)
            - g4g5g5_data: 2D array of g4γ5-g5 correlator values
              (configs × time)
            - min_configurations: Minimum number of configurations
              required

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check minimum number of configurations
        n_configs = g5g5_data.shape[0]
        if n_configs < min_configurations:
            return False, (
                f"Insufficient gauge configurations: {n_configs} "
                f"(minimum required: {min_configurations})"
            )

        # Check data shape consistency
        if g5g5_data.shape != g4g5g5_data.shape:
            return False, (
                f"Shape mismatch: g5g5 {g5g5_data.shape} vs "
                f"g4g5g5 {g4g5g5_data.shape}"
            )

        # Check for invalid values
        if np.any(~np.isfinite(g5g5_data)) or np.any(~np.isfinite(g4g5g5_data)):
            return False, "Input data contains NaN or infinite values"

        # Check minimum time extent for derivatives
        n_time = g5g5_data.shape[1]
        min_time_needed = 2 * self.finite_diff_config["offset"] + 1
        if n_time < min_time_needed:
            return False, (
                f"Insufficient time extent: {n_time} "
                f"(minimum needed for {self.derivative_method.value}: {min_time_needed})"
            )

        return True, ""

    def generate_jackknife_samples(self, data: np.ndarray) -> np.ndarray:
        """
        Generate jackknife samples from input data.

        Args:
            data: 2D array with shape (n_configs, n_time)

        Returns:
            2D array with shape (n_configs, n_time) containing jackknife
            samples
        """
        n_configs, n_time = data.shape
        jackknife_samples = np.zeros_like(data)

        for i in range(n_configs):
            # Create mask excluding the i-th configuration
            mask = np.ones(n_configs, dtype=bool)
            mask[i] = False

            # Calculate average excluding i-th configuration
            jackknife_samples[i] = np.mean(data[mask], axis=0)

        return jackknife_samples

    def calculate_jackknife_statistics(
        self, jackknife_samples: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate jackknife mean and error estimates.

        Args:
            jackknife_samples: 2D array of jackknife samples

        Returns:
            Tuple of (mean_values, error_values)
        """
        # Calculate jackknife mean (should equal sample mean)
        jackknife_mean = np.mean(jackknife_samples, axis=0)

        # Calculate jackknife error estimate
        n_samples = jackknife_samples.shape[0]
        deviations = jackknife_samples - jackknife_mean[np.newaxis, :]
        jackknife_error = np.sqrt(
            (n_samples - 1) / n_samples * np.sum(deviations**2, axis=0)
        )

        return jackknife_mean, jackknife_error

    def calculate_finite_difference_derivative(
        self, correlator_samples: np.ndarray
    ) -> np.ndarray:
        """
        Calculate finite difference derivatives of correlator samples.

        Args:
            correlator_samples: 2D array (n_samples, n_time)

        Returns:
            2D array with derivatives (n_samples, n_time - 2*offset)
        """
        coefficients = self.finite_diff_config["coefficients"]
        denominator = self.finite_diff_config["denominator"]
        offset = self.finite_diff_config["offset"]

        n_samples, n_time = correlator_samples.shape
        n_output_time = n_time - 2 * offset

        if n_output_time <= 0:
            raise ValueError(
                f"Insufficient time points for derivative calculation: "
                f"input {n_time}, need at least {2 * offset + 1}"
            )

        derivative_samples = np.zeros((n_samples, n_output_time))

        # Apply finite difference stencil
        for i in range(n_output_time):
            # Time index in original array (accounting for offset)
            t_center = i + offset

            # Apply stencil
            derivative_value = 0.0
            for j, coeff in enumerate(coefficients):
                t_stencil = t_center + j - len(coefficients) // 2
                derivative_value += coeff * correlator_samples[:, t_stencil]

            derivative_samples[:, i] = derivative_value / denominator

        return derivative_samples

    def process_correlator_group(
        self,
        g5g5_data: np.ndarray,
        g4g5g5_data: np.ndarray,
        group_metadata: Dict[str, Any],
        min_configurations: int = MIN_GAUGE_CONFIGURATIONS,
    ) -> Dict[str, Any]:
        """
        Process a single group of correlator data through complete
        jackknife analysis.

        Args:
            - g5g5_data: 2D array of g5-g5 correlator values
            - g4g5g5_data: 2D array of g4γ5-g5 correlator values
            - group_metadata: Metadata dictionary for this group
            - min_configurations: Minimum number of configurations
              required

        Returns:
            Dictionary containing all processed results
        """
        group_name = group_metadata.get("group_name", "unknown")
        self.logger.info(LOG_MESSAGES["analysis_start"].format(group_name))

        # Validate input data
        is_valid, error_msg = self.validate_input_data(
            g5g5_data, g4g5g5_data, min_configurations
        )
        if not is_valid:
            self.logger.warning(f"Group {group_name}: {error_msg}")
            return {}

        try:
            # Generate jackknife samples for both correlators
            g5g5_jackknife_samples = self.generate_jackknife_samples(g5g5_data)
            g4g5g5_jackknife_samples = self.generate_jackknife_samples(g4g5g5_data)

            # Calculate jackknife statistics
            g5g5_mean, g5g5_error = self.calculate_jackknife_statistics(
                g5g5_jackknife_samples
            )
            g4g5g5_mean, g4g5g5_error = self.calculate_jackknife_statistics(
                g4g5g5_jackknife_samples
            )

            # Calculate derivatives
            g4g5g5_derivative_samples = self.calculate_finite_difference_derivative(
                g4g5g5_jackknife_samples
            )
            g4g5g5_derivative_mean, g4g5g5_derivative_error = (
                self.calculate_jackknife_statistics(g4g5g5_derivative_samples)
            )

            # Package results
            results = {
                # G5-G5 correlator results
                "g5g5_jackknife_samples": g5g5_jackknife_samples,
                "g5g5_mean_values": g5g5_mean,
                "g5g5_error_values": g5g5_error,
                # G4γ5-G5 correlator results
                "g4g5g5_jackknife_samples": g4g5g5_jackknife_samples,
                "g4g5g5_mean_values": g4g5g5_mean,
                "g4g5g5_error_values": g4g5g5_error,
                # G4γ5-G5 derivative results
                "g4g5g5_derivative_jackknife_samples": g4g5g5_derivative_samples,
                "g4g5g5_derivative_mean_values": g4g5g5_derivative_mean,
                "g4g5g5_derivative_error_values": g4g5g5_derivative_error,
                # Metadata
                "n_gauge_configurations": g5g5_data.shape[0],
                "n_time_points": g5g5_data.shape[1],
                "n_derivative_time_points": g4g5g5_derivative_samples.shape[1],
                "derivative_method": self.derivative_method.value,
            }

            self.logger.info(LOG_MESSAGES["analysis_complete"].format(group_name))
            return results

        except Exception as e:
            self.logger.error(f"Error processing group {group_name}: {str(e)}")
            return {}


def create_gvar_arrays(mean_values: np.ndarray, error_values: np.ndarray) -> np.ndarray:
    """
    Create gvar arrays from mean and error values.

    Args:
        - mean_values: Array of mean values
        - error_values: Array of error values

    Returns:
        Array of gvar objects
    """
    return gv.gvar(mean_values, error_values)


def extract_configuration_metadata(
    group_df, configuration_column: str = "Configuration_label"
) -> Dict[str, List]:
    """
    Extract configuration-related metadata from a group DataFrame.

    Args:
        - group_df: DataFrame containing group data
        - configuration_column: Name of configuration label column

    Returns:
        Dictionary with configuration labels and related metadata
    """
    metadata = {}

    if configuration_column in group_df.columns:
        metadata["configuration_labels"] = group_df[configuration_column].tolist()

    # Extract QPB filenames if available
    if "Filename" in group_df.columns:
        metadata["qpb_filenames"] = group_df["Filename"].tolist()
    elif "qpb_log_filename" in group_df.columns:
        metadata["qpb_filenames"] = group_df["qpb_log_filename"].tolist()

    return metadata


def validate_group_for_processing(
    group_df, required_datasets: List[str], logger: Optional[logging.Logger] = None
) -> Tuple[bool, str]:
    """
    Validate that a group has the required datasets for processing.

    Args:
        - group_df: DataFrame representing the group
        - required_datasets: List of required dataset names
        - logger: Optional logger for warnings

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for required columns/datasets
    missing_datasets = []
    for dataset in required_datasets:
        if dataset not in group_df.columns:
            missing_datasets.append(dataset)

    if missing_datasets:
        error_msg = f"Missing required datasets: {missing_datasets}"
        if logger:
            logger.warning(error_msg)
        return False, error_msg

    return True, ""


def prepare_output_metadata(
    input_metadata: Dict[str, Any], processing_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare metadata for output HDF5 group.

    Args:
        - input_metadata: Original metadata from input group
        - processing_results: Results from jackknife processing

    Returns:
        Combined metadata dictionary for output
    """
    output_metadata = input_metadata.copy()

    # Add processing-specific metadata
    output_metadata.update(
        {
            "jackknife_analysis_applied": True,
            "n_gauge_configurations": processing_results.get("n_gauge_configurations"),
            "derivative_method": processing_results.get("derivative_method"),
            "n_time_points_original": processing_results.get("n_time_points"),
            "n_time_points_derivative": processing_results.get(
                "n_derivative_time_points"
            ),
        }
    )

    return output_metadata
