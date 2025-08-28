#!/usr/bin/env python3
"""
Core utilities for correlator analysis.
"""

import numpy as np
import h5py
from library.data.hdf5_analyzer import HDF5Analyzer


def calculate_jackknife_statistics(samples):
    """Calculate mean and error from jackknife samples."""
    n_samples = samples.shape[0]
    mean_values = np.mean(samples, axis=0)
    deviations = samples - mean_values
    variance = np.mean(deviations**2, axis=0)
    error_values = np.sqrt((n_samples - 1) * variance)
    return mean_values, error_values


def find_analysis_groups(file_path, required_datasets):
    """Find groups containing all required datasets."""
    with HDF5Analyzer(file_path) as analyzer:
        valid_groups = []
        with h5py.File(file_path, "r") as f:
            for group_path in analyzer.active_groups:
                if group_path in f:
                    group = f[group_path]
                    if all(dataset in group for dataset in required_datasets):
                        valid_groups.append(group_path)
        return valid_groups


def safe_divide(numerator, denominator):
    """Safe division with invalid value handling."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
        result[~np.isfinite(result)] = 0.0
    return result


def safe_log(x):
    """Safe logarithm with invalid value handling."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(np.maximum(x, 1e-15))
        result[~np.isfinite(result)] = 0.0
    return result


def safe_sqrt(x):
    """Safe square root with negative value handling."""
    return np.sqrt(np.maximum(x, 0.0))


def symmetrize_correlator(correlator):
    """Apply symmetrization: C_sym(t) = 0.5 * (C(t) + C(T-t))."""
    return 0.5 * (correlator + correlator[:, ::-1])


def copy_metadata(input_group, output_group, metadata_list):
    """Copy metadata datasets from input to output group."""
    for metadata_name in metadata_list:
        if metadata_name in input_group:
            item = input_group[metadata_name]
            if isinstance(item, h5py.Dataset):
                output_group.create_dataset(metadata_name, data=item[:])

    # Copy group attributes
    for attr_name, attr_value in input_group.attrs.items():
        output_group.attrs[attr_name] = attr_value


def copy_parent_attributes(input_file, output_file, group_path, processed_parents):
    """Copy parent group attributes (second-to-deepest level constant
    parameters)."""
    parent_path = "/".join(group_path.split("/")[:-1])

    if parent_path and parent_path not in processed_parents:
        if parent_path in input_file:
            # Create parent group in output and copy its attributes
            parent_group = output_file.require_group(parent_path)
            if len(parent_group.attrs) == 0:  # Not already copied
                input_parent = input_file[parent_path]
                for attr_name, attr_value in input_parent.attrs.items():
                    parent_group.attrs[attr_name] = attr_value
        processed_parents.add(parent_path)
