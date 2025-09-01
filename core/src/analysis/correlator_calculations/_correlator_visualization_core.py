#!/usr/bin/env python3
"""
Core functions for correlator analysis visualization.

This module contains all the plotting and data processing functions for
visualizing both PCAC mass and effective mass jackknife samples.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Import configuration
from src.analysis.correlator_calculations._correlator_visualization_config import (
    SAMPLE_PLOT_STYLE,
    AVERAGE_PLOT_STYLE,
    DEFAULT_FIGURE_SIZE,
    DEFAULT_FONT_SIZE,
    SAMPLES_PER_PLOT,
    PLOT_QUALITY,
    get_analysis_config,
    get_sample_color,
    apply_dataset_slicing,
)


def generate_plot_filename(group_name, start_sample, end_sample, analysis_type):
    """Generate filename for multi-sample plot."""
    return (
        f"{group_name}_{analysis_type}_samples_{start_sample:03d}_{end_sample:03d}.png"
    )


def load_gauge_configuration_labels(group, n_samples, logger):
    """Load gauge configuration labels with fallback to indices."""
    try:
        if "gauge_configuration_labels" in group:
            labels_obj = group["gauge_configuration_labels"]
            if isinstance(labels_obj, h5py.Dataset):
                labels = [str(label) for label in labels_obj[()]]
                if len(labels) >= n_samples:
                    return labels[:n_samples]

        # Fallback to sample indices
        logger.warning("Using sample indices as configuration labels")
        return [f"Sample_{i:03d}" for i in range(n_samples)]

    except Exception as e:
        logger.warning(f"Error loading configuration labels: {e}, using indices")
        return [f"Sample_{i:03d}" for i in range(n_samples)]


def validate_dataset_shapes(samples_data, mean_values, error_values, group_path):
    """Validate that dataset shapes are consistent."""
    n_samples, n_time_points = samples_data.shape

    if len(mean_values) != n_time_points:
        raise ValueError(
            f"Shape mismatch in {group_path}: samples have {n_time_points} time points, "
            f"but mean has {len(mean_values)}"
        )

    if len(error_values) != n_time_points:
        raise ValueError(
            f"Shape mismatch in {group_path}: samples have {n_time_points} time points, "
            f"but error has {len(error_values)}"
        )


def create_time_index(n_time_points, time_offset):
    """Create time index array with proper offset."""
    return np.arange(n_time_points) + time_offset


def create_correlator_plot(
    time_index,
    samples_data,
    sample_labels,
    mean_values,
    error_values,
    group_name,
    sample_indices,
    analysis_config,
    logger,
):
    """Create a correlator plot with multiple samples and average."""
    plot_config = analysis_config["plot_config"]

    # Apply dataset-specific slicing
    sliced_time, sliced_samples, sliced_mean, sliced_error = apply_dataset_slicing(
        time_index, samples_data, mean_values, error_values, plot_config
    )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    # Plot each sample
    sample_style = SAMPLE_PLOT_STYLE.copy()
    sample_style.pop("label_suffix", None)  # Remove non-matplotlib parameter

    for i, (sample_data, sample_label) in enumerate(zip(sliced_samples, sample_labels)):
        color = get_sample_color(sample_indices[0] + i)
        ax.plot(
            sliced_time,
            sample_data,
            label=sample_label,
            color=color,
            **sample_style,
        )

    # Plot average with error bars
    average_style = AVERAGE_PLOT_STYLE.copy()
    avg_label = average_style.pop("label", "Jackknife average")

    # Validate data arrays
    if sliced_time is None or sliced_mean is None or sliced_error is None:
        raise ValueError("One or more data arrays is None - cannot create plot")

    if len(sliced_time) == 0 or len(sliced_mean) == 0:
        raise ValueError("Empty data arrays - cannot create plot")

    ax.errorbar(
        sliced_time,
        sliced_mean,
        yerr=sliced_error,
        label=avg_label,
        **average_style,
    )

    # Add zero line if configured
    if plot_config.get("show_zero_line", False):
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    # Set axis properties
    ax.set_yscale(plot_config["y_scale"])
    ax.set_xlabel(plot_config["x_label"], fontsize=DEFAULT_FONT_SIZE)
    ax.set_ylabel(plot_config["y_label"], fontsize=DEFAULT_FONT_SIZE)

    # Add legend
    ax.legend()

    # Set title
    start_idx, end_idx = sample_indices
    title = f"{group_name} - Samples {start_idx} to {end_idx}"
    ax.set_title(title, fontsize=DEFAULT_FONT_SIZE + 2)

    return fig, ax


def create_multi_sample_plots(
    samples_data,
    mean_values,
    error_values,
    config_labels,
    group_name,
    group_path,
    analysis_config,
    group_plots_dir,
    logger,
    verbose,
):
    """Create multiple plots with specified number of samples per
    plot."""
    n_samples, n_time_points = samples_data.shape
    n_plots = (n_samples + SAMPLES_PER_PLOT - 1) // SAMPLES_PER_PLOT
    plots_created = 0

    # Create time index with analysis-specific offset
    time_index = create_time_index(n_time_points, analysis_config["time_offset"])

    if verbose:
        print(
            f"    Creating {n_plots} multi-sample plots ({SAMPLES_PER_PLOT} samples each)"
        )

    for plot_idx in range(n_plots):
        start_sample = plot_idx * SAMPLES_PER_PLOT
        end_sample = min(start_sample + SAMPLES_PER_PLOT, n_samples)

        try:
            # Extract samples for this plot
            plot_samples_data = samples_data[start_sample:end_sample, :]
            plot_sample_labels = config_labels[start_sample:end_sample]

            # Create the plot
            fig, _ = create_correlator_plot(
                time_index,
                plot_samples_data,
                plot_sample_labels,
                mean_values,
                error_values,
                group_name,
                (start_sample, end_sample - 1),
                analysis_config,
                logger,
            )

            # Generate filename
            analysis_type = next(
                k for k, v in analysis_config.items() if v == analysis_config
            )
            filename = generate_plot_filename(
                group_name, start_sample, end_sample - 1, "correlator"
            )

            # Save plot
            full_path = os.path.join(group_plots_dir, filename)
            fig.savefig(full_path, **PLOT_QUALITY)
            plt.close(fig)

            plots_created += 1

            if verbose:
                print(f"      Created plot {plot_idx + 1}/{n_plots}: {filename}")

        except Exception as e:
            logger.error(
                f"Error creating plot {plot_idx} for samples "
                f"{start_sample}-{end_sample-1}: {e}"
            )
            continue

    return plots_created


def load_correlator_datasets(group, analysis_config, group_path, logger):
    """Load correlator datasets based on analysis configuration."""
    # Load samples dataset
    samples_dataset_name = analysis_config["samples_dataset"]
    try:
        samples_obj = group[samples_dataset_name]
        if not isinstance(samples_obj, h5py.Dataset):
            raise ValueError(f"'{samples_dataset_name}' is not a dataset")
        samples_data = samples_obj[()]
    except KeyError:
        raise ValueError(
            f"Dataset '{samples_dataset_name}' not found in group '{group_path}'"
        )

    # Load mean values dataset
    mean_dataset_name = analysis_config["mean_dataset"]
    try:
        mean_obj = group[mean_dataset_name]
        if not isinstance(mean_obj, h5py.Dataset):
            raise ValueError(f"'{mean_dataset_name}' is not a dataset")
        mean_values = mean_obj[()]
    except KeyError:
        raise ValueError(
            f"Dataset '{mean_dataset_name}' not found in group '{group_path}'"
        )

    # Load error values dataset
    error_dataset_name = analysis_config["error_dataset"]
    try:
        error_obj = group[error_dataset_name]
        if not isinstance(error_obj, h5py.Dataset):
            raise ValueError(f"'{error_dataset_name}' is not a dataset")
        error_values = error_obj[()]
    except KeyError:
        raise ValueError(
            f"Dataset '{error_dataset_name}' not found in group '{group_path}'"
        )

    # Validate shapes
    validate_dataset_shapes(samples_data, mean_values, error_values, group_path)

    logger.debug(
        f"Loaded datasets: samples {samples_data.shape}, "
        f"mean {mean_values.shape}, error {error_values.shape}"
    )

    return samples_data, mean_values, error_values


def prepare_group_output_directory(base_plots_dir, group_name, clear_existing, logger):
    """Prepare output directory for a specific group."""
    group_plots_dir = os.path.join(base_plots_dir, group_name)

    if clear_existing and os.path.exists(group_plots_dir):
        import shutil

        shutil.rmtree(group_plots_dir)
        logger.debug(f"Cleared existing plots for group: {group_name}")

    os.makedirs(group_plots_dir, exist_ok=True)

    return group_plots_dir


def process_correlator_group(
    group_path,
    hdf5_file,
    base_plots_dir,
    analysis_config,
    clear_existing,
    logger,
    verbose,
):
    """Process a single correlator group for visualization."""
    try:
        # Verify group exists and is valid
        group = hdf5_file[group_path]
        if not isinstance(group, h5py.Group):
            logger.error(f"Path '{group_path}' is not an HDF5 group")
            return 0
    except KeyError:
        logger.error(f"Group path '{group_path}' not found in HDF5 file")
        return 0

    # Extract group name for directory structure
    group_name = group_path.split("/")[-1]

    try:
        # Load correlator datasets
        samples_data, mean_values, error_values = load_correlator_datasets(
            group, analysis_config, group_path, logger
        )

        # Load configuration labels
        config_labels = load_gauge_configuration_labels(
            group, samples_data.shape[0], logger
        )

        # Prepare output directory
        group_plots_dir = prepare_group_output_directory(
            base_plots_dir, group_name, clear_existing, logger
        )

        # Create multi-sample plots
        plots_created = create_multi_sample_plots(
            samples_data,
            mean_values,
            error_values,
            config_labels,
            group_name,
            group_path,
            analysis_config,
            group_plots_dir,
            logger,
            verbose,
        )

        if verbose:
            print(f"  ✓ Group {group_name}: Created {plots_created} plots")

        return plots_created

    except Exception as e:
        logger.error(f"Error processing group {group_path}: {e}")
        return 0


def prepare_base_output_directory(
    output_directory, analysis_config, clear_existing, logger
):
    """Prepare base output directory for plots."""
    base_plots_dir = os.path.join(
        output_directory, analysis_config["plot_base_directory"]
    )

    if clear_existing and os.path.exists(base_plots_dir):
        logger.info(f"Clearing existing plots directory: {base_plots_dir}")
        import shutil

        shutil.rmtree(base_plots_dir)

    os.makedirs(base_plots_dir, exist_ok=True)
    logger.info(f"Base plots directory: {base_plots_dir}")

    return base_plots_dir


def find_correlator_groups(hdf5_file, analysis_config, logger):
    """Find all groups containing the required correlator datasets."""
    required_datasets = [
        analysis_config["samples_dataset"],
        analysis_config["mean_dataset"],
        analysis_config["error_dataset"],
    ]

    valid_groups = []

    def find_groups(name, obj):
        if isinstance(obj, h5py.Group):
            # Check if this group contains all required datasets
            has_all_datasets = all(dataset in obj for dataset in required_datasets)
            if has_all_datasets:
                valid_groups.append(name)

    hdf5_file.visititems(find_groups)

    logger.info(f"Found {len(valid_groups)} groups with required datasets")
    logger.debug(f"Valid groups: {valid_groups}")

    return valid_groups
