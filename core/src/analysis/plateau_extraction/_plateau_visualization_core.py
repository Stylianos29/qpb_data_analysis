#!/usr/bin/env python3
"""
Core functions for plateau extraction visualization.

This module provides the core plotting functions to create high-quality
multi-panel visualizations of plateau extraction results, matching the
output quality of the original extract_plateau_PCAC_mass.py script.
"""
import os
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
import h5py

# Import library components
from library.visualization.builders.title_builder import PlotTitleBuilder
from library.visualization.managers.file_manager import PlotFileManager
from library.constants.labels import AXES_LABELS_BY_COLUMN_NAME

# Import configuration
from src.analysis.plateau_extraction._plateau_visualization_config import (
    get_layout_config,
    get_plot_styling,
    get_annotation_config,
    get_axes_config,
    get_output_config,
    get_mass_type_for_template,
    get_subplot_grid_size,
)


# =============================================================================
# DATA LOADING AND PROCESSING FUNCTIONS
# =============================================================================


def load_extraction_results_from_group(
    group: h5py.Group,
    analysis_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Load plateau extraction results from an HDF5 group.

    Args:
        - group: HDF5 group containing extraction data
        - analysis_config: Analysis-specific configuration

    Returns:
        List of extraction result dictionaries for each sample
    """
    # Validate that group is indeed an HDF5 group
    if not isinstance(group, h5py.Group):
        raise ValueError(f"Expected h5py.Group, got {type(group)}")

    datasets = analysis_config["input_datasets"]
    time_offset = analysis_config["time_offset"]

    # Validate and load time series data
    if datasets["time_series"] not in group:
        raise ValueError(f"Dataset '{datasets['time_series']}' not found in group")

    time_series_dataset = group[datasets["time_series"]]
    if not isinstance(time_series_dataset, h5py.Dataset):
        raise ValueError(f"'{datasets['time_series']}' is not an HDF5 dataset")

    time_series_data = time_series_dataset[:]
    n_samples, n_time_points = time_series_data.shape

    # Validate and load plateau estimates
    if datasets["plateau_estimates"] not in group:
        raise ValueError(
            f"Dataset '{datasets['plateau_estimates']}' not found in group"
        )

    plateau_estimates_dataset = group[datasets["plateau_estimates"]]
    if not isinstance(plateau_estimates_dataset, h5py.Dataset):
        raise ValueError(f"'{datasets['plateau_estimates']}' is not an HDF5 dataset")

    plateau_estimates = plateau_estimates_dataset[:]

    # Validate and load individual sigma thresholds
    if datasets["sigma_thresholds"] not in group:
        raise ValueError(f"Dataset '{datasets['sigma_thresholds']}' not found in group")

    sigma_thresholds_dataset = group[datasets["sigma_thresholds"]]
    if not isinstance(sigma_thresholds_dataset, h5py.Dataset):
        raise ValueError(f"'{datasets['sigma_thresholds']}' is not an HDF5 dataset")

    sigma_thresholds = sigma_thresholds_dataset[:]

    # Validate and load configuration labels
    config_labels = []
    if datasets["config_labels"] in group:
        config_labels_dataset = group[datasets["config_labels"]]
        if not isinstance(config_labels_dataset, h5py.Dataset):
            raise ValueError(f"'{datasets['config_labels']}' is not an HDF5 dataset")

        labels_data = config_labels_dataset[:]
        config_labels = [
            label.decode("utf-8") if isinstance(label, bytes) else str(label)
            for label in labels_data
        ]
    else:
        # Generate default labels
        config_labels = [f"Sample_{i+1:03d}" for i in range(n_samples)]

    # Load plateau bounds from group attributes
    plateau_start = None
    plateau_end = None
    plateau_error = None

    if "plateau_start" in group.attrs:
        # HDF5 attributes are scalar but could be different numeric
        # types Convert to numpy scalar first, then to Python int/float
        plateau_start_raw = np.asarray(group.attrs["plateau_start"]).item()
        plateau_end_raw = np.asarray(group.attrs["plateau_end"]).item()
        plateau_error_raw = np.asarray(group.attrs.get("plateau_error", 0.0)).item()

        plateau_start = int(plateau_start_raw)
        plateau_end = int(plateau_end_raw)
        plateau_error = float(plateau_error_raw)

    # Create time index array
    time_index = np.arange(n_time_points) + time_offset

    # Package results for each sample
    extraction_results = []
    for i in range(n_samples):
        result = {
            "sample_index": i,
            "config_label": (
                config_labels[i] if i < len(config_labels) else f"Sample_{i+1:03d}"
            ),
            "time_series": time_series_data[i, :],
            "time_index": time_index,
            "plateau_value": plateau_estimates[i],
            "plateau_error": plateau_error,  # Common error for all samples
            "sigma_threshold": sigma_thresholds[i],
            "plateau_bounds": (
                (plateau_start, plateau_end) if plateau_start is not None else None
            ),
        }
        extraction_results.append(result)

    return extraction_results


def split_extractions_into_figures(
    extractions: List[Dict[str, Any]], max_samples_per_figure: int
) -> List[List[Dict[str, Any]]]:
    """
    Split extraction results into groups for separate figures.

    Args:
        - extractions: List of extraction result dictionaries
        - max_samples_per_figure: Maximum samples per figure

    Returns:
        List of extraction result lists (one per figure)
    """
    if not extractions:
        return []

    figures = []
    for i in range(0, len(extractions), max_samples_per_figure):
        end_idx = min(i + max_samples_per_figure, len(extractions))
        figures.append(extractions[i:end_idx])

    return figures


# =============================================================================
# CORE PLOTTING FUNCTIONS
# =============================================================================


def create_multi_panel_figure(
    n_panels: int,
    analysis_config: Dict[str, Any],
    group_metadata: Dict[str, Any],
    title_builder: PlotTitleBuilder,
    layout_config: Dict[str, Any],
) -> Tuple[Figure, List[Axes]]:
    """
    Create a multi-panel figure with proper layout and styling.

    Args:
        - n_panels: Number of panels to create
        - analysis_config: Analysis-specific configuration
        - group_metadata: Metadata for title generation
        - title_builder: PlotTitleBuilder instance
        - layout_config: Layout configuration

    Returns:
        Tuple of (figure, axes_list)
    """
    # Calculate subplot grid
    n_rows, n_cols = get_subplot_grid_size(n_panels)

    # Create figure
    fig_width, fig_height = layout_config["figure_size"]
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Adjust subplot parameters
    spacing = layout_config["subplot_spacing"]
    fig.subplots_adjust(
        hspace=spacing["hspace"],
        wspace=spacing["wspace"],
        top=spacing["top"],
        bottom=spacing["bottom"],
        left=spacing["left"],
        right=spacing["right"],
    )

    # Create subplots
    axes = []
    for i in range(n_panels):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        axes.append(ax)

    # Generate and set main title
    title_parts = {
        "analysis_type": analysis_config["title_prefix"],
        "metadata": group_metadata,
    }

    # Create title using PlotTitleBuilder
    # Use all metadata parameters as tunable params, with proper exclusions
    tunable_params = list(group_metadata.keys())
    excluded_params = {
        # Exclude extraction-specific attributes
        "plateau_extraction_success",
        "n_samples",
        "n_time_points",
        "plateau_mean",
        "plateau_error",
        "plateau_start",
        "plateau_end",
        "sigma_threshold_used",
    }

    main_title = title_builder.build(
        metadata_dict=group_metadata,
        tunable_params=tunable_params,
        excluded=excluded_params,
        leading_substring=analysis_config["title_prefix"],
        wrapping_length=80,
    )

    fig.suptitle(main_title, fontsize=get_plot_styling()["fonts"]["title_size"])

    return fig, axes


def plot_single_panel(
    ax: Axes,
    extraction_result: Dict[str, Any],
    analysis_config: Dict[str, Any],
    is_bottom_panel: bool = False,
) -> None:
    """
    Plot a single panel showing plateau extraction for one sample.

    Args:
        - ax: Matplotlib axes object
        - extraction_result: Data for one sample
        - analysis_config: Analysis-specific configuration
        - is_bottom_panel: Whether this is the bottom panel (for x-axis label)
    """
    styling = get_plot_styling()
    axes_config = get_axes_config()

    # Extract data
    time_index = extraction_result["time_index"]
    time_series = extraction_result["time_series"]
    plateau_value = extraction_result["plateau_value"]
    plateau_bounds = extraction_result["plateau_bounds"]
    config_label = extraction_result["config_label"]

    # Plot time series data (blue dots)
    time_series_style = styling["time_series"]
    ax.plot(
        time_index,
        time_series,
        marker=time_series_style["marker"],
        markersize=time_series_style["markersize"],
        linestyle=time_series_style["linestyle"],
        color=time_series_style["color"],
        alpha=time_series_style["alpha"],
        markerfacecolor=time_series_style["markerfacecolor"],
        markeredgecolor=time_series_style["markeredgecolor"],
        markeredgewidth=time_series_style["markeredgewidth"],
    )

    # Add plateau region (green shaded area)
    if plateau_bounds is not None:
        add_plateau_region_shading(ax, plateau_bounds, time_index, styling)

        # Add plateau fit line (red dashed horizontal line)
        add_plateau_fit_line(ax, plateau_bounds, time_index, plateau_value, styling)

    # Configure axes styling
    configure_axes_styling(ax, styling["axes"])

    # Set axis labels
    set_axis_labels(ax, analysis_config, axes_config, is_bottom_panel)

    # Add plateau information text box
    add_plateau_annotations(ax, extraction_result, analysis_config)

    # Add configuration label
    add_config_label(ax, config_label)

    # Configure ticks
    configure_tick_formatting(ax, axes_config)


def add_plateau_region_shading(
    ax: Axes,
    plateau_bounds: Tuple[int, int],
    time_index: np.ndarray,
    styling: Dict[str, Any],
) -> None:
    """Add green shaded region for plateau fitting range."""
    plateau_start, plateau_end = plateau_bounds

    # Convert bounds to time values
    t_start = time_index[plateau_start]
    t_end = time_index[
        plateau_end - 1
    ]  # end is exclusive, so -1 for last included point

    # Get y-axis limits to fill full height
    y_min, y_max = ax.get_ylim()
    if y_min == y_max:  # Handle case where limits aren't set yet
        y_min, y_max = np.min(ax.get_lines()[0].get_ydata()), np.max(
            ax.get_lines()[0].get_ydata()
        )
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range

    # Add shaded rectangle
    plateau_style = styling["plateau_region"]
    rect = patches.Rectangle(
        (t_start - 0.5, y_min),  # Start at t_start - 0.5 to center on time points
        (t_end - t_start + 1),  # Width spanning the plateau region
        (y_max - y_min),  # Full height
        facecolor=plateau_style["facecolor"],
        alpha=plateau_style["alpha"],
        edgecolor=plateau_style["edgecolor"],
        linewidth=plateau_style["linewidth"],
        linestyle=plateau_style["linestyle"],
        zorder=0,  # Behind data points
    )
    ax.add_patch(rect)


def add_plateau_fit_line(
    ax: Axes,
    plateau_bounds: Tuple[int, int],
    time_index: np.ndarray,
    plateau_value: float,
    styling: Dict[str, Any],
) -> None:
    """Add red dashed horizontal line at plateau value."""
    plateau_start, plateau_end = plateau_bounds

    # Convert bounds to time values
    t_start = time_index[plateau_start]
    t_end = time_index[plateau_end - 1]

    # Draw horizontal line
    fit_style = styling["plateau_fit"]
    ax.plot(
        [t_start - 0.5, t_end + 0.5],  # Extend slightly beyond bounds
        [plateau_value, plateau_value],
        color=fit_style["color"],
        linestyle=fit_style["linestyle"],
        linewidth=fit_style["linewidth"],
        alpha=fit_style["alpha"],
        zorder=2,  # Above shaded region but below data points
    )


def add_plateau_annotations(
    ax: Axes,
    extraction_result: Dict[str, Any],
    analysis_config: Dict[str, Any],
) -> None:
    """Add text box with plateau information."""
    annotation_config = get_annotation_config()
    box_config = annotation_config["plateau_info_box"]

    # Extract values
    plateau_value = extraction_result["plateau_value"]
    plateau_error = extraction_result.get("plateau_error", 0.0)
    sigma_threshold = extraction_result["sigma_threshold"]
    plateau_bounds = extraction_result["plateau_bounds"]

    # Calculate number of fit points
    n_fit_points = 0
    if plateau_bounds is not None:
        n_fit_points = plateau_bounds[1] - plateau_bounds[0]

    # Get mass type for template
    mass_type = get_mass_type_for_template(
        "pcac_mass" if "PCAC" in analysis_config["column_prefix"] else "pion_mass"
    )

    # Format annotation text
    annotation_text = box_config["template"].format(
        mass_type=mass_type,
        plateau_mean=plateau_value,
        plateau_error=plateau_error,
        sigma_threshold=sigma_threshold,
        n_fit_points=n_fit_points,
        precision=box_config["precision"],
    )

    # Add text box
    ax.text(
        box_config["position"]["x"],
        box_config["position"]["y"],
        annotation_text,
        transform=ax.transAxes,
        fontsize=box_config["font_props"]["size"],
        fontfamily=box_config["font_props"]["family"],
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=box_config["bbox_props"],
    )


def add_config_label(ax: Axes, config_label: str) -> None:
    """Add configuration label to top-right corner."""
    annotation_config = get_annotation_config()
    label_config = annotation_config["config_label"]

    if not label_config["show"]:
        return

    label_text = label_config["template"].format(config_label=config_label)

    ax.text(
        label_config["position"]["x"],
        label_config["position"]["y"],
        label_text,
        transform=ax.transAxes,
        fontsize=label_config["font_props"]["size"],
        horizontalalignment=label_config["font_props"]["horizontalalignment"],
        verticalalignment=label_config["font_props"]["verticalalignment"],
    )


def configure_axes_styling(ax: Axes, axes_style: Dict[str, Any]) -> None:
    """Configure axes styling (grid, spines, etc.)."""
    # Configure grid
    if axes_style["grid"]:
        ax.grid(
            True,
            alpha=axes_style["grid_alpha"],
            color=axes_style["grid_color"],
            linestyle=axes_style["grid_linestyle"],
            linewidth=axes_style["grid_linewidth"],
        )

    # Configure spines
    for spine in ax.spines.values():
        spine.set_color(axes_style["spines_color"])
        spine.set_linewidth(axes_style["spines_linewidth"])


def set_axis_labels(
    ax: Axes,
    analysis_config: Dict[str, Any],
    axes_config: Dict[str, Any],
    is_bottom_panel: bool,
) -> None:
    """Set axis labels according to configuration."""
    styling = get_plot_styling()

    # Y-axis label (on all panels or left only)
    y_config = axes_config["y_axis"]
    if y_config["show_label_on"] == "all":
        ax.set_ylabel(
            analysis_config["y_label"], fontsize=styling["fonts"]["axis_label_size"]
        )

    # X-axis label (bottom panel only)
    x_config = axes_config["x_axis"]
    if is_bottom_panel and x_config["show_label_on"] == "bottom_only":
        # Use the time_index label from constants
        x_label = AXES_LABELS_BY_COLUMN_NAME.get(x_config["label_key"], "$t/a$")
        ax.set_xlabel(x_label, fontsize=styling["fonts"]["axis_label_size"])


def configure_tick_formatting(ax: Axes, axes_config: Dict[str, Any]) -> None:
    """Configure tick formatting and limits."""
    styling = get_plot_styling()

    # X-axis: force integer ticks
    x_config = axes_config["x_axis"]
    if x_config["tick_format"] == "integer":
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Set tick label font sizes
    ax.tick_params(
        axis="both", which="major", labelsize=styling["fonts"]["tick_label_size"]
    )

    # Set x-axis to start from 0 if possible
    xlim = ax.get_xlim()
    ax.set_xlim(left=max(0, xlim[0]))


# =============================================================================
# HIGH-LEVEL FIGURE CREATION FUNCTIONS
# =============================================================================


def create_plateau_extraction_figure(
    extractions: List[Dict[str, Any]],
    analysis_config: Dict[str, Any],
    group_metadata: Dict[str, Any],
    title_builder: PlotTitleBuilder,
    figure_index: int = 0,
) -> Figure:
    """
    Create a complete multi-panel figure showing plateau extractions.

    Args:
        - extractions: List of extraction results for this figure
        - analysis_config: Analysis-specific configuration
        - group_metadata: Metadata for title generation
        - title_builder: PlotTitleBuilder instance
        - figure_index: Index of this figure (for multi-figure groups)

    Returns:
        Complete matplotlib Figure
    """
    layout_config = get_layout_config()
    n_panels = len(extractions)

    # Create figure and axes
    fig, axes = create_multi_panel_figure(
        n_panels, analysis_config, group_metadata, title_builder, layout_config
    )

    # Plot each panel
    for i, (ax, extraction) in enumerate(zip(axes, extractions)):
        is_bottom = (
            (i >= n_panels - len(axes) % get_subplot_grid_size(n_panels)[1])
            if n_panels > 1
            else True
        )
        # Simpler bottom detection: is this one of the bottom row panels?
        n_rows, n_cols = get_subplot_grid_size(n_panels)
        is_bottom = i >= (n_rows - 1) * n_cols

        plot_single_panel(ax, extraction, analysis_config, is_bottom)

    # Note: matplotlib only creates the subplots we explicitly request,
    # so no need to hide unused grid positions

    plt.tight_layout()
    return fig


def save_plateau_extraction_figure(
    fig: Figure,
    group_name: str,
    start_idx: int,
    end_idx: int,
    file_manager: PlotFileManager,
    analysis_config: Dict[str, Any],
    layout_config: Dict[str, Any],
) -> str:
    """
    Save plateau extraction figure with proper filename and subdirectory.

    Args:
        - fig: Matplotlib figure to save
        - group_name: Name of the parameter group
        - start_idx: Starting sample index
        - end_idx: Ending sample index
        - file_manager: PlotFileManager instance
        - analysis_config: Analysis-specific configuration
        - layout_config: Layout configuration

    Returns:
        Path to saved file
    """
    output_config = get_output_config()

    # Generate filename with zero-padded indices
    filename = output_config["filename_template"].format(
        group_name=group_name,
        start_idx=start_idx,
        end_idx=end_idx,
        format=output_config["file_format"],
    )

    # Create group-specific subdirectory if enabled
    if layout_config.get("create_group_subdirectories", True):
        # Create subdirectory path: analysis_subdir/group_name/
        plot_subdir = analysis_config["plot_subdirectory"]
        group_subdir_path = f"{plot_subdir}/{group_name}"
        final_subdir_path = file_manager.prepare_subdirectory(group_subdir_path)
    else:
        # Use analysis subdirectory only
        plot_subdir = analysis_config["plot_subdirectory"]
        final_subdir_path = file_manager.prepare_subdirectory(plot_subdir)

    # Construct final output path
    output_path = file_manager.plot_path(
        final_subdir_path,
        filename.rsplit(".", 1)[0],  # Remove extension since plot_path adds it
        format=output_config["file_format"],
    )

    # Save figure
    fig.savefig(
        output_path,
        format=output_config["file_format"],
        dpi=output_config["dpi"],
        bbox_inches=output_config["bbox_inches"],
        pad_inches=output_config["pad_inches"],
        facecolor=output_config["facecolor"],
        transparent=output_config["transparent"],
    )

    plt.close(fig)  # Free memory
    return output_path


# =============================================================================
# MAIN ORCHESTRATION FUNCTIONS
# =============================================================================


def process_group_visualization(
    group: h5py.Group,
    group_name: str,
    analysis_config: Dict[str, Any],
    title_builder: PlotTitleBuilder,
    file_manager: PlotFileManager,
    logger,
) -> List[str]:
    """
    Process visualization for a single parameter group.

    Args:
        - group: HDF5 group containing extraction data
        - group_name: Name of the group
        - analysis_config: Analysis-specific configuration
        - title_builder: PlotTitleBuilder instance
        - file_manager: PlotFileManager instance
        - logger: Logger instance

    Returns:
        List of created plot file paths
    """
    logger.info(f"Processing visualization for group: {group_name}")

    try:
        # Load extraction results
        extractions = load_extraction_results_from_group(group, analysis_config)

        if not extractions:
            logger.warning(f"No extraction results found for group {group_name}")
            return []

        logger.debug(f"Loaded {len(extractions)} extraction results")

        # Extract group metadata for titles
        group_metadata = dict(group.attrs)

        # Split into multiple figures if needed
        layout_config = get_layout_config()
        max_per_figure = layout_config["samples_per_figure"]
        figure_groups = split_extractions_into_figures(extractions, max_per_figure)

        logger.info(
            f"Creating {len(figure_groups)} figures for {len(extractions)} samples"
        )

        # Create and save figures
        saved_paths = []
        for fig_idx, figure_extractions in enumerate(figure_groups):
            # Calculate sample indices for filename
            start_idx = figure_extractions[0]["sample_index"]
            end_idx = figure_extractions[-1]["sample_index"]

            # Create figure
            fig = create_plateau_extraction_figure(
                figure_extractions,
                analysis_config,
                group_metadata,
                title_builder,
                fig_idx,
            )

            # Save figure with group subdirectory
            output_path = save_plateau_extraction_figure(
                fig,
                group_name,
                start_idx,
                end_idx,
                file_manager,
                analysis_config,
                layout_config,
            )

            saved_paths.append(output_path)
            logger.debug(f"Saved figure: {os.path.basename(output_path)}")

        logger.info(
            f"Successfully created {len(saved_paths)} plots for group {group_name}"
        )
        return saved_paths

    except Exception as e:
        logger.error(f"Failed to process visualization for group {group_name}: {e}")
        return []
