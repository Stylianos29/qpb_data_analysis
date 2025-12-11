#!/usr/bin/env python3
"""
Script to analyze pion correlator HDF5 data and create plots grouped by KL_diagonal_order and Kernel_operator_type.

This script:
1. Parses group names to extract parameters (KL_diagonal_order, Kernel_operator_type, etc.)
2. Groups data by KL_diagonal_order and Kernel_operator_type 
3. Extracts g5-g5 correlator data
4. Creates plots with different colors/shapes for each group
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import re
from collections import defaultdict
from library.data.hdf5_analyzer import HDF5Analyzer


def parse_group_name(group_name):
    """
    Parse group name to extract parameters.

    Example: 'KL_Brillouin_cSW0_rho1p0_mu1p0_EpsMSCG1e-12_EpsCG1e-12_m0p15_config0029800_n1_partial_fraction.dat'

    Returns:
        dict: Dictionary with extracted parameters
    """
    params = {}

    print("GROUP_NAME:", group_name)

    # Extract kernel operator type (Wilson or Brillouin)
    if "Standard" in group_name:
        params["Kernel_operator_type"] = "Wilson"
    elif "Brillouin" in group_name:
        params["Kernel_operator_type"] = "Brillouin"
    else:
        params["Kernel_operator_type"] = "Unknown"

    # Extract KL_diagonal_order (n value)
    n_match = re.search(r"_n(\d+)_", group_name)
    if n_match:
        params["KL_diagonal_order"] = int(n_match.group(1))
    else:
        params["KL_diagonal_order"] = None

    # Extract MSCG_epsilon
    mscg_match = re.search(r"EpsMSCG([0-9e\-\.]+)_", group_name)
    if mscg_match:
        eps_str = mscg_match.group(1)
        # Convert scientific notation format (e.g., "1e-12")
        params["MSCG_epsilon"] = float(eps_str)
    else:
        params["MSCG_epsilon"] = None

    # Extract fraction type (single_fraction, partial_fraction, or multiply_up_trick)
    if "multiply_up_trick" in group_name:
        params["fraction_type"] = "multiply_up_trick"
    elif "single_fraction" in group_name:
        params["fraction_type"] = "single_fraction"
    elif "partial_fraction" in group_name:
        params["fraction_type"] = "partial_fraction"
    else:
        params["fraction_type"] = "unknown"

    # Extract other parameters if needed
    # Mass
    mass_match = re.search(r"_m([\d\.p]+)_", group_name)
    if mass_match:
        mass_str = mass_match.group(1).replace("p", ".")
        params["mass"] = float(mass_str)

    # Config
    config_match = re.search(r"_config(\d+)_", group_name)
    if config_match:
        params["config"] = config_match.group(1)

    return params


def analyze_and_plot_pion_correlators(hdf5_file_path):
    """
    Main function to analyze HDF5 pion correlator data and create plots.

    Args:
        hdf5_file_path: Path to the HDF5 file
    """
    # Initialize the analyzer
    print("Loading HDF5 file...")
    analyzer = HDF5Analyzer(hdf5_file_path)

    # Print basic information about the file
    print(f"\n{analyzer}")
    print("\n" + "=" * 60)

    # Get all groups and parse their names
    print("PARSING GROUP NAMES:")
    print("=" * 60)

    grouped_data = defaultdict(
        lambda: defaultdict(list)
    )  # {(kernel_type, n_value, mscg_eps): {fraction_type: [group_paths]}}
    all_params = []

    for group_path in analyzer.active_groups:
        params = parse_group_name(group_path)
        all_params.append((group_path, params))

        kernel_type = params.get("Kernel_operator_type")
        n_value = params.get("KL_diagonal_order")
        mscg_eps = params.get("MSCG_epsilon")
        fraction_type = params.get("fraction_type")

        if (
            kernel_type
            and n_value is not None
            and mscg_eps is not None
            and fraction_type
        ):
            key = (kernel_type, n_value, mscg_eps)
            grouped_data[key][fraction_type].append(group_path)

        print(f"  {group_path}")
        print(
            f"    -> Kernel: {kernel_type}, n: {n_value}, MSCG_eps: {mscg_eps}, fraction: {fraction_type}"
        )

    print(
        f"\nFound {len(grouped_data)} unique (Kernel_operator_type, KL_diagonal_order, MSCG_epsilon) combinations"
    )

    # Extract g5-g5 correlator data
    print("\nExtracting g5-g5 correlator data...")
    correlator_data = {}

    for (kernel_type, n_value, mscg_eps), fraction_groups in grouped_data.items():
        print(f"\nProcessing Kernel={kernel_type}, n={n_value}, MSCG_eps={mscg_eps}")

        group_correlators = {}

        for fraction_type, group_paths in fraction_groups.items():
            print(f"  {fraction_type}: {len(group_paths)} groups")

            fraction_correlators = []
            for group_path in group_paths:
                try:
                    # Get g5-g5 correlator data (no gvar merging needed)
                    g5g5_data = analyzer.dataset_values(
                        "g5-g5",
                        return_gvar=False,  # No gvar merging needed
                        group_path=group_path,
                    )
                    fraction_correlators.append(g5g5_data)
                    print(f"    Successfully extracted g5-g5 from {group_path}")

                except ValueError as e:
                    print(
                        f"    Warning: Could not extract g5-g5 from {group_path}: {e}"
                    )
                    continue

            if fraction_correlators:
                group_correlators[fraction_type] = fraction_correlators

        if group_correlators:
            correlator_data[(kernel_type, n_value, mscg_eps)] = group_correlators

    # Create plots
    if correlator_data:
        create_pion_correlator_plots(correlator_data)
    else:
        print("No correlator data found to plot!")

    return analyzer


def create_pion_correlator_plots(correlator_data):
    """
    Create plots of g5-g5 correlators with one plot per (Kernel_operator_type, KL_diagonal_order, MSCG_epsilon) combination.

    Args:
        correlator_data: Dictionary with (kernel_type, n_value, mscg_eps) as keys
                        and {fraction_type: [correlators]} as values
    """
    from pathlib import Path

    # Create output directory
    output_dir = Path("./test_HDF5Plotter_output")
    output_dir.mkdir(exist_ok=True)

    # Colors and markers for different fraction types
    fraction_colors = {
        "single_fraction": "blue",
        "partial_fraction": "red",
        "multiply_up_trick": "green",
    }
    fraction_markers = {
        "single_fraction": "o",
        "partial_fraction": "s",
        "multiply_up_trick": "d",
    }

    print(
        f"\nCreating plots for {len(correlator_data)} (Kernel, n, MSCG_eps) combinations..."
    )

    plots_created = 0

    # Create one plot per (kernel_type, n_value, mscg_eps) combination
    for (kernel_type, n_value, mscg_eps), fraction_data in sorted(
        correlator_data.items()
    ):
        # Set up plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        print(
            f"  Plot {plots_created + 1}: Kernel={kernel_type}, n={n_value}, MSCG_eps={mscg_eps}"
        )
        print(f"    Fraction types: {list(fraction_data.keys())}")

        for fraction_type, correlators in fraction_data.items():
            color = fraction_colors.get(fraction_type, "gray")
            marker = fraction_markers.get(fraction_type, "^")

            print(f"    {fraction_type}: {len(correlators)} correlators")

            # Plot individual correlators only (no averages)
            for j, correlator in enumerate(correlators):
                if hasattr(correlator, "__len__") and len(correlator) > 0:
                    x_vals = np.arange(len(correlator))

                    # Create label only for first occurrence
                    label = fraction_type.replace("_", " ") if j == 0 else ""

                    ax.plot(
                        x_vals,
                        correlator,
                        color=color,
                        marker=marker,
                        alpha=0.8,
                        markersize=5,
                        linestyle="None",
                        label=label,
                    )

        # Customize plot
        title = f"g5-g5 Pion Correlators: {kernel_type}, n={n_value}, MSCG_eps={mscg_eps:.0e}"
        ax.set_xlabel("Time Index", fontsize=12)
        ax.set_ylabel("g5-g5 Correlator (log scale)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Legend
        ax.legend(fontsize=11, loc="best")

        plt.tight_layout()

        # Save plot
        filename = f"g5g5_pion_correlators_{kernel_type}_n{n_value}_eps{mscg_eps:.0e}.png"  # _{plots_created + 1:03d}
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"    Saved: {filepath}")

        plt.close()
        plots_created += 1

    print(f"\nTotal plots created: {plots_created}")
    print(f"Output directory: {output_dir.absolute()}")


def inspect_pion_file_structure(hdf5_file_path):
    """
    Helper function to inspect the pion correlator HDF5 file structure.

    Args:
        hdf5_file_path: Path to the HDF5 file
    """
    analyzer = HDF5Analyzer(hdf5_file_path)

    print("PION CORRELATOR FILE INSPECTION:")
    print("=" * 60)
    print(f"File: {hdf5_file_path}")
    print(f"Total groups: {len(analyzer._all_deepest_groups)}")
    print()

    print("GROUP STRUCTURE AND DATASETS:")
    print("-" * 40)

    for i, group_path in enumerate(sorted(analyzer.active_groups)):
        params = parse_group_name(group_path)
        print(f"\nGroup {i+1}: {group_path}")
        print(f"  Parsed parameters: {params}")

        # Show available datasets
        try:
            datasets = analyzer._datasets_by_group.get(group_path, [])
            print(f"  Datasets: {datasets}")

            # Show g5-g5 data shape if available
            if "g5-g5" in datasets:
                g5g5_data = analyzer.dataset_values(
                    "g5-g5", return_gvar=False, group_path=group_path
                )
                print(f"  g5-g5 shape: {np.array(g5g5_data).shape}")

        except Exception as e:
            print(f"  Error accessing datasets: {e}")

    return analyzer


# Example usage
if __name__ == "__main__":
    # Update this path to your actual HDF5 file
    file_path = "../../../data_files/processed/invert/KL_single_Vs_partial_fraction_varying_n/pion_correlators_values.h5"

    # Check if file exists
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        print(
            "Please update the file_path variable with the correct path to your HDF5 file."
        )
    else:
        # First, inspect the file structure
        print("STEP 1: Inspecting pion correlator file structure...")
        inspect_analyzer = inspect_pion_file_structure(file_path)
        print("\n" + "=" * 80 + "\n")

        # Then run the full analysis and plotting
        print("STEP 2: Running pion correlator analysis and plotting...")
        main_analyzer = analyze_and_plot_pion_correlators(file_path)

        # Clean up
        inspect_analyzer.close()
        main_analyzer.close()
