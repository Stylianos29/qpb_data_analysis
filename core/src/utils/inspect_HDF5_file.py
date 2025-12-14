"""
HDF5 File Inspector
===================

A command-line utility for comprehensive inspection and analysis of HDF5
files.

This script provides analysis tools for examining HDF5 data structures,
including:
    - Basic information about the file structure and hierarchy
    - Dataset structure analysis with shape and dtype information
    - Gvar dataset pair detection (mean/error pairs)
    - Parameter uniqueness analysis (single vs multi-valued)
    - Optional dataset statistics (min/max/mean values)
    - Partial h5glance tree output for visual reference
    - Structural consistency checking across groups
    - Customizable output formats (text, markdown, LaTeX)

The tool leverages the HDF5Analyzer class for robust analysis and is
designed to produce concise, informative summaries similar to
inspect_csv_file.py.

Usage:
------
    python inspect_HDF5_file.py -hdf5 PATH_TO_HDF5 [options]

Required Arguments:
------------------
    -hdf5, --hdf5_file_path PATH    Path to the HDF5 file to be
    inspected

Optional Arguments:
-----------------
    -out, --output_directory DIR    Directory where output files will be
    saved --output_format FORMAT          Format of output file (txt,
    md, or tex) --output_filename NAME          Custom filename for the
    output file -v, --verbose                   Print output to console
    even when writing
                                    to file
    --dataset_statistics            Show min/max/mean for numeric
    datasets --sample_groups N               Number of groups to show in
    h5glance tree
                                    (default: 2, use 0 to disable)
    --no_uniqueness_report          Skip the parameter uniqueness report

Examples:
--------
    # Basic inspection with console output python inspect_HDF5_file.py
    -hdf5 data.h5

    # Generate a comprehensive report in markdown format python
    inspect_HDF5_file.py -hdf5 data.h5 -out reports/ --output_format md

    # Include dataset statistics python inspect_HDF5_file.py -hdf5
    data.h5 --dataset_statistics

    # Disable h5glance tree preview python inspect_HDF5_file.py -hdf5
    data.h5 --sample_groups 0

Dependencies:
------------
    - h5py: For HDF5 file access
    - numpy: For numerical operations
    - click: For command line interface
    - h5glance: Optional, for tree visualization
    - Custom library modules (HDF5Analyzer)
"""

import sys
import os
import subprocess
from typing import Optional, TextIO, Dict, List, Any, Tuple

import click
import h5py
import numpy as np

from library import HDF5Analyzer
from library.validation.click_validators import (
    hdf5_file,
    directory,
)


def get_h5glance_output(hdf5_file_path: str) -> Optional[str]:
    """
    Get h5glance tree output for the HDF5 file.

    Args:
        hdf5_file_path: Path to the HDF5 file

    Returns:
        h5glance output string or None if h5glance is not available
    """
    try:
        result = subprocess.run(
            ["h5glance", hdf5_file_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def truncate_h5glance_output(
    h5glance_output: str, num_groups: int, deepest_group_prefix: str
) -> str:
    """
    Truncate h5glance output to show only the first N deepest groups
    with datasets.

    Args:
        - h5glance_output: Full h5glance output string
        - num_groups: Number of deepest groups to include
        - deepest_group_prefix: The path prefix for deepest groups

    Returns:
        Truncated h5glance output
    """
    if not h5glance_output or num_groups <= 0:
        return ""

    lines = h5glance_output.strip().split("\n")

    # First pass: identify which lines are deepest-level groups (those
    # with datasets) Deepest groups are followed by dataset lines
    # (containing '[')
    deepest_group_indices = []
    for i, line in enumerate(lines):
        if "(" in line and "attributes)" in line:
            # Check if next line is a dataset (contains '[' for
            # dtype/shape)
            if i + 1 < len(lines) and "[" in lines[i + 1]:
                deepest_group_indices.append(i)

    if not deepest_group_indices:
        # No deepest groups found, return original
        return h5glance_output

    # Calculate the indentation of deepest groups
    if deepest_group_indices:
        first_deepest = deepest_group_indices[0]
        deepest_indent = len(lines[first_deepest]) - len(
            lines[first_deepest].lstrip("│├└─ ")
        )

    # Second pass: build truncated output
    result_lines = []
    groups_included = 0
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this is a deepest-level group
        if i in deepest_group_indices:
            if groups_included < num_groups:
                # Include this group
                result_lines.append(line)
                groups_included += 1
                i += 1

                # Include all datasets (children) of this group
                while i < len(lines):
                    next_line = lines[i]
                    # Check if still a child (dataset line) - has '['
                    # and higher indent
                    next_indent = len(next_line) - len(next_line.lstrip("│├└─ "))

                    # Stop if we hit another group at same level or
                    # shallower
                    if "attributes)" in next_line and next_indent <= deepest_indent:
                        break

                    # Include dataset lines and other children
                    result_lines.append(next_line)
                    i += 1
            else:
                # We've included enough groups, add truncation marker
                result_lines.append("    ... (remaining groups truncated)")
                break
        else:
            # Header/parent group lines - include them
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines)


def format_shape(shape: Tuple) -> str:
    """Format dataset shape for display."""
    if len(shape) == 0:
        return "scalar"
    elif len(shape) == 1:
        return str(shape[0])
    else:
        return " × ".join(str(d) for d in shape)


def format_dtype(dtype) -> str:
    """Format dtype for display."""
    dtype_str = str(dtype)
    if dtype_str.startswith("|S") or dtype_str.startswith("<U"):
        return "string"
    elif "float" in dtype_str:
        return "float64" if "64" in dtype_str else "float32"
    elif "int" in dtype_str:
        return "int64" if "64" in dtype_str else "int32"
    return dtype_str


def get_dataset_statistics(
    hdf5_file: h5py.File, dataset_paths: Dict[str, List[str]]
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate min/max/mean statistics for numeric datasets.

    Args:
        - hdf5_file: Open HDF5 file handle
        - dataset_paths: Dictionary mapping dataset names to their paths

    Returns:
        Dictionary with statistics for each dataset
    """
    stats = {}

    for dataset_name, paths in dataset_paths.items():
        if not paths:
            continue

        # Sample the first path to check dtype
        try:
            sample_dataset = hdf5_file[paths[0]]
            if not np.issubdtype(sample_dataset.dtype, np.number):
                continue  # Skip non-numeric datasets

            all_values = []
            for path in paths:
                data = hdf5_file[path][:]
                all_values.append(data.flatten())

            combined = np.concatenate(all_values)
            stats[dataset_name] = {
                "min": float(np.nanmin(combined)),
                "max": float(np.nanmax(combined)),
                "mean": float(np.nanmean(combined)),
            }
        except Exception:
            continue

    return stats


def check_structural_consistency(
    analyzer: HDF5Analyzer,
) -> Tuple[bool, List[str]]:
    """
    Check if all groups have the same dataset structure.

    Args:
        analyzer: HDF5Analyzer instance

    Returns:
        Tuple of (is_consistent, list of inconsistency messages)
    """
    inconsistencies = []

    # Get datasets by group
    datasets_by_group = analyzer._datasets_by_group

    if not datasets_by_group:
        return True, []

    # Get reference structure from first group
    reference_group = list(datasets_by_group.keys())[0]
    reference_datasets = set(datasets_by_group[reference_group])

    for group_path, datasets in datasets_by_group.items():
        current_datasets = set(datasets)

        if current_datasets != reference_datasets:
            missing = reference_datasets - current_datasets
            extra = current_datasets - reference_datasets

            if missing:
                inconsistencies.append(
                    f"Group '{group_path}' missing datasets: {sorted(missing)}"
                )
            if extra:
                inconsistencies.append(
                    f"Group '{group_path}' has extra datasets: {sorted(extra)}"
                )

    return len(inconsistencies) == 0, inconsistencies


def format_value(value: Any) -> str:
    """Format a value for display in the uniqueness report."""
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return format_value(value.item())
        else:
            return f"array{value.shape}"
    elif isinstance(value, float):
        if value == int(value):
            return str(int(value))
        else:
            return f"{value:.8g}"
    elif isinstance(value, (list, tuple)):
        return str(value)
    else:
        return str(value)


def generate_uniqueness_report(
    analyzer: HDF5Analyzer, max_width: int = 80, output_format: str = "txt"
) -> str:
    """
    Generate a formatted uniqueness report for parameters.

    Args:
        - analyzer: HDF5Analyzer instance
        - max_width: Maximum width of the report
        - output_format: Output format (txt, md, tex)

    Returns:
        Formatted uniqueness report string
    """
    half_width = (max_width - 3) // 2

    single_items = [
        (name, analyzer.unique_value_columns_dictionary[name])
        for name in analyzer.list_of_single_valued_tunable_parameter_names
    ]
    multi_items = [
        (name, analyzer.multivalued_columns_count_dictionary[name])
        for name in analyzer.list_of_multivalued_tunable_parameter_names
    ]

    # Handle empty case
    if not single_items and not multi_items:
        return "No tunable parameters found."

    max_rows = (
        max(len(single_items), len(multi_items)) if single_items or multi_items else 0
    )

    if output_format == "md":
        # Markdown table format
        lines = [
            "| Single-valued fields | Value | Multivalued fields | Count |",
            "|---------------------|-------|-------------------|-------|",
        ]

        for i in range(max_rows):
            single_name = single_items[i][0] if i < len(single_items) else ""
            single_val = (
                format_value(single_items[i][1]) if i < len(single_items) else ""
            )
            multi_name = multi_items[i][0] if i < len(multi_items) else ""
            multi_count = str(multi_items[i][1]) if i < len(multi_items) else ""

            lines.append(
                f"| {single_name} | {single_val} | {multi_name} | {multi_count} |"
            )

        return "\n".join(lines)

    elif output_format == "tex":
        # LaTeX table format
        lines = [
            r"\begin{tabular}{ll|ll}",
            r"\hline",
            r"Single-valued fields & Value & Multivalued fields & Count \\",
            r"\hline",
        ]

        for i in range(max_rows):
            single_name = (
                single_items[i][0].replace("_", r"\_") if i < len(single_items) else ""
            )
            single_val = (
                format_value(single_items[i][1]) if i < len(single_items) else ""
            )
            multi_name = (
                multi_items[i][0].replace("_", r"\_") if i < len(multi_items) else ""
            )
            multi_count = str(multi_items[i][1]) if i < len(multi_items) else ""

            lines.append(
                f"{single_name} & {single_val} & {multi_name} & {multi_count} \\\\"
            )

        lines.extend([r"\hline", r"\end{tabular}"])
        return "\n".join(lines)

    else:
        # Plain text format (default)
        header_left = "Single-valued fields: unique value"
        header_right = "Multivalued fields: No of unique values"
        header = f"{header_left:<{half_width}} | {header_right}"
        separator = "-" * max_width

        lines = [header, separator]

        for i in range(max_rows):
            left_col = ""
            right_col = ""

            if i < len(single_items):
                name, value = single_items[i]
                left_col = f"{name}: {format_value(value)}"

            if i < len(multi_items):
                name, count = multi_items[i]
                right_col = f"{name}: {count}"

            lines.append(f"{left_col:<{half_width}} | {right_col}")

        return "\n".join(lines)


@click.command()
@click.option(
    "-hdf5",
    "--hdf5_file_path",
    "hdf5_file_path",
    required=True,
    callback=hdf5_file.input,
    help="Path to the HDF5 file to be inspected.",
)
@click.option(
    "-out",
    "--output_directory",
    "output_directory",
    default=None,
    callback=directory.can_create,
    help="Directory where output files will be saved.",
)
@click.option(
    "--output_format",
    "output_format",
    default="txt",
    type=click.Choice(["txt", "md", "tex"], case_sensitive=False),
    help="Format of the output file (txt, md, or tex). Default: txt",
)
@click.option(
    "--output_filename",
    default=None,
    type=str,
    help="Custom filename for the output file (without extension).",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Print output to console even when writing to file.",
)
@click.option(
    "--dataset_statistics",
    is_flag=True,
    help="Show min/max/mean statistics for numeric datasets.",
)
@click.option(
    "--sample_groups",
    default=2,
    type=int,
    help="Number of groups to show in h5glance tree preview (0 to disable). Default: 2",
)
@click.option(
    "--no_uniqueness_report",
    is_flag=True,
    help="Skip the parameter uniqueness report section.",
)
def main(
    hdf5_file_path: str,
    output_directory: Optional[str],
    output_format: str,
    output_filename: Optional[str],
    verbose: bool,
    dataset_statistics: bool,
    sample_groups: int,
    no_uniqueness_report: bool,
):
    """
    Inspect and analyze an HDF5 file structure.

    Produces a comprehensive summary including file structure, dataset
    information, parameter analysis, and optional statistics.
    """
    # Determine output file path
    output_file: Optional[TextIO] = None
    if output_directory:
        if output_filename:
            base_filename = output_filename
        else:
            base_filename = os.path.splitext(os.path.basename(hdf5_file_path))[0]
            base_filename = f"{base_filename}_HDF5_summary"

        output_file_path = os.path.join(
            output_directory, f"{base_filename}.{output_format}"
        )
        output_file = open(output_file_path, "w")

    def write_section(title: str):
        """Write a section header."""
        if output_format == "md":
            content = f"\n## {title}\n"
        elif output_format == "tex":
            content = f"\n\\subsection{{{title}}}\n"
        else:
            content = f"\n{title}\n{'=' * len(title)}"

        if output_file:
            output_file.write(f"{content}\n")
            if verbose:
                click.echo(content)
        else:
            click.echo(content)

    def write_content(content: str, print_to_console: bool = True):
        """Write content to output."""
        if output_file:
            output_file.write(f"{content}\n\n")
            if verbose and print_to_console:
                click.echo(content)
        else:
            click.echo(content)

    # Load the HDF5 file
    try:
        file_name = os.path.basename(hdf5_file_path)
        parent_dir = os.path.dirname(hdf5_file_path)

        analyzer = HDF5Analyzer(hdf5_file_path)

        loading_message = (
            f"Successfully loaded HDF5 file: '{file_name}'\n"
            f"from directory: '{parent_dir}'."
        )

        if output_file:
            write_section("File Details")
            write_content(loading_message, print_to_console=True)
        else:
            click.echo(loading_message)

    except Exception as e:
        error_message = f"ERROR: Failed to load HDF5 file: {str(e)}"
        if output_file:
            write_content(error_message)
        click.echo(error_message)
        if output_file:
            output_file.close()
        sys.exit(1)

    # Basic Information Section
    write_section("Basic Information")

    # Convert set to sorted list for consistent access
    all_deepest_groups = sorted(list(analyzer._all_deepest_groups))
    total_groups = len(all_deepest_groups)
    basic_info_lines = [f"Total Groups (deepest level): {total_groups}"]

    # Determine group hierarchy pattern
    if all_deepest_groups:
        sample_path = all_deepest_groups[0]
        hierarchy_parts = sample_path.split("/")[:-1]
        if hierarchy_parts:
            hierarchy_pattern = "/" + "/".join(hierarchy_parts) + "/<group>"
            basic_info_lines.append(f"Group Hierarchy: {hierarchy_pattern}")

    # File size
    file_size = os.path.getsize(hdf5_file_path)
    if file_size < 1024:
        size_str = f"{file_size} B"
    elif file_size < 1024 * 1024:
        size_str = f"{file_size / 1024:.1f} KB"
    elif file_size < 1024 * 1024 * 1024:
        size_str = f"{file_size / (1024 * 1024):.1f} MB"
    else:
        size_str = f"{file_size / (1024 * 1024 * 1024):.2f} GB"
    basic_info_lines.append(f"File Size: {size_str}")

    write_content("\n".join(basic_info_lines))

    # Dataset Structure Section
    write_section("Dataset Structure")

    # Collect dataset info from ALL groups to detect shape variations
    dataset_info: Dict[str, Dict[str, Any]] = {}
    with h5py.File(hdf5_file_path, "r") as hdf5_file_handle:
        for dataset_name, paths in analyzer._dataset_paths.items():
            if not paths:
                continue

            shapes_by_dim: Dict[int, set] = {}
            dtype_str = None

            for path in paths:
                try:
                    ds = hdf5_file_handle[path]
                    if dtype_str is None:
                        dtype_str = format_dtype(ds.dtype)

                    shape = ds.shape
                    for dim_idx, dim_size in enumerate(shape):
                        if dim_idx not in shapes_by_dim:
                            shapes_by_dim[dim_idx] = set()
                        shapes_by_dim[dim_idx].add(dim_size)
                except Exception:
                    continue

            if dtype_str is not None and shapes_by_dim:
                # Build shape string with ranges for varying dimensions
                shape_parts = []
                for dim_idx in sorted(shapes_by_dim.keys()):
                    sizes = shapes_by_dim[dim_idx]
                    if len(sizes) == 1:
                        shape_parts.append(str(list(sizes)[0]))
                    else:
                        min_size = min(sizes)
                        max_size = max(sizes)
                        shape_parts.append(f"{min_size}-{max_size}")

                if not shape_parts:
                    shape_str = "scalar"
                else:
                    shape_str = " × ".join(shape_parts)

                dataset_info[dataset_name] = {
                    "dtype": dtype_str,
                    "shape_str": shape_str,
                }

    if dataset_info:
        num_datasets = len(dataset_info)
        structure_lines = [f"{num_datasets} datasets per group:\n"]

        for name, info in sorted(dataset_info.items()):
            structure_lines.append(f"  {name}: {info['dtype']} [{info['shape_str']}]")

        write_content("\n".join(structure_lines), print_to_console=False)
    else:
        write_content("No datasets found in deepest groups.")

    # Gvar Dataset Pairs Section
    if analyzer._gvar_dataset_pairs:
        write_section("Gvar Dataset Pairs")

        gvar_lines = []
        for base_name, (mean_name, error_name) in sorted(
            analyzer._gvar_dataset_pairs.items()
        ):
            gvar_lines.append(f"  {base_name}: {mean_name} + {error_name}")

        write_content("\n".join(gvar_lines), print_to_console=False)

    # Dataset Statistics Section (optional)
    if dataset_statistics:
        write_section("Dataset Statistics")

        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            stats = get_dataset_statistics(hdf5_file, analyzer._dataset_paths)

        if stats:
            if output_format == "md":
                stats_lines = [
                    "| Dataset | Min | Max | Mean |",
                    "|---------|-----|-----|------|",
                ]
                for name, stat in sorted(stats.items()):
                    stats_lines.append(
                        f"| {name} | {stat['min']:.6g} | {stat['max']:.6g} | {stat['mean']:.6g} |"
                    )
            elif output_format == "tex":
                stats_lines = [
                    r"\begin{tabular}{lrrr}",
                    r"\hline",
                    r"Dataset & Min & Max & Mean \\",
                    r"\hline",
                ]
                for name, stat in sorted(stats.items()):
                    escaped_name = name.replace("_", r"\_")
                    stats_lines.append(
                        f"{escaped_name} & {stat['min']:.6g} & {stat['max']:.6g} & {stat['mean']:.6g} \\\\"
                    )
                stats_lines.extend([r"\hline", r"\end{tabular}"])
            else:
                stats_lines = [
                    f"{'Dataset':<40} {'Min':>12} {'Max':>12} {'Mean':>12}",
                    "-" * 78,
                ]
                for name, stat in sorted(stats.items()):
                    stats_lines.append(
                        f"{name:<40} {stat['min']:>12.6g} {stat['max']:>12.6g} {stat['mean']:>12.6g}"
                    )

            write_content("\n".join(stats_lines), print_to_console=False)
        else:
            write_content("No numeric datasets found for statistics.")

    # Parameter Uniqueness Report Section
    if not no_uniqueness_report:
        write_section("Parameter Uniqueness Report")

        try:
            report = generate_uniqueness_report(
                analyzer, max_width=80, output_format=output_format
            )
            write_content(report, print_to_console=False)
        except Exception as e:
            write_content(f"Could not generate uniqueness report: {str(e)}")

    # Structural Consistency Check
    is_consistent, inconsistencies = check_structural_consistency(analyzer)
    if not is_consistent:
        write_section("Structural Inconsistencies")
        write_content(
            "WARNING: Not all groups have identical dataset structures:\n"
            + "\n".join(f"  - {msg}" for msg in inconsistencies),
            print_to_console=True,
        )

    # Partial h5glance Tree Section
    if sample_groups > 0:
        write_section(f"Structure Preview (first {sample_groups} groups)")

        h5glance_output = get_h5glance_output(hdf5_file_path)
        if h5glance_output:
            # Find the deepest group prefix for truncation
            if all_deepest_groups:
                deepest_prefix = "/".join(all_deepest_groups[0].split("/")[:-1])
            else:
                deepest_prefix = ""

            truncated_output = truncate_h5glance_output(
                h5glance_output, sample_groups, deepest_prefix
            )

            if output_format == "md":
                write_content(f"```\n{truncated_output}\n```", print_to_console=False)
            elif output_format == "tex":
                write_content(
                    f"\\begin{{verbatim}}\n{truncated_output}\n\\end{{verbatim}}",
                    print_to_console=False,
                )
            else:
                write_content(truncated_output, print_to_console=False)
        else:
            write_content("h5glance not available - install with: pip install h5glance")

    # Close analyzer and output file
    analyzer.close()

    if output_file:
        output_file.close()

    click.echo(
        f"   -- Summary of the '{os.path.basename(hdf5_file_path)}' "
        "HDF5 file generated."
    )


if __name__ == "__main__":
    main()
