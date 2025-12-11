"""
inspect_HDF5_file.py

This script is a utility for inspecting the structure and content of HDF5 files. 
It provides a command-line interface to explore the hierarchical organization of 
HDF5 files, including groups, subgroups, datasets, and attributes.

Purpose:
--------
To facilitate the exploration of HDF5 files by listing groups, subgroups, and 
attributes, and displaying the content or a sample of datasets. This tool is 
designed to work with HDF5 files that contain structured scientific or numerical 
data.

Functionality:
--------------
The script offers the following options:
1. --list_groups (-hdf5): Lists all top-level groups in the HDF5 file.
2. --list_subgroups: Displays unique first-level subgroups across all main groups.
3. --list_attributes: Lists attributes of the file or a specific group/dataset.
4. --show_dataset: Prints the content of a specified dataset.
5. --show_sample: Displays the first 5 elements of the specified dataset.

Input:
------
- hdf5_file_path (string): The path to the HDF5 file to be inspected. 
                           This is required for all operations.
- Optional flags:
  - --list_groups: Flag to list top-level groups in the file.
  - --list_subgroups: Flag to display unique first-level subgroups.
  - --list_attributes: Flag to show attributes of the file or a dataset.
  - --show_dataset: Name of the dataset to display.
  - --show_sample: Flag to display only the first 5 elements of the dataset.

Output:
-------
Depending on the options provided, the script outputs:
1. Names of top-level groups and their count.
2. Unique first-level subgroups shared across main groups.
3. Attributes of the file, group, or dataset.
4. Content or sample data of a specified dataset.

Usage:
------
To list all top-level groups:
    python inspect_HDF5_file.py -hdf5 /path/to/file.h5 --list_groups

To list unique first-level subgroups:
    python inspect_HDF5_file.py -hdf5 /path/to/file.h5 --list_subgroups

To view attributes of a specific dataset:
    python inspect_HDF5_file.py -hdf5 /path/to/file.h5 --list_attributes --show_dataset /group/dataset

To display content or a sample of a dataset:
    python inspect_HDF5_file.py -hdf5 /path/to/file.h5 --show_dataset /group/dataset --show_sample

Notes:
------
- Ensure that the specified HDF5 file path is valid and accessible.
- Use caution when displaying large datasets to avoid excessive output.
- The --list_subgroups option assumes that the structure of first-level subgroups 
  is consistent across main groups.
"""

import sys

import click
import h5py

from library.validation.click_validators import hdf5_file
from library import is_valid_file


@click.command()
@click.option(
    "-hdf5",
    "--hdf5_file_path",
    required=True,
    callback=hdf5_file.input,
    help="Path to the HDF5 file to be inspected.",
)
@click.option(
    "--list_groups", is_flag=True, help="List top-level groups in the HDF5 file."
)
@click.option(
    "--list_subgroups",
    is_flag=True,
    help="List first-level subgroups of all main groups.",
)
@click.option(
    "--list_attributes",
    is_flag=True,
    help="List attributes of the specified group or dataset.",
)
@click.option(
    "--show_dataset",
    "dataset_name",
    default=None,
    help="Show the content of the specified dataset. " "Example: '/group/dataset'",
)
@click.option(
    "--show_sample", is_flag=True, help="Show a sample of a dataset (first 5 elements)."
)
def main(
    hdf5_file_path,
    list_groups,
    list_subgroups,
    list_attributes,
    dataset_name,
    show_sample,
):

    # Check provided path to HDF5 file
    if not is_valid_file(hdf5_file_path):
        print("ERROR: Provided path to .csv file is invalid.")
        print("Exiting...")
        sys.exit(1)

    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        if list_groups:
            # List all top-level groups
            print("\nTop-level groups in the HDF5 file:")
            for group_name in hdf5_file:
                print(f"- {group_name}")

        if list_subgroups:
            # List first-level subgroups for each main group
            subgroups = set()
            for group_name in hdf5_file:
                main_group = hdf5_file[group_name]
                for subgroup_name in main_group:
                    subgroups.add(subgroup_name)

            print("\nUnique first-level subgroups across all main groups:")
            for subgroup in sorted(subgroups):
                print(f"- {subgroup}")

        # TODO: Check validity of the provided "dataset_name"
        if list_attributes:
            # List attributes of the whole file (or specific group/dataset)
            if dataset_name:
                # Inspect attributes of a dataset
                dataset = hdf5_file[dataset_name]
                print(f"\nAttributes of dataset {dataset_name}:")
                for attr_name, attr_value in dataset.attrs.items():
                    print(f"  {attr_name}: {attr_value}")
            else:
                # Inspect attributes of the file itself
                print("\nAttributes of the HDF5 file:")
                for attr_name, attr_value in hdf5_file.attrs.items():
                    print(f"  {attr_name}: {attr_value}")

        if dataset_name:
            # Show content of a specific dataset
            dataset = hdf5_file[dataset_name]
            if show_sample:
                # Show a sample of the first 5 elements
                print(f"\nFirst 5 elements of {dataset_name}: {dataset[:5]}")
            else:
                # Show the entire dataset (be cautious with large datasets)
                print(f"\nContent of {dataset_name}: {dataset[:]}")

        # Optionally, count the number of groups
        if not any([list_groups, list_subgroups, list_attributes, dataset_name]):
            # If no flags are provided, print the number of main groups
            print(f"\nTotal number of top-level groups: {len(hdf5_file)}")


if __name__ == "__main__":
    main()
