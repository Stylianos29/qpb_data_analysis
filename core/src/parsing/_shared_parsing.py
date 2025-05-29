"""
Private shared processing utilities for data file processing scripts.

This module contains common functionality used by both parse_log_files and
parse_correlator_files.py. It is marked as private (underscore prefix) and
should not be part of the public API.
"""

import os
import pandas as pd
import h5py
from library import filesystem_utilities


def _classify_parameters_by_uniqueness(scalar_params_list):
    """
    Classify scalar parameters into constant and multivalued categories based on
    their uniqueness across all files.

    Args:
        scalar_params_list (list): List of parameter dictionaries from all files

    Returns:
        tuple: (dataframe, constant_params_dict, multivalued_params_list)
            - dataframe: pandas DataFrame of all parameters
            - constant_params_dict: Parameters with single unique value across
              files
            - multivalued_params_list: Parameters with multiple unique values
              across files
    """
    # Convert to DataFrame for analysis
    dataframe = pd.DataFrame(scalar_params_list)

    # Get unique value counts for each column
    unique_counts = dataframe.nunique()

    # Classify parameters
    constant_params_list = unique_counts[unique_counts == 1].index.tolist()
    multivalued_params_list = unique_counts[unique_counts > 1].index.tolist()

    # Create dictionary of constant parameters
    constant_params_dict = {col: dataframe[col].iloc[0] for col in constant_params_list}

    return dataframe, constant_params_dict, multivalued_params_list


def _create_hdf5_structure_with_constant_params(
    hdf5_file, constant_params_dict, base_directory, target_directory, logger
):
    """
    Create HDF5 group structure and add constant parameters as attributes.

    Args:
        - hdf5_file (h5py.File): Open HDF5 file handle
        - constant_params_dict (dict): Constant parameters to store as
          attributes
        - base_directory (str): Base directory for HDF5 structure
        - target_directory (str): Target directory for HDF5 structure
        - logger: Logger instance

    Returns:
        h5py.Group: The data files set group with constant parameters as
        attributes
    """
    # Create HDF5 group structure mirroring directory hierarchy
    data_files_set_group = filesystem_utilities.create_hdf5_group_structure(
        hdf5_file, base_directory, target_directory, logger
    )

    # Add constant parameters as attributes to the second-to-deepest level
    for param_name, param_value in constant_params_dict.items():
        data_files_set_group.attrs[param_name] = param_value

    return data_files_set_group


def _export_dataframe_to_csv(dataframe, output_path, logger, description="parameters"):
    """
    Export a pandas DataFrame to CSV file with logging.

    Args:
        - dataframe (pd.DataFrame): DataFrame to export
        - output_path (str): Full path to output CSV file
        - logger: Logger instance
        - description (str): Description for logging message
    """
    dataframe.to_csv(output_path, index=False)
    logger.info(
        f"Extracted {description} are stored in the "
        f"'{os.path.basename(output_path)}' file."
    )


def _export_arrays_to_hdf5_with_proper_structure(
    constant_params_dict,
    multivalued_params_list,
    arrays_dict,
    scalar_params_list,
    hdf5_path,
    base_directory,
    target_directory,
    logger,
    description="arrays",
):
    """
    Export arrays to HDF5 file following the project's structure protocol.

    This function implements the correct HDF5 structure:
    - Second-to-deepest level: constant parameters as attributes
    - File-level groups: multivalued parameters as attributes + arrays as
      datasets

    Args:
        - constant_params_dict (dict): Constant parameters to store as
          attributes at second-to-deepest level
        - multivalued_params_list (list): Names of parameters that vary across
          files
        - arrays_dict (dict): Dictionary with structure {filename: {array_name:
          numpy_array, ...}}
        - scalar_params_list (list): List of all scalar parameter dictionaries
          per file
        - hdf5_path (str): Full path to output HDF5 file
        - base_directory (str): Base directory for HDF5 structure
        - target_directory (str): Target directory for HDF5 structure
        - logger: Logger instance
        - description (str): Description for logging message
    """
    # Create a lookup dictionary for quick access to each file's parameters
    file_params_lookup = {params["Filename"]: params for params in scalar_params_list}

    with h5py.File(hdf5_path, "w") as hdf5_file:
        # Create HDF5 structure with constant parameters at second-to-deepest level
        data_files_set_group = _create_hdf5_structure_with_constant_params(
            hdf5_file, constant_params_dict, base_directory, target_directory, logger
        )

        # Create file-level groups and store arrays + multivalued parameters
        for filename, arrays in arrays_dict.items():
            file_group = data_files_set_group.create_group(filename)

            # Get this file's scalar parameters
            file_params = file_params_lookup.get(filename, {})

            # Add ONLY multivalued parameters as attributes to the file group
            for param_name in multivalued_params_list:
                if param_name in file_params:
                    file_group.attrs[param_name] = file_params[param_name]

            # Store arrays as datasets
            for array_name, array_data in arrays.items():
                file_group.create_dataset(array_name, data=array_data)

    logger.info(
        f"Extracted {description} are stored in the "
        f"'{os.path.basename(hdf5_path)}' file."
    )


def _check_parameter_mismatches(
    source1_params,
    source2_params,
    context_info,
    logger,
    source1_name="source1",
    source2_name="source2",
):
    """
    Check for mismatches between two parameter dictionaries and log warnings.

    Args:
        - source1_params (dict): First set of parameters
        - source2_params (dict): Second set of parameters
        - context_info (str): Context information for logging (e.g., filename)
        - logger: Logger instance
        - source1_name (str): Name of first source for logging
        - source2_name (str): Name of second source for logging
    """
    for param_name in source1_params.keys() & source2_params.keys():
        if source1_params[param_name] != source2_params[param_name]:
            logger.warning(
                f"Mismatch for '{param_name}' parameter in {context_info}. "
                f"{source1_name} value: {source1_params[param_name]}, "
                f"{source2_name} value: {source2_params[param_name]}."
            )
