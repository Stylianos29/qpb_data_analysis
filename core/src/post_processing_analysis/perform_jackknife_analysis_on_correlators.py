# TODO: Write a detailed introductory commentary
# TODO: Add more logging messages
"""
post_processing_analysis/process_qpb_log_files_extracted_values.py

Summary:

Input:

Output:

Functionality:

Usage:
"""

import os
import sys

import click
import numpy as np
import gvar as gv
import h5py
import copy

from library import (
    momentum_correlator,
    jackknife_analysis,
    filesystem_utilities,
    data_processing,
    custom_plotting,
    PROCESSED_DATA_FILES_DIRECTORY,
)


@click.command()
@click.option(
    "-in_params_csv",
    "--input_parameter_values_csv_file_path",
    "input_parameter_values_csv_file_path",
    required=True,
    callback=filesystem_utilities.validate_input_csv_file,
    help=(
        "Path to input .csv file containing processed info extracted from qpb "
        "log files."
    ),
)
@click.option(
    "-in_cors_hdf5",
    "--input_correlators_hdf5_file_path",
    "input_correlators_hdf5_file_path",
    required=True,
    callback=filesystem_utilities.validate_input_HDF5_file,
    help="Path to input HDF5 file containing extracted correlators values.",
)
@click.option(
    "-out_dir",
    "--output_files_directory",
    "output_files_directory",
    default=None,
    callback=validate_input_directory,
    help="Path to directory where all output files will be stored.",
)
@click.option(
    "-plots_dir",
    "--plots_directory",
    "plots_directory",
    default="../../../output/plots",
    callback=validate_input_directory,
    help="Path to the output directory for storing plots.",
)
@click.option(
    "-sym_cors",
    "--symmetrize_correlators",
    "symmetrize_correlators",
    is_flag=True,
    default=False,
    help="Enable symmetrizing the jackknife resampled correlator values.",
)
@click.option(
    "-plot_g5g5",
    "--plot_g5g5_correlators",
    "plot_g5g5_correlators",
    is_flag=True,
    default=False,
    help="Enable plotting g5g5 correlator values.",
)
@click.option(
    "-plot_g4g5g5",
    "--plot_g4g5g5_correlators",
    "plot_g4g5g5_correlators",
    is_flag=True,
    default=False,
    help="Enable plotting g4g5g5 correlator values.",
)
@click.option(
    "-plot_g4g5g5_der",
    "--plot_g4g5g5_derivative_correlators",
    "plot_g4g5g5_derivative_correlators",
    is_flag=True,
    default=False,
    help="Enable plotting g4g5g5 derivative correlator values.",
)
@click.option(
    "-out_hdf5_name",
    "--output_hdf5_filename",
    "output_hdf5_filename",
    default="correlators_jackknife_analysis.h5",
    callback=filesystem_utilities.validate_output_HDF5_filename,
    help="Specific name for the output HDF5 file.",
)
@click.option(
    "-log_on",
    "--enable_logging",
    "enable_logging",
    is_flag=True,
    default=False,
    help="Enable logging.",
)
@click.option(
    "-log_file_dir",
    "--log_file_directory",
    "log_file_directory",
    default=None,
    callback=filesystem_utilities.validate_script_log_file_directory,
    help="Directory where the script's log file will be stored.",
)
@click.option(
    "-log_name",
    "--log_filename",
    "log_filename",
    default=None,
    callback=filesystem_utilities.validate_script_log_filename,
    help="Specific name for the script's log file.",
)
def main(
    input_parameter_values_csv_file_path,
    input_correlators_hdf5_file_path,
    output_files_directory,
    plots_directory,
    symmetrize_correlators,
    plot_g5g5_correlators,
    plot_g4g5g5_correlators,
    plot_g4g5g5_derivative_correlators,
    output_hdf5_filename,
    enable_logging,
    log_file_directory,
    log_filename,
):
    # HANDLE EMPTY INPUT ARGUMENTS

    # If no output directory is provided, use the directory of an input file
    if output_files_directory is None:
        output_files_directory = os.path.dirname(input_correlators_hdf5_file_path)

    # INITIATE LOGGING

    # Setup logging
    logger = filesystem_utilities.LoggingWrapper(
        log_file_directory, log_filename, enable_logging
    )

    # Log script start
    logger.initiate_script_logging()

    # CREATE PLOTS SUBDIRECTORIES

    # If there's any request for plots then create main plots directory if it
    # does not exist
    if any(
        [
            plot_g5g5_correlators,
            plot_g4g5g5_correlators,
            plot_g4g5g5_derivative_correlators,
        ]
    ):
        plots_main_subdirectory = filesystem_utilities.create_subdirectory(
            plots_directory, "Correlators_jackknife_analysis"
        )

    # Create then specific deeper-level plots subdirectories if requested
    if plot_g5g5_correlators:
        g5_g5_plots_base_name = "g5g5_correlator"
        g5_g5_plots_subdirectory = filesystem_utilities.create_subdirectory(
            plots_main_subdirectory,
            g5_g5_plots_base_name + "_values",
            clear_contents=True,
        )
        logger.info("Subdirectory for g5g5 correlator plots created.")

    if plot_g4g5g5_correlators:
        g4g5_g5_plots_base_name = "g4g5g5_correlator"
        g4g5_g5_plots_subdirectory = filesystem_utilities.create_subdirectory(
            plots_main_subdirectory,
            g4g5_g5_plots_base_name + "_values",
            clear_contents=True,
        )
        logger.info("Subdirectory for g4g5g5 correlator plots created.")

    if plot_g4g5g5_derivative_correlators:
        g4g5_g5_derivative_plots_base_name = "g4g5g5_derivative_correlator"
        g4g5_g5_derivative_plots_subdirectory = (
            filesystem_utilities.create_subdirectory(
                plots_main_subdirectory,
                g4g5_g5_derivative_plots_base_name + "_values",
                clear_contents=True,
            )
        )
        logger.info("Subdirectory for g4g5g5 derivative correlator plots created.")

    # IMPORT DATASETS AND METADATA

    # Construct output HDF5 file path
    output_correlators_jackknife_analysis_hdf5_file_path = os.path.join(
        output_files_directory, output_hdf5_filename
    )

    # Open the input HDF5 file for reading and the output HDF5 file for writing
    with h5py.File(input_correlators_hdf5_file_path, "r") as hdf5_file_read, h5py.File(
        output_correlators_jackknife_analysis_hdf5_file_path, "w"
    ) as hdf5_file_write:

        # Construct the path to the processed data files set directory
        processed_data_files_set_directory = os.path.dirname(
            input_correlators_hdf5_file_path
        )

        # The top HDF5 file groups (for both HDF5 files) mirror the directory
        # structure of the data files set directory itself and its parent
        # directories relative to the 'PROCESSED_DATA_FILES_DIRECTORY' directory
        input_data_files_set_group = filesystem_utilities.get_hdf5_target_group(
            hdf5_file_read,
            PROCESSED_DATA_FILES_DIRECTORY,
            processed_data_files_set_directory,
            logger=None,
        )
        logger.info("Top groups of the input HDF5 file identified.")

        # Check if input HDF5 file contains any data before creating the output
        # HDF5 file, otherwise con
        if not filesystem_utilities.has_subgroups(input_data_files_set_group):
            logger.critical(
                "Input HDF5 file does not contain any data to analyze!", to_console=True
            )
            sys.exit(1)

        output_data_files_set_group = filesystem_utilities.create_hdf5_group_structure(
            hdf5_file_write,
            PROCESSED_DATA_FILES_DIRECTORY,
            processed_data_files_set_directory,
            logger,
        )
        logger.info("Top groups of the output HDF5 file created.")

        # Analyze input processed parameter values .csv file
        parameter_values_dataframe = data_processing.load_csv(
            input_parameter_values_csv_file_path
        )
        analyzer = data_processing.DataFrameAnalyzer(parameter_values_dataframe)

        # Store names and values of tunable parameters with unique values
        single_valued_fields_dictionary = copy.deepcopy(
            analyzer.single_valued_fields_dictionary
        )

        # Initialize a list of parameter names with multiple unique values
        rest_of_the_multivalued_field_names_list = list(
            analyzer.multivalued_fields_dictionary.keys()
        )

        # CONSTRUCT LIST OF RELEVANT MULTIVALUED TUNABLE PARAMETERS FOR GROUPING

        # Groupings will be based on tunable parameters with more than one
        # unique values (multivalued)
        tunable_multivalued_parameter_names_list = copy.deepcopy(
            analyzer.list_of_tunable_multivalued_parameter_names
        )

        # Exclude "MPI_geometry" from tunable multivalued parameters list
        # NOTE: The "MPI_geometry" parameter affects only computation speed,
        # not the results. Excluding it ensures proper grouping based on all
        # relevant tunable multivalued parameters.
        tunable_multivalued_parameter_names_list = [
            parameter_name
            for parameter_name in tunable_multivalued_parameter_names_list
            if parameter_name != "MPI_geometry"
        ]

        # Exclude "Configuration_label" from tunable multivalued parameters list
        # NOTE: In this jackknife resampling analysis, the final grouping must
        # be based on "Configuration_label" values.
        tunable_multivalued_parameter_names_list = [
            parameter_name
            for parameter_name in tunable_multivalued_parameter_names_list
            if parameter_name != "Configuration_label"
        ]

        # Reduce initial "rest_of_the_multivalued_field_names_list" to
        # containing the rest of the multivalued parameter names not included in
        # "tunable_multivalued_parameter_names_list"
        rest_of_the_multivalued_field_names_list = [
            parameter_name
            for parameter_name in rest_of_the_multivalued_field_names_list
            if parameter_name not in tunable_multivalued_parameter_names_list
        ]
        # Remove additionally "Filename" and "Configuration_label" since these
        # are strings and need to be used explicitly for specific checks
        rest_of_the_multivalued_field_names_list = [
            parameter_name
            for parameter_name in rest_of_the_multivalued_field_names_list
            if parameter_name != "Filename" or parameter_name != "Configuration_label"
        ]

        # LOOP OVER ALL RELEVANT TUNABLE PARAMETERS GROUPINGS

        # Include counting the iterations for later use
        for jackknife_analysis_index, (
            combination_of_values,
            dataframe_group,
        ) in enumerate(
            parameter_values_dataframe.groupby(
                tunable_multivalued_parameter_names_list, observed=True
            )
        ):
            # Define a unique name for each grouping as a separate jackknife
            # analysis
            correlators_jackknife_analysis_group_name = (
                f"Correlators_jackknife_analysis_{jackknife_analysis_index}"
            )

            # STORE PARAMETER VALUES AND DATASETS FOR THE CURRENT GROUPING

            # Store specific tunable multivalued parameter names and values in a
            # dedicated metadata dictionary for later use
            if not isinstance(combination_of_values, tuple):
                combination_of_values = [combination_of_values]
            metadata_dictionary = dict(
                zip(tunable_multivalued_parameter_names_list, combination_of_values)
            )
            logger.info(
                f"{correlators_jackknife_analysis_group_name} for values: "
                f"{metadata_dictionary}."
            )

            # Construct an overall (both single- and multivalued) tunable
            # parameter values dictionary for the current grouping
            tunable_parameter_values_dictionary = copy.deepcopy(
                single_valued_fields_dictionary
            )

            # TODO: Store in lists
            rest_of_the_multivalued_parameters_dictionary = {}
            for parameter_name in rest_of_the_multivalued_field_names_list:
                list_name = str(parameter_name) + "_values_list"
                rest_of_the_multivalued_parameters_dictionary[list_name] = (
                    dataframe_group[parameter_name].to_numpy()
                )

            # List qpb log filenames and configuration labels for the current
            # grouping
            list_of_qpb_log_filenames = dataframe_group["Filename"].tolist()
            list_of_configuration_labels = dataframe_group[
                "Configuration_label"
            ].tolist()

            # Precautionary check if the length of these list is the same
            if len(list_of_qpb_log_filenames) != len(list_of_configuration_labels):
                logger.warning(
                    f"{correlators_jackknife_analysis_group_name}: The number "
                    "of qpb log files is not equal to the number of gauge "
                    "links configuration labels.",
                    to_console=True,
                )

            # Ensure that the current grouping encompasses more than 2 gauge
            # configurations
            number_of_gauge_configurations = len(list_of_configuration_labels)
            if number_of_gauge_configurations <= 1:
                logger.warning(
                    f"{correlators_jackknife_analysis_group_name}: "
                    "Cannot perform Jackknife analysis with sample size 1. "
                    "Skipping...",
                    to_console=True,
                )
                continue

            # Pass "g5-g5" and "g4g5-g5" correlator values datasets for the
            # current grouping to lists
            g5_g5_correlator_values_grouped_by_gauge_configurations_list = []
            g4g5_g5_correlator_values_grouped_by_gauge_configurations_list = []
            for qpb_log_filename in list_of_qpb_log_filenames:

                correlators_filename = qpb_log_filename.replace(".txt", ".dat")

                # Precautionary check if qpb_log_filename exists as a subgroup
                # in input_data_files_set_group. This ensures it is both present
                # and of type h5py.Group (not just a dataset).
                if (
                    correlators_filename not in input_data_files_set_group
                    or not isinstance(
                        input_data_files_set_group[correlators_filename], h5py.Group
                    )
                ):
                    logger.warning(
                        f"{correlators_jackknife_analysis_group_name}: "
                        "There are no datasets stored for a "
                        f"'{correlators_filename}' correlators data file "
                        f"corresponding to {qpb_log_filename}."
                        "Skipping...",
                        to_console=True,
                    )
                    number_of_gauge_configurations -= 1
                    continue

                correlators_data_file_group = input_data_files_set_group[
                    correlators_filename
                ]
                if "g5-g5" in correlators_data_file_group.keys():
                    g5_g5_dataset = correlators_data_file_group["g5-g5"][:]
                    g5_g5_correlator_values_grouped_by_gauge_configurations_list.append(
                        g5_g5_dataset
                    )

                if "g4g5-g5" in correlators_data_file_group.keys():
                    g4g5_g5_dataset = correlators_data_file_group["g4g5-g5"][:]
                    g4g5_g5_correlator_values_grouped_by_gauge_configurations_list.append(
                        g4g5_g5_dataset
                    )

            # Check again number of gauge configurations in case it has been
            # modified
            if number_of_gauge_configurations <= 1:
                logger.warning(
                    f"{correlators_jackknife_analysis_group_name}: "
                    "Cannot perform Jackknife analysis with sample size 1."
                    "Skipping...",
                    to_console=True,
                )
                continue

            metadata_dictionary["Number_of_gauge_configurations"] = (
                number_of_gauge_configurations
            )
            # TODO: Not the best strategy
            tunable_parameter_values_dictionary.update(metadata_dictionary)

            # Convert the lists of 1D arrays into a 2D NumPy array
            g5_g5_correlator_values_grouped_by_gauge_configurations_2D_array = (
                np.vstack(g5_g5_correlator_values_grouped_by_gauge_configurations_list)
            )
            g4g5_g5_correlator_values_grouped_by_gauge_configurations_2D_array = (
                np.vstack(
                    g4g5_g5_correlator_values_grouped_by_gauge_configurations_list
                )
            )

            ### JACKKNIFE ANALYSIS ###

            # JACKKNIFE ANALYSIS OF G5-G5 CORRELATORS

            g5_g5_correlator_jackknife_analyzer = jackknife_analysis.JackknifeAnalysis(
                g5_g5_correlator_values_grouped_by_gauge_configurations_2D_array
            )

            # Calculate jackknife samples of the g5-g5 correlators
            jackknife_samples_of_g5_g5_correlator_2D_array = (
                g5_g5_correlator_jackknife_analyzer.jackknife_replicas_of_original_2D_array
            )

            # Calculate jackknife average of the g5-g5 correlator samples
            jackknife_average_of_g5_g5_correlator_array = (
                g5_g5_correlator_jackknife_analyzer.jackknife_average
            )
            # Symmetrize the jackknife average correlator if requested
            if symmetrize_correlators:
                jackknife_average_of_g5_g5_correlator_array = (
                    momentum_correlator.symmetrization(
                        jackknife_average_of_g5_g5_correlator_array
                    )
                )
                logger.info(
                    "Jackknife average of the g5-g5 correlator was symmetrized."
                )

            # JACKKNIFE ANALYSIS OF G4G5-G5 CORRELATORS

            g4g5_g5_correlator_jackknife_analyzer = (
                jackknife_analysis.JackknifeAnalysis(
                    g4g5_g5_correlator_values_grouped_by_gauge_configurations_2D_array
                )
            )

            # Calculate jackknife samples of the g4g5-g5 correlators
            jackknife_samples_of_g4g5_g5_correlator_2D_array = (
                g4g5_g5_correlator_jackknife_analyzer.jackknife_replicas_of_original_2D_array
            )

            # Calculate jackknife average of the g4g5-g5 correlators
            jackknife_average_of_g4g5_g5_correlator_array = (
                g4g5_g5_correlator_jackknife_analyzer.jackknife_average
            )
            # Symmetrize the jackknife average correlator if requested
            if symmetrize_correlators:
                jackknife_average_of_g4g5_g5_correlator_array = (
                    momentum_correlator.symmetrization(
                        jackknife_average_of_g4g5_g5_correlator_array
                    )
                )
                logger.info(
                    "Jackknife average of the g4g5-g5 correlator was symmetrized."
                )

            # G4G5-G5 DERIVATE CORRELATORS CALCULATION

            # Calculate jackknife samples of the g4g5-g5 derivative correlators
            jackknife_samples_of_g4g5_g5_derivative_correlator_2D_array = np.array(
                [
                    momentum_correlator.centered_difference_correlator_derivative(
                        jackknife_replica
                    )
                    for jackknife_replica in jackknife_samples_of_g4g5_g5_correlator_2D_array
                ]
            )

            # Calculate the average of the jackknife samples of the g4g5-g5
            # derivate correlators
            jackknife_average_of_g4g5_g5_derivative_array = (
                jackknife_analysis.calculate_jackknife_average_array(
                    jackknife_samples_of_g4g5_g5_derivative_correlator_2D_array
                )
            )
            # Symmetrize the jackknife average correlator if requested
            if symmetrize_correlators:
                jackknife_average_of_g4g5_g5_derivative_array = (
                    momentum_correlator.symmetrization(
                        jackknife_average_of_g4g5_g5_derivative_array
                    )
                )
                logger.info(
                    "Jackknife average of the g4g5-g5 derivative correlator "
                    "was symmetrized."
                )

            # Calculate an alternative definition of the average of the g4g5-g5
            # derivative correlator using the jackknife average of the g4g5-g5
            # correlator directly
            average_g4g5_g5_derivative_from_average_g4g5_g5_correlator_array = (
                momentum_correlator.centered_difference_correlator_derivative(
                    jackknife_average_of_g4g5_g5_correlator_array
                )
            )

            # PCAC MASS CORRELATOR CALCULATION

            # Calculate jackknife samples of the PCAC mass correlator values
            # NOTE: The PCAC mass is defined as half the ratio of the g4g5-g5
            # derivative correlator values over the g5-g5 correlator values
            jackknife_samples_of_PCAC_mass_correlator_values_list = []
            for index in range(len(jackknife_samples_of_g5_g5_correlator_2D_array)):
                # TODO: Investigate "RuntimeWarning: divide by zero encountered in true_divide 0.5"
                jackknife_replica_of_PCAC_mass_values = (
                    0.5
                    * jackknife_samples_of_g4g5_g5_derivative_correlator_2D_array[index]
                    / jackknife_samples_of_g5_g5_correlator_2D_array[index]
                )

                jackknife_samples_of_PCAC_mass_correlator_values_list.append(
                    jackknife_replica_of_PCAC_mass_values
                )
            jackknife_samples_of_PCAC_mass_correlator_values_2D_array = np.array(
                jackknife_samples_of_PCAC_mass_correlator_values_list
            )

            # Calculate the average of the jackknife samples of the g4g5-g5
            # derivate correlators
            jackknife_average_of_PCAC_mass_correlator_array = (
                jackknife_analysis.calculate_jackknife_average_array(
                    jackknife_samples_of_PCAC_mass_correlator_values_2D_array
                )
            )
            # Symmetrize the jackknife average correlator if requested
            if symmetrize_correlators:
                jackknife_average_of_PCAC_mass_correlator_array = (
                    momentum_correlator.symmetrization(
                        jackknife_average_of_PCAC_mass_correlator_array
                    )
                )
                logger.info(
                    "Jackknife average of the PCAC mass correlator was " "symmetrized."
                )

            # Calculate an alternative definition of the average of the PCAC
            # mass correlator values using the jackknife averages of the g4g5-g5
            # derivative and the g5-g5 correlators directly
            # TODO: Check divide by zero error
            # PCAC_mass_correlator_from_average_correlators_array = (
            #     0.5
            #     * jackknife_average_of_g4g5_g5_derivative_array
            #     / jackknife_average_of_g5_g5_correlator_array
            # )

            ### PLOTS ###

            if plot_g5g5_correlators:
                custom_plotting.plot_correlator(
                    jackknife_average_of_g5_g5_correlator_array,
                    xlabel="$t/a$",
                    ylabel="$C_{\\gamma_5\!-\!\\gamma_5}(t)$",
                    base_name=g5_g5_plots_base_name,
                    subdirectory=g5_g5_plots_subdirectory,
                    metadata_dict=metadata_dictionary,
                    tunable_parameters_dict=tunable_parameter_values_dictionary,
                    yaxis_log_scale=True,
                )

            if plot_g4g5g5_correlators:
                custom_plotting.plot_correlator(
                    jackknife_average_of_g4g5_g5_correlator_array,
                    xlabel="$t/a$",
                    ylabel="$C_{\\gamma_4\\gamma_5\!-\!\\gamma_5}(t)$",
                    base_name=g4g5_g5_plots_base_name,
                    subdirectory=g4g5_g5_plots_subdirectory,
                    metadata_dict=metadata_dictionary,
                    tunable_parameters_dict=tunable_parameter_values_dictionary,
                    starting_time=1,
                )

            if plot_g4g5g5_derivative_correlators:
                custom_plotting.plot_correlator(
                    jackknife_average_of_g4g5_g5_derivative_array,
                    xlabel="$t/a$",
                    ylabel="$C_{\\partial\\gamma_4\\gamma_5\!-\!\\gamma_5}(t)$",
                    base_name=g4g5_g5_derivative_plots_base_name,
                    subdirectory=g4g5_g5_derivative_plots_subdirectory,
                    metadata_dict=metadata_dictionary,
                    tunable_parameters_dict=tunable_parameter_values_dictionary,
                    starting_time=1,
                )

            ### EXPORT DATA TO OUTPUT HDF5 FILE ###

            # Create subgroup for current grouping / jackknife analysis
            correlators_jackknife_analysis_hdf5_group = (
                output_data_files_set_group.create_group(
                    correlators_jackknife_analysis_group_name
                )
            )

            # STORE PARAMETER VALUES

            # Store unique values as attributes to current analysis HDF5 group
            for parameter_name, parameter_value in metadata_dictionary.items():
                correlators_jackknife_analysis_hdf5_group.attrs[parameter_name] = (
                    parameter_value
                )

            for key in rest_of_the_multivalued_parameters_dictionary:
                correlators_jackknife_analysis_hdf5_group.create_dataset(
                    key,
                    data=rest_of_the_multivalued_parameters_dictionary[key],
                ).attrs["Description"] = (
                    key.replace("_", " ") + "."
                )

            # Convert lists of strings to variable-length UTF-8 encoded data
            qpb_log_filenames_array = np.array(
                list_of_qpb_log_filenames, dtype=h5py.string_dtype(encoding="utf-8")
            )
            configuration_labels_array = np.array(
                list_of_configuration_labels, dtype=h5py.string_dtype(encoding="utf-8")
            )

            # Store the list of configuration labels for the current grouping
            correlators_jackknife_analysis_hdf5_group.create_dataset(
                "List_of_gauge_configuration_labels",
                data=configuration_labels_array,
            ).attrs["Description"] = "List of gauge links configuration labels."

            # Store the list of qpb log filenames for the current grouping
            correlators_jackknife_analysis_hdf5_group.create_dataset(
                "List_of_qpb_log_filenames",
                data=qpb_log_filenames_array,
            ).attrs["Description"] = "List of qpb log filenames."

            # STORE G5-G5 CORRELATOR VALUES

            # Store g5-g5 correlator jackknife sample's 2D array as a dataset
            correlators_jackknife_analysis_hdf5_group.create_dataset(
                "Jackknife_samples_of_g5_g5_correlator_2D_array",
                data=jackknife_samples_of_g5_g5_correlator_2D_array,
            ).attrs["Description"] = (
                "A 2D Numpy array containing the jackknife replicas of the "
                "original g5-g5 correlator values grouped by gauge links "
                "configuration for specific tunable parameter values."
            )
            # Store the jackknife average of the g5-g5 correlator samples as two
            # separate datasets: one for the mean and one for the error
            correlators_jackknife_analysis_hdf5_group.create_dataset(
                "Jackknife_average_of_g5_g5_correlator_mean_values",
                data=gv.mean(jackknife_average_of_g5_g5_correlator_array),
            ).attrs["Description"] = (
                "A Numpy array containing the mean values of the the "
                "jackknife average of the g5g-g5 correlator jackknife samples."
            )
            correlators_jackknife_analysis_hdf5_group.create_dataset(
                "Jackknife_average_of_g5_g5_correlator_error_values",
                data=gv.sdev(jackknife_average_of_g5_g5_correlator_array),
            ).attrs["Description"] = (
                "A Numpy array containing the error values of the the "
                "jackknife average of the g5g-g5 correlator jackknife samples."
            )

            # STORE G4G5-G5 CORRELATOR VALUES

            # Store g4g5-g5 correlator jackknife sample's 2D array as a dataset
            correlators_jackknife_analysis_hdf5_group.create_dataset(
                "Jackknife_samples_of_g4g5_g5_correlator_2D_array",
                data=jackknife_samples_of_g4g5_g5_correlator_2D_array,
            ).attrs["Description"] = (
                "A 2D Numpy array containing the jackknife replicas of the "
                "original g4g5-g5 correlator values grouped by gauge links "
                "configuration for specific tunable parameter values."
            )
            # Store the jackknife average of the g4g5-g5 correlator samples as
            # two separate datasets: one for the mean and one for the error
            correlators_jackknife_analysis_hdf5_group.create_dataset(
                "Jackknife_average_of_g4g5_g5_correlator_mean_values",
                data=gv.mean(jackknife_average_of_g4g5_g5_correlator_array),
            ).attrs["Description"] = (
                "A Numpy array containing the mean values of the the "
                "jackknife average of the g4g5g-g5 correlator jackknife samples."
            )
            correlators_jackknife_analysis_hdf5_group.create_dataset(
                "Jackknife_average_of_g4g5_g5_correlator_error_values",
                data=gv.sdev(jackknife_average_of_g4g5_g5_correlator_array),
            ).attrs["Description"] = (
                "A Numpy array containing the error values of the the "
                "jackknife average of the g4g5g-g5 correlator jackknife samples."
            )

            # STORE G4G5-G5 DERIVATIVE CORRELATOR VALUES

            # Store g4g5-g5 derivate correlator jackknife sample's 2D array as a
            # dataset
            correlators_jackknife_analysis_hdf5_group.create_dataset(
                "Jackknife_samples_of_g4g5_g5_derivative_correlator_2D_array",
                data=jackknife_samples_of_g4g5_g5_derivative_correlator_2D_array,
            ).attrs["Description"] = (
                "A 2D Numpy array containing the jackknife replicas of the "
                "g4g5-g5 derivative correlator values grouped by gauge links "
                "configuration for specific tunable parameter values. The "
                "first derivative of the g4g5-g5 correlator was calculated "
                "using forth-order centered finite difference approximation."
            )
            # Store the jackknife average of the g4g5-g5 derivative correlator
            # samples as two separate datasets: one for the mean and one for the
            # error
            correlators_jackknife_analysis_hdf5_group.create_dataset(
                "Jackknife_average_of_g4g5_g5_derivative_correlator_mean_values",
                data=gv.mean(jackknife_average_of_g4g5_g5_derivative_array),
            ).attrs["Description"] = (
                "A Numpy array containing the mean values of the the jackknife"
                "average of the g4g5g-g5 derivative correlator jackknife samples."
            )
            correlators_jackknife_analysis_hdf5_group.create_dataset(
                "Jackknife_average_of_g4g5_g5_derivative_correlator_error_values",
                data=gv.sdev(jackknife_average_of_g4g5_g5_derivative_array),
            ).attrs["Description"] = (
                "A Numpy array containing the error values of the the jackknife"
                "average of the g4g5g-g5 derivative correlator jackknife samples."
            )

            # STORE PCAC MASS CORRELATOR VALUES

            # Store dataset and attach brief description
            correlators_jackknife_analysis_hdf5_group.create_dataset(
                "Jackknife_samples_of_PCAC_mass_correlator_values_2D_array",
                data=jackknife_samples_of_PCAC_mass_correlator_values_2D_array,
            ).attrs["Description"] = (
                "A 2D Numpy array containing the jackknife replicas of the "
                "PCAC mass correlator values grouped by gauge links "
                "configuration for specific tunable parameter values. The PCAC "
                "mass correlator is calculated by half the ratio of the"
                "g4g5-g5 derivative correlator over the g5-g5 correlator values."
            )
            # Store the jackknife average of the PCAC mass correlator samples as
            # two separate datasets: one for the mean and one for the error
            correlators_jackknife_analysis_hdf5_group.create_dataset(
                "Jackknife_average_of_PCAC_mass_correlator_mean_values",
                data=gv.mean(jackknife_average_of_PCAC_mass_correlator_array),
            ).attrs["Description"] = (
                "A Numpy array containing the mean values of the the jackknife"
                "average of the PCAC mass correlator jackknife samples."
            )
            correlators_jackknife_analysis_hdf5_group.create_dataset(
                "Jackknife_average_of_PCAC_mass_correlator_error_values",
                data=gv.sdev(jackknife_average_of_PCAC_mass_correlator_array),
            ).attrs["Description"] = (
                "A Numpy array containing the error values of the the jackknife"
                "average of the PCAC mass correlator jackknife samples."
            )

        # Store unique parameter values as attributes to data files set group
        for (
            parameter_name,
            parameter_value,
        ) in single_valued_fields_dictionary.items():
            output_data_files_set_group.attrs[parameter_name] = parameter_value

    click.echo("   -- Correlators jackknife analysis completed.")

    # Terminate logging
    logger.terminate_script_logging()


if __name__ == "__main__":
    main()
