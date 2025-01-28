import os

import click
import numpy as np
import gvar as gv
import logging
import h5py

from library import (
    momentum_correlator,
    jackknife_analysis,
    filesystem_utilities,
    data_processing,
    constants,
)


@click.command()
@click.option(
    "--input_parameter_values_csv_file_path",
    "input_parameter_values_csv_file_path",
    "-in_param_csv",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help=(
        "Path to input .csv file containing processed info extracted from qpb "
        "log files."
    ),
)
@click.option(
    "--input_correlators_hdf5_file_path",
    "input_correlators_hdf5_file_path",
    "-cor_hdf5",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to input HDF5 file containing extracted correlators values.",
)
@click.option(
    "--output_files_directory",
    "output_files_directory",
    "-out_dir",
    default=None,
    callback=filesystem_utilities.validate_directory,
    help="Path to directory where all output files will be stored.",
)
@click.option(
    "--output_hdf5_filename",
    "output_hdf5_filename",
    "-hdf5",
    default="PCAC_mass_correlator_values.h5",
    callback=filesystem_utilities.validate_output_HDF5_filename,
    help="Specific name for the output HDF5 file.",
)
@click.option(
    "--disable-logging",
    is_flag=True,
    help="Disable logging entirely.",
)
@click.option(
    "--log_file_directory",
    "log_file_directory",
    "-log_file_dir",
    default=None,
    callback=filesystem_utilities.validate_directory,
    help="Directory where the script's log file will be stored.",
)
@click.option(
    "--log_filename",
    "log_filename",
    "-log",
    default=None,
    callback=filesystem_utilities.validate_script_log_filename,
    help="Specific name for the script's log file.",
)
def main(
    input_parameter_values_csv_file_path,
    input_correlators_hdf5_file_path,
    output_files_directory,
    output_hdf5_filename,
    log_file_directory,
    log_filename,
):

    # HANDLE EMPTY INPUT ARGUMENTS

    # If no output directory is provided, use the directory of the input file
    if output_files_directory is None:
        output_files_directory = os.path.dirname(input_correlators_hdf5_file_path)

    # Specify current script's log file directory
    if log_file_directory is None:
        log_file_directory = output_files_directory

    # INITIATE LOGGING

    filesystem_utilities.setup_logging(log_file_directory, log_filename)

    # # Create a logger instance for the current script using the script's name.
    # logger = logging.getLogger(__name__)

    # Get the script's filename
    script_name = os.path.basename(__file__)

    # Initiate logging
    logging.info(f"Script '{script_name}' execution initiated.")

    # CREATE PLOTS SUBDIRECTORIES

    # if plot_g5g5_correlators:
    #     g5_g5_plots_base_name = "g5g5_correlator"
    #     g5_g5_plots_subdirectory = filesystem_utilities.create_subdirectory(
    #         plots_main_subdirectory,
    #         g5_g5_plots_base_name + "s",
    #         clear_contents=True,
    #     )

    # if plot_g4g5g5_correlators:
    #     g4g5_g5_plots_base_name = "g4g5g5_correlator"
    #     g4g5_g5_plots_subdirectory = filesystem_utilities.create_subdirectory(
    #         plots_main_subdirectory,
    #         g4g5_g5_plots_base_name + "s",
    #         clear_contents=True,
    #     )

    # if plot_g4g5g5_derivative_correlators:
    #     g4g5_g5_derivative_plots_base_name = "g4g5g5_derivative_correlator"
    #     g4g5_g5_derivative_plots_subdirectory = (
    #         filesystem_utilities.create_subdirectory(
    #             plots_main_subdirectory,
    #             g4g5_g5_derivative_plots_base_name + "s",
    #             clear_contents=True,
    #         )
    #     )

    # PCAC MASS CORRELATOR VALUES CALCULATION

    # Construct output HDF5 file path
    output_PCAC_mass_correlator_hdf5_file_path = os.path.join(
        output_files_directory, output_hdf5_filename
    )

    # Open the input HDF5 file for reading and the output HDF5 file for writing
    with h5py.File(input_correlators_hdf5_file_path, "r") as hdf5_file_read, h5py.File(
        output_PCAC_mass_correlator_hdf5_file_path, "w"
    ) as hdf5_file_write:

        # Initialize group structure of the output HDF5 file
        # NOTE: The assumption here is that the name of the raw data files
        # directory represents the data files set and its parent directory the
        # qpb main program that generated the data files
        parent_directory_name, last_directory_name = (
            filesystem_utilities.extract_directory_names(output_files_directory)
        )
        qpb_main_program_group = hdf5_file_write.create_group(parent_directory_name)
        data_files_set_group = qpb_main_program_group.create_group(last_directory_name)

        # Select input HDF5 file's group to read
        input_qpb_main_program_group = hdf5_file_read[parent_directory_name]
        input_data_files_set_group = input_qpb_main_program_group[last_directory_name]

        # ANALYZE .CSV FILE

        qpb_log_files_dataframe = data_processing.load_csv(
            input_parameter_values_csv_file_path
        )

        new_analyzer = data_processing.DataFrameAnalyzer(qpb_log_files_dataframe)
        print(new_analyzer.list_of_tunable_multivalued_parameter_names)

        # Create an instance of DataAnalyzer
        analyzer = data_processing.DataAnalyzer(qpb_log_files_dataframe)

        multivalued_fields_list = analyzer.get_multivalued_fields()
        multivalued_fields_list = list(
            set(multivalued_fields_list)
            - {"Filename", "Configuration_label", "Kernel_operator_type"}
        )

        # Store unique values as attributes to top-level output HDF5 groups
        for key, value in analyzer.fields_with_unique_values_dictionary.items():
            data_files_set_group.attrs[key] = value

        # Set excluded fields
        excluded_fields = {
            "Filename",
            *constants.OUTPUT_QUANTITY_NAMES_LIST,
            "Configuration_label",
        }
        analyzer.set_excluded_fields(excluded_fields)

        # Get valid (non-empty) dataframe groups with their metadata
        valid_groups_with_metadata = analyzer.get_valid_dataframe_groups_with_metadata()

        # Now process the valid dataframe groups
        for analysis_index, group_data in enumerate(
            valid_groups_with_metadata, start=1
        ):
            dataframe_group = group_data["dataframe_group"]
            metadata = group_data["metadata"]

            # Now 'group' contains the subset for this combination of values
            list_of_qpb_log_filenames = dataframe_group["Filename"].tolist()

            list_of_configuration_labels = dataframe_group[
                "Configuration_label"
            ].tolist()

            number_of_gauge_configurations = len(list_of_qpb_log_filenames)

            if number_of_gauge_configurations == 1:
                print("Cannot perform Jackknife analysis with sample size 1.")
                print("Skipping...")
                continue

            # Define a unique name for each top-level HDF5 group
            PCAC_mass_correlator_analysis_group_name = (
                f"PCAC_mass_correlator_analysis_{analysis_index}"
            )
            PCAC_mass_correlator_hdf5_group = data_files_set_group.create_group(
                PCAC_mass_correlator_analysis_group_name
            )

            PCAC_mass_correlator_hdf5_group.attrs["Number_of_gauge_configurations"] = (
                number_of_gauge_configurations
            )

            # Store unique values as attributes to current analysis HDF5 group
            for key, value in metadata.items():
                PCAC_mass_correlator_hdf5_group.attrs[key] = value

            # Initialize a dictionary to store multivalued fields values
            multivalued_fields_dictionary = {}
            for multivalued_field in multivalued_fields_list:
                multivalued_fields_dictionary[multivalued_field] = []

            # Pass "g5-g5" and "g4g5-g5" datasets, corresponding to different
            # gauge links configuration files, to respective lists common for
            # the current grouping of parameters
            g5_g5_correlator_values_per_configuration_list = []
            g4g5_g5_correlator_values_per_configuration_list = []

            for qpb_log_filename in list_of_qpb_log_filenames:

                # List values of multivalued fields per config labels sets
                for multivalued_field in multivalued_fields_list:
                    filtered_values = dataframe_group.loc[
                        dataframe_group["Filename"] == qpb_log_filename,
                        multivalued_field,
                    ]
                    if not filtered_values.empty:
                        multivalued_fields_dictionary[multivalued_field].append(
                            filtered_values.iloc[0]
                        )
                    # else:
                    # TODO: Work edge cases better
                    # multivalued_fields_dictionary[multivalued_field].append(None)

                correlators_filename = qpb_log_filename.replace(".txt", ".dat")

                # Cautionary check if the item is a
                if not correlators_filename in input_data_files_set_group:
                    # TODO: Log warning
                    continue

                filename_group = input_data_files_set_group[correlators_filename]

                if "g5-g5" in filename_group.keys():
                    g5_g5_dataset = filename_group["g5-g5"][:]
                    g5_g5_correlator_values_per_configuration_list.append(g5_g5_dataset)

                if "g4g5-g5" in filename_group.keys():
                    g4g5_g5_dataset = filename_group["g4g5-g5"][:]
                    g4g5_g5_correlator_values_per_configuration_list.append(
                        g4g5_g5_dataset
                    )

            # TODO: comment
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "list_of_configuration_labels",
                data=list_of_configuration_labels,
            ).attrs["Description"] = "List of configuration labels"

            for multivalued_field in multivalued_fields_list:
                dataset = multivalued_fields_dictionary[multivalued_field]
                if len(set(dataset)) == 1:
                    continue
                # Store dataset and attach brief description
                PCAC_mass_correlator_hdf5_group.create_dataset(
                    multivalued_field + "_values_array",
                    data=dataset,
                ).attrs["Description"] = (
                    multivalued_field.replace("_", " ")
                    + " values array per configuration label set"
                )

            # Convert the list of 1D arrays into a 2D NumPy array
            g5_g5_correlator_values_per_configuration_2D_array = np.vstack(
                g5_g5_correlator_values_per_configuration_list
            )
            g4g5_g5_correlator_values_per_configuration_2D_array = np.vstack(
                g4g5_g5_correlator_values_per_configuration_list
            )

            # JACKKNIFE ANALYSIS OF THE g5-g5

            # Jackknife analysis of the g5-g5 correlator values
            jackknife_analyzed_g5_g5_correlator_values_per_configuration_object = (
                jackknife_analysis.JackknifeAnalysis(
                    g5_g5_correlator_values_per_configuration_2D_array
                )
            )

            # Jackknife samples of the g5-g5 correlators
            jackknife_samples_of_g5_g5_correlator_2D_array = (
                jackknife_analyzed_g5_g5_correlator_values_per_configuration_object.jackknife_replicas_of_original_2D_array
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_samples_of_g5_g5_correlator_2D_array",
                data=jackknife_samples_of_g5_g5_correlator_2D_array,
            ).attrs["Description"] = "Jackknife samples."

            # Jackknife average of the g5-g5 correlators
            jackknife_average_of_g5_g5_correlator = (
                jackknife_analyzed_g5_g5_correlator_values_per_configuration_object.jackknife_average
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_average_of_g5_g5_correlator_mean_values",
                data=gv.mean(jackknife_average_of_g5_g5_correlator),
            ).attrs["Description"] = "Average from Jackknife samples. Mean values."
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_average_of_g5_g5_correlator_error_values",
                data=gv.sdev(jackknife_average_of_g5_g5_correlator),
            ).attrs["Description"] = "Average from Jackknife samples. Error values."

            # JACKKNIFE ANALYSIS OF THE g4g5-g5 PION CORRELATORS

            # Jackknife analysis of the g4g5-g5 correlator values
            jackknife_analyzed_g4g5_g5_correlator_values_per_configuration_object = (
                jackknife_analysis.JackknifeAnalysis(
                    g4g5_g5_correlator_values_per_configuration_2D_array
                )
            )

            # Jackknife samples of the g4g5-g5 correlators
            jackknife_samples_of_g4g5_g5_correlator_2D_array = (
                jackknife_analyzed_g4g5_g5_correlator_values_per_configuration_object.jackknife_replicas_of_original_2D_array
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_samples_of_g4g5_g5_correlator_2D_array",
                data=jackknife_samples_of_g4g5_g5_correlator_2D_array,
            ).attrs["Description"] = "Jackknife samples."

            # Jackknife average of the g4g5-g5 correlators
            jackknife_average_of_g4g5_g5_correlator = (
                jackknife_analyzed_g4g5_g5_correlator_values_per_configuration_object.jackknife_average
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_average_of_g4g5_g5_correlator_mean_values",
                data=gv.mean(jackknife_average_of_g4g5_g5_correlator),
            ).attrs["Description"] = "Average from Jackknife samples. Mean values."
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_average_of_g4g5_g5_correlator_error_values",
                data=gv.sdev(jackknife_average_of_g4g5_g5_correlator),
            ).attrs["Description"] = "Average from Jackknife samples. Error values."

            # PCAC MASS CORRELATOR CALCULATION

            """ 
            NOTE: The PCAC mass is defined as the ratio of the g4g5-g5
            derivative correlator values over the g5-g5 correlator values
            """

            # Jackknife samples of the g4g5-g5 derivate correlators
            jackknife_samples_of_g4g5_g5_derivative_correlator_2D_list = []
            for index in range(len(jackknife_samples_of_g4g5_g5_correlator_2D_array)):
                jackknife_samples_of_g4g5_g5_derivative_correlator_2D_list.append(
                    momentum_correlator.centered_difference_correlator_derivative(
                        jackknife_samples_of_g4g5_g5_correlator_2D_array[index]
                    )
                )
            jackknife_samples_of_g4g5_g5_derivative_correlator_2D_array = np.array(
                jackknife_samples_of_g4g5_g5_derivative_correlator_2D_list
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_samples_of_g4g5_g5_derivative_correlator_2D_array",
                data=jackknife_samples_of_g4g5_g5_derivative_correlator_2D_array,
            ).attrs["Description"] = "Centered difference derivative."

            # g4g5-g5 derivative correlator from the jackknife average of g4g5-g5 correlator
            g4g5_g5_derivative_correlator_from_jackknife_average_of_g4g5_g5_correlator = momentum_correlator.centered_difference_correlator_derivative(
                jackknife_average_of_g4g5_g5_correlator
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "g4g5_g5_derivative_correlator_from_jackknife_average_of_g4g5_g5_correlator_mean_values",
                data=gv.mean(
                    g4g5_g5_derivative_correlator_from_jackknife_average_of_g4g5_g5_correlator
                ),
            ).attrs[
                "Description"
            ] = "Centered difference derivative from average of g4g5-g5 correlator. Mean values."
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "g4g5_g5_derivative_correlator_from_jackknife_average_of_g4g5_g5_correlator_error_values",
                data=gv.sdev(
                    g4g5_g5_derivative_correlator_from_jackknife_average_of_g4g5_g5_correlator
                ),
            ).attrs[
                "Description"
            ] = "Centered difference derivative from average of g4g5-g5 correlator. Error values."

            # Jackknife samples of the time-dependent PCAC mass values
            jackknife_samples_of_PCAC_mass_correlator_values_list = []
            for index in range(len(jackknife_samples_of_g5_g5_correlator_2D_array)):

                jackknife_sample_of_PCAC_mass_values = (
                    0.5
                    * jackknife_samples_of_g4g5_g5_derivative_correlator_2D_array[index]
                    / jackknife_samples_of_g5_g5_correlator_2D_array[index]
                )

                jackknife_sample_of_PCAC_mass_values = (
                    momentum_correlator.symmetrization(
                        jackknife_sample_of_PCAC_mass_values
                    )
                )

                jackknife_samples_of_PCAC_mass_correlator_values_list.append(
                    jackknife_sample_of_PCAC_mass_values
                )
            jackknife_samples_of_PCAC_mass_correlator_values_2D_array = np.array(
                jackknife_samples_of_PCAC_mass_correlator_values_list
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_samples_of_PCAC_mass_correlator_values_2D_array",
                data=jackknife_samples_of_PCAC_mass_correlator_values_2D_array,
            ).attrs["Description"] = "Jackknife samples."

            # Jackknife average of the time-dependent PCAC mass values
            jackknife_average_of_PCAC_mass_correlator_values_array = gv.gvar(
                np.average(
                    jackknife_samples_of_PCAC_mass_correlator_values_2D_array,
                    axis=0,
                ),
                np.sqrt(number_of_gauge_configurations - 1)
                * np.std(
                    jackknife_samples_of_PCAC_mass_correlator_values_2D_array,
                    axis=0,
                    ddof=0,
                ),
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_average_of_PCAC_mass_correlator_values_array_mean_values",
                data=gv.mean(jackknife_average_of_PCAC_mass_correlator_values_array),
            ).attrs["Description"] = "Average from Jackknife samples. Mean values."
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "jackknife_average_of_PCAC_mass_correlator_values_array_error_values",
                data=gv.sdev(jackknife_average_of_PCAC_mass_correlator_values_array),
            ).attrs["Description"] = "Average from Jackknife samples. Error values."

            # Time-dependent PCAC mass values from the jackknife averages of the correlators
            PCAC_mass_correlator_values_from_jackknife_averages_of_correlators_array = (
                0.5
                * g4g5_g5_derivative_correlator_from_jackknife_average_of_g4g5_g5_correlator
                / jackknife_average_of_g5_g5_correlator
            )
            PCAC_mass_correlator_values_from_jackknife_averages_of_correlators_array = momentum_correlator.symmetrization(
                PCAC_mass_correlator_values_from_jackknife_averages_of_correlators_array
            )
            # Store dataset and attach brief description
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "PCAC_mass_correlator_values_from_jackknife_averages_of_correlators_array_mean_values",
                data=gv.mean(
                    PCAC_mass_correlator_values_from_jackknife_averages_of_correlators_array
                ),
            ).attrs["Description"] = "Average from average correlators. Mean values."
            PCAC_mass_correlator_hdf5_group.create_dataset(
                "PCAC_mass_correlator_values_from_jackknife_averages_of_correlators_array_error_values",
                data=gv.sdev(
                    PCAC_mass_correlator_values_from_jackknife_averages_of_correlators_array
                ),
            ).attrs["Description"] = "Average from average correlators. Error values."

    print("   -- PCAC mass correlator analysis completed.")

    # Terminate logging
    logging.info(f"Script '{script_name}' execution terminated successfully.")


if __name__ == "__main__":
    main()
