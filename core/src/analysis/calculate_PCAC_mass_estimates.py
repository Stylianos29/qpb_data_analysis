# TODO: Write a detailed introductory commentary
# TODO: Add more logging messages
# TODO: There should be a centralized class/function for plotting
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
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import gvar as gv
import lsqfit
import pandas as pd
import h5py
import copy

from library import (
    filesystem_utilities,
    effective_mass,
    custom_plotting,
    jackknife_analysis,
    momentum_correlator,
    PROCESSED_DATA_FILES_DIRECTORY,
    validate_input_directory,
    validate_input_script_log_filename,
)


@click.command()
@click.option(
    "-in_jack_hdf5",
    "--input_correlators_jackknife_analysis_hdf5_file_path",
    "input_correlators_jackknife_analysis_hdf5_file_path",
    required=True,
    callback=filesystem_utilities.validate_input_HDF5_file,
    help="Path to HDF5 file containing extracted correlators values.",
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
    "-plot_PCAC_corr",
    "--plot_PCAC_mass_correlators",
    "plot_PCAC_mass_correlators",
    is_flag=True,
    default=False,
    help="Enable plotting PCAC mass correlator values.",
)
@click.option(
    "-zoom_in",
    "--zoom_in_PCAC_mass_correlators_plots",
    "zoom_in_PCAC_mass_correlators_plots",
    is_flag=True,
    default=False,
    help="Enable zooming in on PCAC mass correlators plots.",
)
@click.option(
    "-out_csv_name",
    "--output_PCAC_mass_estimates_csv_filename",
    "output_PCAC_mass_estimates_csv_filename",
    default="PCAC_mass_estimates.csv",
    callback=filesystem_utilities.validate_output_csv_filename,
    help="Specific name for the output .csv file.",
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
    callback=validate_input_script_log_filename,
    help="Specific name for the script's log file.",
)
def main(
    input_correlators_jackknife_analysis_hdf5_file_path,
    output_files_directory,
    plots_directory,
    plot_PCAC_mass_correlators,
    zoom_in_PCAC_mass_correlators_plots,
    output_PCAC_mass_estimates_csv_filename,
    enable_logging,
    log_file_directory,
    log_filename,
):
    # HANDLE EMPTY INPUT ARGUMENTS

    # If no output directory is provided, use the directory of the input file
    if output_files_directory is None:
        output_files_directory = os.path.dirname(
            input_correlators_jackknife_analysis_hdf5_file_path
        )

    # INITIATE LOGGING

    # Setup logging
    logger = filesystem_utilities.LoggingWrapper(
        log_file_directory, log_filename, enable_logging
    )

    # Log script start
    logger.initiate_script_logging()

    # CREATE PLOTS SUBDIRECTORIES

    # Create plots main subdirectory if it does not exist if requested
    plots_main_subdirectory = filesystem_utilities.create_subdirectory(
        plots_directory,
        "PCAC_mass_estimates_calculation",
        clear_contents=True,
    )
    if plot_PCAC_mass_correlators:

        # Create deeper-level subdirectories if they do not exist
        PCAC_mass_plots_base_name = "PCAC_mass_correlator"
        PCAC_mass_plots_subdirectory = filesystem_utilities.create_subdirectory(
            plots_main_subdirectory,
            PCAC_mass_plots_base_name + "_values",
            clear_contents=True,
        )
        logger.info("Subdirectory for PCAC mass correlator plots created.")

    # IMPORT DATASETS AND METADATA

    # Open the input HDF5 file for reading and the output HDF5 file for writing
    with h5py.File(
        input_correlators_jackknife_analysis_hdf5_file_path, "r"
    ) as hdf5_file:

        # Construct the path to the processed data files set directory
        processed_data_files_set_directory = os.path.dirname(
            input_correlators_jackknife_analysis_hdf5_file_path
        )
        # The top HDF5 file groups (for both HDF5 files) mirror the directory
        # structure of the data files set directory itself and its parent
        # directories relative to the 'PROCESSED_DATA_FILES_DIRECTORY' directory
        input_data_files_set_group = filesystem_utilities.get_hdf5_target_group(
            hdf5_file,
            PROCESSED_DATA_FILES_DIRECTORY,
            processed_data_files_set_directory,
            logger=None,
        )
        logger.info("Top groups of the input HDF5 file identified.")

        # Check if input HDF5 file contains any data before initiating analysis
        if not filesystem_utilities.has_subgroups(input_data_files_set_group):
            logger.critical(
                "Input HDF5 file does not contain any data to analyze!", to_console=True
            )
            sys.exit(1)

        # Extract attributes of top-level groups into a dictionary.
        # NOTE: By construction the attributes of the top level groups are the
        # values of the parameters with a single unique value
        single_valued_parameters_dictionary = {
            parameter: attribute
            for parameter, attribute in input_data_files_set_group.attrs.items()
        }

        # List to pass values to dataframe
        PCAC_mass_estimates_list = []
        # Loop over all PCAC mass correlator jackknife analysis groups
        for (
            correlators_jackknife_analysis_group_name,
            correlators_jackknife_analysis_group,
        ) in input_data_files_set_group.items():

            # Cautionary check if the item is a HDF5 group
            if not isinstance(correlators_jackknife_analysis_group, h5py.Group):
                logger.warning(
                    f"'{correlators_jackknife_analysis_group_name}' is not a "
                    "subgroup of the input HDF5 file.",
                    to_console=True,
                )
                continue

            # STORE PARAMETER VALUES AND DATASETS FOR CURRENT JACKKNIFE ANALYSIS

            # Initialize the parameters values dictionary
            parameters_value_dictionary = copy.deepcopy(
                single_valued_parameters_dictionary
            )

            # Create a separate metadata dictionary containing attribute values
            # from the current jackknife analysis subgroup for later use.
            # NOTE: By design, the attributes of jackknife analysis subgroups
            # correspond to specific values of the multivalued parameters used
            # in forming the jackknife analysis dataframe groupings. Other
            # multivalued parameters that were not used in these groupings were
            # not stored as attributes. Instead, they were stored as datasets,
            # containing lists of values for each specific jackknife analysis
            # grouping.
            metadata_dictionary = {}
            for (
                parameter,
                attribute,
            ) in correlators_jackknife_analysis_group.attrs.items():
                metadata_dictionary[parameter] = attribute

            # Merge the metadata dictionary into the parameters values dictionary.
            parameters_value_dictionary.update(metadata_dictionary)

            # Store jackknife analysis identifier
            parameters_value_dictionary["Jackknife_analysis_identifier"] = (
                correlators_jackknife_analysis_group_name
            )

            # Import jackknife samples datasets of PCAC mass correlators
            jackknife_samples_of_PCAC_mass_correlators_2D_array = (
                correlators_jackknife_analysis_group[
                    "Jackknife_samples_of_PCAC_mass_correlator_values_2D_array"
                ][:]
            )
            if (
                np.isnan(jackknife_samples_of_PCAC_mass_correlators_2D_array).any()
                or np.isinf(jackknife_samples_of_PCAC_mass_correlators_2D_array).any()
            ):
                logger.warning(
                    f"{correlators_jackknife_analysis_group_name}: Jackknife "
                    "samples of PCAC mass correlators arrays contain NaN "
                    "or inf values and cannot be processed. Skipping..."
                )
                continue
            logger.info(
                f"Jackknife samples of PCAC mass correlators were loaded as a "
                "NumPy 2D array."
            )

            # VALIDATE VALUES OF IMPORTANT PARAMETERS

            # Ensuring the important parameter values of temporal lattice size
            # and number of gauge configurations are stored and available
            temporal_lattice_size = np.shape(
                jackknife_samples_of_PCAC_mass_correlators_2D_array
            )[1]
            if "Temporal_lattice_size" not in parameters_value_dictionary:
                parameters_value_dictionary["Temporal_lattice_size"] = (
                    temporal_lattice_size
                )
            elif (
                parameters_value_dictionary["Temporal_lattice_size"]
                != temporal_lattice_size
            ):
                logger.warning(
                    f"{correlators_jackknife_analysis_group_name}: Discrepancy "
                    "between the stored temporal lattice size value and the "
                    "size of the PCAC mass correlators NumPy 2D array.",
                    to_console=True,
                )

            number_of_gauge_configurations = np.shape(
                jackknife_samples_of_PCAC_mass_correlators_2D_array
            )[0]
            if "Number_of_gauge_configurations" not in parameters_value_dictionary:
                parameters_value_dictionary["Number_of_gauge_configurations"] = (
                    number_of_gauge_configurations
                )
            elif (
                parameters_value_dictionary["Number_of_gauge_configurations"]
                != number_of_gauge_configurations
            ):
                logger.warning(
                    f"{correlators_jackknife_analysis_group_name}: Discrepancy "
                    "between the stored number of gauge configurations value "
                    "and the size of the PCAC mass correlators NumPy 2D array.",
                    to_console=True,
                )
            logger.info(
                "The values for temporal lattice size and the number of gauge "
                "configurations were validated."
            )

            # CALCULATE FURTHER USEFUL QUANTITIES

            if (
                "Adjusted_average_core_hours_per_spinor_values_list"
                in correlators_jackknife_analysis_group
                and isinstance(
                    correlators_jackknife_analysis_group[
                        "Adjusted_average_core_hours_per_spinor_values_list"
                    ],
                    h5py.Dataset,
                )
            ):
                parameters_value_dictionary[
                    "Adjusted_average_core_hours_per_spinor_per_configuration"
                ] = np.average(
                    correlators_jackknife_analysis_group[
                        "Adjusted_average_core_hours_per_spinor_values_list"
                    ][:]
                )
            logger.info(
                "The adjusted average core hours per spinor per configuration "
                "was calculated."
            )

            if (
                "Average_number_of_MV_multiplications_per_spinor_values_list"
                in correlators_jackknife_analysis_group
                and isinstance(
                    correlators_jackknife_analysis_group[
                        "Average_number_of_MV_multiplications_per_spinor_values_list"
                    ],
                    h5py.Dataset,
                )
            ):
                parameters_value_dictionary[
                    "Average_number_of_MV_multiplications_per_spinor_per_configuration"
                ] = np.average(
                    correlators_jackknife_analysis_group[
                        "Average_number_of_MV_multiplications_per_spinor_values_list"
                    ][:]
                )
            logger.info(
                "The average number of mv multiplications per spinor per "
                "configuration was calculated."
            )

            logger.info(
                f"{correlators_jackknife_analysis_group_name}: Parameter "
                "values dictionary created and filled."
            )

            # TRUNCATE PCAC MASS CORRELATORS

            # jackknife_average_of_correlators_jackknife_array = (
            #     momentum_correlator.symmetrization(
            #         jackknife_average_of_correlators_jackknife_array
            #     )
            # )

            # Ignore the second half of the PCAC mass correlators arrays since
            # they are by construction symmetrized
            upper_index_cut = temporal_lattice_size // 2 + 1
            jackknife_samples_of_PCAC_mass_correlators_2D_array = np.array(
                [
                    momentum_correlator.symmetrization(correlators_jackknife_replica)[
                        :upper_index_cut
                    ]
                    for (
                        correlators_jackknife_replica
                    ) in jackknife_samples_of_PCAC_mass_correlators_2D_array
                ]
            )
            logger.info(
                "All PCAC mass correlator jackknife replicas were symmetrized, "
                f"then truncated at index {upper_index_cut}. Values at and  "
                "beyond this index were discarded."
            )

            jackknife_average_of_correlators_jackknife_array = (
                jackknife_analysis.calculate_jackknife_average_array(
                    jackknife_samples_of_PCAC_mass_correlators_2D_array
                )
            )

            # CALCULATE JACKKNIFE AVERAGE MASS CORRELATORS

            jackknife_average_correlators_jackknife_array = (
                jackknife_analysis.jackknife_average(
                    jackknife_samples_of_PCAC_mass_correlators_2D_array,
                    use_covariance=True,
                )
            )

            # jackknife_average_correlators_jackknife_array = np.mean(
            #     jackknife_samples_of_PCAC_mass_correlators_2D_array, axis=0
            # )

            # # Restrict the calculation range to a possible plateau range
            # calculation_range = np.arange(
            #     temporal_lattice_size // 4 - temporal_lattice_size // 8,
            #     temporal_lattice_size // 4 + temporal_lattice_size // 8,
            # )
            # integrated_autocorrelation_time = (
            #     jackknife_analysis.calculate_integrated_autocorrelation_time(
            #         jackknife_average_correlators_jackknife_array[calculation_range]
            #     )
            # )

            # if integrated_autocorrelation_time < 1:
            #     integrated_autocorrelation_time = 1

            integrated_autocorrelation_time = 1

            # jackknife_average_correlators_jackknife_array = gv.gvar(
            #     jackknife_average_correlators_jackknife_array,
            #     jackknife_analysis.jackknife_correlated_error(
            #         jackknife_samples_of_PCAC_mass_correlators_2D_array,
            #         integrated_autocorrelation_time,
            #     ),
            # )

            # PLATEAU RANGE FOR PCAC MASS CORRELATOR
            # NEW!!!!!!!!!!!!!!!
            sigma_thresholds = [2, 3, 4, 5, 7]
            plateau_found = False

            for sigma_threshold in sigma_thresholds:
                try:
                    plateau_start, plateau_final, _ = jackknife_analysis.detect_plateau_region(
                        jackknife_average_correlators_jackknife_array, 
                        sigma_threshold=sigma_threshold
                    )
                    plateau_found = True
                    break
                except:
                    continue

            if not plateau_found:
                logger.critical("Could not detect plateau region with any sigma threshold", to_console=True)
                sys.exit(1)

            sigma_criterion_factor = 1.5
            plateau_indices_list = []
            # TODO: Why this number?
            minimum_number_of_data_points = temporal_lattice_size // 8
            while len(plateau_indices_list) < minimum_number_of_data_points:
                plateau_indices_list = effective_mass.plateau_indices_range(
                    jackknife_average_correlators_jackknife_array,
                    sigma_criterion_factor,
                    3,
                )
                sigma_criterion_factor += 0.5

            # PLATEAU FIT ON EVERY REPLICA DATASET

            PCAC_mass_plateau_fit_guess = [
                np.mean(
                    gv.mean(
                        jackknife_average_correlators_jackknife_array[
                            plateau_indices_list
                        ]
                    )
                )
            ]

            plateau_fit_PCAC_mass_estimates_list = []
            for (
                correlators_jackknife_replica
            ) in jackknife_samples_of_PCAC_mass_correlators_2D_array:
                y = gv.gvar(
                    correlators_jackknife_replica,
                    gv.sdev(jackknife_average_correlators_jackknife_array),
                )
                x = np.arange(len(y))
                PCAC_mass_plateau_state_fit = lsqfit.nonlinear_fit(
                    data=(x[plateau_indices_list], y[plateau_indices_list]),
                    p0=PCAC_mass_plateau_fit_guess,
                    fcn=effective_mass.plateau_fit_function,
                    debug=True,
                )
                plateau_fit_PCAC_mass_estimates_list.append(
                    PCAC_mass_plateau_state_fit.p
                )

            PCAC_mass_estimate = gv.gvar(
                jackknife_analysis.weighted_mean(
                    gv.mean(plateau_fit_PCAC_mass_estimates_list),
                    gv.sdev(plateau_fit_PCAC_mass_estimates_list),
                    np.sqrt(number_of_gauge_configurations)
                    * np.sqrt(2 * integrated_autocorrelation_time),
                    # TODO: Still need a justification for including this:
                    # * np.sqrt(len(plateau_indices_list)),
                )
            )

            # New method
            PCAC_mass_estimate, _ = jackknife_analysis.plateau_estimate(
                jackknife_samples_of_PCAC_mass_correlators_2D_array,
                plateau_start,
                plateau_final,
                method="simple",
                use_inverse_variance=True,
            )

            # PLOT PCAC MASS CORRELATORS

            if plot_PCAC_mass_correlators:
                starting_time = 5
                y = jackknife_average_correlators_jackknife_array[starting_time:]
                x = np.arange(starting_time, len(y) + starting_time)

                fig, ax = plt.subplots()
                ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

                # Remove excluded title fields from parameters_value_dictionary
                excluded_title_fields = [
                    "Number_of_vectors",
                    "Delta_Min",
                    "Delta_Max",
                    "Lanczos_epsilon",
                    "Number_of_gauge_configurations",
                ]
                filtered_parameters_value_dictionary = {
                    k: v
                    for k, v in parameters_value_dictionary.items()
                    if k not in excluded_title_fields
                }

                plot_title = custom_plotting.DataPlotter._construct_plot_title(
                    None,
                    leading_substring="",
                    # metadata_dictionary={},
                    metadata_dictionary=metadata_dictionary,
                    title_width=90,
                    fields_unique_value_dictionary=filtered_parameters_value_dictionary,
                    additional_excluded_fields=[
                        "Adjusted_average_core_hours_per_spinor_per_configuration",
                        "Average_number_of_MV_multiplications_per_spinor_per_configuration",
                    ],
                )
                ax.set_title(f"{plot_title}", pad=8)

                ax.set(
                    xlabel="$t/a$",
                    ylabel="a$m_{PCAC}(t)$",
                )
                fig.subplots_adjust(left=0.15)  # Adjust left margin

                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                ax.errorbar(
                    x, gv.mean(y), yerr=gv.sdev(y), fmt=".", markersize=8, capsize=10
                )

                # if zoom_in_PCAC_mass_correlators_plots:
                #     y_limits = [0.9 * y[0].mean, 1.1 * PCAC_mass_estimate.mean]
                #     if y[0] > y[temporal_lattice_size // 4]:
                #         y_limits = [y_limits[1], y_limits[0]]
                #     ax.set_ylim(y_limits)

                plateau_range_minimum = x[plateau_indices_list[0] - starting_time]
                plateau_range_maximum = x[plateau_indices_list[-1] - starting_time]

                # ax.axvline(plateau_range_minimum, color="black")
                # ax.axvline(plateau_range_maximum, color="black")

                label_string = (
                    f"Plateau fit:\n"
                    f"- fit range: t/a$\\in[${plateau_range_minimum}, "
                    f"{plateau_range_maximum}$]$,\n- $m^{{best\!-\!fit}}_{{PCAC}}$"
                    f"={PCAC_mass_estimate:.3f}\n"
                )

                ax.hlines(
                    y=PCAC_mass_estimate.mean,
                    xmin=plateau_range_minimum,
                    xmax=plateau_range_maximum,
                    color="r",
                    linestyle="--",
                    label=label_string,
                )

                ax.fill_between(
                    np.arange(plateau_range_minimum, plateau_range_maximum + 1),
                    PCAC_mass_estimate.mean - PCAC_mass_estimate.sdev,
                    PCAC_mass_estimate.mean + PCAC_mass_estimate.sdev,
                    color="r",
                    alpha=0.2,
                )

                # if y[0] > y[temporal_lattice_size // 4]:
                #     ax.legend(loc="upper center")
                # else:
                #     ax.legend(loc="lower center")

                current_plots_base_name = PCAC_mass_plots_base_name
                plot_path = custom_plotting.DataPlotter._generate_plot_path(
                    None,
                    PCAC_mass_plots_subdirectory,
                    current_plots_base_name,
                    metadata_dictionary,
                    single_valued_fields_dictionary=single_valued_parameters_dictionary,
                )

                fig.savefig(plot_path)
                plt.close()

            parameters_value_dictionary["PCAC_mass_estimate"] = (
                PCAC_mass_estimate.mean,
                PCAC_mass_estimate.sdev,
            )

            PCAC_mass_estimates_list.append(parameters_value_dictionary)

        # EXPORT CALCULATED DATA

        # Check if list is empty before exporting
        if not PCAC_mass_estimates_list:
            logger.warning(
                "PCAC mass estimates analysis produced no results.", to_console=True
            )
            sys.exit(1)

        # Create a DataFrame from the extracted data
        PCAC_mass_estimates_dataframe = pd.DataFrame(PCAC_mass_estimates_list)

        # Construct output .csv file path
        csv_file_full_path = os.path.join(
            output_files_directory, output_PCAC_mass_estimates_csv_filename
        )
        # Export the DataFrame to a CSV file
        PCAC_mass_estimates_dataframe.to_csv(csv_file_full_path, index=False)

    click.echo("   -- PCAC mass estimates calculation completed!")

    # Terminate logging
    logger.terminate_script_logging()


if __name__ == "__main__":
    main()
