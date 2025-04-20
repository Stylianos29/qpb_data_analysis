# TODO: Write a detailed introductory commentary
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
import logging
import pandas as pd
import h5py
import copy

from library import (
    filesystem_utilities,
    effective_mass,
    custom_plotting,
    momentum_correlator,
    jackknife_analysis,
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
    "-plot_g5g5",
    "--plot_g5g5_correlators",
    "plot_g5g5_correlators",
    is_flag=True,
    default=False,
    help="Enable plotting g5-g5 correlator values.",
)
@click.option(
    "-plot_eff_mass",
    "--plot_effective_mass_correlators",
    "plot_effective_mass_correlators",
    is_flag=True,
    default=False,
    help="Enable plotting effective mass correlator values.",
)
@click.option(
    "-zoom_in",
    "--zoom_in_effective_mass_correlators_plots",
    "zoom_in_effective_mass_correlators_plots",
    is_flag=True,
    default=False,
    help="Enable zooming in on effective mass correlators plots.",
)
@click.option(
    "-out_csv_name",
    "--output_pion_effective_mass_estimates_csv_filename",
    "output_pion_effective_mass_estimates_csv_filename",
    default="pion_effective_mass_estimates.csv",
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
    plot_g5g5_correlators,
    plot_effective_mass_correlators,
    zoom_in_effective_mass_correlators_plots,
    output_pion_effective_mass_estimates_csv_filename,
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

    # Create main plots directory if does not exist if any plots were requested
    if any([plot_g5g5_correlators, plot_effective_mass_correlators]):
        plots_main_subdirectory = filesystem_utilities.create_subdirectory(
            plots_directory,
            "Effective_mass_calculation",
        )

    # Create deeper-level subdirectories if specific plots were requested
    if plot_g5g5_correlators:
        g5g5_plots_base_name = "g5g5_correlator"
        g5g5_plots_subdirectory = filesystem_utilities.create_subdirectory(
            plots_main_subdirectory,
            g5g5_plots_base_name + "_values",
            clear_contents=True,
        )
        logger.info("Subdirectory for g5-g5 correlator plots created.")
    if plot_effective_mass_correlators:
        effective_mass_plots_base_name = "effective_mass_correlator"
        effective_mass_plots_subdirectory = filesystem_utilities.create_subdirectory(
            plots_main_subdirectory,
            effective_mass_plots_base_name + "_values",
            clear_contents=True,
        )
        logger.info("Subdirectory for pion effective mass correlator plots created.")

    # IMPORT DATASETS AND METADATA

    # Open the input HDF5 file for reading
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

        # Ensure if input HDF5 file contains any data before initiating analysis
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
        pion_effective_mass_estimates_list = []
        # Loop over all PCAC mass correlator Jackknife analysis groups
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

            # Import jackknife average of the g5-g5 correlator from both mean
            # and error values datasets

            # TODO: Assign these datasets names to a centralized constant such
            # that any changes to names do not break down the whole script
            jackknife_average_g5g5_correlator_array = gv.gvar(
                correlators_jackknife_analysis_group[
                    "Jackknife_average_of_g5_g5_correlator_mean_values"
                ][:],
                correlators_jackknife_analysis_group[
                    "Jackknife_average_of_g5_g5_correlator_error_values"
                ][:],
            )
            logger.info(
                "Jackknife average of g5-g5 correlator dataset was loaded as "
                "a NumPy array."
            )

            # Import Jackknife replicas datasets of g5-g5 correlators
            jackknife_samples_of_g5g5_correlators_2D_array = (
                correlators_jackknife_analysis_group[
                    "Jackknife_samples_of_g5_g5_correlator_2D_array"
                ][:]
            )
            logger.info(
                "Jackknife samples of g5-g5 correlators datasets were loaded "
                "as a NumPy 2D array."
            )

            # SYMMETRIZE G5-G5 CORRELATORS

            # Precautionary symmetrization of the g5g-g5 correlators
            jackknife_average_g5g5_correlator_array = (
                momentum_correlator.symmetrization(
                    jackknife_average_g5g5_correlator_array
                )
            )
            logger.info("Jackknife average of g5-g5 correlators was symmetrized.")

            jackknife_samples_of_g5g5_correlators_2D_array = np.array(
                [
                    momentum_correlator.symmetrization(g5_g5_correlator_replica)
                    for g5_g5_correlator_replica in jackknife_samples_of_g5g5_correlators_2D_array
                ]
            )
            logger.info("Jackknife samples of g5-g5 correlators were symmetrized.")

            # CALCULATE EFFECTIVE MASS CORRELATORS

            # Halved effective mass replica correlator values from jackknife
            # replicas of g5g5 correlator values
            effective_mass_correlator_replicas_2D_array = np.array(
                [
                    effective_mass.calculate_two_state_periodic_effective_mass_correlator(
                        g5_g5_correlator_replica
                    )
                    for g5_g5_correlator_replica in jackknife_samples_of_g5g5_correlators_2D_array
                ]
            )
            logger.info(
                "Jackknife samples of pion effective mass correlators were "
                "calculated."
            )

            jackknife_average_effective_mass_correlator = np.mean(
                effective_mass_correlator_replicas_2D_array, axis=0
            )

            integrated_autocorrelation_time = (
                jackknife_analysis.calculate_integrated_autocorrelation_time(
                    jackknife_average_effective_mass_correlator
                )
            )

            integrated_autocorrelation_time = 1

            jackknife_average_effective_mass_correlator = gv.gvar(
                jackknife_average_effective_mass_correlator,
                jackknife_analysis.jackknife_correlated_error(
                    effective_mass_correlator_replicas_2D_array,
                    integrated_autocorrelation_time,
                ),
            )
            logger.info(
                "Jackknife average of the pion effective mass correlators was "
                "calculated."
            )

            # VALIDATE VALUES OF IMPORTANT PARAMETERS

            # Ensuring the important parameter values of temporal lattice size
            # and number of gauge configurations are stored and available
            temporal_lattice_size = np.shape(
                jackknife_samples_of_g5g5_correlators_2D_array
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
                jackknife_samples_of_g5g5_correlators_2D_array
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
                "Adjusted_average_core_hours_per_spinor"
                in correlators_jackknife_analysis_group
                and isinstance(
                    correlators_jackknife_analysis_group[
                        "Adjusted_average_core_hours_per_spinor"
                    ],
                    h5py.Dataset,
                )
            ):
                parameters_value_dictionary[
                    "Adjusted_average_core_hours_per_spinor_per_configuration"
                ] = np.average(
                    correlators_jackknife_analysis_group[
                        "Adjusted_average_core_hours_per_spinor"
                    ][:]
                )
            logger.info(
                "The adjusted average core hours per spinor per configuration "
                "was calculated."
            )

            if (
                "Average_number_of_MV_multiplications_per_spinor_values_array"
                in correlators_jackknife_analysis_group
                and isinstance(
                    correlators_jackknife_analysis_group[
                        "Average_number_of_MV_multiplications_per_spinor_values_array"
                    ],
                    h5py.Dataset,
                )
            ):
                parameters_value_dictionary[
                    "Average_number_of_MV_multiplications_per_spinor_per_configuration"
                ] = np.average(
                    correlators_jackknife_analysis_group[
                        "Average_number_of_MV_multiplications_per_spinor_values_array"
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

            ### CALCULATE PLATEAU AND TWO-STATE RANGE FOR EFFECTIVE MASS FIT ###
            """
            In this section The aim.
            """

            # CALCULATE GUESS PARAMETERS FOR G5G5 CORRELATORS SINGLE-STATE FITS
            """
            Fitting begins with a naive guess of the amplitude of the g5-g5
            time-dependent correlator C(t) and the pion effective mass estimate.
            Assuming that either the left or the right symmetric part of C(t)
            has a A e^(-m t) form.
            """

            # NOTE: Usually the time-dependent pion effective mass values at
            # about t=T/4 are a good estimate of the plateau fit effective mass
            # estimate, and this the reason it can be used as a naive effective
            # mass guess for the single-state g5-g5 correlator, and
            # correspondingly for its amplitude.
            guess_index = temporal_lattice_size // 4

            # Pion effective mass guess
            single_state_non_periodic_effective_mass_correlator = effective_mass.calculate_single_state_non_periodic_effective_mass_correlator(
                jackknife_average_g5g5_correlator_array
            )
            effective_mass_guess = gv.mean(
                single_state_non_periodic_effective_mass_correlator[guess_index]
            )

            # Amplitude factor guess
            amplitude_factor_guess = (
                momentum_correlator.amplitude_of_single_state_non_periodic_correlator(
                    jackknife_average_g5g5_correlator_array,
                    effective_mass_guess,
                    guess_index,
                )
            )
            amplitude_factor_guess = gv.mean(amplitude_factor_guess)

            # EXTRACT BETTER GUESS PARAMETERS FROM SINGLE STATE NON PERIODIC FIT
            """
            Improve fitting guess parameters by fitting on the jackknife average
            of the g5-g5 correlator a single-state non-periodic exponential
            function using the naive guesses for pion effective mass estimate
            and aptitude factor previously calculated.

            The choice of the single-state non-periodic exponential function to
            be used for this fitting was made for simplicity. However, its
            non-periodicity restricts its usefulness to half the (periodic)
            g5-g5 correlator.
            """

            y = jackknife_average_g5g5_correlator_array
            x = np.arange(len(y))

            # Set arbitrarily (T//4-T//8, T//4+T//8) as the fitting range

            # NOTE: The reason this range was chose is because it is expected to
            # exhibit the smoothest behavior for single-state exponential
            # fitting
            single_state_non_periodic_fit_range = np.arange(
                guess_index - guess_index // 2, guess_index + guess_index // 2
            )
            # Use the previously calculated pion effective mass estimate and
            # aptitude factor as guess parameters for fitting
            g5_g5_correlator_single_state_non_periodic_fit_p0 = [
                amplitude_factor_guess,
                effective_mass_guess,
            ]
            g5_g5_correlator_single_state_fit = lsqfit.nonlinear_fit(
                data=(
                    x[single_state_non_periodic_fit_range],
                    y[single_state_non_periodic_fit_range],
                ),
                p0=g5_g5_correlator_single_state_non_periodic_fit_p0,
                fcn=momentum_correlator.single_state_non_periodic_correlator,
                debug=True,
            )

            # PLOT G5-G5 CORRELATOR FOR ASSESSING GUESS PARAMETERS IF REQUESTED
            """
            Plotting the jackknife average of the g5-g5 correlator, if
            requested, to evaluate the fit of the single-state non-periodic
            exponential function and assess the quality of the resulting
            best-fit parameters.
            """

            if plot_g5g5_correlators:

                # Exclude first point for a more symmetrical shape
                starting_time = 1
                y = jackknife_average_g5g5_correlator_array[starting_time:]
                x = np.arange(starting_time, len(y) + starting_time)

                fig, ax = plt.subplots()

                ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

                ax.set(xlabel="$t/a$", ylabel="$C(t)$")
                ax.set_yscale("log")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                plot_title = custom_plotting.DataPlotter._construct_plot_title(
                    None,
                    leading_substring="",
                    metadata_dictionary=metadata_dictionary,
                    title_width=100,
                    fields_unique_value_dictionary=parameters_value_dictionary,
                )
                ax.set_title(f"{plot_title}", pad=8)

                ax.errorbar(
                    x, gv.mean(y), yerr=gv.sdev(y), fmt=".", markersize=8, capsize=10
                )

                ax.axvline(
                    x[single_state_non_periodic_fit_range.min()],
                    color="green",
                    linestyle="--",
                    alpha=0.5,
                )
                ax.axvline(
                    x[single_state_non_periodic_fit_range.max()],
                    color="green",
                    linestyle="--",
                    alpha=0.5,
                )

                # Plot single-state periodic curve using fit non-periodic parameters
                x_data = np.linspace(min(x), max(x), 100)
                label_string = (
                    f"Single-state non-periodic fit:\n"
                    f"- fit range: t/a$\\in[${x[single_state_non_periodic_fit_range.min()]}, "
                    f"{x[single_state_non_periodic_fit_range.max()]}$]$, \n"
                    f"- $\\chi^2$/dof={g5_g5_correlator_single_state_fit.chi2:.2f}"
                    f"/{g5_g5_correlator_single_state_fit.dof}="
                    f"{g5_g5_correlator_single_state_fit.chi2 / g5_g5_correlator_single_state_fit.dof:.3f},\n"
                    f"- $m^{{best\\!-\!fit}}_{{eff.}}$={g5_g5_correlator_single_state_fit.p[1]:.3f}"
                )
                ax.plot(
                    x_data,
                    momentum_correlator.single_state_periodic_correlator(
                        x_data, gv.mean(g5_g5_correlator_single_state_fit.p)
                    ),
                    "r--",
                    label=label_string,
                )

                ax.legend(loc="upper center", framealpha=1.0)

                current_plots_base_name = "g5g5_average_correlator"
                plot_path = custom_plotting.DataPlotter._generate_plot_path(
                    None,
                    g5g5_plots_subdirectory,
                    current_plots_base_name,
                    metadata_dictionary,
                    single_valued_fields_dictionary=parameters_value_dictionary,
                )
                fig.savefig(plot_path)
                plt.close()

            # PLATEAU RANGE FOR EFFECTIVE MASS CORRELATOR
            """
            Use the resulting best-fit parameters from the the single-state
            non-periodic exponential function fit on the jackknife average of
            the g5-g5 correlator to calculate 
            """

            sigma_criterion_factor = 1.0
            plateau_indices_list = []
            # TODO: Why 5?
            while len(plateau_indices_list) < 5:
                plateau_indices_list = effective_mass.plateau_indices_range(
                    jackknife_average_effective_mass_correlator, sigma_criterion_factor
                )
                sigma_criterion_factor += 0.5

            # CALCULATE GUESS PARAMETERS FOR EFFECTIVE MASS TWO-STATE FITS

            y = jackknife_average_effective_mass_correlator[
                : plateau_indices_list[-1] + 1
            ]
            x = np.arange(len(y))

            # Use best fit values from g5g5 correlator single-state fit
            amplitude_factor_guess = gv.mean(g5_g5_correlator_single_state_fit.p[0])
            effective_mass_guess = gv.mean(g5_g5_correlator_single_state_fit.p[1])

            c_guess = np.log(
                (y[1] - amplitude_factor_guess * np.exp(-effective_mass_guess))
                / (y[2] - amplitude_factor_guess * np.exp(-2 * effective_mass_guess))
            )
            c_guess = gv.mean(c_guess)

            r_guess = (np.exp(np.sum(y) - len(y) * effective_mass_guess) - 1) / np.exp(
                -c_guess
            )
            r_guess = gv.mean(r_guess)

            effective_mass_two_state_fit_p0 = [effective_mass_guess, r_guess, c_guess]

            # CALCULATE BETTER GUESS PARAMETERS AND FIT RANGE

            optimum_Q = 0  # Initialize optimum p-value variable
            # Investigate the optimum lower index of the fit range. Maintain a
            # distance of at least two point from the plateau range
            for lower_index_cut in range(plateau_indices_list[0] - 1):
                effective_mass_two_state_fit = lsqfit.nonlinear_fit(
                    data=(x[lower_index_cut:], y[lower_index_cut:]),
                    p0=effective_mass_two_state_fit_p0,
                    fcn=effective_mass.two_state_fit_function,
                    debug=True,
                )

                if (
                    (optimum_Q < effective_mass_two_state_fit.Q)
                    and (effective_mass_two_state_fit.p[2] > 0)
                    and (effective_mass_two_state_fit.p[1] > 0)
                ):
                    optimum_Q = effective_mass_two_state_fit.Q
                    effective_mass_two_state_optimum_fit = effective_mass_two_state_fit
                    optimum_lower_index = lower_index_cut

            effective_mass_two_state_fit_p0 = gv.mean(
                effective_mass_two_state_optimum_fit.p
            )

            # TWO-STATE FIT ON EVERY REPLICA DATASET

            two_state_effective_mass_estimates_list = []
            for (
                effective_mass_correlator_replica
            ) in effective_mass_correlator_replicas_2D_array:
                y = gv.gvar(
                    effective_mass_correlator_replica,
                    gv.sdev(jackknife_average_effective_mass_correlator),
                )
                y = y[optimum_lower_index : plateau_indices_list[-1]]
                x = np.arange(optimum_lower_index, len(y) + optimum_lower_index)
                effective_mass_two_state_fit = lsqfit.nonlinear_fit(
                    data=(x, y),
                    p0=effective_mass_two_state_fit_p0,
                    fcn=effective_mass.two_state_fit_function,
                    debug=True,
                )
                two_state_effective_mass_estimates_list.append(
                    effective_mass_two_state_fit.p[0]
                )

            pion_effective_mass_estimate = gv.gvar(
                jackknife_analysis.weighted_mean(
                    gv.mean(two_state_effective_mass_estimates_list),
                    gv.sdev(two_state_effective_mass_estimates_list),
                    np.sqrt(number_of_gauge_configurations)
                    * np.sqrt(2 * integrated_autocorrelation_time),
                    # TODO: Still need a justification for including this:
                    # * np.sqrt(len(plateau_indices_list)),
                )
            )

            # PLOT EFFECTIVE MASS CORRELATORS
            if plot_effective_mass_correlators:
                y = jackknife_average_effective_mass_correlator
                x = np.arange(len(y))

                fig, ax = plt.subplots()
                ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

                plot_title = custom_plotting.DataPlotter._construct_plot_title(
                    None,
                    leading_substring="",
                    # metadata_dictionary={},
                    metadata_dictionary=metadata_dictionary,
                    title_width=110,
                    fields_unique_value_dictionary=single_valued_parameters_dictionary,
                    additional_excluded_fields=[
                        "Average_calculation_time_per_spinor_per_configuration",
                        "Average_number_of_MV_multiplications_per_spinor_per_configuration",
                    ],
                )
                ax.set_title(f"{plot_title}", pad=8)

                if zoom_in_effective_mass_correlators_plots:
                    ax.set_ylim(
                        [
                            0.5 * pion_effective_mass_estimate.mean,
                            1.5 * pion_effective_mass_estimate.mean,
                        ]
                    )

                ax.set(
                    xlabel="$t/a$",
                    ylabel=(
                        "$m_{eff.}\!(t)=\\frac{1}{2}\\log\!\\left( "
                        "\\frac{C(t-1)+\\sqrt{C^2(t-1)-C^2(T//2)})}{C(t+1)"
                        "+\\sqrt{C^2(t+1)-C^2(T//2)})} \\right) $"
                    ),
                )
                fig.subplots_adjust(left=0.15)  # Adjust left margin

                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                ax.errorbar(
                    x, gv.mean(y), yerr=gv.sdev(y), fmt=".", markersize=8, capsize=10
                )

                label_string = (
                    f"Plateau fit:\n- fit range: t/a$\\in[${x[plateau_indices_list[0]]}, "
                    f"{x[plateau_indices_list[-1]]}$]$,\n- $m^{{best\!-\!fit}}_{{eff.}}$"
                    f"={pion_effective_mass_estimate:.3f}\n"
                )

                ax.hlines(
                    y=pion_effective_mass_estimate.mean,
                    xmin=x[plateau_indices_list[0]],
                    xmax=x[plateau_indices_list[-1]],
                    color="r",
                    linestyle="--",
                    label=label_string,
                )

                ax.fill_between(
                    x[plateau_indices_list],
                    pion_effective_mass_estimate.mean
                    - pion_effective_mass_estimate.sdev,
                    pion_effective_mass_estimate.mean
                    + pion_effective_mass_estimate.sdev,
                    color="r",
                    alpha=0.2,
                )

                ax.legend(loc="upper right")

                current_plots_base_name = "effective_mass_correlator"
                plot_path = custom_plotting.DataPlotter._generate_plot_path(
                    None,
                    effective_mass_plots_subdirectory,
                    current_plots_base_name,
                    metadata_dictionary,
                    single_valued_fields_dictionary=single_valued_parameters_dictionary,
                )

                fig.savefig(plot_path)
                plt.close()

            # EXPORT CALCULATED DATA

            parameters_value_dictionary["Pion_effective_mass_estimate"] = (
                pion_effective_mass_estimate.mean,
                pion_effective_mass_estimate.sdev,
            )

            pion_effective_mass_estimates_list.append(parameters_value_dictionary)

        # Create a DataFrame from the extracted data
        pion_effective_mass_estimates_dataframe = pd.DataFrame(
            pion_effective_mass_estimates_list
        )

        # Construct output .csv file path
        csv_file_full_path = os.path.join(
            output_files_directory, output_pion_effective_mass_estimates_csv_filename
        )

        # Export the DataFrame to a CSV file
        pion_effective_mass_estimates_dataframe.to_csv(csv_file_full_path, index=False)

    click.echo("   -- Pion effective mass estimates calculation completed.")

    # Terminate logging
    logger.terminate_script_logging()


if __name__ == "__main__":
    main()
