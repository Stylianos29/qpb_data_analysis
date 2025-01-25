import os
import sys
import itertools

import click
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
import lsqfit
import logging
import pandas as pd
import h5py
import copy
from collections import Counter

from library import filesystem_utilities, plotting, data_processing, fit_functions


@click.command()
@click.option(
    "--input_PCAC_mass_estimates_csv_file_path",
    "input_PCAC_mass_estimates_csv_file_path",
    "-PCAC_csv",
    # default="/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/invert/Chebyshev_several_config_varying_N/PCAC_mass_estimates.csv",
    required=True,
    help="Path to .csv file containing PCAC mass estimates.",
)
@click.option(
    "--input_PCAC_mass_correlator_hdf5_file_path",
    "input_PCAC_mass_correlator_hdf5_file_path",
    "-PCAC_hdf5",
    # default="/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/invert/KL_several_config_and_mu_varying_m/PCAC_mass_correlator_values.h5",
    required=True,
    help="Path to input HDF5 file containing extracted correlators values.",
)
@click.option(
    "--output_files_directory",
    "output_files_directory",
    "-out_dir",
    default=None,
    help="Path to directory where all output files will be stored.",
)
@click.option(
    "--plots_directory",
    "plots_directory",
    "-plots_dir",
    default="../../output/plots",
    # default="/nvme/h/cy22sg1/qpb_data_analysis/output/plots/invert/Chebyshev_several_config_varying_N",
    help="Path to the output directory for storing plots.",
)
@click.option(
    "-plot_critical",
    "--plot_critical_bare_mass",
    "plot_critical_bare_mass",
    is_flag=True,
    default=False,
    help="Enable plotting critical bare mass.",
)
@click.option(
    "-apply_fits",
    "--fit_for_critical_bare_mass",
    "fit_for_critical_bare_mass",
    is_flag=True,
    default=False,
    help="Enable performing fits for the calculation of critical bare mass.",
)
@click.option(
    "-annotate",
    "--annotate_data_points",
    "annotate_data_points",
    is_flag=True,
    default=False,
    help="Enable annotating the data points.",
)
@click.option(
    "--output_critical_bare_mass_csv_filename",
    "output_critical_bare_mass_csv_filename",
    "-out_csv",
    default="critical_bare_mass_from_PCAC_mass_estimates.csv",
    help="Specific name for the output .csv files.",
)
@click.option(
    "--log_file_directory",
    "log_file_directory",
    "-log_file_dir",
    default=None,
    help="Directory where the script's log file will be stored.",
)
@click.option(
    "--log_filename",
    "log_filename",
    "-log",
    default=None,
    help="Specific name for the script's log file.",
)
def main(
    input_PCAC_mass_estimates_csv_file_path,
    input_PCAC_mass_correlator_hdf5_file_path,
    output_files_directory,
    plots_directory,
    plot_critical_bare_mass,
    fit_for_critical_bare_mass,
    annotate_data_points,
    output_critical_bare_mass_csv_filename,
    log_file_directory,
    log_filename,
):
    # VALIDATE INPUT ARGUMENTS

    if not filesystem_utilities.is_valid_file(input_PCAC_mass_estimates_csv_file_path):
        error_message = "Passed correlator values HDF5 file path is invalid."
        print("ERROR:", error_message)
        sys.exit(1)

    # If no output directory is provided, use the directory of the input file
    if output_files_directory is None:
        output_files_directory = os.path.dirname(
            input_PCAC_mass_estimates_csv_file_path
        )
    # Check validity if the provided
    elif not filesystem_utilities.is_valid_file(output_files_directory):
        error_message = (
            "Passed output files directory path is invalid " "or not a directory."
        )
        print("ERROR:", error_message)
        print("Exiting...")
        sys.exit(1)

    if not filesystem_utilities.is_valid_directory(plots_directory):
        error_message = "The specified plots directory path is invalid."
        print("ERROR:", error_message)
        sys.exit(1)

    # Specify current script's log file directory
    if log_file_directory is None:
        log_file_directory = output_files_directory
    elif not filesystem_utilities.is_valid_directory(log_file_directory):
        error_message = (
            "Passed directory path to store script's log file is "
            "invalid or not a directory."
        )
        print("ERROR:", error_message)
        print("Exiting...")
        sys.exit(1)

    # Get the script's filename
    script_name = os.path.basename(__file__)

    if log_filename is None:
        log_filename = script_name.replace(".py", "_python_script.log")

    # Check for proper extensions in provided output filenames
    if not output_critical_bare_mass_csv_filename.endswith(".csv"):
        output_critical_bare_mass_csv_filename = (
            output_critical_bare_mass_csv_filename + ".csv"
        )
    if not log_filename.endswith(".log"):
        log_filename = log_filename + ".log"

    # INITIATE LOGGING

    filesystem_utilities.setup_logging(log_file_directory, log_filename)

    # # Create a logger instance for the current script using the script's name.
    # logger = logging.getLogger(__name__)

    # Get the script's filename
    script_name = os.path.basename(__file__)

    # Initiate logging
    logging.info(f"Script '{script_name}' execution initiated.")

    # CREATE PLOTS SUBDIRECTORIES

    if plot_critical_bare_mass:
        # Create main plots directory if it does not exist
        plots_main_subdirectory = filesystem_utilities.create_subdirectory(
            plots_directory,
            "critical_bare_mass_calculation",
        )

        # Create deeper-level subdirectories if requested
        critical_bare_mass_plots_base_name = "critical_bare_mass"
        critical_bare_mass_plots_subdirectory = filesystem_utilities.create_subdirectory(
            plots_main_subdirectory,
            critical_bare_mass_plots_base_name + "_from_PCAC_mass",
            # clear_contents=True,
        )

    # IMPORT DATASETS AND METADATA

    # Boolean variable indicating he 
    critical_bare_mass_values_calculated = False

    # Load the .csv file to a dataframe
    PCAC_mass_estimates_dataframe = data_processing.load_csv(
        input_PCAC_mass_estimates_csv_file_path
    )

    analyzer = data_processing.DataFrameAnalyzer(PCAC_mass_estimates_dataframe)

    single_valued_fields_dictionary = analyzer.single_valued_fields_dictionary

    tunable_multivalued_parameters_list = (
        analyzer.list_of_tunable_multivalued_parameter_names
    )

    # TODO: Rethink this strategy of excluding "MPI_geometry" manually
    tunable_multivalued_parameters_list = [
        item for item in tunable_multivalued_parameters_list if item != "MPI_geometry"
    ]

    # Remove "Bare_mass" from the list of tunable multivalued parameters
    if "Bare_mass" not in tunable_multivalued_parameters_list:
        error_message = (
            "Critical bare mass analysis cannot be performed without a range "
            "of bare mass values data."
        )
        print("ERROR:", error_message)
        sys.exit(1)
    tunable_multivalued_parameters_list.remove("Bare_mass")

    # Initialize list with parameter values for the output dataframe
    critical_bare_mass_values_list = []

    # Open the input HDF5 file for reading and the output HDF5 file for writing
    with h5py.File(input_PCAC_mass_correlator_hdf5_file_path, "r") as hdf5_file_read:

        # Initialize group structure of the output HDF5 file
        # NOTE: The assumption here is that the name of the raw data files
        # directory represents the data files set (or experiment) and its parent
        # directory the qpb main program that generated the data files
        parent_directory_name, last_directory_name = (
            filesystem_utilities.extract_directory_names(output_files_directory)
        )

        # Select input HDF5 file's group to read
        input_qpb_main_program_group = hdf5_file_read[parent_directory_name]
        input_data_files_set_group = input_qpb_main_program_group[last_directory_name]

        for value, group in PCAC_mass_estimates_dataframe.groupby(
            tunable_multivalued_parameters_list
        ):
            # Check for a minimum amount of data point
            if group["Bare_mass"].nunique() < 3:
                # TODO: Log warning
                continue

            critical_bare_mass_values_calculated = True

            list_of_Jackknife_analysis_identifiers = group[
                "Jackknife_analysis_identifier"
            ].to_list()

            # Construct a flattened list of all the labels of all the gauge
            # links configurations used to calculate the PCAC mass estimates and
            # eventually the critical bare mass
            list_of_configuration_labels = list(
                itertools.chain.from_iterable(
                    [
                        input_data_files_set_group[Jackknife_analysis_identifier][
                            "list_of_configuration_labels"
                        ][:]
                        for Jackknife_analysis_identifier in list_of_Jackknife_analysis_identifiers
                    ]
                )
            )

            # Look for duplicates
            if len(list_of_configuration_labels) != len(
                set(list_of_configuration_labels)
            ):
                counter = Counter(list_of_configuration_labels)
                # Get duplicates
                duplicates_dictionary = {
                    label: count for label, count in counter.items() if count > 1
                }
                # TODO: Log warning
                print(
                    "WARNING:",
                    "The following gauge links configurations have been used "
                    "more than once in the calculation of the critical bare mass:",
                )
                print("Configuration label, counts")
                for label, count in duplicates_dictionary.items():
                    print(label, count)

            # Initialize the parameters values dictionary
            parameters_value_dictionary = copy.deepcopy(single_valued_fields_dictionary)

            # Store for later use
            if not isinstance(value, tuple):
                value = [value]
            metadata_dictionary = dict(zip(tunable_multivalued_parameters_list, value))

            # Append metadata dictionary
            parameters_value_dictionary.update(metadata_dictionary)

            number_of_gauge_configurations_array = group[
                "Number_of_gauge_configurations"
            ].to_numpy()

            x = group["Bare_mass"].to_numpy()
            y = gv.gvar(group["PCAC_mass_estimate"].to_numpy())

            # FITS

            if fit_for_critical_bare_mass:

                # Find indices of min(x) and max(x)
                min_index = np.argmin(x)
                max_index = np.argmax(x)

                # Get y values corresponding to min(x) and max(x)
                y_min = y[min_index]
                y_max = y[max_index]

                slope = (y_max - y_min) / (np.max(x) - np.min(x))
                x_intercept = y_min / slope + np.min(x)
                linear_fit_p0 = gv.mean([slope, x_intercept])
                linear_fit = lsqfit.nonlinear_fit(
                    data=(x, y),
                    p0=linear_fit_p0,
                    fcn=fit_functions.linear_function,
                    debug=True,
                )
                critical_bare_mass_value = linear_fit.p[1]

            # PLOT

            if plot_critical_bare_mass:
                fig, ax = plt.subplots()
                ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

                plot_title = plotting.DataPlotter._construct_plot_title(
                    None,
                    leading_substring="",
                    metadata_dictionary=metadata_dictionary,
                    title_width=100,
                    fields_unique_value_dictionary=parameters_value_dictionary,
                    additional_excluded_fields=[
                        "Average_calculation_time_per_spinor_per_configuration",
                        "Average_number_of_MV_multiplications_per_spinor_per_configuration",
                    ],
                )
                ax.set_title(f"{plot_title}", pad=8)

                ax.set(xlabel="a$m_{{bare}}$", ylabel="a$m_{PCAC}$")

                ax.axvline(0, color="black")  # x = 0
                ax.axhline(0, color="black")  # y = 0

                ax.errorbar(
                    x, gv.mean(y), yerr=gv.sdev(y), fmt=".", markersize=8, capsize=10
                )

                if fit_for_critical_bare_mass:

                    # Linear fit
                    if gv.mean(critical_bare_mass_value) > 0:
                        margin = 0.06
                    else:
                        margin = -0.06
                    x_data = np.linspace(
                        gv.mean(critical_bare_mass_value) * (1 - margin),
                        max(gv.mean(x)) * (1 + np.abs(margin)),
                        100,
                    )
                    y_data = fit_functions.linear_function(x_data, linear_fit.p)
                    label_string = (
                        f"Linear fit:\n"
                        f"- $\\chi^2$/dof={linear_fit.chi2:.2f}/{linear_fit.dof}="
                        f"{linear_fit.chi2/linear_fit.dof:.4f}\n"
                        f"- a$m^{{critical}}_{{bare}}$={critical_bare_mass_value:.5f}"
                    )
                    plt.plot(
                        x_data,
                        gv.mean(y_data),
                        "r--",
                        label=label_string,
                    )
                    ax.fill_between(
                        x_data,
                        gv.mean(y_data) - gv.sdev(critical_bare_mass_value),
                        gv.mean(y_data) + gv.sdev(critical_bare_mass_value),
                        color="r",
                        alpha=0.2,
                    )

                    ax.legend(loc="upper left")

                if annotate_data_points:
                    for index, sample_size in enumerate(
                        number_of_gauge_configurations_array
                    ):
                        ax.annotate(
                            f"{sample_size}",
                            (x[index], gv.mean(y[index])),
                            xytext=(-40, 10),
                            textcoords="offset pixels",
                            bbox=dict(facecolor="none", edgecolor="black"),
                            arrowprops=dict(arrowstyle="->"),
                            # connectionstyle="arc,angleA=0,armA=50,rad=10")
                        )

                current_plots_base_name = critical_bare_mass_plots_base_name
                plot_path = plotting.DataPlotter._generate_plot_path(
                    None,
                    critical_bare_mass_plots_subdirectory,
                    current_plots_base_name,
                    metadata_dictionary,
                    single_valued_fields_dictionary=single_valued_fields_dictionary,
                )

                fig.savefig(plot_path)
                plt.close()

            # EXPORT CALCULATED DATA
            if fit_for_critical_bare_mass:
                parameters_value_dictionary["Critical_bare_mass"] = (
                    critical_bare_mass_value.mean,
                    critical_bare_mass_value.sdev,
                )

            critical_bare_mass_values_list.append(parameters_value_dictionary)

    # EXPORT CALCULATED DATA

    if critical_bare_mass_values_calculated:
        # Create a DataFrame from the extracted data
        critical_bare_mass_dataframe = pd.DataFrame(critical_bare_mass_values_list)

        # Construct output .csv file path
        csv_file_full_path = os.path.join(
            output_files_directory, output_critical_bare_mass_csv_filename
        )

        # Export the DataFrame to a CSV file
        critical_bare_mass_dataframe.to_csv(csv_file_full_path, index=False)

        # Terminate logging
        logging.info(f"Script '{script_name}' execution terminated successfully.")
    
        print(
            "   -- Critical bare mass values calculation from PCAC mass "
            "estimates completed."
        )

    else:
        print("   -- WARNING", "No output .csv file was generated.")


if __name__ == "__main__":
    main()
