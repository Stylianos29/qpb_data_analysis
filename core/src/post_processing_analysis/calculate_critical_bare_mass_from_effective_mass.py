import os
import sys

import click
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
import lsqfit
import logging
import pandas as pd
import copy

from library import filesystem_utilities, plotting, data_processing, fit_functions

ANNOTATE_DATA_POINTS = True


@click.command()
@click.option(
    "--input_effective_mass_csv_file_path",
    "input_effective_mass_csv_file_path",
    "-eff_csv",
    # default="/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/invert/KL_several_config_varying_n/Pion_effective_mass_estimates.csv",
    required=True,
    help="Path to .csv file containing effective mass estimates.",
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
    # default="/nvme/h/cy22sg1/qpb_data_analysis/output/plots/invert/KL_several_config_varying_n",
    help="Path to the output directory for storing plots.",
)
@click.option(
    "-plot_critical",
    "--plot_critical_bare_mass",
    "plot_critical_bare_mass",
    is_flag=True,
    default=False,
    # TODO: Work it out better
    help="Enable plotting critical bare mass.",
)
@click.option(
    "-fits",
    "--fit_for_critical_bare_mass",
    "fit_for_critical_bare_mass",
    is_flag=True,
    default=False,
    # TODO: Work it out better
    help="Enable performing fits for the calculation of critical bare mass.",
)
@click.option(
    "--output_critical_bare_mass_csv_filename",
    "output_critical_bare_mass_csv_filename",
    "-out_csv",
    default="critical_bare_mass_from_pion_effective_mass.csv",
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
    input_effective_mass_csv_file_path,
    output_files_directory,
    plots_directory,
    plot_critical_bare_mass,
    fit_for_critical_bare_mass,
    output_critical_bare_mass_csv_filename,
    log_file_directory,
    log_filename,
):
    # VALIDATE INPUT ARGUMENTS

    if not filesystem_utilities.is_valid_file(input_effective_mass_csv_file_path):
        error_message = "Passed correlator values HDF5 file path is invalid."
        print("ERROR:", error_message)
        sys.exit(1)

    # If no output directory is provided, use the directory of the input file
    if output_files_directory is None:
        output_files_directory = os.path.dirname(input_effective_mass_csv_file_path)
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
            plots_directory, "critical_bare_mass_calculation"
        )

        # Create deeper-level subdirectories if requested
        critical_bare_mass_plots_base_name = "critical_bare_mass"
        critical_bare_mass_plots_subdirectory = filesystem_utilities.create_subdirectory(
            plots_main_subdirectory,
            critical_bare_mass_plots_base_name + "_from_pion_effective_mass",
            # clear_contents=True,
        )

    # IMPORT DATASETS AND METADATA

    effective_mass_estimates_dataframe = data_processing.load_csv(
        input_effective_mass_csv_file_path
    )

    analyzer = data_processing.DataFrameAnalyzer(effective_mass_estimates_dataframe)

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

    # ?

    critical_bare_mass_values_list = []
    for value, group in effective_mass_estimates_dataframe.groupby(
        tunable_multivalued_parameters_list
    ):
        # Check for a minimum amount of data point
        if group["Bare_mass"].nunique() < 3:
            # TODO: Log warning
            continue

        # Initialize the parameters values dictionary
        parameters_value_dictionary = copy.deepcopy(single_valued_fields_dictionary)

        # Store for later use
        metadata_dictionary = dict(zip(tunable_multivalued_parameters_list, value))

        # Append metadata dictionary
        parameters_value_dictionary.update(metadata_dictionary)

        number_of_gauge_configurations_array = group[
            "Number_of_gauge_configurations"
        ].to_numpy()

        x = group["Bare_mass"].to_numpy()
        y = gv.gvar(group["Pion_effective_mass_estimate"].to_numpy())
        y = np.square(y)

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

            # Quadratic fit
            FIT_QUADRATIC_FUNCTION = False
            if len(y):
                FIT_QUADRATIC_FUNCTION = True

                quadratic_fit_p0 = [
                    gv.mean(linear_fit.p[0]),
                    gv.mean(critical_bare_mass_value),
                    0.1 * gv.mean(critical_bare_mass_value),
                ]
                quadratic_fit = lsqfit.nonlinear_fit(
                    data=(x, y),
                    p0=quadratic_fit_p0,
                    fcn=fit_functions.quadratic_function,
                    debug=True,
                )

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
            )
            ax.set_title(f"{plot_title}", pad=8)

            ax.set(xlabel="a$m_{{bare}}$", ylabel="$m^2_{eff.}$")

            ax.axvline(0, color="black")  # x = 0
            ax.axhline(0, color="black")  # y = 0

            L = parameters_value_dictionary["Spatial_lattice_size"]
            effective_mass_squared_lower_limit = (4 / L) ** 2
            ax.axhline(
                effective_mass_squared_lower_limit,
                color="blue",
                linestyle="--",
                label="Finite volume $m^2_{eff.}$ threshold",
            )

            ax.errorbar(
                x, gv.mean(y), yerr=gv.sdev(y), fmt=".", markersize=8, capsize=10
            )

            if fit_for_critical_bare_mass:

                # Linear fit
                margin = 0.06
                x_data = np.linspace(
                    gv.mean(critical_bare_mass_value) * (1 - margin),
                    max(gv.mean(x)) * (1 + margin),
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

                # Quadratic fit
                if FIT_QUADRATIC_FUNCTION:
                    # margin =
                    x_data = np.linspace(
                        # min(gv.mean(x)) * (1 - margin),
                        gv.mean(critical_bare_mass_value) * (1 - margin),
                        max(gv.mean(x)) * (1 + margin),
                        100,
                    )
                    y_data = fit_functions.quadratic_function(x_data, quadratic_fit.p)

                    label_string = (
                        f"Quadratic fit:\n"
                        f"- $\\chi^2$/dof={quadratic_fit.chi2:.2f}/{quadratic_fit.dof}="
                        f"{quadratic_fit.chi2/quadratic_fit.dof:.2f}\n"
                        # f"- a$m^{{critical}}_{{bare}}$={quadratic_fit.p[1]:.5f}"
                    )
                    plt.plot(x_data, gv.mean(y_data), "g--", label=label_string)
                    ax.fill_between(
                        x_data,
                        gv.mean(y_data) - gv.sdev(y_data),
                        gv.mean(y_data) + gv.sdev(y_data),
                        color="g",
                        alpha=0.2,
                    )

                if ANNOTATE_DATA_POINTS:
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

                ax.legend(loc="upper left")

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
        "   -- Critical bare mass values calculation from pion "
        "effective mass estimates completed."
    )


if __name__ == "__main__":
    main()
