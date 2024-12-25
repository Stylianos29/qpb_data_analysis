import os
import sys
import ast

import numpy as np
import click
import pandas as pd
import logging
import h5py

from library import filesystem_utilities
from library import data_processing


@click.command()
@click.option(
    "--input_qpb_log_files_csv_file_path",
    "input_qpb_log_files_csv_file_path",
    "-in_csv_path",
    default=None,
    help="Path to .csv file containing extracted info from qpb log files sets.",
)
@click.option(
    "--input_qpb_log_files_hdf5_file_path",
    "input_qpb_log_files_hdf5_file_path",
    "-in_hdf5_path",
    default=None,
    help="Path to HDF5 file containing extracted info from qpb log files sets.",
)
@click.option(
    "--output_qpb_log_files_csv_filename",
    "output_qpb_log_files_csv_filename",
    "-out_csv",
    default=None,
    help="Specific name for the output .csv file.",
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
    input_qpb_log_files_csv_file_path,
    input_qpb_log_files_hdf5_file_path,
    output_qpb_log_files_csv_filename,
    log_file_directory,
    log_filename,
):

    # VALIDATE INPUT ARGUMENTS

    if not filesystem_utilities.is_valid_file(input_qpb_log_files_csv_file_path):
        error_message = "Passed qpb log files .csv file path is invalid!."
        print("ERROR:", error_message)
        sys.exit(1)

    if not filesystem_utilities.is_valid_file(input_qpb_log_files_hdf5_file_path):
        error_message = "Passed correlator values HDF5 file path is invalid!."
        print("ERROR:", error_message)
        sys.exit(1)

    if output_qpb_log_files_csv_filename is None:
        output_qpb_log_files_csv_filename = os.path.basename(
            input_qpb_log_files_csv_file_path
        )

    # Specify current script's log file directory
    if log_file_directory is None:
        log_file_directory = os.path.dirname(input_qpb_log_files_csv_file_path)
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
        log_filename = script_name.replace(".py", ".log")

    # Check for proper extensions in provided output filenames
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

    # PROCESS EXTRACTED QPB PARAMETERS AND OUTPUT VALUES

    qpb_log_files_dataframe = pd.read_csv(input_qpb_log_files_csv_file_path)

    # Replace term "Standard" with "Wilson".
    if "Kernel_operator_type" in qpb_log_files_dataframe.columns:
        qpb_log_files_dataframe["Kernel_operator_type"] = qpb_log_files_dataframe[
            "Kernel_operator_type"
        ].replace("Standard", "Wilson")

    # Ensure the "Configuration_label" field has 7-digit strings
    if "Configuration_label" in qpb_log_files_dataframe.columns:
        qpb_log_files_dataframe["Configuration_label"] = (
            qpb_log_files_dataframe["Configuration_label"].astype(str).str.zfill(7)
        )

    # Ensure "QCD_beta_value" field values are formatted as strings with two
    # decimal places
    if "QCD_beta_value" in qpb_log_files_dataframe.columns:
        qpb_log_files_dataframe["QCD_beta_value"] = qpb_log_files_dataframe[
            "QCD_beta_value"
        ].apply(lambda x: f"{float(x):.2f}")

    # Extract 'Temporal_lattice_size' and 'Spatial_lattice_size'
    if "Lattice_geometry" in qpb_log_files_dataframe.columns:
        qpb_log_files_dataframe["Temporal_lattice_size"] = qpb_log_files_dataframe[
            "Lattice_geometry"
        ].apply(lambda x: int(ast.literal_eval(x)[0]))
        qpb_log_files_dataframe["Spatial_lattice_size"] = qpb_log_files_dataframe[
            "Lattice_geometry"
        ].apply(lambda x: int(ast.literal_eval(x)[1]))

        # Remove 'Lattice_geometry'
        qpb_log_files_dataframe.drop(columns=["Lattice_geometry"], inplace=True)

    # Modify the "MPI_geometry" field
    if "MPI_geometry" in qpb_log_files_dataframe.columns:
        qpb_log_files_dataframe["MPI_geometry"] = qpb_log_files_dataframe[
            "MPI_geometry"
        ].apply(
            lambda x: str(
                ast.literal_eval(x)[1:]
            )  # Remove the first element and convert back to string
        )

    if not "Number_of_vectors" in qpb_log_files_dataframe.columns:
        qpb_log_files_dataframe["Number_of_vectors"] = 1

    # APE_iterations with Initial_APE_iterations
    if "Initial_APE_iterations" in qpb_log_files_dataframe.columns:
        if "APE_iterations" in qpb_log_files_dataframe.columns:
            qpb_log_files_dataframe["APE_iterations"] = pd.to_numeric(
                qpb_log_files_dataframe["APE_iterations"], errors="coerce"
            ) + pd.to_numeric(
                qpb_log_files_dataframe["Initial_APE_iterations"], errors="coerce"
            )
            # Remove 'Initial_APE_iterations'
            qpb_log_files_dataframe.drop(
                columns=["Initial_APE_iterations"], inplace=True
            )
        else:
            qpb_log_files_dataframe.rename(
                columns={"Initial_APE_iterations": "APE_iterations"}, inplace=True
            )

    # Ensure "Clover_coefficient" field values are integers (0 or 1)
    if "Clover_coefficient" in qpb_log_files_dataframe.columns:
        qpb_log_files_dataframe["Clover_coefficient"] = qpb_log_files_dataframe[
            "Clover_coefficient"
        ].astype(int)

    # Take the square root of the "Solver_epsilon" value
    if "Solver_epsilon" in qpb_log_files_dataframe.columns:
        qpb_log_files_dataframe["Solver_epsilon"] = np.sqrt(
            qpb_log_files_dataframe["Solver_epsilon"].apply(float)
        )

    # Take the square root of the "CG_epsilon" value
    if "CG_epsilon" in qpb_log_files_dataframe.columns:
        qpb_log_files_dataframe["CG_epsilon"] = np.sqrt(
            qpb_log_files_dataframe["CG_epsilon"].apply(float)
        )

    # Take the square root of the "MSCG_epsilon" value
    if "MSCG_epsilon" in qpb_log_files_dataframe.columns:
        qpb_log_files_dataframe["MSCG_epsilon"] = np.sqrt(
            qpb_log_files_dataframe["MSCG_epsilon"].apply(float)
        )

    if ("MSCG_epsilon" in qpb_log_files_dataframe.columns) and not (
        ("Outer solver epsilon" in qpb_log_files_dataframe.columns)
    ):
        if "Solver_epsilon" in qpb_log_files_dataframe.columns:
            qpb_log_files_dataframe = qpb_log_files_dataframe.drop(
                columns="Solver_epsilon"
            )

    #
    if ("Minimum_eigenvalue_squared" in qpb_log_files_dataframe.columns) and (
        "Maximum_eigenvalue_squared" in qpb_log_files_dataframe.columns
    ):
        qpb_log_files_dataframe["Condition_number"] = qpb_log_files_dataframe[
            "Maximum_eigenvalue_squared"
        ].apply(float) / qpb_log_files_dataframe["Minimum_eigenvalue_squared"].apply(
            float
        )

    # Take the square root of the "Minimum_eigenvalue_squared" value
    if "Minimum_eigenvalue_squared" in qpb_log_files_dataframe.columns:
        qpb_log_files_dataframe["Minimum_eigenvalue"] = np.sqrt(
            qpb_log_files_dataframe["Minimum_eigenvalue_squared"].apply(float)
        )
        # # Remove "Minimum_eigenvalue_squared"
        # qpb_log_files_dataframe.drop(
        #     columns=["Minimum_eigenvalue_squared"], inplace=True
        # )

    # Take the square root of the "Maximum_eigenvalue_squared" value
    if "Maximum_eigenvalue_squared" in qpb_log_files_dataframe.columns:
        qpb_log_files_dataframe["Maximum_eigenvalue"] = np.sqrt(
            qpb_log_files_dataframe["Maximum_eigenvalue_squared"].apply(float)
        )
        # # Remove "Maximum_eigenvalue_squared"
        # qpb_log_files_dataframe.drop(
        #     columns=["Maximum_eigenvalue_squared"], inplace=True
        # )

    # INPUT HDF5 FILE

    # Calculate average calculation result
    calculation_result_per_vector_dictionary = (
        data_processing.extract_HDF5_datasets_to_dictionary(
            input_qpb_log_files_hdf5_file_path, "Calculation_result_per_vector"
        )
    )
    if calculation_result_per_vector_dictionary:
        # TODO: Check length more comprehensively using a set:
        # lengths = {len(arr) for arr in my_dict.values()}

        # Ensure all dictionary values (NumPy arrays) have length greater than one
        all_lengths_greater_than_one = all(
            len(array) > 1
            for array in calculation_result_per_vector_dictionary.values()
        )
        if all_lengths_greater_than_one:
            # Calculate the average value and its error
            average_calculation_result_dictionary = {
                filename: (
                    (
                        np.average(dataset),
                        np.std(dataset, ddof=1) / np.sqrt(len(dataset)),
                    )
                )
                for filename, dataset in calculation_result_per_vector_dictionary.items()
            }
            # Add a new column with the dictionary values
            qpb_log_files_dataframe["Average_calculation_result"] = (
                qpb_log_files_dataframe["Filename"].map(
                    average_calculation_result_dictionary
                )
            )
        else:
            # Alternatively check if all NumPy arrays have length one
            all_lengths_equal_to_one = all(
                len(array) == 1
                for array in calculation_result_per_vector_dictionary.values()
            )
            if all_lengths_equal_to_one:
                # Pass the single value to the DataFrame without error
                Calculation_result_with_no_error_dictionary = {
                    filename: dataset[0]
                    for filename, dataset in calculation_result_per_vector_dictionary.items()
                }
                # Add a new column with the dictionary values
                qpb_log_files_dataframe["Calculation_result_with_no_error"] = (
                    qpb_log_files_dataframe["Filename"].map(
                        Calculation_result_with_no_error_dictionary
                    )
                )

    # Calculate average number of MSCG iterations
    total_number_of_MSCG_iterations_dictionary = (
        data_processing.extract_HDF5_datasets_to_dictionary(
            input_qpb_log_files_hdf5_file_path, "Total_number_of_MSCG_iterations"
        )
    )
    if total_number_of_MSCG_iterations_dictionary:
        # Calculate the average value and its error
        average_number_of_MSCG_iterations_dictionary = {
            filename: np.sum(dataset)
            / qpb_log_files_dataframe["Number_of_vectors"].unique()[0]
            for filename, dataset in total_number_of_MSCG_iterations_dictionary.items()
        }
        # Add a new column with the dictionary values
        qpb_log_files_dataframe["Average_number_of_MSCG_iterations"] = (
            qpb_log_files_dataframe["Filename"].map(
                average_number_of_MSCG_iterations_dictionary
            )
        )

    # Extract MS shifts
    MS_expansion_shifts_dictionary = (
        data_processing.extract_HDF5_datasets_to_dictionary(
            input_qpb_log_files_hdf5_file_path, "MS_expansion_shifts"
        )
    )
    if MS_expansion_shifts_dictionary:
        # Calculate the average value and its error
        MS_expansion_shifts_dictionary = {
            filename: list(np.unique(dataset))
            for filename, dataset in MS_expansion_shifts_dictionary.items()
        }
        # Add a new column with the dictionary values
        qpb_log_files_dataframe["MS_expansion_shifts"] = qpb_log_files_dataframe[
            "Filename"
        ].map(MS_expansion_shifts_dictionary)

    # # Calculate average CG calculation time per spinor
    # CG_total_calculation_time_per_spinor_dictionary = (
    #     data_processing.extract_HDF5_datasets_to_dictionary(
    #         input_qpb_log_files_hdf5_file_path, "CG_total_calculation_time_per_spinor"
    #     )
    # )
    # if CG_total_calculation_time_per_spinor_dictionary:
    #     # Calculate the average value and its error
    #     average_CG_calculation_time_per_spinor_dictionary = {
    #         filename: np.average(dataset)
    #         for filename, dataset in CG_total_calculation_time_per_spinor_dictionary.items()
    #     }
    #     # Add a new column with the dictionary values
    #     qpb_log_files_dataframe["Average_CG_calculation_time_per_spinor"] = (
    #         qpb_log_files_dataframe["Filename"].map(
    #             average_CG_calculation_time_per_spinor_dictionary
    #         )
    #     )

    # # Calculate average number of CG iterations per spinor
    # total_number_of_CG_iterations_per_spinor_dictionary = (
    #     data_processing.extract_HDF5_datasets_to_dictionary(
    #         input_qpb_log_files_hdf5_file_path,
    #         "Total_number_of_CG_iterations_per_spinor",
    #     )
    # )
    # if total_number_of_CG_iterations_per_spinor_dictionary:
    #     # Calculate the average value and its error
    #     average_number_of_CG_iterations_per_spinor_dictionary = {
    #         filename: np.average(dataset)
    #         for filename, dataset in total_number_of_CG_iterations_per_spinor_dictionary.items()
    #     }
    #     # Add a new column with the dictionary values
    #     qpb_log_files_dataframe["Average_number_of_CG_iterations_per_spinor"] = (
    #         qpb_log_files_dataframe["Filename"].map(
    #             average_number_of_CG_iterations_per_spinor_dictionary
    #         )
    #     )

    # Construct the output .csv file path
    output_qpb_log_files_csv_file_path = os.path.join(
        os.path.dirname(input_qpb_log_files_csv_file_path),
        output_qpb_log_files_csv_filename,
    )

    # Export modified DataFrame back to the same .csv file
    qpb_log_files_dataframe.to_csv(output_qpb_log_files_csv_file_path, index=False)

    print("   -- Processing QPB log files extracted information completed.")

    # Terminate logging
    logging.info(f"Script '{script_name}' execution terminated successfully.")


if __name__ == "__main__":
    main()
