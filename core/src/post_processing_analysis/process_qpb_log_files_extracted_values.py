import os
import sys
import ast

import numpy as np
import click
import pandas as pd
import logging

from library import filesystem_utilities


@click.command()
@click.option(
    "--input_qpb_log_files_csv_file_path",
    "input_qpb_log_files_csv_file_path",
    "-in_csv_path",
    default=None,
    help="Path to .csv file containing extracted info from qpb log files sets.",
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
    default="TEST.log",
    help="Specific name for the script's log file.",
)
def main(
    input_qpb_log_files_csv_file_path,
    output_qpb_log_files_csv_filename,
    log_file_directory,
    log_filename,
):

    # PERFORM VALIDITY CHECKS ON INPUT ARGUMENTS

    if not filesystem_utilities.is_valid_file(input_qpb_log_files_csv_file_path):
        error_message = "Passed qpb log files .csv file path is invalid!."
        print("ERROR:", error_message)
        sys.exit(1)

    if output_qpb_log_files_csv_filename is None:
        output_qpb_log_files_csv_filename = os.path.basename(
            input_qpb_log_files_csv_file_path
        )

    # Specify current script's log file directory
    if log_file_directory is None:
        log_file_directory = os.path.dirname(input_qpb_log_files_csv_file_path)
        print(log_file_directory)
    elif not filesystem_utilities.is_valid_directory(log_file_directory):
        error_message = (
            "Passed directory path to store script's log file is "
            "invalid or not a directory."
        )
        print("ERROR:", error_message)
        print("Exiting...")
        sys.exit(1)

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
