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
import ast

import numpy as np
import click
import pandas as pd
import logging

from library import (
    filesystem_utilities,
    data_processing,
    validate_input_directory,
    validate_input_script_log_filename,
)


@click.command()
@click.option(
    "-in_param_csv",
    "--input_single_valued_csv_file_path",
    "input_single_valued_csv_file_path",
    required=True,
    callback=filesystem_utilities.validate_input_csv_file,
    help=(
        "Path to .csv file containing extracted single-valued parameter "
        "values from qpb log files."
    ),
)
@click.option(
    "-in_param_hdf5",
    "--input_multivalued_hdf5_file_path",
    "input_multivalued_hdf5_file_path",
    required=True,
    callback=filesystem_utilities.validate_input_HDF5_file,
    help=(
        "Path to .csv file containing extracted multivalued parameter "
        "values from qpb log files."
    ),
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
    "-out_csv_name",
    "--output_csv_filename",
    "output_csv_filename",
    default="processed_qpb_log_files_extracted_values.csv",
    callback=filesystem_utilities.validate_output_csv_filename,
    help=(
        "Specific name for the output .csv file containing processed values of "
        "single-valued parameters from qpb log files."
    ),
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
    input_single_valued_csv_file_path,
    input_multivalued_hdf5_file_path,
    output_files_directory,
    output_csv_filename,
    enable_logging,
    log_file_directory,
    log_filename,
):
    # HANDLE EMPTY INPUT ARGUMENTS

    # If no output directory is provided, use the directory of the input file
    if output_files_directory is None:
        output_files_directory = output_files_directory = os.path.dirname(
            input_single_valued_csv_file_path
        )

    # INITIATE LOGGING

    # Setup logging
    logger = filesystem_utilities.LoggingWrapper(
        log_file_directory, log_filename, enable_logging
    )

    # Log script start
    logger.initiate_script_logging()

    ## PROCESS EXTRACTED SINGLE-VALUED PARAMETER VALUES ##

    # Load single-valued parameter values .csv file to a dataframe
    single_valued_parameters_dataframe = pd.read_csv(input_single_valued_csv_file_path)

    # STRING-VALUED PARAMETERS

    # Replace term "Standard" with "Wilson".
    if "Kernel_operator_type" in single_valued_parameters_dataframe.columns:
        single_valued_parameters_dataframe["Kernel_operator_type"] = (
            single_valued_parameters_dataframe["Kernel_operator_type"].replace(
                "Standard", "Wilson"
            )
        )
    logging.info(
        f"Replaced 'Kernel_operator_type' string values 'Standard' with 'Wilson'."
    )

    # Ensure the "Configuration_label" field has 7-digit strings
    if "Configuration_label" in single_valued_parameters_dataframe.columns:
        single_valued_parameters_dataframe["Configuration_label"] = (
            single_valued_parameters_dataframe["Configuration_label"]
            .astype(str)
            .str.zfill(7)
        )
    logging.info(
        f"Replaced 'Kernel_operator_type' string values 'Standard' with 'Wilson'."
    )

    # Ensure "QCD_beta_value" field values are formatted as strings with two
    # decimal places
    if "QCD_beta_value" in single_valued_parameters_dataframe.columns:
        single_valued_parameters_dataframe["QCD_beta_value"] = (
            single_valued_parameters_dataframe["QCD_beta_value"].apply(
                lambda x: f"{float(x):.2f}"
            )
        )

    # Extract 'Temporal_lattice_size' and 'Spatial_lattice_size'
    if "Lattice_geometry" in single_valued_parameters_dataframe.columns:
        single_valued_parameters_dataframe["Temporal_lattice_size"] = (
            single_valued_parameters_dataframe["Lattice_geometry"].apply(
                lambda x: int(ast.literal_eval(x)[0])
            )
        )
        single_valued_parameters_dataframe["Spatial_lattice_size"] = (
            single_valued_parameters_dataframe["Lattice_geometry"].apply(
                lambda x: int(ast.literal_eval(x)[1])
            )
        )

        # Remove 'Lattice_geometry'
        single_valued_parameters_dataframe.drop(
            columns=["Lattice_geometry"], inplace=True
        )

    # Modify the "MPI_geometry" field
    if "MPI_geometry" in single_valued_parameters_dataframe.columns:
        single_valued_parameters_dataframe[
            "MPI_geometry"
        ] = single_valued_parameters_dataframe["MPI_geometry"].apply(
            lambda x: str(
                ast.literal_eval(x)[1:]
            )  # Remove the first element and convert back to string
        )

    # INTEGER-TYPE PARAMETERS

    if (
        "MPI_geometry" in single_valued_parameters_dataframe.columns
        and "Threads_per_process" in single_valued_parameters_dataframe.columns
    ):
        single_valued_parameters_dataframe["Number_of_cores"] = (
            single_valued_parameters_dataframe.apply(
                lambda row: np.prod(ast.literal_eval(row["MPI_geometry"]))
                * row["Threads_per_process"],
                axis=1,
            )
        )

    # TODO: Unacceptable!
    if not "Number_of_vectors" in single_valued_parameters_dataframe.columns:
        single_valued_parameters_dataframe["Number_of_vectors"] = 1

    # APE_iterations with Initial_APE_iterations
    if "Initial_APE_iterations" in single_valued_parameters_dataframe.columns:
        if "APE_iterations" in single_valued_parameters_dataframe.columns:
            single_valued_parameters_dataframe["APE_iterations"] = pd.to_numeric(
                single_valued_parameters_dataframe["APE_iterations"], errors="coerce"
            ) + pd.to_numeric(
                single_valued_parameters_dataframe["Initial_APE_iterations"],
                errors="coerce",
            )
            # Remove 'Initial_APE_iterations'
            single_valued_parameters_dataframe.drop(
                columns=["Initial_APE_iterations"], inplace=True
            )
        else:
            single_valued_parameters_dataframe.rename(
                columns={"Initial_APE_iterations": "APE_iterations"}, inplace=True
            )

    # Ensure "Clover_coefficient" field values are integers (0 or 1)
    if "Clover_coefficient" in single_valued_parameters_dataframe.columns:
        single_valued_parameters_dataframe["Clover_coefficient"] = (
            single_valued_parameters_dataframe["Clover_coefficient"].astype(int)
        )

    # FLOAT-TYPE PARAMETERS

    # Take the square root of the "Solver_epsilon" value
    if "Solver_epsilon" in single_valued_parameters_dataframe.columns:
        single_valued_parameters_dataframe["Solver_epsilon"] = np.sqrt(
            single_valued_parameters_dataframe["Solver_epsilon"].apply(float)
        )

    # Take the square root of the "CG_epsilon" value
    if "CG_epsilon" in single_valued_parameters_dataframe.columns:
        single_valued_parameters_dataframe["CG_epsilon"] = np.sqrt(
            single_valued_parameters_dataframe["CG_epsilon"].apply(float)
        )

    # Take the square root of the "MSCG_epsilon" value
    if "MSCG_epsilon" in single_valued_parameters_dataframe.columns:
        single_valued_parameters_dataframe["MSCG_epsilon"] = np.sqrt(
            single_valued_parameters_dataframe["MSCG_epsilon"].apply(float)
        )

    if ("MSCG_epsilon" in single_valued_parameters_dataframe.columns) and not (
        ("Outer solver epsilon" in single_valued_parameters_dataframe.columns)
    ):
        if "Solver_epsilon" in single_valued_parameters_dataframe.columns:
            single_valued_parameters_dataframe = (
                single_valued_parameters_dataframe.drop(columns="Solver_epsilon")
            )

    #
    if (
        "Minimum_eigenvalue_squared" in single_valued_parameters_dataframe.columns
    ) and ("Maximum_eigenvalue_squared" in single_valued_parameters_dataframe.columns):
        single_valued_parameters_dataframe[
            "Condition_number"
        ] = single_valued_parameters_dataframe["Maximum_eigenvalue_squared"].apply(
            float
        ) / single_valued_parameters_dataframe[
            "Minimum_eigenvalue_squared"
        ].apply(
            float
        )

    ## PROCESS EXTRACTED MULTIVALUED PARAMETER VALUES ##

    # TODO: extract_HDF5_datasets_to_dictionary() custom function opens the
    # input HDF5 file every time it is called. Better open the file only once.

    # Calculate average calculation result
    calculation_result_per_vector_dictionary = (
        data_processing.extract_HDF5_datasets_to_dictionary(
            input_multivalued_hdf5_file_path, "Calculation_result_per_vector"
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
                        float(np.average(dataset)),
                        float(np.std(dataset, ddof=1) / np.sqrt(len(dataset))),
                    )
                )
                for filename, dataset in calculation_result_per_vector_dictionary.items()
            }
            # Add a new column with the dictionary values
            single_valued_parameters_dataframe[
                "Average_calculation_result"
            ] = single_valued_parameters_dataframe["Filename"].map(
                # lambda x: (float(average_calculation_result_dictionary[x][0]), float(average_calculation_result_dictionary[x][1]))
                average_calculation_result_dictionary.get
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
                    filename: float(dataset[0])
                    for filename, dataset in calculation_result_per_vector_dictionary.items()
                }
                # Add a new column with the dictionary values
                single_valued_parameters_dataframe[
                    "Calculation_result_with_no_error"
                ] = single_valued_parameters_dataframe["Filename"].map(
                    Calculation_result_with_no_error_dictionary
                )

    if "Main_program_type" in single_valued_parameters_dataframe.columns:
        main_program_type = str(
            single_valued_parameters_dataframe["Main_program_type"].unique()[0]
        )
        if "Average_calculation_result" in single_valued_parameters_dataframe.columns:
            main_program_type = main_program_type.replace("_values", "")
            new_column_name = "Average_" + main_program_type + "_values"
            single_valued_parameters_dataframe.rename(
                columns={"Average_calculation_result": new_column_name}, inplace=True
            )
        elif (
            "Calculation_result_with_no_error"
            in single_valued_parameters_dataframe.columns
        ):
            main_program_type = main_program_type.replace("_values", "")
            new_column_name = main_program_type.capitalize() + "_value_with_no_error"
            single_valued_parameters_dataframe.rename(
                columns={"Calculation_result_with_no_error": new_column_name},
                inplace=True,
            )

    # Calculate average number of MSCG iterations
    total_number_of_MSCG_iterations_dictionary = (
        data_processing.extract_HDF5_datasets_to_dictionary(
            input_multivalued_hdf5_file_path, "Total_number_of_MSCG_iterations"
        )
    )
    if total_number_of_MSCG_iterations_dictionary:
        # Calculate the average value and its error
        average_number_of_MSCG_iterations_dictionary = {
            filename: float(
                # TODO: Change it later!
                np.average(dataset)
                # / single_valued_parameters_dataframe["Number_of_vectors"].unique()[0]
            )
            for filename, dataset in total_number_of_MSCG_iterations_dictionary.items()
        }
        # Add a new column with the dictionary values
        # Forward operator applications case
        if not "Number_of_spinors" in single_valued_parameters_dataframe.columns:
            single_valued_parameters_dataframe[
                "Average_number_of_MSCG_iterations_per_vector"
            ] = single_valued_parameters_dataframe["Filename"].map(
                average_number_of_MSCG_iterations_dictionary
            )
        # Inversions case
        else:
            single_valued_parameters_dataframe[
                "Average_number_of_MSCG_iterations_per_spinor"
            ] = (
                single_valued_parameters_dataframe["Filename"].map(
                    average_number_of_MSCG_iterations_dictionary
                )
                / single_valued_parameters_dataframe["Number_of_spinors"].unique()[0]
            )

    # Calculate average number of MSCG iterations
    number_of_kernel_applications_per_MSCG_dictionary = (
        data_processing.extract_HDF5_datasets_to_dictionary(
            input_multivalued_hdf5_file_path, "Number_of_kernel_applications_per_MSCG"
        )
    )
    if number_of_kernel_applications_per_MSCG_dictionary:
        #
        total_number_of_kernel_applications_dictionary = {
            filename: int(np.sum(dataset)) + len(dataset)
            for filename, dataset in number_of_kernel_applications_per_MSCG_dictionary.items()
        }
        # Add a new column with the dictionary values
        # Forward operator applications case
        if not "Number_of_spinors" in single_valued_parameters_dataframe.columns:
            single_valued_parameters_dataframe[
                "Average_number_of_MV_multiplications_per_vector"
            ] = (
                single_valued_parameters_dataframe["Filename"].map(
                    total_number_of_kernel_applications_dictionary
                )
                / single_valued_parameters_dataframe["Number_of_vectors"].unique()[0]
            )
        # Inversions case
        else:
            single_valued_parameters_dataframe[
                "Average_number_of_MV_multiplications_per_spinor"
            ] = (
                single_valued_parameters_dataframe["Filename"].map(
                    total_number_of_kernel_applications_dictionary
                )
                / single_valued_parameters_dataframe["Number_of_spinors"].unique()[0]
            )

    # Extract MS shifts
    MS_expansion_shifts_dictionary = (
        data_processing.extract_HDF5_datasets_to_dictionary(
            input_multivalued_hdf5_file_path, "MS_expansion_shifts"
        )
    )
    if MS_expansion_shifts_dictionary:
        # Convert the dictionary values to lists of floats
        MS_expansion_shifts_dictionary = {
            filename: [float(value) for value in np.unique(dataset)]
            for filename, dataset in MS_expansion_shifts_dictionary.items()
        }
        # Add a new column with the dictionary values
        single_valued_parameters_dataframe["MS_expansion_shifts"] = (
            single_valued_parameters_dataframe["Filename"].map(
                MS_expansion_shifts_dictionary
            )
        )

    # # Calculate average CG calculation time per spinor
    # CG_total_calculation_time_per_spinor_dictionary = (
    #     data_processing.extract_HDF5_datasets_to_dictionary(
    #         input_multivalued_hdf5_file_path, "CG_total_calculation_time_per_spinor"
    #     )
    # )
    # if CG_total_calculation_time_per_spinor_dictionary:
    #     # Calculate the average value and its error
    #     average_CG_calculation_time_per_spinor_dictionary = {
    #         filename: np.average(dataset)
    #         for filename, dataset in CG_total_calculation_time_per_spinor_dictionary.items()
    #     }
    #     # Add a new column with the dictionary values
    #     single_valued_parameters_dataframe["Average_CG_calculation_time_per_spinor"] = (
    #         single_valued_parameters_dataframe["Filename"].map(
    #             average_CG_calculation_time_per_spinor_dictionary
    #         )
    #     )

    # Calculate average number of CG iterations per spinor
    total_number_of_CG_iterations_per_spinor_dictionary = (
        data_processing.extract_HDF5_datasets_to_dictionary(
            input_multivalued_hdf5_file_path,
            "Total_number_of_CG_iterations_per_spinor",
        )
    )
    if total_number_of_CG_iterations_per_spinor_dictionary:
        # Calculate the average value and its error
        average_number_of_CG_iterations_per_spinor_dictionary = {
            filename: float(np.average(dataset))
            for filename, dataset in total_number_of_CG_iterations_per_spinor_dictionary.items()
        }
        # Add a new column with the dictionary values
        single_valued_parameters_dataframe[
            "Average_number_of_CG_iterations_per_spinor"
        ] = single_valued_parameters_dataframe["Filename"].map(
            average_number_of_CG_iterations_per_spinor_dictionary
        )

    # Forward operator applications
    if not "Number_of_spinors" in single_valued_parameters_dataframe.columns:
        # Chebyshev case
        if (
            single_valued_parameters_dataframe["Overlap_operator_method"].unique()[0]
            == "Chebyshev"
        ):
            single_valued_parameters_dataframe[
                "Average_number_of_MV_multiplications_per_vector"
            ] = (
                2
                * single_valued_parameters_dataframe[
                    "Total_number_of_Lanczos_iterations"
                ]
                + 1
            ) + (
                2 * single_valued_parameters_dataframe["Number_of_Chebyshev_terms"] - 1
            )

        # KL case
        # elif (
        #     single_valued_parameters_dataframe["Overlap_operator_method"].unique()[0]
        #     == "KL"
        # ):
        #     if (
        #         "Average_number_of_MSCG_iterations_per_vector"
        #         in single_valued_parameters_dataframe.columns
        #     ):
        #         single_valued_parameters_dataframe[
        #             "Average_number_of_MV_multiplications_per_vector"
        #         ] = (
        #             2
        #             * single_valued_parameters_dataframe[
        #                 "Average_number_of_MSCG_iterations_per_vector"
        #             ]
        #             + 1
        #         )
    # Inversions
    else:
        # Chebyshev case
        if (
            single_valued_parameters_dataframe["Overlap_operator_method"].unique()[0]
            == "Chebyshev"
        ):
            single_valued_parameters_dataframe[
                "Average_number_of_MV_multiplications_per_spinor"
            ] = (
                2
                * single_valued_parameters_dataframe[
                    "Total_number_of_Lanczos_iterations"
                ]
                + 1
            ) + (
                2
                * single_valued_parameters_dataframe[
                    "Average_number_of_CG_iterations_per_spinor"
                ]
                + 1
            ) * (
                2 * single_valued_parameters_dataframe["Number_of_Chebyshev_terms"] - 1
            )

        # elif (
        #     single_valued_parameters_dataframe["Overlap_operator_method"].unique()[0]
        #     == "KL"
        # ):
        #     single_valued_parameters_dataframe[
        #         "Average_number_of_MV_multiplications_per_spinor"
        #     ] = (
        #         2
        #         * single_valued_parameters_dataframe[
        #             "Average_number_of_MSCG_iterations_per_spinor"
        #         ]
        #         + 1
        #     )

    # Forward operator applications
    if "Total_calculation_time" in single_valued_parameters_dataframe.columns:
        if not "Number_of_spinors" in single_valued_parameters_dataframe.columns:

            if "Total_overhead_time" in single_valued_parameters_dataframe:
                single_valued_parameters_dataframe[
                    "Average_wall_clock_time_per_vector"
                ] = (
                    single_valued_parameters_dataframe["Total_calculation_time"]
                    - single_valued_parameters_dataframe["Total_overhead_time"]
                ) / single_valued_parameters_dataframe[
                    "Number_of_vectors"
                ] + single_valued_parameters_dataframe[
                    "Total_overhead_time"
                ]
            else:
                single_valued_parameters_dataframe[
                    "Average_wall_clock_time_per_vector"
                ] = (
                    single_valued_parameters_dataframe["Total_calculation_time"]
                ) / single_valued_parameters_dataframe[
                    "Number_of_vectors"
                ]

            if "Number_of_cores" in single_valued_parameters_dataframe.columns:
                single_valued_parameters_dataframe["Average_core_hours_per_vector"] = (
                    single_valued_parameters_dataframe["Number_of_cores"]
                    * single_valued_parameters_dataframe[
                        "Average_wall_clock_time_per_vector"
                    ]
                    / 3600
                )

                single_valued_parameters_dataframe[
                    "Adjusted_average_core_hours_per_vector"
                ] = single_valued_parameters_dataframe.apply(
                    lambda row: (
                        row["Average_core_hours_per_vector"] * 0.87
                        if row["Number_of_cores"] == 256
                        else (
                            row["Average_core_hours_per_vector"] * 1.13
                            if row["Number_of_cores"] == 512
                            else (
                                row["Average_core_hours_per_vector"] * 0.98
                                if row["Number_of_cores"] == 768
                                else row["Average_core_hours_per_vector"]
                            )
                        )
                    ),
                    axis=1,
                )

        # Inversions
        else:

            if "Total_overhead_time" in single_valued_parameters_dataframe:
                single_valued_parameters_dataframe[
                    "Average_wall_clock_time_per_spinor"
                ] = (
                    single_valued_parameters_dataframe["Total_calculation_time"]
                    - single_valued_parameters_dataframe["Total_overhead_time"]
                ) / single_valued_parameters_dataframe[
                    "Number_of_spinors"
                ] + single_valued_parameters_dataframe[
                    "Total_overhead_time"
                ]
            else:
                single_valued_parameters_dataframe[
                    "Average_wall_clock_time_per_spinor"
                ] = (
                    single_valued_parameters_dataframe["Total_calculation_time"]
                ) / single_valued_parameters_dataframe[
                    "Number_of_spinors"
                ]

            if "Number_of_cores" in single_valued_parameters_dataframe.columns:
                single_valued_parameters_dataframe["Average_core_hours_per_spinor"] = (
                    single_valued_parameters_dataframe["Number_of_cores"]
                    * single_valued_parameters_dataframe[
                        "Average_wall_clock_time_per_spinor"
                    ]
                    / 3600
                )

                single_valued_parameters_dataframe[
                    "Adjusted_average_core_hours_per_spinor"
                ] = single_valued_parameters_dataframe.apply(
                    lambda row: (
                        row["Average_core_hours_per_spinor"] * 0.87
                        if row["Number_of_cores"] == 256
                        else (
                            row["Average_core_hours_per_spinor"] * 1.13
                            if row["Number_of_cores"] == 512
                            else (
                                row["Average_core_hours_per_spinor"] * 0.98
                                if row["Number_of_cores"] == 768
                                else row["Average_core_hours_per_spinor"]
                            )
                        )
                    ),
                    axis=1,
                )

    # EXPORT PROCESSED VALUES

    # Construct the output .csv file path
    output_qpb_log_files_csv_file_path = os.path.join(
        output_files_directory,
        output_csv_filename,
    )

    # Export modified DataFrame back to the same .csv file
    single_valued_parameters_dataframe.to_csv(
        output_qpb_log_files_csv_file_path, index=False
    )

    # Terminate logging
    logger.terminate_script_logging()

    click.echo("   -- Processing extracted values from qpb log files completed.")


if __name__ == "__main__":
    main()
