import unittest

import pandas as pd

from library import data_processing
from library import constants

LOG_FILES_DATA_CSV_FILE_PATH = "/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/invert/KL_several_m_varying_EpsCG_and_EpsMSCG/qpb_log_files_single_valued_parameters.csv"
log_files_dataframe = pd.read_csv(LOG_FILES_DATA_CSV_FILE_PATH)


def test_get_fields_with_multiple_values():
    # Test extraction of fields with multiple unique values, excluding specified
    # fields

    list_of_fields_with_multiple_values = (
        data_processing.get_fields_with_multiple_values(
            log_files_dataframe, {"Filename", "Plaquette"}
        )
    )

    assert list_of_fields_with_multiple_values == [
        "Configuration_label",
        "Bare_mass",
        "Kappa_value",
        "Maximum_solver_iterations",
        "CG_epsilon",
        "MSCG_epsilon",
    ]


def test_get_fields_with_unique_values():
    # Test extraction of fields with a single unique values, excluding specified
    # fields

    fields_with_unique_values_dictionary = (
        data_processing.get_fields_with_unique_values(log_files_dataframe)
    )

    assert fields_with_unique_values_dictionary == {
        "Kernel_operator_type": "Standard",
        "Lattice_geometry": "(48,24,24,24)",
        "QCD_beta_value": 6.2,
        "Initial_APE_iterations": 1,
        "APE_alpha": 0.72,
        "APE_iterations": 0,
        "Rho_value": 1.0,
        "Clover_coefficient": 0.0,
        "KL_diagonal_order": 1,
        "KL_scaling_factor": 1.0,
        "Overlap_operator_method": "KL",
    }


if __name__ == "__main__":
    # unittest.main()

    input_qpb_log_files_csv_file_path = "/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/invert/Chebyshev_several_m_varying_EpsLanczos/qpb_log_files_single_valued_parameters.csv"

    # Assuming your CSV data is loaded here
    qpb_log_files_dataframe = pd.read_csv(input_qpb_log_files_csv_file_path)

    # Create an instance of DataFrameAnalyzer
    analyzer = data_processing.DataFrameAnalyzer(qpb_log_files_dataframe)

    # Set excluded fields
    excluded_fields = {
        "Filename",
        *constants.OUTPUT_VALUES_LIST,
        "Lanczos_epsilon",
        # "Configuration_label",
    }
    analyzer.set_excluded_fields(excluded_fields)

    # Get valid (non-empty) dataframe groups
    valid_dataframe_groups = analyzer.get_valid_dataframe_groups()

    # Now process the valid dataframe groups
    for analysis_index, dataframe_group in enumerate(valid_dataframe_groups, start=1):
        # Perform your complicated analysis on `dataframe_group` here
        print(f"Processing group {analysis_index}:\n", dataframe_group.head())
