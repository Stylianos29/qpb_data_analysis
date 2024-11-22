import unittest
import sys

import pandas as pd

sys.path.append("../")
from library import data_processing


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
    unittest.main()
