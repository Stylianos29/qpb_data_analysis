import unittest
import sys

import pytest # type: ignore
import pandas as pd

from library import data_processing


log_files_data_csv_file_path="/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/invert/KL_several_m_varying_EpsCG_and_EpsMSCG/qpb_log_files_single_valued_parameters.csv"
log_files_dataframe = pd.read_csv(log_files_data_csv_file_path)

def test_get_fields_with_multiple_values():
    # Test extraction of fields with multiple unique values, excluding specified
    # fields

    list_of_fields_with_multiple_values = data_processing.get_fields_with_multiple_values(log_files_dataframe, {"Filename", "Plaquette"})

    assert list_of_fields_with_multiple_values == ['Configuration_label', 'Bare_mass', 'Kappa_value', 'Maximum_solver_iterations', 'CG_epsilon', 'MSCG_epsilon']

if __name__ == '__main__':
    unittest.main()
