import unittest
import h5py

from library import plotting
from library import data_processing


def test_construct_plot_subtitle():
    field_unique_values_dictionary = {
        "APE_alpha": 0.72,
        "APE_iterations": 0,
        "CG_epsilon": 1e-12,
        "Clover_coefficient": 0.0,
        "Initial_APE_iterations": 1,
        "KL_diagonal_order": 1,
        "Kernel_operator_type": "Wilson",
        "Lattice_geometry": "(48,24,24,24)",
        "MSCG_epsilon": 1e-14,
        "Overlap_operator_method": "KL",
        "QCD_beta_value": 6.2,
        "Rho_value": 1.0,
        "Solver_epsilon": 1e-12,
        "Bare_mass": 0.13,
        "KL_scaling_factor": 7.0,
    }

    print(plotting.construct_plot_subtitle(field_unique_values_dictionary))


if __name__ == "__main__":
    unittest.main()

