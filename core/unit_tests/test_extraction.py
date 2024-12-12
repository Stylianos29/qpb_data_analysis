import unittest
import sys

import pytest

from library import extraction


def test_extract_parameters_values_from_filename_valid():
    # Test extraction from a valid filename
    filename = "KL_Brillouin_mu1p0_rho1p0_cSW0_EpsCG1e-16_config0000200_n3.txt"
    extracted = extraction.extract_parameters_values_from_filename(filename)
    
    # Check that all expected parameters are extracted
    assert extracted["Overlap_operator_method"] == "KL"
    assert extracted["Kernel_operator_type"] == "Brillouin"
    assert extracted["KL_scaling_factor"] == 1.0
    assert extracted["Rho_value"] == 1.0
    assert extracted["Clover_coefficient"] == 0.0
    assert extracted["CG_epsilon"] == 1e-16
    assert extracted["Configuration_label"] == "0000200"
    assert extracted["KL_diagonal_order"] == 3

def test_extract_parameters_values_from_filename_type_conversion():
    # Test type conversion
    filename = "KL_Brillouin_mu1p0_rho1p0_cSW0_EpsCG1e-16_config0000200_n10.txt"
    extracted = extraction.extract_parameters_values_from_filename(filename)
    
    # Check that extracted values are of correct types
    assert isinstance(extracted["KL_scaling_factor"], float)
    assert isinstance(extracted["Rho_value"], float)
    assert isinstance(extracted["KL_diagonal_order"], int)

@pytest.mark.parametrize("filename, expected_value", [
    ("KL_Brillouin_mu1p0_rho1p0_cSW0_EpsCG1e-16_config0000200_n10.txt", 10),
    ("KL_Brillouin_mu1p0_rho2p5_cSW0_EpsCG1e-16_config0000300_n5.txt", 5),
])
def test_extract_parameters_values_from_filename_parametrized(filename, expected_value):
    # Test the number of KL iterations
    extracted = extraction.extract_parameters_values_from_filename(filename)
    assert extracted["KL_diagonal_order"] == expected_value

def test_extract_parameters_values_from_filename_additional_text():
    # Test additional text extraction
    filename = "KL_invert_Brillouin_mu1p0_rhoInvalid_cSW0_EpsCG1e-16_config0000200_n10_TrueResidual.txt"
    extracted = extraction.extract_parameters_values_from_filename(filename)

    # Check that additional text part is as expected
    assert extracted["Additional_text"] == ['invert', 'rhoInvalid', 'TrueResidual']


if __name__ == '__main__':
    unittest.main()
