import unittest
import sys

import pytest # type: ignore

sys.path.append('../')
from library import extraction


# Sample filenames for testing
valid_filename = "KL_Brillouin_mu1p0_rho1p0_cSW0_EpsCG1e-16_config0000200_n3.txt"
missing_param_filename = "KL_Brillouin_mu1p0_rho1p0_cSW0_config0000200_n3.txt"  # Missing some params
incorrect_param_filename = "KL_Brillouin_mu1p0_rhoInvalid_cSW0_EpsCG1e-16_config0000200_n3.txt"  # Invalid rho value

def test_filename_extraction_valid():
    # Test extraction from a valid filename
    extracted = extraction.filename_extraction(valid_filename)
    
    # Check that all expected parameters are extracted
    assert extracted["Overlap_operator_method"] == "KL"
    assert extracted["Kernel_operator_type"] == "Brillouin"
    assert extracted["KL_scaling_factor"] == 1.0
    assert extracted["Rho_value"] == 1.0
    assert extracted["Clover_coefficient"] == 0.0
    assert extracted["CG_epsilon"] == 1e-16
    assert extracted["Configuration_label"] == "0000200"
    assert extracted["KL_iterations"] == 3

def test_filename_extraction_missing_params():
    # Test a filename with missing parameters
    extracted = extraction.filename_extraction(missing_param_filename)
    
    # Check that missing parameters are not present in the dictionary
    assert "Delta_Min" not in extracted
    assert "Delta_Max" not in extracted

def test_filename_extraction_type_conversion():
    # Test type conversion
    filename = "KL_Brillouin_mu1p0_rho1p0_cSW0_EpsCG1e-16_config0000200_n10.txt"
    extracted = extraction.filename_extraction(filename)
    
    # Check that extracted values are of correct types
    assert isinstance(extracted["KL_scaling_factor"], float)
    assert isinstance(extracted["Rho_value"], float)

@pytest.mark.parametrize("filename, expected_value", [
    ("KL_Brillouin_mu1p0_rho1p0_cSW0_EpsCG1e-16_config0000200_n10.txt", 10),
    ("KL_Brillouin_mu1p0_rho2p5_cSW0_EpsCG1e-16_config0000300_n5.txt", 5),
])
def test_filename_extraction_parametrized(filename, expected_value):
    # Test the number of KL iterations
    extracted = extraction.filename_extraction(filename)
    assert extracted["KL_iterations"] == expected_value


if __name__ == '__main__':
    unittest.main()