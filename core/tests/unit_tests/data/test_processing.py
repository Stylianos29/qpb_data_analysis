# TODO: Fix the failed test functions

import os
import pytest

import pandas as pd

from library.data.processing import load_csv
from library.constants import ROOT


# CONSTANTS

TEST_CSV_FILE_PATH = os.path.join(
    ROOT,
    "core/tests/mock_data/valid/"
    "KL_several_m_varying_EpsCG_and_EpsMSCG_processed_parameter_values.csv",
)
TEST_DATAFRAME = load_csv(TEST_CSV_FILE_PATH)


# TESTS


def test_load_csv():
    # Test basic loading functionality
    df = load_csv(TEST_CSV_FILE_PATH)

    # Check that a DataFrame is returned
    assert isinstance(df, pd.DataFrame), "load_csv should return a pandas DataFrame"

    # Check that the DataFrame is not empty
    assert not df.empty, "DataFrame should not be empty"

    # Check expected columns exist (adjust based on your actual CSV structure)
    expected_columns = [
        "Kernel_operator_type",
        "Bare_mass",
        "CG_epsilon",
        "MSCG_epsilon",
    ]
    for col in expected_columns:
        assert col in df.columns, f"Expected column '{col}' not found in DataFrame"

    # Test that data types are correctly applied
    # For categorical data
    assert pd.api.types.is_categorical_dtype(
        df["Kernel_operator_type"]
    ), "Kernel_operator_type should be categorical"

    # Test that custom converters were applied
    # Example: For a QCD_beta_value column that should be formatted to 2 decimal places
    if "QCD_beta_value" in df.columns:
        # Check if values are strings formatted with 2 decimal places
        for value in df["QCD_beta_value"]:
            assert isinstance(
                value, str
            ), "QCD_beta_value should be converted to string"
            if "." in value:  # If it has a decimal point
                decimal_places = len(value.split(".")[1])
                assert (
                    decimal_places == 2
                ), "QCD_beta_value should have 2 decimal places"

    # Test with custom dtype_mapping
    custom_dtype = {"Bare_mass": float}
    df_custom_dtype = load_csv(TEST_CSV_FILE_PATH, dtype_mapping=custom_dtype)
    assert pd.api.types.is_float_dtype(
        df_custom_dtype["Bare_mass"]
    ), "Custom dtype_mapping should be applied"

    # Test with custom converters_mapping
    custom_converter = {"Bare_mass": lambda x: float(x) * 2}
    df_custom_converter = load_csv(
        TEST_CSV_FILE_PATH, converters_mapping=custom_converter
    )
    # Check if values are doubled compared to original DataFrame
    if "Bare_mass" in df.columns and "Bare_mass" in df_custom_converter.columns:
        original_values = df["Bare_mass"].astype(float)
        converted_values = df_custom_converter["Bare_mass"]
        # Check a few values to see if they've been doubled
        for i in range(min(5, len(original_values))):
            assert (
                abs(converted_values.iloc[i] - original_values.iloc[i] * 2) < 1e-10
            ), "Custom converter should double Bare_mass values"


def test_load_csv_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_csv("non_existent_file.csv")


def test_load_csv_malformed():
    # Path to a malformed CSV file in your test directory
    malformed_csv_path = os.path.join(
        ROOT, "core/tests/mock_data/invalid/malformed_csv.csv"
    )

    # The exact exception might vary depending on how your function handles errors
    with pytest.raises(Exception):
        load_csv(malformed_csv_path)


def test_load_csv_empty():
    # Path to an empty CSV file
    empty_csv_path = os.path.join(ROOT, "core/tests/mock_data/valid/empty.csv")

    # Load the empty CSV
    df = load_csv(empty_csv_path)

    # Check that a DataFrame is returned but it's empty
    assert isinstance(df, pd.DataFrame)
    assert df.empty


if __name__ == "__main__":
    pass
