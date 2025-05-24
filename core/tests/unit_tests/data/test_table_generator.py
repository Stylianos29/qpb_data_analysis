import os

from library.data.processing import load_csv
from library.data.table_generator import TableGenerator
from library.constants import ROOT


# CONSTANTS

TEST_CSV_FILE_PATH = os.path.join(
    ROOT,
    "core/tests/mock_data/valid/"
    "KL_several_m_varying_EpsCG_and_EpsMSCG_processed_parameter_values.csv",
)
TEST_DATAFRAME = load_csv(TEST_CSV_FILE_PATH)
TEST_TABLE_GENERATOR = TableGenerator(TEST_DATAFRAME)


# TESTS


def test_generate_column_uniqueness_report_against_golden_master():
    report = TEST_TABLE_GENERATOR.generate_column_uniqueness_report()

    # Path to a pre-approved golden master file
    golden_master_path = os.path.join(
        ROOT, "core/tests/mock_data/golden_masters/column_uniqueness_report.md"
    )

    # Compare with golden master
    with open(golden_master_path, "r") as f:
        golden_master = f.read()

    # You might want to normalize line endings or whitespace before comparison
    assert report.strip() == golden_master.strip()


if __name__ == "__main__":
    pass
