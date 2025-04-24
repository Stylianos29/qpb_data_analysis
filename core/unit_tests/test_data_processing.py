import pandas as pd
import numpy as np

from library import load_csv, DataFrameAnalyzer, TableGenerator


# CONSTANTS

TEST_DATAFRAME = load_csv(
    "./mock_data/valid_csv_files/KL_several_m_varying_EpsCG_and_EpsMSCG_processed_parameter_values.csv"
)
TEST_ANALYZER = DataFrameAnalyzer(TEST_DATAFRAME)
TEST_TABLE_GENERATOR = TableGenerator(TEST_DATAFRAME)

# TESTS


def test_original_dataframe_preservation():
    """Test if the original_dataframe attribute matches the input dataframe."""
    pd.testing.assert_frame_equal(TEST_ANALYZER.original_dataframe, TEST_DATAFRAME)


def test_list_of_dataframe_column_names():
    """Test if list_of_dataframe_column_names matches the expected list of
    column names."""
    expected_columns = [
        "Filename",
        "Main_program_type",
        "Kernel_operator_type",
        "MPI_geometry",
        "Threads_per_process",
        "Configuration_label",
        "QCD_beta_value",
        "APE_alpha",
        "APE_iterations",
        "Rho_value",
        "Bare_mass",
        "Clover_coefficient",
        "Plaquette",
        "Number_of_spinors",
        "KL_diagonal_order",
        "KL_scaling_factor",
        "Total_calculation_time",
        "Overlap_operator_method",
        "CG_epsilon",
        "MSCG_epsilon",
        "Temporal_lattice_size",
        "Spatial_lattice_size",
        "Number_of_cores",
        "Number_of_vectors",
        "Average_number_of_MSCG_iterations_per_spinor",
        "MS_expansion_shifts",
        "Average_number_of_CG_iterations_per_spinor",
        "Average_number_of_MV_multiplications_per_spinor",
        "Average_wall_clock_time_per_spinor",
        "Average_core_hours_per_spinor",
        "Adjusted_average_core_hours_per_spinor",
    ]
    assert TEST_ANALYZER.list_of_dataframe_column_names == expected_columns


def test_list_of_tunable_parameter_names():
    """Test if list_of_tunable_parameter_names_from_dataframe matches the
    expected tunable parameters."""
    expected_tunable_parameters = [
        "APE_alpha",
        "APE_iterations",
        "Bare_mass",
        "CG_epsilon",
        "Clover_coefficient",
        "Configuration_label",
        "Kernel_operator_type",
        "KL_diagonal_order",
        "KL_scaling_factor",
        "Main_program_type",
        "MPI_geometry",
        "MSCG_epsilon",
        "Number_of_spinors",
        "Number_of_vectors",
        "Overlap_operator_method",
        "QCD_beta_value",
        "Rho_value",
    ]
    assert set(TEST_ANALYZER.list_of_tunable_parameter_names_from_dataframe) == set(
        expected_tunable_parameters
    )


def test_list_of_output_quantity_names():
    """Test if list_of_output_quantity_names_from_dataframe matches the expected
    output quantities."""
    expected_output_quantities = [
        "Adjusted_average_core_hours_per_spinor",
        "Average_core_hours_per_spinor",
        "Average_number_of_CG_iterations_per_spinor",
        "Average_number_of_MSCG_iterations_per_spinor",
        "Average_number_of_MV_multiplications_per_spinor",
        "Average_wall_clock_time_per_spinor",
        "Filename",
        "MS_expansion_shifts",
        "Number_of_cores",
        "Plaquette",
        "Spatial_lattice_size",
        "Temporal_lattice_size",
        "Threads_per_process",
        "Total_calculation_time",
    ]
    assert set(TEST_ANALYZER.list_of_output_quantity_names_from_dataframe) == set(
        expected_output_quantities
    )


def test_concatenated_parameters_and_quantities():
    """Test if concatenating tunable parameters and output quantities matches
    all column names."""
    concatenated_list = (
        TEST_ANALYZER.list_of_tunable_parameter_names_from_dataframe
        + TEST_ANALYZER.list_of_output_quantity_names_from_dataframe
    )
    assert set(concatenated_list) == set(TEST_ANALYZER.list_of_dataframe_column_names)


def test_unique_value_columns_dictionary():
    """Test if unique_value_columns_dictionary contains the expected values."""
    expected_dict = {
        "Main_program_type": "invert",
        "Kernel_operator_type": "Wilson",
        "QCD_beta_value": "6.20",
        "APE_alpha": np.float64(0.72),
        "APE_iterations": np.int64(1),
        "Rho_value": np.float64(1.0),
        "Clover_coefficient": np.int64(0),
        "Number_of_spinors": np.int64(12),
        "KL_diagonal_order": np.int64(1),
        "KL_scaling_factor": np.float64(1.0),
        "Overlap_operator_method": "KL",
        "Temporal_lattice_size": np.int64(48),
        "Spatial_lattice_size": np.int64(24),
        "Number_of_vectors": np.int64(1),
        "MS_expansion_shifts": "(0.333333)",
    }
    assert TEST_ANALYZER.unique_value_columns_dictionary == expected_dict


def test_multivalued_columns_count_dictionary():
    """Test if multivalued_columns_count_dictionary contains the expected
    counts."""
    expected_dict = {
        "Filename": 48,
        "MPI_geometry": 3,
        "Threads_per_process": 2,
        "Configuration_label": 6,
        "Bare_mass": 2,
        "Plaquette": 6,
        "Total_calculation_time": 48,
        "CG_epsilon": 4,
        "MSCG_epsilon": 4,
        "Number_of_cores": 3,
        "Average_number_of_MSCG_iterations_per_spinor": 9,
        "Average_number_of_CG_iterations_per_spinor": 24,
        "Average_number_of_MV_multiplications_per_spinor": 9,
        "Average_wall_clock_time_per_spinor": 48,
        "Average_core_hours_per_spinor": 48,
        "Adjusted_average_core_hours_per_spinor": 48,
    }
    assert TEST_ANALYZER.multivalued_columns_count_dictionary == expected_dict


def test_list_of_single_valued_column_names():
    """Test if list_of_single_valued_column_names matches the expected list of
    columns with single unique values."""
    expected_single_valued_columns = [
        "Main_program_type",
        "Kernel_operator_type",
        "QCD_beta_value",
        "APE_alpha",
        "APE_iterations",
        "Rho_value",
        "Clover_coefficient",
        "Number_of_spinors",
        "KL_diagonal_order",
        "KL_scaling_factor",
        "Overlap_operator_method",
        "Temporal_lattice_size",
        "Spatial_lattice_size",
        "Number_of_vectors",
        "MS_expansion_shifts",
    ]
    assert set(TEST_ANALYZER.list_of_single_valued_column_names) == set(
        expected_single_valued_columns
    )


def test_list_of_multivalued_column_names():
    """Test if list_of_multivalued_column_names matches the expected list of
    columns with multiple unique values."""
    expected_multivalued_columns = [
        "Filename",
        "MPI_geometry",
        "Threads_per_process",
        "Configuration_label",
        "Bare_mass",
        "Plaquette",
        "Total_calculation_time",
        "CG_epsilon",
        "MSCG_epsilon",
        "Number_of_cores",
        "Average_number_of_MSCG_iterations_per_spinor",
        "Average_number_of_CG_iterations_per_spinor",
        "Average_number_of_MV_multiplications_per_spinor",
        "Average_wall_clock_time_per_spinor",
        "Average_core_hours_per_spinor",
        "Adjusted_average_core_hours_per_spinor",
    ]
    assert set(TEST_ANALYZER.list_of_multivalued_column_names) == set(
        expected_multivalued_columns
    )


def test_concatenated_single_and_multivalued_columns():
    """Test if concatenating single and multivalued column lists matches all
    column names."""
    concatenated_list = (
        TEST_ANALYZER.list_of_single_valued_column_names
        + TEST_ANALYZER.list_of_multivalued_column_names
    )
    assert set(concatenated_list) == set(TEST_ANALYZER.list_of_dataframe_column_names)


def test_list_of_single_valued_tunable_parameter_names():
    """Test if list_of_single_valued_tunable_parameter_names contains the
    expected parameters."""
    expected_list = [
        "Rho_value",
        "Kernel_operator_type",
        "QCD_beta_value",
        "APE_iterations",
        "Overlap_operator_method",
        "Clover_coefficient",
        "Main_program_type",
        "KL_diagonal_order",
        "Number_of_vectors",
        "APE_alpha",
        "KL_scaling_factor",
        "Number_of_spinors",
    ]
    assert set(TEST_ANALYZER.list_of_single_valued_tunable_parameter_names) == set(
        expected_list
    )


def test_list_of_multivalued_tunable_parameter_names():
    """Test if list_of_multivalued_tunable_parameter_names contains the expected
    parameters."""
    expected_list = [
        "MPI_geometry",
        "Configuration_label",
        "MSCG_epsilon",
        "CG_epsilon",
        "Bare_mass",
    ]
    assert set(TEST_ANALYZER.list_of_multivalued_tunable_parameter_names) == set(
        expected_list
    )


def test_concatenated_single_and_multivalued_tunable_parameters():
    """Test if concatenating single-valued and multi-valued tunable parameter
    lists matches all tunable parameters."""
    concatenated_list = (
        TEST_ANALYZER.list_of_single_valued_tunable_parameter_names
        + TEST_ANALYZER.list_of_multivalued_tunable_parameter_names
    )
    assert set(concatenated_list) == set(
        TEST_ANALYZER.list_of_tunable_parameter_names_from_dataframe
    )


def test_list_of_single_valued_output_quantity_names():
    """Test if list_of_single_valued_output_quantity_names is empty as
    expected."""
    excepted_list = [
        "MS_expansion_shifts",
        "Temporal_lattice_size",
        "Spatial_lattice_size",
    ]
    assert set(TEST_ANALYZER.list_of_single_valued_output_quantity_names) == set(
        excepted_list
    )


def test_list_of_multivalued_output_quantity_names():
    """Test if list_of_multivalued_output_quantity_names contains the expected
    quantities."""
    expected_list = [
        "Average_number_of_CG_iterations_per_spinor",
        "Adjusted_average_core_hours_per_spinor",
        "Number_of_cores",
        "Threads_per_process",
        "Filename",
        "Average_number_of_MV_multiplications_per_spinor",
        "Total_calculation_time",
        "Average_core_hours_per_spinor",
        "Plaquette",
        "Average_wall_clock_time_per_spinor",
        "Average_number_of_MSCG_iterations_per_spinor",
    ]
    assert set(TEST_ANALYZER.list_of_multivalued_output_quantity_names) == set(
        expected_list
    )


def test_concatenated_single_and_multivalued_output_quantities():
    """Test if concatenating single-valued and multi-valued output quantity
    lists matches all output quantities."""
    concatenated_list = (
        TEST_ANALYZER.list_of_single_valued_output_quantity_names
        + TEST_ANALYZER.list_of_multivalued_output_quantity_names
    )
    assert set(concatenated_list) == set(
        TEST_ANALYZER.list_of_output_quantity_names_from_dataframe
    )


def test_group_by_multivalued_tunable_parameters():
    """Test if group_by_multivalued_tunable_parameters returns a
    DataFrameGroupBy object."""
    filter_params = [
        "Bare_mass",
        "MPI_geometry",
        "CG_epsilon",
        "MSCG_epsilon",
        "Configuration_label",
    ]
    grouped = TEST_ANALYZER.group_by_multivalued_tunable_parameters(
        filter_out_parameters_list=filter_params
    )
    assert isinstance(grouped, pd.core.groupby.generic.DataFrameGroupBy)


def test_group_by_multivalued_tunable_parameters_no_filter():
    """Test if group_by_multivalued_tunable_parameters returns a
    DataFrameGroupBy object when no filter is provided."""
    grouped = TEST_ANALYZER.group_by_multivalued_tunable_parameters()
    assert isinstance(grouped, pd.core.groupby.generic.DataFrameGroupBy)


def test_restore_original_dataframe():
    """Test if restore_original_dataframe restores the dataframe to its original
    state after modification."""
    # Modify the dataframe
    TEST_ANALYZER.dataframe.drop("Filename", axis=1, inplace=True)

    # Restore the original dataframe
    TEST_ANALYZER.restore_original_dataframe()

    # Verify that the dataframe matches the original
    pd.testing.assert_frame_equal(TEST_ANALYZER.dataframe, TEST_DATAFRAME)


if __name__ == "__main__":
    pass
