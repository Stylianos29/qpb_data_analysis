from library import data_processing

INPUT_CSV_FILE_PATH = "/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/sign_squared_values/Chebyshev_several_configs_varying_Ν/qpb_log_files_single_valued_parameters.csv"
INPUT_HDF5_FILE_PATH = "/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/sign_squared_values/Chebyshev_several_configs_varying_Ν/qpb_log_files_multivalued_parameters.h5"
PLOTS_DIRECTORY = "/nvme/h/cy22sg1/qpb_data_analysis/output/plots/sign_squared_values/Chebyshev_several_configs_varying_Ν"

qpb_log_files_dataframe = data_processing.load_csv(INPUT_CSV_FILE_PATH)

analyzer = data_processing.DataFrameAnalyzer(qpb_log_files_dataframe)

