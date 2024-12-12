import os
import shutil

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import textwrap
import gvar as gv

from library import data_processing
from library import constants
from library import plotting


# Main Script
INPUT_CSV_FILE_PATH = "/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/invert/Chebyshev_several_m_varying_EpsLanczos/PCAC_mass_estimates.csv"
PLOTS_DIRECTORY = "/nvme/h/cy22sg1/qpb_data_analysis/output/plots/invert/Chebyshev_several_m_varying_EpsLanczos"

qpb_log_files_dataframe = data_processing.load_csv(INPUT_CSV_FILE_PATH)
analyzer = data_processing.DataFrameAnalyzer(qpb_log_files_dataframe)

# Process each pair of variables
variable_pairs = [
    ("Lanczos_epsilon", "PCAC_mass_estimate"),
    ("Number_of_Chebyshev_terms", "PCAC_mass_estimate"),
]

for x_var, y_var in variable_pairs:
    plotting.process_variable_pair(analyzer, PLOTS_DIRECTORY, x_var, y_var)
