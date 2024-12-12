import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import ast
import textwrap
import lsqfit
import gvar as gv
from scipy.optimize import curve_fit

from library import filesystem_utilities, data_processing, plotting, fit_functions


# Define the power law function
def power_law(x, a, b):
    return a * x**b


WORKSPACE_DIRECTORY_FULL_PATH = "/nvme/h/cy22sg1/qpb_data_analysis"
CSV_FILE_FULL_PATH = os.path.join(WORKSPACE_DIRECTORY_FULL_PATH,
                                  "data_files/processed/invert/KL_several_config_and_mu_varying_m/PCAC_mass_estimates.csv")
PLOTTING_DIRECTORY = os.path.join(WORKSPACE_DIRECTORY_FULL_PATH,
                                  "output/plots/invert/KL_several_config_and_mu_varying_m")

QPB_LOG_FILES_CSV_FILE_FULL_PATH = os.path.join(WORKSPACE_DIRECTORY_FULL_PATH,
                                  "/nvme/h/cy22sg1/qpb_data_analysis/data_files/processed/invert/KL_several_config_and_mu_varying_m/qpb_log_files_single_valued_parameters.csv")

# Check if given .csv file full path is valid
if not (os.path.exists(QPB_LOG_FILES_CSV_FILE_FULL_PATH) and os.path.isfile(CSV_FILE_FULL_PATH)):
    print("Invalid .csv file directory.\nExiting")
    sys.exit()

# Check if given .csv file full path is valid
if not (os.path.exists(CSV_FILE_FULL_PATH) and os.path.isfile(CSV_FILE_FULL_PATH)):
    print("Invalid .csv file directory.\nExiting")
    sys.exit()

# Check if given plots directories is valid
if not (os.path.exists(PLOTTING_DIRECTORY) and os.path.isdir(PLOTTING_DIRECTORY)):
    print("Invalid plots directory.\nExiting...")
    sys.exit()

# Import .csv files
KL_several_config_and_mu_varying_m_dataframe = pd.read_csv(CSV_FILE_FULL_PATH,
                                            dtype={'Clover_coefficient': int,
                                                    # 'QCD_beta_value': str
                                                    }
                                                )
qpb_log_files_dataframe = pd.read_csv(QPB_LOG_FILES_CSV_FILE_FULL_PATH,
                                            dtype={'Clover_coefficient': int,
                                                    # 'QCD_beta_value': str
                                                    }
                                                )

# Get a list of columns with more than one unique value
columns_with_multiple_unique_values = {col: KL_several_config_and_mu_varying_m_dataframe[col].nunique() for col in KL_several_config_and_mu_varying_m_dataframe.columns if KL_several_config_and_mu_varying_m_dataframe[col].nunique() > 1}
print("\nFields with multiple unique values:")
print(columns_with_multiple_unique_values)
# Get a list of columns with a single unique value
columns_with_a_unique_value = [col for col in KL_several_config_and_mu_varying_m_dataframe.columns if KL_several_config_and_mu_varying_m_dataframe[col].nunique() == 1]
print("\nFields with a unique value:")
print(columns_with_a_unique_value)

# Get a list of columns with more than one unique value
columns_with_multiple_unique_values = {col: qpb_log_files_dataframe[col].nunique() for col in qpb_log_files_dataframe.columns if qpb_log_files_dataframe[col].nunique() > 1}
print("\nFields with multiple unique values:")
print(columns_with_multiple_unique_values)
# Get a list of columns with a single unique value
columns_with_a_unique_value = [col for col in qpb_log_files_dataframe.columns if qpb_log_files_dataframe[col].nunique() == 1]
print("\nFields with a unique value:")
print(columns_with_a_unique_value)

# Extract unique values
unique_APE_alpha = KL_several_config_and_mu_varying_m_dataframe['APE_alpha'].unique()[0]
unique_APE_iterations = KL_several_config_and_mu_varying_m_dataframe['APE_iterations'].unique()[0]
unique_clover_coefficient = KL_several_config_and_mu_varying_m_dataframe['Clover_coefficient'].unique()[0]
# unique_KL_diagonal_order = KL_several_config_and_mu_varying_m_dataframe['KL_diagonal_order'].unique()[0]
unique_overlap_operator_method = KL_several_config_and_mu_varying_m_dataframe['Overlap_operator_method'].unique()[0]
unique_QCD_beta_value = KL_several_config_and_mu_varying_m_dataframe['QCD_beta_value'].unique()[0]
unique_rho_value = KL_several_config_and_mu_varying_m_dataframe['Rho_value'].unique()[0]

KL_several_config_and_mu_varying_m_dataframe['PCAC_mass_estimate'] = KL_several_config_and_mu_varying_m_dataframe['PCAC_mass_estimate'].apply(ast.literal_eval)


plotting_subdirectory = os.path.join(PLOTTING_DIRECTORY, 'PCAC_mass_against_bare_mass_values_for_various_KL_scaling_factors')
os.makedirs(plotting_subdirectory, exist_ok=True)

for kernel_operator_type, kernel_operator_type_group in KL_several_config_and_mu_varying_m_dataframe.groupby('Kernel_operator_type'):
        
    for KL_diagonal_order, KL_diagonal_order_group in kernel_operator_type_group.groupby('KL_diagonal_order'):

        for CG_epsilon, CG_epsilon_group in KL_diagonal_order_group.groupby('CG_epsilon'):

            if CG_epsilon != 1e-6:
                continue

            for MSCG_epsilon, MSCG_epsilon_group in CG_epsilon_group.groupby('MSCG_epsilon'):

                if MSCG_epsilon != 1e-7:
                    continue

                fig, ax = plt.subplots()
                plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

                plot_title = ""
                parameter_values_subtitle=f"KL {kernel_operator_type} $n$={KL_diagonal_order}, $\\beta$={unique_QCD_beta_value:.2f}, $c_{{SW}}$={unique_clover_coefficient}, $\\rho$={unique_rho_value}, APE iters={unique_APE_iterations}, $\\epsilon_{{MSCG}}$={MSCG_epsilon}, $\\epsilon_{{CG}}$={CG_epsilon}"
                wrapper = textwrap.TextWrapper(width=90, initial_indent="   ")
                wrapped_parameter_values_subtitle = wrapper.fill(parameter_values_subtitle)
                
                ax.set_title(f'{plot_title}\n{wrapped_parameter_values_subtitle}', pad=7)

                # Set x-axis ticks to integer values only
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                ax.set(xlabel='a$m_{{bare}}$', ylabel='a$m_{PCAC}$')
                plt.subplots_adjust(left=0.11) # Adjust left margin

                ax.axhline(0, color='black') # y = 0
                ax.axvline(0, color='black') # x = 0


                # Get the unique KL_scaling_factor values, sorted in descending order
                sorted_KL_scaling_factors = sorted(
                    MSCG_epsilon_group['KL_scaling_factor'].unique(), reverse=True
                )

                # Iterate over the sorted KL_scaling_factors
                for KL_scaling_factor in sorted_KL_scaling_factors:
                    # Filter the group for the current KL_scaling_factor
                    KL_scaling_factor_group = MSCG_epsilon_group[
                        MSCG_epsilon_group['KL_scaling_factor'] == KL_scaling_factor
                    ]

                    if KL_scaling_factor > 2.0:
                        continue
                    
                    filtered_qpb_log_files_dataframe = qpb_log_files_dataframe[
                        (qpb_log_files_dataframe["Kernel_operator_type"] == kernel_operator_type) &
                        (qpb_log_files_dataframe["KL_diagonal_order"] == KL_diagonal_order) &
                        (qpb_log_files_dataframe["CG_epsilon"] == CG_epsilon) &
                        (qpb_log_files_dataframe["MSCG_epsilon"] == MSCG_epsilon) &
                        (qpb_log_files_dataframe["KL_scaling_factor"] == KL_scaling_factor)
                    ]

                    # number_of_nodes = np.prod(ast.literal_eval(filtered_qpb_log_files_dataframe['MPI_geometry'].unique()[0]))
                    MPI_geometry = filtered_qpb_log_files_dataframe['MPI_geometry'].unique()[0]

                    bare_mass_values_array = np.sort(filtered_qpb_log_files_dataframe['Bare_mass'].unique())

                    average_CG_calculation_time_per_bare_mass_array = filtered_qpb_log_files_dataframe.groupby('Bare_mass')[
                            'Average_CG_calculation_time_per_spinor'
                        ].mean().to_numpy()
                    
                    # Fit the power law to the data
                    popt, pcov = curve_fit(power_law, bare_mass_values_array, average_CG_calculation_time_per_bare_mass_array)
                    a, b = popt  # Extract the parameters

                    # Extrapolate to x = 0.17
                    reference_bare_mass_value = 0.17
                    average_CG_calculation_time_at_reference_bare_mass = power_law(reference_bare_mass_value, a, b)

                    
                    # print(kernel_operator_type, KL_diagonal_order, bare_mass_values_array,
                        #   average_CG_calculation_time_per_bare_mass_array)

                # for KL_scaling_factor, KL_scaling_factor_group in MSCG_epsilon_group.groupby('KL_scaling_factor'):

                    bare_mass_values_array = KL_scaling_factor_group['Bare_mass'].to_numpy()
                    PCAC_mass_estimates_array = gv.gvar(KL_scaling_factor_group['PCAC_mass_estimate'].to_numpy())

                    ax.errorbar(bare_mass_values_array, gv.mean(PCAC_mass_estimates_array), yerr=gv.sdev(PCAC_mass_estimates_array), fmt='.', markersize=8, capsize=10, label=f'$\\mu$={KL_scaling_factor}, $\\bar{{t}}_{{0.17}} = {average_CG_calculation_time_at_reference_bare_mass:.2f}$ sec, {MPI_geometry}')

                    if len(bare_mass_values_array) > 2:
                    
                        # Linear fit
                        min_index = 0
                        x = bare_mass_values_array[min_index:]
                        y = PCAC_mass_estimates_array[min_index:]
                        # # The initial estimate for the effective mass equals the value
                        slope = (max(gv.mean(y)) - min(gv.mean(y)))/(max(gv.mean(x)) - min(gv.mean(x)))
                        linear_fit_p0 = [slope, min(gv.mean(y))/slope+min(x)]
                        linear_fit = lsqfit.nonlinear_fit(data=(x, y), p0=linear_fit_p0, fcn=fit_functions.linear_function, debug=True)



                        # critical_mass_value = linear_fit.p[1]
                        # critical_mass_value = gv.gvar(-test_intercept/test_slop, np.sqrt(np.abs(-test_intercept_error/test_slop)))
                        
                        error = np.sqrt(np.sum(((gv.mean(y) - fit_functions.linear_function(x, gv.mean(linear_fit.p)))/1)**2)/len(y)) #gv.sdev(y)
                        critical_mass_value = gv.gvar(gv.mean(linear_fit.p[1]), error)


                        margin = 0.03
                        x_data = np.linspace(min(gv.mean(x))*(-margin), max(gv.mean(x))*(1+margin), 100)
                        y_data = fit_functions.linear_function(x_data, linear_fit.p)
                        kappa_critical_linear_fit = 0.5/(critical_mass_value+4)
                        plt.plot(x_data, gv.mean(y_data), '--')
                        # , label=f'linear fit ($\\chi^2$/dof={linear_fit.chi2:.2f}/{linear_fit.dof}={linear_fit.chi2/linear_fit.dof:.4f}):\n$a m_c$={critical_mass_value:.5f}, $\\kappa_c$={kappa_critical_linear_fit:.6f}')
                        ax.fill_between(x_data, gv.mean(y_data) - gv.sdev(critical_mass_value), gv.mean(y_data) + gv.sdev(critical_mass_value), color='r', alpha=0.2)

                ax.legend(loc="upper left")

                plot_path = os.path.join(plotting_subdirectory, f'PCAC_mass_against_bare_mass_values_for_various_KL_scaling_factors_{kernel_operator_type}_EpsMSCG{MSCG_epsilon}_EpsCG{CG_epsilon}_cSW{unique_clover_coefficient}_rho{unique_rho_value}_n{KL_diagonal_order}'.replace(".", "p")+'.png')
                fig.savefig(plot_path)
                plt.close()

