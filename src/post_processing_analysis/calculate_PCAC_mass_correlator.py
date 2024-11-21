import os
import sys

import click # type: ignore
import numpy as np
import gvar as gv # type: ignore
import pandas as pd
import ast
import logging
import h5py

sys.path.append('../')
from library import momentum_correlator, jackknife_analysis, filesystem_utilities


def is_valid_directory(directory_path):
    # Check if passed directory path exists and it is indeed a directory
    return os.path.exists(directory_path) and os.path.isdir(directory_path)


@click.command()
@click.option("--log_files_data_csv_file_path", "log_files_data_csv_file_path",
              "-log_csv", default="../csv_files/log_files_data.csv",
            help="Path to the .csv file containing the extracted info from log files.")
@click.option("--pion_correlator_values_hdf5_file_path",
              "pion_correlator_values_hdf5_file_path",
              "-pion_hdf5", default="../hdf5_files/pion_correlator_values.h5",
            help="Path to the HDF5 file containing the extracted pion correlator values.")
@click.option("--jackknife_analysis_for_PCAC_mass_values_hdf5_file_path",
              "jackknife_analysis_for_PCAC_mass_values_hdf5_file_path",
              "-PCAC_hdf5", default="../hdf5_files/jackknife_PCAC_mass_values.h5",
            help="Path to the output HDF5 file with the jackknife PCAC mass values.")
def main(log_files_data_csv_file_path, pion_correlator_values_hdf5_file_path,
                        jackknife_analysis_for_PCAC_mass_values_hdf5_file_path):

    # PERFORM VALIDITY CHECKS ON INPUT ARGUMENTS

    if not filesystem_utilities.is_valid_file(log_files_data_csv_file_path):
        # logging.error("The log files data csv file path is invalid!")
        sys.exit(1)

    if not filesystem_utilities.is_valid_file(pion_correlator_values_hdf5_file_path):
        # logging.error("The pion correlator values HDF5 file path is invalid!")
        sys.exit(1)

    if not is_valid_directory(os.path.dirname(
                    jackknife_analysis_for_PCAC_mass_values_hdf5_file_path)):
        # logging.error("The HDF5 files directory path is invalid!")
        sys.exit(1)

    # EXTRACT USEFUL ATTRIBUTES OF THE WORKING DATASET OF PION CORRELATOR

    # Load the CSV file into a DataFrame
    log_files_dataframe = pd.read_csv(log_files_data_csv_file_path)

    # Extract unique values
    # TODO: Check uniqueness of the operator method
    unique_operator_method = log_files_dataframe['Operator_method'].unique()[0]
    # unique_operator_type = log_files_dataframe['Operator_type'].unique()[0]
    unique_lattice_geometry = log_files_dataframe['Lattice_geometry'].unique()[0]
    unique_QCD_beta_value = log_files_dataframe['QCD_beta_value'].unique()[0]
    unique_APE_alpha = log_files_dataframe['APE_alpha'].unique()[0]
    unique_APE_iterations = log_files_dataframe['APE_iterations'].unique()[0]
    unique_rho_value = log_files_dataframe['Rho_value'].unique()[0]
    unique_clover_coefficient = log_files_dataframe['Clover_coefficient'].unique()[0]
    unique_KL_iterations = log_files_dataframe['KL_iterations'].unique()[0]
    # unique_KL_scaling_factor = log_files_dataframe['KL_scaling_factor'].unique()[0]
    # unique_CG_epsilon = log_files_dataframe['CG_epsilon'].unique()[0]
    # unique_MSCG_epsilon = log_files_dataframe['MSCG_epsilon'].unique()[0]
    # unique_bare_mass = log_files_dataframe['Bare_mass'].unique()[0]

    # Calculate further useful information
    number_of_gauge_configurations = log_files_dataframe['Configuration_label'].nunique()
    temporal_direction_lattice_size = int(ast.literal_eval(unique_lattice_geometry)[0])

    # TODO: Write comments
    with h5py.File(jackknife_analysis_for_PCAC_mass_values_hdf5_file_path, 'w') as hdf5_file_write, \
        h5py.File(pion_correlator_values_hdf5_file_path, 'r') as hdf5_file_read:

        # Add a global attributes to the file
        hdf5_file_write.attrs['Operator_method'] = unique_operator_method
        hdf5_file_write.attrs['QCD_beta_value'] = unique_QCD_beta_value
        hdf5_file_write.attrs['APE_alpha'] = unique_APE_alpha
        hdf5_file_write.attrs['APE_iterations'] = unique_APE_iterations
        hdf5_file_write.attrs['Rho_value'] = unique_rho_value
        hdf5_file_write.attrs['Clover_coefficient'] = unique_clover_coefficient
        hdf5_file_write.attrs['KL_iterations'] = unique_KL_iterations
        # hdf5_file_write.attrs['KL_scaling_factor'] = unique_KL_scaling_factor
        # hdf5_file_write.attrs['Bare_mass'] = unique_bare_mass

        hdf5_file_write.attrs['number_of_gauge_configurations'] = number_of_gauge_configurations
        hdf5_file_write.attrs['temporal_direction_lattice_size'] = temporal_direction_lattice_size

        PCAC_mass_correlator_values_hdf5_group = hdf5_file_write.create_group('PCAC_mass_correlator_values')

        # EXTRACT THE g5-g5 AND THE g4g5-g5 PION CORRELATORS FROM HDF5 FILE

        # TODO: Write comments
        for operator_type, operator_type_group in log_files_dataframe.groupby('Operator_type'):

            operator_type_hdf5_group = PCAC_mass_correlator_values_hdf5_group.create_group(operator_type)

            for KL_scaling_factor, KL_scaling_factor_group in operator_type_group.groupby('KL_scaling_factor'):

                KL_scaling_factor_hdf5_group = operator_type_hdf5_group.create_group(str(KL_scaling_factor))

                for bare_mass, bare_mass_group in KL_scaling_factor_group.groupby('Bare_mass'):

                    bare_mass_hdf5_group = KL_scaling_factor_hdf5_group.create_group(str(bare_mass))

                    for CG_epsilon, CG_epsilon_group in bare_mass_group.groupby('CG_epsilon'):

                        CG_epsilon_hdf5_group = bare_mass_hdf5_group.create_group(str(CG_epsilon))

                        # for MSCG_epsilon, MSCG_epsilon_group in CG_epsilon_group.groupby('MSCG_epsilon'):

                        #     MSCG_epsilon_hdf5_group = CG_epsilon_hdf5_group.create_group(str(MSCG_epsilon))

                        g5_g5_correlator_values_per_configuration_list = []
                        g4g5_g5_correlator_values_per_configuration_list = []

                        # Iterate over all top-level groups (filenames)
                        # for filename in hdf5_file_read.keys():
                        for filename in CG_epsilon_group['Filename'].tolist():

                            # TODO: Write comment
                            filename = filename.replace(".txt", ".dat")
                            filename_group = hdf5_file_read[filename]

                            if 'g5-g5' in filename_group.keys():
                                g5_g5_data = filename_group['g5-g5'][:]
                                g5_g5_correlator_values_per_configuration_list.append(g5_g5_data)

                            if 'g4g5-g5' in filename_group.keys():
                                g4g5_g5_data = filename_group['g4g5-g5'][:]
                                g4g5_g5_correlator_values_per_configuration_list.append(g4g5_g5_data)

                        # Convert the list of 1D arrays into a 2D NumPy array
                        g5_g5_correlator_values_per_configuration_2D_array = np.vstack(g5_g5_correlator_values_per_configuration_list)
                        g4g5_g5_correlator_values_per_configuration_2D_array = np.vstack(g4g5_g5_correlator_values_per_configuration_list)

                        # JACKKNIFE ANALYSIS OF THE g5-g5 and g4g5-g5 PION CORRELATORS

                        # Jackknife analysis of the g5-g5 correlator values
                        jackknife_analyzed_g5_g5_correlator_values_per_configuration_object = jackknife_analysis.JackknifeAnalysis(g5_g5_correlator_values_per_configuration_2D_array)

                        # Jackknife samples of the g5-g5 correlators
                        jackknife_samples_of_g5_g5_correlator_2D_array = jackknife_analyzed_g5_g5_correlator_values_per_configuration_object.jackknife_replicas_of_original_2D_array

                        # Jackknife average of the g5-g5 correlators
                        jackknife_average_of_g5_g5_correlator = jackknife_analyzed_g5_g5_correlator_values_per_configuration_object.jackknife_average

                        # Jackknife analysis of the g4g5-g5 correlator values 
                        jackknife_analyzed_g4g5_g5_correlator_values_per_configuration_object = jackknife_analysis.JackknifeAnalysis(g4g5_g5_correlator_values_per_configuration_2D_array)

                        # Jackknife samples of the g4g5-g5 correlators
                        jackknife_samples_of_g4g5_g5_correlator_2D_array = jackknife_analyzed_g4g5_g5_correlator_values_per_configuration_object.jackknife_replicas_of_original_2D_array

                        # Jackknife average of the g4g5-g5 correlators
                        jackknife_average_of_g4g5_g5_correlator = jackknife_analyzed_g4g5_g5_correlator_values_per_configuration_object.jackknife_average

                        # PCAC MASS CALCULATION

                        """ The PCAC mass is defined as the ratio of the g4g5-g5 derivative correlator values over the g5-g5 correlator values """

                        # Jackknife samples of the g4g5-g5 derivate correlators
                        jackknife_samples_of_g4g5_g5_derivative_correlator_2D_list = list()
                        for index in range(len(jackknife_samples_of_g4g5_g5_correlator_2D_array)):
                            jackknife_samples_of_g4g5_g5_derivative_correlator_2D_list.append(momentum_correlator.centered_difference_correlator_derivative(jackknife_samples_of_g4g5_g5_correlator_2D_array[index]))
                        jackknife_samples_of_g4g5_g5_derivative_correlator_2D_array = np.array(jackknife_samples_of_g4g5_g5_derivative_correlator_2D_list)

                        # g4g5-g5 derivative correlator from the jackknife average of g4g5-g5 correlator
                        g4g5_g5_derivative_correlator_from_jackknife_average_of_g4g5_g5_correlator = momentum_correlator.centered_difference_correlator_derivative(jackknife_average_of_g4g5_g5_correlator)

                        # Jackknife samples of the time-dependent PCAC mass values
                        jackknife_samples_of_time_dependent_PCAC_mass_values_list = list()
                        for index in range(len(jackknife_samples_of_g5_g5_correlator_2D_array)):

                            jackknife_sample_of_time_dependent_PCAC_mass_values = 0.5*jackknife_samples_of_g4g5_g5_derivative_correlator_2D_array[index]/jackknife_samples_of_g5_g5_correlator_2D_array[index]

                            jackknife_sample_of_time_dependent_PCAC_mass_values = momentum_correlator.symmetrization(jackknife_sample_of_time_dependent_PCAC_mass_values)
                            
                            jackknife_samples_of_time_dependent_PCAC_mass_values_list.append(jackknife_sample_of_time_dependent_PCAC_mass_values)
                        jackknife_samples_of_time_dependent_PCAC_mass_values_2D_array = np.array(jackknife_samples_of_time_dependent_PCAC_mass_values_list)

                        # Jackknife average of the time-dependent PCAC mass values
                        jackknife_average_of_time_dependent_PCAC_mass_values_array = gv.gvar(
                            np.average(jackknife_samples_of_time_dependent_PCAC_mass_values_2D_array, axis=0),
                            np.sqrt(number_of_gauge_configurations-1)*np.std(jackknife_samples_of_time_dependent_PCAC_mass_values_2D_array, axis=0, ddof=0)
                            )

                        # Time-dependent PCAC mass values from the jackknife averages of the correlators
                        time_dependent_PCAC_mass_values_from_jackknife_averages_of_correlators_array = 0.5*g4g5_g5_derivative_correlator_from_jackknife_average_of_g4g5_g5_correlator/jackknife_average_of_g5_g5_correlator

                        time_dependent_PCAC_mass_values_from_jackknife_averages_of_correlators_array = momentum_correlator.symmetrization(time_dependent_PCAC_mass_values_from_jackknife_averages_of_correlators_array)

                        CG_epsilon_hdf5_group.create_dataset('jackknife_samples_of_PCAC_mass_correlator_values', data=jackknife_samples_of_time_dependent_PCAC_mass_values_2D_array)
                        CG_epsilon_hdf5_group.create_dataset('jackknife_average_of_PCAC_mass_correlator_values_means', data=gv.mean(jackknife_average_of_time_dependent_PCAC_mass_values_array))
                        CG_epsilon_hdf5_group.create_dataset('jackknife_average_of_PCAC_mass_correlator_values_error', data=gv.sdev(jackknife_average_of_time_dependent_PCAC_mass_values_array))

    print("* PCAC mass correlator values jackknife analysis completed.")


if __name__ == "__main__":
    main()
