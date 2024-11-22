import numpy as np
import gvar as gv


def jackknife_averages(jackknife_replicas_array):

    average = np.average(jackknife_replicas_array)
    error = np.sqrt(len(jackknife_replicas_array)-1)*np.std(jackknife_replicas_array, ddof=0)

    return gv.gvar(average, error)


class JackknifeAnalysis:

    def __init__(self, original_2D_array):

        self.original_2D_array = np.array(original_2D_array)

        self.sample_size = self.original_2D_array.shape[0]
        self.dataset_size = self.original_2D_array.shape[1]

        self.jackknife_average = self.jackknife_replicas()
    
    def jackknife_replicas(self):

        jackknife_replicas_2D_list = list()
        for index_to_remove in range(self.sample_size):

            reduced_original_2D_array = np.delete(self.original_2D_array, index_to_remove, axis=0)

            jackknife_replica = np.average(reduced_original_2D_array, axis=0)

            jackknife_replicas_2D_list.append(jackknife_replica)

        self.jackknife_replicas_of_original_2D_array = np.array(jackknife_replicas_2D_list)

        jackknife_average = np.average(self.jackknife_replicas_of_original_2D_array, axis=0)

        covariance_matrix = (self.sample_size-1)*np.cov(self.jackknife_replicas_of_original_2D_array, rowvar=False, ddof=0)

        jackknife_error = np.sqrt(self.sample_size-1)*np.std(self.jackknife_replicas_of_original_2D_array, ddof=0, axis=0)

        # print(jackknife_error**2)
        # print(covariance_matrix)

        return gv.gvar(jackknife_average, jackknife_error)
        # return gv.gvar(jackknife_average, covariance_matrix)
    
