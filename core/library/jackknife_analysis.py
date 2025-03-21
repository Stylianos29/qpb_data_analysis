import numpy as np
import gvar as gv


def jackknife_averages(jackknife_replicas_array):

    average = np.average(jackknife_replicas_array)
    error = np.sqrt(len(jackknife_replicas_array) - 1) * np.std(
        jackknife_replicas_array, ddof=0
    )

    return gv.gvar(average, error)


class JackknifeAnalysis:

    def __init__(self, original_2D_array):

        self.original_2D_array = np.array(original_2D_array)

        self.sample_size = self.original_2D_array.shape[0]
        self.dataset_size = self.original_2D_array.shape[1]

        self.jackknife_average = self.jackknife_replicas()

    def jackknife_replicas(self):

        jackknife_replicas_2D_list = []
        for index_to_remove in range(self.sample_size):

            reduced_original_2D_array = np.delete(
                self.original_2D_array, index_to_remove, axis=0
            )

            jackknife_replica = np.average(reduced_original_2D_array, axis=0)

            jackknife_replicas_2D_list.append(jackknife_replica)

        self.jackknife_replicas_of_original_2D_array = np.array(
            jackknife_replicas_2D_list
        )

        jackknife_average = np.average(
            self.jackknife_replicas_of_original_2D_array, axis=0
        )

        covariance_matrix = (self.sample_size - 1) * np.cov(
            self.jackknife_replicas_of_original_2D_array, rowvar=False, ddof=0
        )

        jackknife_error = np.sqrt(self.sample_size - 1) * np.std(
            self.jackknife_replicas_of_original_2D_array, ddof=0, axis=0
        )

        return gv.gvar(jackknife_average, jackknife_error)
        # return gv.gvar(jackknife_average, covariance_matrix)


def calculate_autocorrelation(data):
    n = len(data)
    mean = np.mean(data)
    variance = np.var(data)
    autocorrelation = np.correlate(data - mean, data - mean, mode="full")[n - 1 :] / (
        variance * n
    )

    return autocorrelation


def calculate_integrated_autocorrelation_time(data, max_lag=None):
    autocorrelation = calculate_autocorrelation(data)
    if max_lag is None:
        max_lag = len(autocorrelation) // 2
    tau_int = 1 + 2 * np.sum(autocorrelation[1:max_lag])

    return tau_int


def jackknife_correlated_error(jackknife_replicas, tau_int=0.5):
    """
    Calculate the jackknife standard error for correlated data, including
    the effect of the integrated autocorrelation time.

    Parameters:
        jackknife_replicas (numpy.ndarray): A 2D array where each row is a jackknife replica
                                            and each column corresponds to a data point.
        tau_int (float or numpy.ndarray): The integrated autocorrelation time. Can be a scalar
                                          (applied to all data points) or a 1D array (one per column).

    Returns:
        numpy.ndarray: The jackknife standard error for each column of the replicas.
    """
    N = jackknife_replicas.shape[0]  # Number of jackknife replicas

    # Mean of the jackknife replicas (column-wise)
    jackknife_mean = np.mean(jackknife_replicas, axis=0)

    # Variance from jackknife formula: (N-1)/N * Σ((replica - mean)**2)
    variance = (N - 1) / N * np.sum((jackknife_replicas - jackknife_mean) ** 2, axis=0)

    # Adjust for the integrated autocorrelation time
    variance *= 2 * tau_int

    # Standard error is the square root of the adjusted variance
    return np.sqrt(variance)


def calculate_jackknife_average_array(jackknife_samples_2D_array):
    """
    Computes the jackknife average and standard error for a given 2D array of jackknife samples.

    Parameters:
    - jackknife_samples_2D_array (np.ndarray): A 2D array where each row represents a jackknife sample.
    - number_of_gauge_configurations (int): The total number of gauge configurations used in the jackknife resampling.

    Returns:
    - gvar.GVar: A generalized variable object containing the mean and standard error of the jackknife samples.
    """

    number_of_gauge_configurations = np.shape(jackknife_samples_2D_array)[0]

    return gv.gvar(
        np.average(jackknife_samples_2D_array, axis=0),
        np.sqrt(number_of_gauge_configurations - 1)
        * np.std(jackknife_samples_2D_array, ddof=0, axis=0),
    )


def weighted_mean(values, errors, factor=1):
    """
    Calculate the weighted mean and its error for a set of values and uncertainties.

    Parameters:
    - values (array-like): A list or NumPy array of numerical values.
    - errors (array-like): A list or NumPy array of uncertainties associated with the values.

    Returns:
    - tuple: Weighted mean and its associated uncertainty as a tuple (mean, error).
    """
    values = np.asarray(values)
    errors = np.asarray(errors)

    if len(values) != len(errors):
        raise ValueError("The length of values and errors must be the same.")
    if np.any(errors <= 0):
        raise ValueError("All errors must be positive.")

    # Compute weights
    weights = 1 / errors**2

    # Calculate weighted mean
    weighted_mean = np.sum(weights * values) / np.sum(weights)

    # Calculate the error of the weighted mean
    weighted_mean_error = np.sqrt(1 / np.sum(weights))

    return weighted_mean, weighted_mean_error * factor
