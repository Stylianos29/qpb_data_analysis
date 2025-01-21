import numpy as np
import lsqfit
import gvar as gv
from scipy.stats import t
import warnings

warnings.filterwarnings("ignore")


def linear_function(x, p):
    """Parameters: 1. p[0]: slope 2. p[1]: x-intercept (critical mass)."""

    return p[0] * (x - p[1])


def quadratic_function(x, p):
    """Looks like linear fit."""

    return (p[2] * x + p[0]) * (x - p[1])


def simple_exponential_function(x, p):
    """TODO:"""

    return p[0] * np.exp(-p[1] * x)


def critical_mass_optimum_range(
    bare_mass_values_array, average_squared_effective_mass_estimates_array, sample_size
):
    """Assumption: sorted data"""

    x = bare_mass_values_array
    y = average_squared_effective_mass_estimates_array

    # Linear fit
    # # The initial estimate for the effective mass equals the value
    slope = (max(gv.mean(y)) - min(gv.mean(y))) / (max(gv.mean(x)) - min(gv.mean(x)))
    linear_fit_p0 = [slope, -min(gv.mean(y)) / slope + min(x)]
    linear_fit = lsqfit.nonlinear_fit(
        data=(x, y), p0=linear_fit_p0, fcn=linear_function, debug=True
    )
    # Quadratic fit
    quadratic_fit_p0 = [
        gv.mean(linear_fit.p[0]),
        gv.mean(linear_fit.p[1]),
        0.1 * gv.mean(linear_fit.p[1]),
    ]
    quadratic_fit = lsqfit.nonlinear_fit(
        data=(x, y), p0=quadratic_fit_p0, fcn=quadratic_function, debug=True
    )

    total_number_of_data_points = len(bare_mass_values_array) + 1
    minimum_upper_index_cut = 4
    maximum_number_of_parameters = 2
    maximum_value = 0  # Arbitrary value
    min_key = tuple()
    min_key = (0, maximum_number_of_parameters + 1)
    fit_parameters = list()

    for upper_index_cut in range(minimum_upper_index_cut, total_number_of_data_points):
        # Maintain a distance with upper index such that enough data points are used for fitting
        # for lower_index_cut in range(upper_index_cut-(maximum_number_of_parameters-1)):

        lower_index_cut = 0

        # print(f'range=({lower_index_cut}, {upper_index_cut})')

        x = bare_mass_values_array[lower_index_cut:upper_index_cut]
        y = average_squared_effective_mass_estimates_array[
            lower_index_cut:upper_index_cut
        ]

        # Linear fit
        # # The initial estimate for the effective mass equals the value
        slope = (max(gv.mean(y)) - min(gv.mean(y))) / (
            max(gv.mean(x)) - min(gv.mean(x))
        )
        linear_fit_p0 = [slope, -min(gv.mean(y)) / slope + min(x)]
        linear_fit = lsqfit.nonlinear_fit(
            data=(x, y), p0=linear_fit_p0, fcn=linear_function, debug=True
        )

        key = (lower_index_cut, upper_index_cut)

        difference = linear_fit.p[1] - quadratic_fit.p[1]

        t_statistic = gv.mean(difference) / gv.sdev(
            (difference) / np.sqrt(total_number_of_data_points - 1)
        )

        degrees_of_freedom = total_number_of_data_points - 2

        p_value = t.sf(np.abs(t_statistic), degrees_of_freedom) * 2

        # print(key, t_statistic, 1-p_value)

        # p_value = 1 - quadratic_fit.Q

        if p_value > maximum_value:
            maximum_value = p_value
            min_key = key
            # fit_parameters = linear_fit

    return min_key


def chi_squared_minimized_plateau_fit(data_array):

    squared_data_array = np.square(data_array)
    inverse_squared_data_array = 1 / squared_data_array

    numerator = np.sum(data_array * inverse_squared_data_array)
    denominator = np.sum(inverse_squared_data_array)

    return numerator / denominator
