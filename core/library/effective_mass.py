import numpy as np
import scipy.stats as stats
import lsqfit
import warnings


warnings.filterwarnings("ignore")


def merge_contiguous_ranges(indices, maximum_gap=2):
    # Step 1: Split indices into contiguous sublists
    contiguous_ranges = []
    current_range = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_range.append(indices[i])
        else:
            contiguous_ranges.append(current_range)
            current_range = [indices[i]]
    contiguous_ranges.append(current_range)  # Add the last range

    # Step 2: Find the largest list
    largest_range_index = max(
        range(len(contiguous_ranges)),
        key=lambda i: len(contiguous_ranges[i]),
    )
    largest_range = contiguous_ranges[largest_range_index]

    # Step 3: Merge preceding lists
    i = largest_range_index - 1
    while i >= 0:
        current_max = contiguous_ranges[i][-1]
        next_min = largest_range[0]

        if next_min - current_max <= maximum_gap:
            # Merge and fill the gap
            largest_range = (
                contiguous_ranges[i]
                + list(range(current_max + 1, next_min))
                + largest_range
            )
            contiguous_ranges[i] = []  # Mark this range as merged
        else:
            break
        i -= 1

    # Step 4: Merge succeeding lists
    i = largest_range_index + 1
    while i < len(contiguous_ranges):
        current_min = contiguous_ranges[i][0]
        next_max = largest_range[-1]

        if current_min - next_max <= maximum_gap:
            # Merge and fill the gap
            largest_range = (
                largest_range
                + list(range(next_max + 1, current_min))
                + contiguous_ranges[i]
            )
            contiguous_ranges[i] = []  # Mark this range as merged
        else:
            break
        i += 1

    return largest_range


def plateau_indices_range(dataset, sigma_criterion_factor=0.5, maximum_gap=2):

    dy_dx_centered = (dataset[2:] - dataset[:-2]) / 2
    plateau_indices_centered = [
        i + 1
        for i, dy in enumerate(dy_dx_centered)
        if abs(dy.mean) <= sigma_criterion_factor * dy.sdev
    ]

    if len(plateau_indices_centered) > 0:
        return merge_contiguous_ranges(plateau_indices_centered, maximum_gap)
    return plateau_indices_centered


def calculate_single_state_non_periodic_effective_mass_correlator(
    g5g5_correlator_array,
):
    """Calculates the m parameter of the single-state non-period correlator function: m = log(C(t)/C(t+1))."""

    shifted_forward_g5g5_correlator_array = np.roll(g5g5_correlator_array, shift=-1)

    return np.log(g5g5_correlator_array / shifted_forward_g5g5_correlator_array)


def calculate_two_state_periodic_effective_mass_correlator(
    g5g5_correlator_array, lowering_factor=0.99, truncate_half=True
):

    middle_value = np.min(g5g5_correlator_array) * lowering_factor

    temporal_lattice_size = np.shape(g5g5_correlator_array)[0]

    shifted_backward_array = np.roll(g5g5_correlator_array, shift=+1)
    shifted_forward_array = np.roll(g5g5_correlator_array, shift=-1)

    # Remove extreme elements, they cannot be used in calculations
    shifted_backward_array = shifted_backward_array[1:-1]
    shifted_forward_array = shifted_forward_array[1:-1]

    if truncate_half:
        upper_index_cut = (temporal_lattice_size - 2) // 2
        shifted_backward_array = shifted_backward_array[:upper_index_cut]
        shifted_forward_array = shifted_forward_array[:upper_index_cut]

    numerator = shifted_backward_array + np.sqrt(
        np.square(shifted_backward_array) - middle_value**2
    )
    denominator = shifted_forward_array + np.sqrt(
        np.square(shifted_forward_array) - middle_value**2
    )

    return 0.5 * np.log(numerator / denominator)


def effective_mass_periodic_case_function_old(
    correlator_array, middle_value_array=None
):

    lattice_size = np.shape(correlator_array)[0]

    if middle_value_array is None:
        middle_value_lowering_correction = 0.01
        middle_value_array = np.min(correlator_array) * (
            1 - middle_value_lowering_correction
        )

    upper_index_cut = lattice_size // 2 - 0 * 1
    shifted_backward_array = np.roll(correlator_array, shift=+1)
    shifted_backward_array = shifted_backward_array[1:upper_index_cut]
    shifted_forward_array = np.roll(correlator_array, shift=-1)
    shifted_forward_array = shifted_forward_array[1:upper_index_cut]

    numerator = shifted_backward_array + np.sqrt(
        np.square(shifted_backward_array) - middle_value_array**2
    )
    denominator = shifted_forward_array + np.sqrt(
        np.square(shifted_forward_array) - middle_value_array**2
    )

    return 0.5 * np.log(numerator / denominator)


def plateau_fit_function(x, p):

    return np.full(len(x), p)


def two_state_fit_function(x, p):

    x = np.array(x)
    ratio = (1 + p[1] * np.exp(-p[2] * x)) / (1 + p[1] * np.exp(-p[2] * (x + 1)))
    result = p[0] + np.log(ratio)
    return result


def symmetric_correlator_optimum_fit_range():
    pass


def optimum_range(xrange, effective_mass, fitting_function_name, fit_p0):

    temporal_direction_lattice_size = len(effective_mass)
    number_of_parameters = len(fit_p0)

    chi_square_dict = dict()
    minimum_value = 1000  # Arbitrary value
    min_key = tuple()
    fit_parameters = list()
    # for upper_index_cut in range(temporal_direction_lattice_size):
    upper_index_cut = temporal_direction_lattice_size - 1
    # Maintain a distance with upper index such that enough data points are used for fitting
    for lower_index_cut in range(upper_index_cut - (number_of_parameters - 1)):

        x = np.array(xrange[lower_index_cut:upper_index_cut])
        y = effective_mass[lower_index_cut:upper_index_cut]

        fit = lsqfit.nonlinear_fit(
            data=(x, y), p0=fit_p0, fcn=fitting_function_name, debug=True
        )

        if fit.dof == 0:
            continue

        key = (lower_index_cut, upper_index_cut)

        # print(key)

        # (fit.dof, fit.chi2)

        # Specify the degrees of freedom
        degrees_of_freedom = fit.dof

        # Specify the cumulative probability
        cumulative_probability = 0.95

        chi_square_value = stats.chi2.ppf(cumulative_probability, degrees_of_freedom)

        # chi_square_dict[key] = 1 - fit.Q

        p_value = 1 - fit.Q

        if p_value < minimum_value:
            minimum_value = p_value
            min_key = key
            fit_parameters = fit

        # print(fit.chi2, chi_square_value)

    # min_value = min(chi_square_dict.values())
    # min_key = [key for key, value in chi_square_dict.items() if value == min_value][0]

    return min_key, fit_parameters
