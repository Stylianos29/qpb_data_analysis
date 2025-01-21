import numpy as np


def symmetrization(values_list):
    reverse = values_list[::-1]

    return 0.5 * (values_list + np.roll(reverse, shift=+1))


def single_state_non_periodic_correlator(t, p):
    """Implements function: C(t) = A*exp(m*t). Parameters A and m are passed using the array parameters array p: A = p[0] and m = p[1]."""

    return p[0] * np.exp(-p[1] * t)


def amplitude_of_single_state_non_periodic_correlator(
    correlator_array, effective_mass, time_index
):
    """Calculates the m parameter of the single-state non-period correlator function: A = C(t)*exp(m*t)."""

    return correlator_array[time_index] * np.exp(effective_mass * time_index)


def single_state_periodic_correlator(t, p):
    """Implements function: C(t) = A*[exp(-m*t) + exp(-m*(T-t))].
    Parameters A and m are passed with the array p: A = p[0] and m = p[1].
    TODO: What about the third parameter?"""
    T = 48
    return p[0] * (np.exp(-p[1] * t) + np.exp(-p[1] * (T - t)))


def single_state_doubly_periodic_correlator(t, p):
    """Implements function: C(t) = A*[exp(-m*t) + exp(-m*(T-t)) + exp(-m*(T+t)) + exp(-m*(2T-t))].
    Parameters A and m are passed with the array p: A = p[0] and m = p[1].
    TODO: What about the third parameter?"""
    T = 48
    return p[0] * (
        np.exp(-p[1] * t)
        + np.exp(-p[1] * (T - t))
        + np.exp(-p[1] * (T + t))
        + np.exp(-p[1] * (2 * T - t))
    )


def two_state_periodic_correlator(t, p):
    """Implements function: C(t) = A*[exp(-m*t) + B*exp(-c*t) + B*exp(-c*(T-t)) + exp(-m*(T-t))].
    Parameters A and m are passed with the array p: A = p[0], B = p[2], m = p[1], and c = p[3].
    TODO: What about the third parameter?"""
    T = 48
    # return p[0]*(np.exp(-p[1]*t -p[2]*t*t) + np.exp(-p[1]*(T-t) -p[2]*(T-t)*(T-t)))
    return p[0] * (
        np.exp(-p[1] * t)
        + p[2] * np.exp(-p[3] * t)
        + np.exp(-p[1] * (T - t))
        + p[2] * np.exp(-p[3] * (T - t))
    )


def centered_difference_correlator_derivative(correlator_values_array):
    """Implements centered difference approximation of the derivative expression
    with Dx^4 error:
    f'(x) = [-f(x+2Δx) + 8f(x+Δx) - 8f(x-Δx) + f(x-2Δx)]/(12Δx) + O(Δx^4) .
    """

    return -(1 / 12) * (
        -np.roll(correlator_values_array, shift=-2)
        + 8 * np.roll(correlator_values_array, shift=-1)
        - 8 * np.roll(correlator_values_array, shift=+1)
        + np.roll(correlator_values_array, shift=+2)
    )
