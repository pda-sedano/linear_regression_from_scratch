import numpy as np
from linear_regression.exceptions import UnsolvableError


def generate_noisy_data(n_data_points, n_independent_variables, beta, x_range, noise_stddev):
    """
    Generates 2d data points based on a linear relationship with added Gaussian noise.
    :param n_data_points: The number of data points to generate.
    :param n_independent_variables: The number of independent variables in each data point.
    :param beta: The vector of coefficients for the independent variables.
    :param x_range: The range from which to draw x-values.
    :param noise_stddev: The standard deviation of the Gaussian noise to add to each y-value.
    :return: Vectors of x and y values.
    """

    x = np.random.uniform(*x_range, (n_data_points, n_independent_variables))
    y = x @ beta + np.random.normal(scale=noise_stddev, size=n_data_points)
    return x, y


def solve_upper_triangular(a, b):
    """
    Solves the linear equation ax = b for x.
    :param a: An n x n upper triangular matrix.
    :param b: An n-dimensional vector.
    :return: A vector x for which ax = b.
    """

    tracker = np.zeros(a.shape[1])
    result = np.zeros(a.shape[1])

    for row, val in zip(a[::-1], b[::-1]):
        unset_var_indices = np.where((tracker == 0) & (row != 0))[0]

        # What to do if there are no unset vars?
        if len(unset_var_indices) == 0:
            if np.isclose(result @ row, val, rtol=0):
                break
            else:
                raise UnsolvableError('The given values of a and b result in an unsolvable equation.')

        tracker[unset_var_indices] = 1
        result[unset_var_indices[1:]] = 1
        i = unset_var_indices[0]
        result[i] = (val - result @ row) / row[i]

    return result
