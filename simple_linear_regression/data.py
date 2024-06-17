import numpy as np


def generate_noisy_data(n_points, slope, x_range, noise_stddev):
    """
    :param n_points: The number of data points to generate
    :param slope: The slope of the line
    :param x_range: The range from which to draw x-values
    :param noise_stddev: The standard deviation of the Gaussian noise to add to each y-value
    :return: Vectors of x and y values
    """

    x = np.random.uniform(*x_range, n_points)
    y = slope * x + np.random.normal(scale=noise_stddev, size=n_points)
    return x, y
