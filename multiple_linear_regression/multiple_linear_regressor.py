import numpy as np
from utils import solve_upper_triangular
from linear_regression.exceptions import RegressionModelNotFitError


class MultipleLinearRegressor:
    """Performs multiple linear regression."""

    def __init__(self):
        self.beta = None

    def predict(self, x):
        """
        Gives predicted y-values from a given array of x-values.
        :param x: Vector or matrix of x-values.
        :return: Vector of predicted y-values.
        """

        if self.beta is None:
            raise RegressionModelNotFitError('Oh no! The model has not been fit!')

        return x @ self.beta

    def fit(self, x, y):
        """
        Fits the model based on a matrix of x-values and vector of corresponding y-values.
        :param x: Matrix of x-values.
        :param y: Vector of y-values.
        :return: The sum-of-squares error of the fitted model.
        """

        x_t = x.transpose()
        q, r = np.linalg.qr(x_t @ x)
        vec = np.linalg.inv(q) @ x_t @ y
        self.beta = solve_upper_triangular(r, vec)
        diff = self.predict(x) - y
        return diff @ diff
