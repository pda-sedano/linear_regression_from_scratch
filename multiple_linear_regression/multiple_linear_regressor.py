import numpy as np
from utils import solve_upper_triangular
from exceptions import RegressionModelNotFitError


class MultipleLinearRegressor:
    """Performs multiple linear regression."""

    def __init__(self):
        self.beta = None

    def predict(self, x):
        """
        Gives a predicted set of values from a given set of input values
        :param x: 1D or 2D array of values
        :return: 1D array of predictions
        """

        if self.beta is None:
            raise RegressionModelNotFitError('The model has not been fit!')

        return x @ self.beta

    def fit(self, x, y):
        """
        Fits the model to some data
        :param x: 2D array of input values
        :param y: 1D array of predictions
        :return: The MSE of the model
        """

        x_t = x.transpose()

        q, r = np.linalg.qr(x_t @ x)
        vec = np.linalg.inv(q) @ x_t @ y

        self.beta = solve_upper_triangular(r, vec)

        diff = self.predict(x) - y
        return diff @ diff
