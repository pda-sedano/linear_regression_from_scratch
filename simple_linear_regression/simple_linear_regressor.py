from linear_regression.exceptions import RegressionModelNotFitError


class SimpleLinearRegressor:
    """Performs simple linear regression."""

    def __init__(self):
        self.k = None

    def predict(self, x):
        """
        Gives predicted y-values from an input x-value, or vector of x-values.
        :param x: The input value(s).
        :return: The predicted y-value(s).
        """

        if self.k is None:
            raise RegressionModelNotFitError('Oh no! The model has not been fit!')

        return self.k * x

    def fit(self, x, y):
        """
        Fits the model based on the given vectors of x and y values.
        :param x: A vector of x-values.
        :param y: A vector of y-values.
        :return: The sum-of-squares error of the fitted model.
        """

        self.k = x @ y / (x @ x)
        diff = self.predict(x) - y
        return diff @ diff
