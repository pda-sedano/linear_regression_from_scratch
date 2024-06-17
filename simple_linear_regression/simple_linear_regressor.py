class SimpleLinearRegressor:
    """Performs simple linear regression."""

    def __init__(self):
        self.m = None

    def predict(self, x):
        """Gives a predicted set of values from a given set of input values.\n
        x - Number or array."""

        if self.m is None:
            raise TypeError('Regression model not fit!')

        return self.m * x

    def fit(self, x, y):
        """Fits the model and returns the sum-of-squares error of the fit.\n
        x - Array of x values.\n
        y - Array of y values."""

        self.m = x @ y / (x @ x)
        diff = self.predict(x) - y
        return diff @ diff
