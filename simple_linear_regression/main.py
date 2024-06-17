import numpy as np
from matplotlib import pyplot as plt
from data import generate_noisy_data
from simple_linear_regressor import SimpleLinearRegressor


def main():
    x_range = np.array([0, 5])
    x, y = generate_noisy_data(n_points=20, slope=0.42, x_range=x_range, noise_stddev=0.5)
    plt.scatter(x, y)

    regressor = SimpleLinearRegressor()
    fit = regressor.fit(x, y)
    slope = regressor.k
    plt.plot(x_range, regressor.predict(x_range), color='red')
    plt.text(3, 0, f'Error: {"{:.2f}".format(fit)}\nPredicted slope: {"{:.2f}".format(slope)}')
    plt.show()


if __name__ == '__main__':
    main()
