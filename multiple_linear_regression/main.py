import numpy as np
from utils import generate_noisy_data
from multiple_linear_regressor import MultipleLinearRegressor


def main():
    regressor = MultipleLinearRegressor()
    x, y = generate_noisy_data(n_data_points=500,
                               n_independent_variables=10,
                               beta=np.array([-10, 5, -8, -2, 1, -3, 4, -5, -1, 3]),
                               x_range=np.array([-100, 100]),
                               noise_stddev=50)
    sse = regressor.fit(x, y)
    print(f'Sum Squared Error: {sse}')
    print(f'Beta: {regressor.beta}')


if __name__ == '__main__':
    main()
