import numpy as np
from utils import generate_noisy_data
from multiple_linear_regressor import MultipleLinearRegressor


def main():
    regressor = MultipleLinearRegressor()
    x, y = generate_noisy_data(100, 3, np.array([5, 1, 3]), np.array([0, 100]), 50)
    regressor.fit(x, y)
    print(f'lstsq = {np.linalg.lstsq(x, y)}')

    for datum in x:
        print(regressor.predict(datum))


if __name__ == '__main__':
    main()
