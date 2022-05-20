import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from src import bootstrapped_plot


def make_polynomial_regression(data_subset, data_full, ax):
    # Plot all points
    ax.scatter(data_full[:, 0], data_full[:, 1])

    # Plot regression model on bootstrapped subset
    poly = PolynomialFeatures(degree=2)
    features = poly.fit_transform(data_subset[:, 0].reshape(-1, 1))
    lr = LinearRegression()
    lr.fit(features, data_subset[:, 1])

    xs = np.linspace(-10, 10, 1000)
    xs_features = poly.transform(xs.reshape(-1, 1))
    ax.plot(xs, lr.predict(xs_features), c='r')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)


if __name__ == '__main__':
    np.random.seed(0)

    dataset = np.random.randn(100, 2)
    noise = np.random.randn(len(dataset)) * 2.5
    dataset[:, 1] = dataset[:, 0] * 1.5 + 2 + noise

    mat = bootstrapped_plot(
        make_polynomial_regression,
        dataset,
        m=100,
        output_image_path='bootstrapped_polynomial_regression.png',
        output_animation_path='bootstrapped_polynomial_regression.gif'
    )

    plt.figure()
    plt.matshow(mat)
    plt.axis('off')
    plt.show()
