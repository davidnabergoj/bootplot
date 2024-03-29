import numpy as np
from sklearn.linear_model import LinearRegression

from bootplot import bootplot


def make_linear_regression(data_subset, data_full, ax):
    # Plot full dataset
    ax.scatter(data_full[:, 0], data_full[:, 1])

    # Plot regression line trained on the subset
    lr = LinearRegression()
    lr.fit(data_subset[:, 0].reshape(-1, 1), data_subset[:, 1])
    xs = np.linspace(-10, 10, 1000)
    ax.plot(xs, lr.predict(xs.reshape(-1, 1)), c='r')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)


if __name__ == '__main__':
    np.random.seed(0)

    dataset = np.random.randn(100, 2)
    noise = np.random.randn(len(dataset)) * 2.5
    dataset[:, 1] = dataset[:, 0] * 1.5 + 2 + noise

    bootplot(
        make_linear_regression,
        dataset,
        m=100,
        output_image_path='bootstrapped_linear_regression.png',
        output_animation_path='bootstrapped_linear_regression.gif',
        verbose=True
    )
