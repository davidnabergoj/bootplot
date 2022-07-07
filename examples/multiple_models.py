import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import pandas as pd

from bootplot import bootplot


def make_regression(data_subset, data_full, ax):
    # Plot full dataset
    ax.scatter(data_full[:, 0], data_full[:, 1], alpha=0.2, label='Full dataset')

    # Plot regression line trained on the subset
    lr = LinearRegression()
    lr.fit(data_subset[:, 0].reshape(-1, 1), data_subset[:, 1])
    svr = SVR()
    svr.fit(data_subset[:, 0].reshape(-1, 1), data_subset[:, 1])

    xs = np.linspace(-10, 10, 1000)
    ax.plot(xs, svr.predict(xs.reshape(-1, 1)), c='tab:orange', linestyle='dashed', label='SVR')
    ax.plot(xs, lr.predict(xs.reshape(-1, 1)), c='tab:green', label='Linear regression')

    ax.legend(loc='lower right', shadow=True)


if __name__ == '__main__':
    np.random.seed(0)

    df = pd.DataFrame()

    dataset = np.random.randn(100, 2)
    noise = np.random.randn(len(dataset)) * 2.5
    dataset[:, 1] = dataset[:, 0] * 1.5 + 2 + noise

    bootplot(
        make_regression,
        dataset,
        m=100,
        contrast_modifier=3.0,
        output_image_path='bootstrapped_regression_multiple_models.png',
        output_animation_path='bootstrapped_regression_multiple_models.gif',
        xlim=(-10, 10),
        ylim=(-10, 10),
        verbose=True
    )
