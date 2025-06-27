import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import pandas as pd

from bootplot import bootplot


def make_regression(data_subset, data_full, ax):
    # Plot full dataset
    ax.scatter(data_full[:, 0], data_full[:, 1], alpha=0.2, label='Full dataset')

    # Plot regression models trained on the subset
    lr = LinearRegression()
    lr.fit(data_subset[:, 0].reshape(-1, 1), data_subset[:, 1])
    svr = SVR()
    svr.fit(data_subset[:, 0].reshape(-1, 1), data_subset[:, 1])

    xs = np.linspace(-10, 10, 1000)
    ax.plot(xs, svr.predict(xs.reshape(-1, 1)), c='tab:orange', linestyle='dashed', label='SVR')
    ax.plot(xs, lr.predict(xs.reshape(-1, 1)), c='tab:green', label='Linear regression')

    ax.legend(loc='lower right', shadow=True)
    lr_rmse = np.sqrt(np.mean(np.square(data_subset[:, 1] - lr.predict(data_subset[:, 0].reshape(-1, 1)))))
    svr_rmse = np.sqrt(np.mean(np.square(data_subset[:, 1] - svr.predict(data_subset[:, 0].reshape(-1, 1)))))
    ax.text(
        -3.6, 2.6,
        f'RMSE (LR): {lr_rmse:.4f}\nRMSE (SVR): {svr_rmse:.4f}',
        ha='left',
        va='top',
        bbox=dict(facecolor='none', edgecolor='black', pad=6),
        family='monospace'
    )

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 3)


if __name__ == '__main__':
    np.random.seed(0)

    n = 40
    x = np.random.uniform(low=-1.7, high=1.5, size=n)
    noise = np.random.randn(n) * 0.05
    y = x ** 4 - 2 * x ** 2 + x + noise
    dataset = np.c_[x, y]

    bootplot(
        make_regression,
        dataset,
        m=100,
        output_image_path='bootstrapped_regression_multiple_models.png',
        output_animation_path='bootstrapped_regression_multiple_models.gif',
        verbose=True
    )
