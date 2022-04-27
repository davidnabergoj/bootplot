import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from src import bootstrapped_plot


def make_plot_demo_text(data, ax, xlim=(-10, 10), ylim=(-10, 10)):
    lr = LinearRegression()
    lr.fit(data[:, 0].reshape(-1, 1), data[:, 1])
    xs = np.linspace(xlim[0], xlim[1], 10)
    ax.text(
        0, -8,
        f'Target mean: {float(np.mean(data[:, 1])):.4f}', fontsize=12, ha='center',
        bbox=dict(facecolor='none', edgecolor='black', pad=10.0)
    )
    ax.plot(xs, lr.predict(xs.reshape(-1, 1)), c='r')
    ax.scatter(data[:, 0], data[:, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # ax.axis('off')


if __name__ == '__main__':
    np.random.seed(0)

    plt.rcParams["figure.figsize"] = (5, 5)

    dataset = np.random.randn(100, 2)
    noise = np.random.randn(len(dataset)) * 2.5
    dataset[:, 1] = (dataset[:, 0] * 1.5 + noise) / 5 + 2.551
    mat = bootstrapped_plot(make_plot_demo_text, dataset, m=100, out_file='bootstrapped_text.png')

    plt.figure()
    plt.matshow(mat)
    plt.axis('off')
    plt.show()
