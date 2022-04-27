import numpy as np
import matplotlib.pyplot as plt

from src import bootstrapped_plot


def make_plot_demo_point_plot(data, ax, xlim=(-1.25, 1.25)):
    plt.scatter(data[:, 1], data[:, 0], alpha=0.1)
    ax.set_xlim(*xlim)
    # ax.axis('off')


if __name__ == '__main__':
    np.random.seed(0)

    plt.rcParams["figure.figsize"] = (5, 5)

    n_observations = 100
    n_classes = 4
    dataset = []
    for i in range(n_observations):
        for j in range(n_classes):
            dataset.append([j, np.tanh(j - 0.5 + np.random.randn() * 0.25)])
    dataset = np.array(dataset)
    mat = bootstrapped_plot(make_plot_demo_point_plot, dataset, m=100, out_file='bootstrapped_point_plot.png')

    plt.figure()
    plt.matshow(mat)
    plt.axis('off')
    plt.show()
