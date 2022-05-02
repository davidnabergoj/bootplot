import numpy as np
import matplotlib.pyplot as plt

from src import bootstrapped_plot, bootstrapped_animation


def make_plot_demo_point_plot(data, ax):
    plt.scatter(np.mean(data.ravel()), 0)
    # plt.scatter(data.ravel()[0], 0)
    ax.set_xlim(-3, 3)
    # ax.axis('off')


if __name__ == '__main__':
    np.random.seed(0)

    plt.rcParams["figure.figsize"] = (5, 5)
    plt.rcParams["lines.markersize"] = 15

    n_observations = 1000
    dataset = np.random.uniform(low=-5.5, high=5.5, size=(n_observations, 1))

    bootstrapped_animation(make_plot_demo_point_plot, dataset, m=1000, out_file='bootstrapped_point_plot_single.gif', sort_type="pca")
    mat = bootstrapped_plot(make_plot_demo_point_plot, dataset, m=100, out_file='bootstrapped_point_plot_single.png')

    plt.figure()
    plt.matshow(mat)
    plt.axis('off')
    plt.show()
