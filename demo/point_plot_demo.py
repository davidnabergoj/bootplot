import numpy as np
import matplotlib.pyplot as plt

from src import bootstrapped_plot, bootstrapped_animation

N_CLASSES = 7


def make_plot_demo_point_plot(data, ax, n_classes=N_CLASSES):
    for i in range(n_classes):
        mask = data[:, 0] == i
        plt.scatter(np.mean(data[mask, 1]), i)
    ax.set_xlim(-N_CLASSES, N_CLASSES)
    ax.set_ylim(-N_CLASSES + N_CLASSES / 2, N_CLASSES + N_CLASSES / 2)
    # ax.axis('off')


if __name__ == '__main__':
    np.random.seed(0)

    plt.rcParams["figure.figsize"] = (5, 5)

    n_observations = 300
    n_classes = N_CLASSES
    dataset = []
    for i in range(n_observations):
        for j in range(n_classes):
            # dataset.append([j, np.tanh(j - 0.5 + np.random.randn() * 10.25) + np.random.randn() * 2])
            dataset.append([j, np.random.randn() * j ** .75 + (j - N_CLASSES // 2 - 1) ** 1.5 - 0.6])
    dataset = np.array(dataset)

    bootstrapped_animation(make_plot_demo_point_plot, dataset, m=1000, out_file='bootstrapped_point_plot.gif',
                           sort_type="pca", animation_duration=100)
    mat = bootstrapped_plot(make_plot_demo_point_plot, dataset, m=100, out_file='bootstrapped_point_plot.png')

    plt.figure()
    plt.matshow(mat)
    plt.axis('off')
    plt.show()
