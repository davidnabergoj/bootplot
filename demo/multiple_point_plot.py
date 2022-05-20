import numpy as np
import matplotlib.pyplot as plt

from src import bootstrapped_plot


def make_point_plot(data_subset, data_full, ax):
    n_classes = len(np.unique(data_full[:, 0]))

    for i in range(n_classes):
        if i in data_subset[:, 0]:
            mask = data_subset[:, 0] == i
            ax.scatter(np.mean(data_subset[mask, 1]), i)

    ax.set_xlim(-n_classes, n_classes)
    ax.set_ylim(-1, n_classes)


if __name__ == '__main__':
    np.random.seed(0)

    n_observations = 300
    n_classes = 7
    dataset = np.zeros((n_observations, 2), dtype=np.float32)
    for i in range(n_observations):
        j = float(np.random.randint(n_classes))
        dataset[i] = [j, np.random.randn() * j]

    mat = bootstrapped_plot(
        make_point_plot,
        dataset,
        m=100,
        output_image_path='bootstrapped_multiple_point_plot.png',
        output_animation_path='bootstrapped_multiple_point_plot.gif',
        sort_type="pca",
        verbose=True
    )

    plt.figure()
    plt.matshow(mat)
    plt.axis('off')
    plt.show()
