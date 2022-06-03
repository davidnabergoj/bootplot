import numpy as np
from src.bootplot.base import bootplot


def make_point_plot(data_subset, data_full, ax):
    # Get info from the full dataset
    n_classes = len(np.unique(data_full[:, 0]))

    # Plot subset
    for i in range(n_classes):
        if i in data_subset[:, 0]:
            mask = data_subset[:, 0] == i
            ax.scatter(np.mean(data_subset[mask, 1]), i)

    # Define global axis settings
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

    bootplot(
        make_point_plot,
        dataset,
        m=100,
        output_image_path='bootstrapped_multiple_point_plot.png',
        output_animation_path='bootstrapped_multiple_point_plot.gif',
        sort_type="pca",
        verbose=True
    )
