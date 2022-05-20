import numpy as np
import matplotlib.pyplot as plt

from src import bootstrapped_plot


def make_plot_demo_point_plot(data_subset, data_full, ax):
    ax.scatter(np.mean(data_subset), 0, s=100)
    ax.set_xlim(-3, 3)


if __name__ == '__main__':
    np.random.seed(0)

    n_observations = 1000
    dataset = np.random.uniform(low=-5.5, high=5.5, size=(n_observations,))

    mat = bootstrapped_plot(
        make_plot_demo_point_plot,
        dataset,
        m=100,
        output_image_path='bootstrapped_single_point_plot.png',
        output_animation_path='bootstrapped_single_point_plot.gif',
        sort_type="pca",
        verbose=True
    )

    plt.figure()
    plt.matshow(mat)
    plt.axis('off')
    plt.show()
