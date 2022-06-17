import numpy as np
from bootplot import bootplot


def make_plot_demo_point_plot(data_subset, data_full, ax):
    # Plot subset
    ax.scatter(np.mean(data_subset), 0, s=100)


if __name__ == '__main__':
    np.random.seed(0)

    n_observations = 1000
    dataset = np.random.uniform(low=-5.5, high=5.5, size=(n_observations,))

    mat = bootplot(
        make_plot_demo_point_plot,
        dataset,
        m=100,
        output_image_path='bootstrapped_single_point_plot.png',
        output_animation_path='bootstrapped_single_point_plot.gif',
        sort_type="pca",
        xlim=(-3, 3),
        verbose=True
    )
