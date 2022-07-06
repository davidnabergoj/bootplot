import numpy as np
from bootplot import bootplot


def make_bar_chart(data_subset, data_full, ax):
    n_classes = 20

    # Plot subset
    bar_ticks, bar_counts = np.unique(data_subset, return_counts=True)
    bar_ticks_adjusted = np.arange(n_classes)
    bar_counts_adjusted = np.zeros_like(bar_ticks_adjusted, dtype=np.float32)
    bar_counts_adjusted[bar_ticks] = bar_counts
    ax.bar(bar_ticks_adjusted, bar_counts_adjusted)


if __name__ == '__main__':
    np.random.seed(0)

    dataset = np.random.binomial(20, 0.3, size=1000)
    bootplot(
        make_bar_chart,
        dataset,
        m=100,
        output_image_path='bootstrapped_bar_chart.png',
        output_animation_path='bootstrapped_bar_chart.mp4',
        verbose=True,
        ylim=(0, 300)
    )
