import numpy as np
from bootplot import bootplot


def make_pie_chart(data_subset, data_full, ax):
    # Get info from the full dataset
    n_classes = np.max(data_full) + 1

    # Plot subset
    ticks, counts = np.unique(data_subset, return_counts=True)
    ticks_adjusted = np.zeros(n_classes)
    ticks_adjusted[ticks] = counts
    wedges, _ = ax.pie(x=ticks_adjusted, explode=0.1 * np.ones(n_classes))

    ax.legend(
        wedges,
        list(range(n_classes)),
        loc='upper right',
        bbox_to_anchor=(1.1, 1.1, 0, 0),
        shadow=True,
        title='Class',
        prop={'family': 'monospace'}
    )

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)


if __name__ == '__main__':
    np.random.seed(0)

    dataset = np.random.binomial(7, 0.2, size=1000)

    bootplot(
        make_pie_chart,
        dataset,
        m=100,
        output_image_path='bootstrapped_pie_chart.png',
        output_animation_path='bootstrapped_pie_chart.gif',
        verbose=True
    )
