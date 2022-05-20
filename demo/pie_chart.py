import numpy as np
import matplotlib.pyplot as plt

from src import bootstrapped_plot


def make_pie_chart(data_subset, data_full, ax):
    n_classes = len(np.unique(data_full))

    ticks, counts = np.unique(data_subset, return_counts=True)
    ticks_adjusted = np.zeros(n_classes)
    ticks_adjusted[ticks] = counts

    wedges, _ = ax.pie(x=ticks_adjusted, explode=0.1 * np.ones(n_classes))

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.legend(
        wedges,
        list(range(n_classes)),
        loc='upper right',
        bbox_to_anchor=(1.1, 1.1, 0, 0),
        shadow=True,
        title='Class'
    )


if __name__ == '__main__':
    np.random.seed(0)

    dataset = np.random.binomial(7, 0.2, size=1000)

    mat = bootstrapped_plot(
        make_pie_chart,
        dataset,
        m=100,
        output_image_path='bootstrapped_pie_chart.png',
        output_animation_path='bootstrapped_pie_chart.gif',
        verbose=True
    )

    plt.figure()
    plt.matshow(mat)
    plt.axis('off')
    plt.show()
