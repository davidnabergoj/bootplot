import numpy as np
import matplotlib.pyplot as plt

from src import bootstrapped_plot, bootstrapped_animation


def make_plot_demo_bar(data, ax, ylim=(0, 100), n_classes=6):
    bar_ticks, bar_counts = np.unique(data, return_counts=True)
    bar_ticks_adjusted = np.arange(n_classes)
    bar_counts_adjusted = np.zeros_like(bar_ticks_adjusted, dtype=np.float32)
    bar_counts_adjusted[bar_ticks] = bar_counts
    ax.bar(bar_ticks_adjusted, bar_counts_adjusted)
    ax.set_ylim(*ylim)
    # ax.axis('off')


if __name__ == '__main__':
    np.random.seed(0)

    plt.rcParams["figure.figsize"] = (5, 5)

    dataset = np.random.binomial(5, 0.5, size=200)
    mat = bootstrapped_plot(make_plot_demo_bar, dataset, m=100, out_file='bootstrapped_bar_chart.png')

    plt.figure()
    plt.matshow(mat)
    plt.axis('off')
    plt.show()

    bootstrapped_animation(make_plot_demo_bar, dataset, m=100, out_file='bootstrapped_bar_chart.gif')
