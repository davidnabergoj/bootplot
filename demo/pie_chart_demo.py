import numpy as np
import matplotlib.pyplot as plt

from src import bootstrapped_plot, bootstrapped_animation


def make_plot_demo_pie(data, ax, n_classes=6):
    bar_ticks, bar_counts = np.unique(data, return_counts=True)
    bar_ticks_adjusted = np.zeros(n_classes)
    for bt, bc in zip(bar_ticks, bar_counts):
        bar_ticks_adjusted[bt] = bc
    wedges, _ = ax.pie(x=bar_ticks_adjusted, explode=0.2 * np.ones(n_classes))
    ax.legend(wedges, list(range(n_classes)), loc='upper right', bbox_to_anchor=(1.1, 1.1, 0, 0), shadow=True,
              title='Class')


if __name__ == '__main__':
    np.random.seed(0)

    plt.rcParams["figure.figsize"] = (5, 5)

    dataset = np.random.binomial(5, 0.5, size=200)
    mat = bootstrapped_plot(make_plot_demo_pie, dataset, m=100, out_file='bootstrapped_pie_chart.png')

    plt.figure()
    plt.matshow(mat)
    plt.axis('off')
    plt.show()

    bootstrapped_animation(make_plot_demo_pie, dataset, m=100, out_file='bootstrapped_pie_chart.gif')
