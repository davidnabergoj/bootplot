import numpy as np
from sklearn.linear_model import LinearRegression
from src import bootstrapped_plot


def make_plot_demo_text(data_subset, data_full, ax):
    ax.scatter(data_full[:, 0], data_full[:, 1])

    lr = LinearRegression()
    lr.fit(data_subset[:, 0].reshape(-1, 1), data_subset[:, 1])
    xs = np.linspace(-10, 10, 1000)
    ax.text(
        0, -8,
        f'Target mean: {float(np.mean(data_subset[:, 1])):.4f}',
        fontsize=12,
        ha='center',
        bbox=dict(facecolor='none', edgecolor='black', pad=10.0)
    )
    ax.plot(xs, lr.predict(xs.reshape(-1, 1)), c='r')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)


if __name__ == '__main__':
    np.random.seed(0)

    dataset = np.random.randn(100, 2)
    noise = np.random.randn(len(dataset)) * 2.5
    dataset[:, 1] = (dataset[:, 0] * 1.5 + noise) / 5 + 2.551

    bootstrapped_plot(
        make_plot_demo_text,
        dataset,
        m=100,
        output_image_path='bootstrapped_text.png',
        output_animation_path='bootstrapped_text.gif',
        verbose=True
    )
