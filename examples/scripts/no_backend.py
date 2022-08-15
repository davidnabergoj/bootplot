import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from bootplot import bootplot
from bootplot.backend.matplotlib import plot_to_array, close_figure


# Suppose you want to do everything in the plot function
# Your plot function receives the data subset and the full data
# Your plot function should return a numpy.ndarray (image)


def make_linear_regression(data_subset, data_full):
    fig, ax = plt.subplots()

    # Plot full dataset
    ax.scatter(data_full[:, 0], data_full[:, 1])

    # Plot regression line trained on the subset
    lr = LinearRegression()
    lr.fit(data_subset[:, 0].reshape(-1, 1), data_subset[:, 1])
    xs = np.linspace(-10, 10, 1000)
    ax.plot(xs, lr.predict(xs.reshape(-1, 1)), c='r')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # We have some pre-made functions to handle converting plots to arrays and closing the plots.
    # If you are using a custom plotting library, you need to handle this step yourself. A straightforward way is to
    #  implement a custom backend using the bootplot.backend.base.Backend abstract class.
    image = plot_to_array(fig)
    close_figure(fig)

    return image


if __name__ == '__main__':
    np.random.seed(0)

    dataset = np.random.randn(100, 2)
    noise = np.random.randn(len(dataset)) * 2.5
    dataset[:, 1] = dataset[:, 0] * 1.5 + 2 + noise

    bootplot(
        make_linear_regression,
        dataset,
        m=100,
        output_image_path='no_backend.png',
        output_animation_path='no_backend.gif',
        verbose=True,
        backend='basic'
    )
