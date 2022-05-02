import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from src import bootstrapped_plot, bootstrapped_animation


def make_plot_demo_lin_reg(data, ax, xlim=(-10, 10), ylim=(-10, 10)):
    lr = LinearRegression()
    lr.fit(data[:, 0].reshape(-1, 1), data[:, 1])
    xs = np.linspace(xlim[0], xlim[1], 10)
    ax.plot(xs, lr.predict(xs.reshape(-1, 1)), c='r')
    # ax.scatter(data[:, 0], data[:, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # ax.axis('off')


def make_plot_demo_lin_reg_with_resample(data, ax, xlim=(-10, 10), ylim=(-10, 10)):
    lr = LinearRegression()
    resample_mask = np.random.randint(0, len(data), size=len(data))
    lr.fit(data[resample_mask, 0].reshape(-1, 1), data[resample_mask, 1])
    xs = np.linspace(xlim[0], xlim[1], 10)
    ax.plot(xs, lr.predict(xs.reshape(-1, 1)), c='r')
    ax.scatter(data[:, 0], data[:, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


if __name__ == '__main__':
    np.random.seed(0)

    plt.rcParams["figure.figsize"] = (5, 5)

    dataset = np.random.randn(100, 2)
    noise = np.random.randn(len(dataset)) * 2.5
    dataset[:, 1] = dataset[:, 0] * 1.5 + 2 + noise

    bootstrapped_animation(make_plot_demo_lin_reg, dataset, m=400, out_file='bootstrapped_lin_reg.gif')
    mat = bootstrapped_plot(make_plot_demo_lin_reg, dataset, m=100, out_file='bootstrapped_lin_reg.png')

    plt.figure()
    plt.matshow(mat)
    plt.axis('off')
    plt.show()

    # Same thing, but plotting all points
    bootstrapped_animation(make_plot_demo_lin_reg_with_resample, dataset, m=400, out_file='bootstrapped_lin_reg_v2.gif',
                           resample_in_advance=False)
    mat = bootstrapped_plot(make_plot_demo_lin_reg_with_resample, dataset, m=100,
                            out_file='bootstrapped_lin_reg_v2.png',
                            resample_in_advance=False)

    plt.figure()
    plt.matshow(mat)
    plt.axis('off')
    plt.show()
